#!/usr/bin/env python3
"""CS781 Capstone, Phase 4: Court Judgment Prediction, CJPE (Darsh Kedia, 250311).

References:
  [1] Malik et al. 2021, ILDC for CJPE (ACL 2021)
  [2] Chalkidis et al. 2019, Neural Legal Judgment Prediction in English
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_core as C

PAD = 0
NC = 2


def find(data, name):
    for p in (os.path.join(data, name), os.path.join(data, "Train", name), os.path.join(data, "Test", name)):
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(data, "**", name), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} under {data}")


def load_raw(data, need_train=True):
    test = [{"id": str(r["id"]), "text": r["text"]} for r in json.load(open(find(data, "test.json"), encoding="utf-8"))]
    tids = {r["id"] for r in test}
    dev_all = json.load(open(find(data, "dev.json"), encoding="utf-8"))
    dev, seen = [], set()
    for r in dev_all:
        if str(r["id"]) in tids or r["text"] in seen:
            continue
        seen.add(r["text"])
        dev.append({"id": str(r["id"]), "text": r["text"], "label": int(r["label"])})
    train = []
    if need_train:
        seen = set()
        for r in json.load(open(find(data, "train.json"), encoding="utf-8")):
            if r["text"] in seen or str(r["id"]) in tids:
                continue
            seen.add(r["text"])
            train.append({"id": str(r["id"]), "text": r["text"], "label": int(r["label"])})
        cnt = Counter(r["id"] for r in train)
        k = Counter()
        for r in train:
            if cnt[r["id"]] > 1:
                k[r["id"]] += 1
                r["id"] = f"{r['id']}#{k[r['id']]}"
    info = {"train": len(train), "dev_clean": len(dev), "dev_removed_as_test": sum(str(r["id"]) in tids for r in dev_all),
            "test": len(test)}
    return {"train": train, "dev": dev, "test": test}, info


_WORD = re.compile(r"[a-z]+")
_BREAK = re.compile(r"([a-z]+)\n([a-z]+)")


def word_counts(texts):
    c = Counter()
    for t in texts:
        c.update(_WORD.findall(t.lower()))
    return c


def clean_text(t, counts):
    def fix(m):
        a, b = m.group(1), m.group(2)
        ab = (a + b).lower()
        return a + b if counts.get(ab, 0) > min(counts.get(a.lower(), 0), counts.get(b.lower(), 0)) else a + " " + b
    t = _BREAK.sub(fix, t)
    t = t.replace("\n", " ")
    return re.sub(r"[ \t]+", " ", t).strip()


def prep(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    data, info = load_raw(args.data, need_train=not args.no_train)
    counts = word_counts(r["text"] for r in data["dev"] + data["test"])
    out = {"info": info, "keep_head": args.keep_head, "keep_tail": args.keep_tail,
           "tokenizer": os.path.abspath(args.tokenizer)}
    t0 = time.time()
    for split, docs in data.items():
        recs = {}
        for k in range(0, len(docs), 512):
            part = docs[k:k + 512]
            encs = tok.encode_batch([clean_text(d["text"], counts) for d in part], add_special_tokens=False)
            for d, e in zip(part, encs):
                ids = e.ids
                recs[d["id"]] = {"head": torch.tensor(ids[:args.keep_head], dtype=torch.int16),
                                 "tail": torch.tensor(ids[-args.keep_tail:], dtype=torch.int16),
                                 "n_tok": len(ids), "label": d.get("label")}
        out[split] = recs
        print(f"prep {split}: {len(recs)} docs, {time.time() - t0:.0f}s", flush=True)
    lens = sorted(r["n_tok"] for r in out["dev"].values())
    out["info"]["dev_median_tokens"] = lens[len(lens) // 2]
    torch.save(out, args.cache)
    print(json.dumps(out["info"]))


def chunks_of(rec, chunk_len, n_chunks, head_chunks):
    tail = rec["tail"].tolist()[-(chunk_len * n_chunks):]
    n_tok = rec["n_tok"]
    tail_chunks = []
    end = len(tail)
    while end > 0 and len(tail_chunks) < n_chunks:
        s = max(0, end - chunk_len)
        tail_chunks.append(tail[s:end])
        end = s
    tail_chunks = tail_chunks[::-1]
    pos = list(range(len(tail_chunks) - 1, -1, -1))
    heads = []
    covered = len(tail)
    if head_chunks and n_tok > covered:
        h = rec["head"].tolist()[:min(head_chunks * chunk_len, n_tok - covered)]
        heads = [h[i:i + chunk_len] for i in range(0, len(h), chunk_len)]
    return heads + tail_chunks, list(range(n_chunks, n_chunks + len(heads))) + pos


def collate(recs, ids, args, teacher=None):
    chunks, pos, owner, labels, tl = [], [], [], [], []
    for b, i in enumerate(ids):
        ch, ps = chunks_of(recs[i], args.chunk_len, args.n_chunks, args.head_chunks)
        chunks += ch
        pos += ps
        owner += [b] * len(ch)
        lab = recs[i]["label"]
        labels.append(-100 if lab is None else lab)
        if teacher is not None:
            tl.append(teacher[i])
    T = max(len(c) for c in chunks)
    x = torch.full((len(chunks), T), PAD, dtype=torch.long)
    valid = torch.zeros((len(chunks), T), dtype=torch.bool)
    for k, c in enumerate(chunks):
        x[k, :len(c)] = torch.tensor(c, dtype=torch.long)
        valid[k, :len(c)] = True
    out = {"ids": x, "valid": valid, "pos": torch.tensor(pos), "owner": torch.tensor(owner),
           "n_docs": len(ids), "labels": torch.tensor(labels)}
    if teacher is not None:
        out["teacher"] = torch.stack(tl)
    return out


class DocHead(nn.Module):

    def __init__(self, H, max_pos, hidden=384, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(max_pos, H)
        nn.init.zeros_(self.pos.weight)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(H, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.out = nn.Linear(2 * hidden, NC)

    def forward(self, v, pos, owner, n_docs):
        v = self.drop(v + self.pos(pos))
        counts = torch.bincount(owner, minlength=n_docs)
        K = int(counts.max())
        x = v.new_zeros(n_docs, K, v.shape[-1])
        slot = torch.cat([torch.arange(int(c), device=v.device) for c in counts])
        x[owner, slot] = v
        packed = nn.utils.rnn.pack_padded_sequence(x, counts.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=K)
        mask = torch.arange(K, device=v.device)[None, :] < counts[:, None]
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        return self.out(self.drop((a[..., None] * h).sum(1)))


class CJPEModel(nn.Module):
    def __init__(self, backbone, max_pos, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = DocHead(backbone.d_model, max_pos, dropout=dropout)

    def chunk_vectors(self, ids, valid):
        h = self.backbone(ids, valid).float()
        m = valid[..., None].to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, b, device):
        v = self.chunk_vectors(b["ids"].to(device), b["valid"].to(device))
        return self.head(v, b["pos"].to(device), b["owner"].to(device), b["n_docs"])


def build(args):
    bb = C.Backbone(args.ckpt, args.code_dir, keep_layers=args.keep_layers or None,
                    bidir=args.bidir, grad_ckpt=args.grad_ckpt)
    peft = C.apply_peft(bb, args.method, args.rank, args.alpha, tuple(args.targets.split(",")),
                        train_norms=args.train_norms)
    return CJPEModel(bb, args.n_chunks + args.head_chunks, args.dropout), peft


def macro_f1(gold, pred):
    f = []
    for c in (0, 1):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / 2


def margins(logp, ids):
    return torch.stack([logp[i][1] - logp[i][0] for i in ids])


def score_logp(recs, logp, thr=0.0):
    ids = [i for i in recs if i in logp]
    pred = (margins(logp, ids) > thr).long().tolist()
    return macro_f1([recs[i]["label"] for i in ids], pred)


def tune_thr(recs, logp):
    ids = [i for i in recs if i in logp]
    m = margins(logp, ids)
    cands = sorted(set([0.0] + torch.quantile(m, torch.linspace(0.2, 0.8, 121)).tolist()))
    best = max((score_logp(recs, logp, t), -abs(t), t) for t in cands)
    return best[2], best[0]


def predictions(test_ids, logp, thr=0.0):
    return [{"id": i, "pred": int(logp[i][1] - logp[i][0] > thr)} for i in test_ids]


def validate(test_ids, pred):
    ids = [p["id"] for p in pred]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert set(ids) == set(test_ids), f"ids differ: missing {len(set(test_ids) - set(ids))}, extra {len(set(ids) - set(test_ids))}"
    for p in pred:
        assert isinstance(p["id"], str) and type(p["pred"]) is int and p["pred"] in (0, 1), p
    return f"ok: {len(ids)} docs, {sum(p['pred'] for p in pred)} predicted 1"


def write_submission(pred, out_dir):
    pj = os.path.join(out_dir, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(pred, f)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")


def batches_by_length(recs, ids, args, batch_docs):
    order = sorted(ids, key=lambda i: -min(recs[i]["n_tok"], args.chunk_len * (args.n_chunks + args.head_chunks)))
    return [order[k:k + batch_docs] for k in range(0, len(order), batch_docs)]


@torch.no_grad()
def doc_logp(model, recs, ids, args, device, amp):
    model.eval()
    out = {}
    for chunk in batches_by_length(recs, ids, args, args.eval_batch):
        b = collate(recs, chunk, args)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(b, device)
        lp = F.log_softmax(logits.float(), -1).cpu()
        for j, i in enumerate(chunk):
            out[i] = lp[j]
    return out


@torch.no_grad()
def chunk_features(model, recs, ids, args, device, amp, log):
    model.eval()
    feats, t0 = {}, time.time()
    bl = batches_by_length(recs, ids, args, args.eval_batch)
    for n, chunk in enumerate(bl):
        b = collate(recs, chunk, args)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            v = model.chunk_vectors(b["ids"].to(device), b["valid"].to(device)).float().cpu()
        for j, i in enumerate(chunk):
            sel = b["owner"] == j
            feats[i] = (v[sel].half(), b["pos"][sel])
        if n % 20 == 0:
            log(f"  features {n + 1}/{len(bl)} batches, {time.time() - t0:.0f}s")
    return feats


def head_batch(feats, recs, ids, device, teacher=None):
    v = torch.cat([feats[i][0] for i in ids]).float().to(device)
    pos = torch.cat([feats[i][1] for i in ids]).to(device)
    owner = torch.cat([torch.full((len(feats[i][1]),), k) for k, i in enumerate(ids)]).to(device)
    lab = torch.tensor([recs[i]["label"] for i in ids], device=device)
    t = torch.stack([teacher[i] for i in ids]).to(device) if teacher is not None else None
    return v, pos, owner, lab, t


def load_teacher(dirs):
    tot = None
    for d in dirs:
        t = torch.load(os.path.join(d, "train_logp.pt"))
        tot = t if tot is None else {k: tot[k] + t[k] for k in tot if k in t}
    return {k: torch.log_softmax(v / len(dirs), -1) for k, v in tot.items()}


def train(args):
    C.set_seed(args.seed)
    rng = random.Random(args.seed)
    device, amp = C.device_and_amp()
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "train.log"), "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    cache = torch.load(args.cache, weights_only=False)
    tr, dv, te = cache["train"], cache["dev"], cache["test"]
    assert not set(dv) & set(te), "dev must not contain test docs"
    pool = sorted(tr)
    if args.full:
        tr = {**tr, **dv}
        pool = sorted(tr)
    teacher = load_teacher(args.teacher) if args.teacher else None
    if teacher is not None:
        pool = [i for i in pool if i in teacher]
    if args.limit_docs:
        pool = random.Random(0).sample(pool, min(args.limit_docs, len(pool)))
        dv = dict(list(dv.items())[:args.limit_docs])
    dev_ids = [] if args.full else sorted(dv)
    test_ids = [r["id"] for r in json.load(open(find(args.data, "test.json"), encoding="utf-8"))]
    assert set(test_ids) == set(te), "cache does not match test.json"
    model, peft = build(args)
    model.to(device)
    n_tr = C.count_trainable(model)
    per_epoch = min(args.docs_per_epoch or len(pool), len(pool))
    lab_counts = Counter(tr[i]["label"] for i in pool)
    log(json.dumps({"arch": model.backbone.arch, "layers_kept": len(model.backbone.layers_kept), **peft,
                    "trainable_total": n_tr, "train_pool": len(pool), "docs_per_epoch": per_epoch,
                    "pool_labels": dict(lab_counts), "dev_docs": len(dev_ids), "device": str(device),
                    "amp": str(amp), "tokens_per_doc": args.chunk_len * (args.n_chunks + args.head_chunks)}))
    n = sum(lab_counts.values())
    cw = torch.tensor([n / (2 * max(lab_counts.get(c, 1), 1)) for c in (0, 1)], device=device)
    best, best_state, hist = -1.0, None, []
    feats = None
    if args.method == "head":
        need = sorted(set(pool) | set(dev_ids) | set(test_ids))
        if args.docs_per_epoch and args.method == "head":
            need = sorted(set(random.Random(args.seed).sample(pool, per_epoch)) | set(dev_ids) | set(test_ids))
            pool = [i for i in pool if i in set(need)]
        allrecs = {**tr, **dv, **te}
        log(f"method head: caching frozen chunk features for {len(need)} docs")
        feats = chunk_features(model, allrecs, need, args, device, amp, log)
    groups = C.param_groups(model, args.lr, args.lr_head, args.lr_adapter, args.weight_decay)
    opt = torch.optim.AdamW(groups)
    steps = args.epochs * math.ceil(per_epoch / (args.batch_docs * args.accum))
    sched = C.warmup_linear(opt, steps, args.warmup)
    stepper = C.AmpStep(opt, amp)
    params = [p for p in model.parameters() if p.requires_grad]

    def dev_logp_now():
        if feats is not None:
            return head_logp(model, feats, dev_ids, device)
        return doc_logp(model, dv, dev_ids, args, device, amp)

    for ep in range(1, args.epochs + 1):
        model.train()
        ids = rng.sample(pool, per_epoch)
        t0, tot, nb = time.time(), 0.0, 0
        for k in range(0, len(ids), args.batch_docs):
            chunk = ids[k:k + args.batch_docs]
            if feats is not None:
                v, pos, owner, lab, tl = head_batch(feats, tr, chunk, device, teacher)
                logits = model.head(v, pos, owner, len(chunk))
            else:
                b = collate(tr, chunk, args, teacher)
                lab = b["labels"].to(device)
                tl = b["teacher"].to(device) if teacher is not None else None
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    logits = model(b, device)
            loss = F.cross_entropy(logits.float(), lab, weight=cw)
            if tl is not None:
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * C.kd_loss(logits, tl, args.kd_T)
            (stepper.scaler.scale(loss / args.accum) if stepper.scaler else loss / args.accum).backward()
            nb += 1
            tot += loss.item()
            if nb % args.accum == 0 or k + args.batch_docs >= len(ids):
                if stepper.scaler:
                    stepper.scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(params, 1.0)
                    stepper.scaler.step(opt)
                    stepper.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if nb % args.log_every == 0:
                log(f"  ep {ep} batch {nb}/{math.ceil(len(ids) / args.batch_docs)} loss {tot / nb:.4f} "
                    f"{time.time() - t0:.0f}s")
        row = {"epoch": ep, "loss": round(tot / max(nb, 1), 4), "sec": round(time.time() - t0, 1)}
        if dev_ids:
            dlp = dev_logp_now()
            f1 = score_logp(dv, dlp)
            row["dev_f1"] = round(f1, 4)
            row["dev_f1_tuned_thr"] = round(tune_thr(dv, dlp)[1], 4)
            if row["dev_f1_tuned_thr"] > best:
                best, best_state = row["dev_f1_tuned_thr"], C.trainable_state(model)
        hist.append(row)
        log(json.dumps(row))
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    run = {"args": vars(args), "peft": peft, "trainable": n_tr, "history": hist, "arch": model.backbone.arch,
           "layers_kept": model.backbone.layers_kept, "data": cache["info"]}
    thr = 0.0
    if dev_ids:
        dlp = dev_logp_now()
        torch.save(dlp, os.path.join(args.out, "dev_logp.pt"))
        f1 = score_logp(dv, dlp)
        thr, f1t = tune_thr(dv, dlp)
        run.update(dev_f1=round(f1, 4), dev_f1_tuned=round(f1t, 4), threshold=thr)
        log(f"dev_clean Macro-F1 {f1:.4f} at threshold 0, {f1t:.4f} at tuned threshold {thr:.3f}")
    if feats is not None:
        tlp = head_logp(model, feats, test_ids, device)
    else:
        tlp = doc_logp(model, te, test_ids, args, device, amp)
    torch.save(tlp, os.path.join(args.out, "test_logp.pt"))
    pred = predictions(test_ids, tlp, thr)
    log(validate(test_ids, pred))
    write_submission(pred, args.out)
    if args.save_train_logp:
        ids = sorted(cache["train"])
        ids = random.Random(1234).sample(ids, min(args.save_train_logp, len(ids)))
        log(f"teacher: log-probs on {len(ids)} train docs")
        tl = doc_logp(model, cache["train"], ids, args, device, amp) if feats is None else head_logp(model, feats, [i for i in ids if i in feats], device)
        torch.save(tl, os.path.join(args.out, "train_logp.pt"))
    if args.save_model:
        C.save_trainable(model, os.path.join(args.out, "model.pt"), run)
    run["test_pred_1"] = sum(p["pred"] for p in pred)
    C.write_json(run, os.path.join(args.out, "run.json"))
    log(f"wrote {args.out}/submission.zip")


@torch.no_grad()
def head_logp(model, feats, ids, device, bs=256):
    model.eval()
    out = {}
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        v = torch.cat([feats[i][0] for i in chunk]).float().to(device)
        pos = torch.cat([feats[i][1] for i in chunk]).to(device)
        owner = torch.cat([torch.full((len(feats[i][1]),), j) for j, i in enumerate(chunk)]).to(device)
        lp = F.log_softmax(model.head(v, pos, owner, len(chunk)).float(), -1).cpu()
        for j, i in enumerate(chunk):
            out[i] = lp[j]
    return out


def dev_labels(data):
    data_raw, _ = load_raw(data, need_train=False)
    return {r["id"]: {"label": r["label"]} for r in data_raw["dev"]}, [r["id"] for r in data_raw["test"]]


def ensemble(args):
    dv, test_ids = dev_labels(args.data)
    tl, dl, n = None, None, len(args.runs)
    for r in args.runs:
        t = torch.load(os.path.join(r, "test_logp.pt"))
        tl = t if tl is None else {k: tl[k] + t[k] for k in tl}
        dp = os.path.join(r, "dev_logp.pt")
        if dl is not False and os.path.exists(dp):
            d = torch.load(dp)
            dl = d if dl is None else {k: dl[k] + d[k] for k in dl if k in d}
        else:
            dl = False
    tl = {k: v / n for k, v in tl.items()}
    os.makedirs(args.out, exist_ok=True)
    info, thr = {"runs": args.runs}, 0.0
    if dl:
        dl = {k: v / n for k, v in dl.items()}
        info["dev_f1"] = round(score_logp(dv, dl), 4)
        thr, f1t = tune_thr(dv, dl)
        info.update(dev_f1_tuned=round(f1t, 4), threshold=thr)
        torch.save(dl, os.path.join(args.out, "dev_logp.pt"))
    pred = predictions(test_ids, tl, thr)
    print(validate(test_ids, pred))
    write_submission(pred, args.out)
    torch.save(tl, os.path.join(args.out, "test_logp.pt"))
    C.write_json(info, os.path.join(args.out, "run.json"))
    print(json.dumps(info))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--keep_head", type=int, default=1024)
    p.add_argument("--keep_tail", type=int, default=4096)
    p.add_argument("--no_train", action="store_true", help="skip train.json (smoke tests)")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--ckpt", required=True, help="Phase 1 export (model_best.pt) of any arch")
    t.add_argument("--code_dir", default=None, help="folder holding cs781_p1.py")
    t.add_argument("--method", default="full", choices=C.METHODS)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=float, default=32)
    t.add_argument("--targets", default=",".join(C.DEFAULT_TARGETS))
    t.add_argument("--train_norms", action="store_true")
    t.add_argument("--bidir", action="store_true", help="gpt: drop the causal mask")
    t.add_argument("--keep_layers", type=int, default=0, help="student: keep this many evenly spaced layers")
    t.add_argument("--chunk_len", type=int, default=512)
    t.add_argument("--n_chunks", type=int, default=4, help="tail chunks (aligned to the document end)")
    t.add_argument("--head_chunks", type=int, default=0, help="extra chunks from the start of the document")
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--docs_per_epoch", type=int, default=12000, help="fresh random sample each epoch (0 = all)")
    t.add_argument("--batch_docs", type=int, default=4)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--eval_batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-5, help="backbone LR (full / norms)")
    t.add_argument("--lr_adapter", type=float, default=5e-4, help="LoRA / LORMA LR")
    t.add_argument("--lr_head", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--warmup", type=float, default=0.06)
    t.add_argument("--teacher", action="append", help="run dir(s) with train_logp.pt (distillation)")
    t.add_argument("--kd_alpha", type=float, default=0.5)
    t.add_argument("--kd_T", type=float, default=2.0)
    t.add_argument("--full", action="store_true", help="train on train+dev_clean, no dev selection, threshold 0")
    t.add_argument("--save_train_logp", type=int, default=0, help="teacher: log-probs on this many train docs")
    t.add_argument("--save_model", action="store_true")
    t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--log_every", type=int, default=200)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--limit_docs", type=int, default=0, help="smoke tests only")
    e = sub.add_parser("ensemble")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("runs", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("pred")
    v = sub.add_parser("validate")
    v.add_argument("--data", required=True)
    v.add_argument("pred")
    args = ap.parse_args(argv)
    if args.cmd == "prep":
        prep(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "ensemble":
        ensemble(args)
    elif args.cmd == "score":
        dv, _ = dev_labels(args.data)
        if os.path.isdir(args.pred):
            lp = torch.load(os.path.join(args.pred, "dev_logp.pt"))
            print("dev_clean Macro-F1 @0:", round(score_logp(dv, lp), 4), "tuned:", tune_thr(dv, lp))
        else:
            pred = {p["id"]: p["pred"] for p in json.load(open(args.pred))}
            ids = [i for i in dv if i in pred]
            print(f"dev_clean docs scored: {len(ids)}  Macro-F1 {macro_f1([dv[i]['label'] for i in ids], [pred[i] for i in ids]):.4f}")
    else:
        _, test_ids = dev_labels(args.data)
        print(validate(test_ids, json.load(open(args.pred))))


if __name__ == "__main__":
    main()
