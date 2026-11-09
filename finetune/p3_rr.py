#!/usr/bin/env python3
"""CS781 Capstone, Phase 3: Rhetorical Role labelling (Darsh Kedia, 250311).

One label per sentence, 13 labels, CL + IT domains pooled, Macro-F1.

Model (hierarchical, in the spirit of Bhattacharya et al. 2019 / Kalamkar et al. 2022 [5]):
  1. Consecutive sentences of a document are packed into windows of up to `max_len`
     tokens (each sentence truncated to `max_sent_tokens`, prefixed with <sep>), so the
     Phase 1 backbone sees each sentence together with its neighbours.
  2. Backbone (any Phase 1 arch, any technique from ft_core) -> token states.
  3. Mean pooling over each sentence's tokens -> sentence vectors.
  4. A BiLSTM over the window's sentence vectors (document context, also gives the
     causal archs right-hand context) -> linear layer -> 13 logits.
  Windows overlap at inference; each sentence's log-probs are averaged over windows.
  Loss: class-weighted cross-entropy (inverse sqrt frequency, for Macro-F1), plus an
  optional distillation term against a teacher's per-sentence log-probs.

CLI
  train     python p3_rr.py train --data DIR --out RUN --tokenizer tokenizer.json --ckpt model_best.pt
            [--method full|lora|lorma_plus|lorma_pi|norms|head] [--epochs 8] [--full]
            [--keep_layers 6 --teacher RUN_T [--teacher RUN_T2] --kd_alpha 0.5 --kd_T 2]
            Trains (on train, picking the best dev epoch; or on train+dev with --full),
            then writes RUN/prediction.json, RUN/submission.zip, RUN/test_logp.pt,
            RUN/dev_logp.pt (if dev was held out), RUN/train_logp.pt (with --save_train_logp,
            for use as a distillation teacher) and RUN/run.json.
  ensemble  python p3_rr.py ensemble --data DIR --out OUT RUN1 RUN2 ... [--bias_from RUN]
  score     python p3_rr.py score --data DIR RUN_OR_PRED.json      (dev Macro-F1)
  validate  python p3_rr.py validate --data DIR prediction.json

[5] Bhattacharya et al. 2019, Identification of Rhetorical Roles of Sentences in Indian
    Legal Judgments (hierarchical BiLSTM); Kalamkar et al. 2022, Corpus for Automatic
    Structuring of Legal Documents; Malik et al. 2022, Semantic Segmentation of Legal
    Documents via Rhetorical Roles (the CL/IT data); Joshi et al. 2024, IL-TUR.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_core as C

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent",
          "PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled",
          "RulingByLowerCourt", "RatioOfTheDecision", "RulingByPresentCourt", "None"]
L2I = {l: i for i, l in enumerate(LABELS)}
NC = len(LABELS)
PAD, SEP = 0, 5   # Phase 1 tokenizer special ids


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def find(data, name):
    for p in (os.path.join(data, name), os.path.join(data, "Train", name), os.path.join(data, "Test", name)):
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(data, "**", name), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} under {data}")


def load_labelled(path, domain):
    docs = []
    for d in json.load(open(path, encoding="utf-8")):
        assert len(d["text"]) == len(d["labels"]), d["id"]
        docs.append({"id": str(d["id"]), "sents": d["text"], "labels": [L2I[l] for l in d["labels"]],
                     "domain": domain})
    return docs


def load_test(path, domain):
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        data = {str(d["id"]): d["text"] for d in data}
    return [{"id": str(k), "sents": v, "labels": None, "domain": domain} for k, v in data.items()]


def load_phase3(data):
    out = {}
    for split in ("train", "dev"):
        out[split] = sum((load_labelled(find(data, f"{dom}_{split}.json"), dom) for dom in ("CL", "IT")), [])
    out["test"] = sum((load_test(find(data, f"{dom}_test.json"), dom) for dom in ("CL", "IT")), [])
    return out


class Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)

    def encode_sents(self, sents, max_sent_tokens):
        """Per sentence: [<sep>] + head of its tokens (long sentences keep head and tail)."""
        out = []
        for enc in self.tok.encode_batch([" " + s.strip() for s in sents], add_special_tokens=False):
            ids = enc.ids
            if len(ids) > max_sent_tokens - 1:
                h = (max_sent_tokens - 1) * 3 // 4
                ids = ids[:h] + ids[len(ids) - (max_sent_tokens - 1 - h):]
            out.append([SEP] + ids)
        return out


def encode_docs(tok, docs, max_sent_tokens):
    for d in docs:
        d["ids"] = tok.encode_sents(d["sents"], max_sent_tokens)
    return docs


def pack(lengths, start, max_len, max_sents):
    """Sentences start.. that fit in max_len tokens (at least one). Returns end index."""
    end, tot = start, 0
    while end < len(lengths) and end - start < max_sents and (tot + lengths[end] <= max_len or end == start):
        tot += lengths[end]
        end += 1
    return end


def train_windows(docs, max_len, max_sents, rng):
    """Non-overlapping windows; the first window of each doc gets a random size so the
    boundaries move every epoch (cheap augmentation)."""
    wins = []
    for di, d in enumerate(docs):
        lens = [len(x) for x in d["ids"]]
        s, first = 0, True
        while s < len(lens):
            cap = rng.randint(max_len // 4, max_len) if first else max_len
            e = pack(lens, s, cap, max_sents)
            wins.append((di, s, e))
            s, first = e, False
    return wins


def eval_windows(docs, max_len, max_sents, overlap=0.5):
    """Overlapping windows: the next window starts at the first sentence past `overlap`
    of the current window's tokens."""
    wins = []
    for di, d in enumerate(docs):
        lens = [len(x) for x in d["ids"]]
        s = 0
        while s < len(lens):
            e = pack(lens, s, max_len, max_sents)
            wins.append((di, s, e))
            if e >= len(lens):
                break
            acc, nxt = 0, s
            while nxt < e - 1 and acc + lens[nxt] <= overlap * sum(lens[s:e]):
                acc += lens[nxt]
                nxt += 1
            s = max(nxt, s + 1)
    return wins


def collate(docs, wins, teacher=None):
    seqs, sids, labels, tlp = [], [], [], []
    S = max(e - s for _, s, e in wins)
    for di, s, e in wins:
        d = docs[di]
        ids, sid = [], []
        for k in range(s, e):
            ids += d["ids"][k]
            sid += [k - s] * len(d["ids"][k])
        seqs.append(ids)
        sids.append(sid)
        lab = (d["labels"][s:e] if d["labels"] is not None else [-100] * (e - s)) + [-100] * (S - (e - s))
        labels.append(lab)
        if teacher is not None:
            t = teacher[d["id"]][s:e]
            tlp.append(torch.cat([t, torch.zeros(S - (e - s), NC)]))
    T = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    sid = torch.full((len(seqs), T), -1, dtype=torch.long)
    for b, (x, y) in enumerate(zip(seqs, sids)):
        ids[b, :len(x)] = torch.tensor(x)
        sid[b, :len(y)] = torch.tensor(y)
    out = {"ids": ids, "valid": sid >= 0, "sid": sid, "labels": torch.tensor(labels),
           "n_sent": torch.tensor([e - s for _, s, e in wins])}
    if teacher is not None:
        out["teacher"] = torch.stack(tlp)
    return out


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class RRModel(nn.Module):
    def __init__(self, backbone, context="lstm", ctx_hidden=384, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        H = backbone.d_model
        self.drop = nn.Dropout(dropout)
        self.context = context
        if context == "lstm":
            self.ctx = nn.LSTM(H, ctx_hidden, batch_first=True, bidirectional=True)
            out = 2 * ctx_hidden
        else:
            self.ctx, out = None, H
        self.head = nn.Linear(out, NC)

    def forward(self, ids, valid, sid, n_sent):
        h = self.backbone(ids, valid).float()                    # [B, T, H]
        B, T, H = h.shape
        S = int(n_sent.max())
        idx = torch.where(sid >= 0, sid, torch.full_like(sid, S))  # pad tokens -> dummy slot S
        pooled = h.new_zeros(B, S + 1, H).scatter_add_(1, idx[..., None].expand(B, T, H), h)
        cnt = h.new_zeros(B, S + 1).scatter_add_(1, idx, torch.ones_like(idx, dtype=h.dtype))
        x = self.drop((pooled / cnt.clamp(min=1)[..., None])[:, :S])
        if self.ctx is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, n_sent.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = self.ctx(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=S)
            x = self.drop(x)
        return self.head(x)                                      # [B, S, NC]


def build(args, ckpt=None):
    bb = C.Backbone(ckpt or args.ckpt, args.code_dir, keep_layers=args.keep_layers or None,
                    bidir=args.bidir, grad_ckpt=args.grad_ckpt)
    peft = C.apply_peft(bb, args.method, args.rank, args.alpha, tuple(args.targets.split(",")),
                        train_norms=args.train_norms)
    return RRModel(bb, args.context, dropout=args.dropout), peft


# --------------------------------------------------------------------------------------
# Metrics / output
# --------------------------------------------------------------------------------------
def macro_f1(gold, pred, n=NC):
    """sklearn f1_score(average='macro', labels=all 13, zero_division=0)."""
    tp, fp, fn = [0] * n, [0] * n, [0] * n
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1 = [2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]) if tp[i] else 0.0 for i in range(n)]
    return sum(f1) / n, {LABELS[i]: round(f1[i], 4) for i in range(n)}


def score_logp(docs, logp, bias=None):
    gold, pred = [], []
    for d in docs:
        lp = logp[d["id"]] + (bias if bias is not None else 0)
        gold += d["labels"]
        pred += lp.argmax(-1).tolist()
    return macro_f1(gold, pred)


def tune_bias(docs, logp, rounds=3, grid=(-2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3)):
    """Per-label additive log-prob offsets that maximise dev Macro-F1 (coordinate ascent).
    Small dev set, so the grid is coarse; check the gain is real before trusting it."""
    bias = torch.zeros(NC)
    best = score_logp(docs, logp, bias)[0]
    for _ in range(rounds):
        for c in range(NC):
            for v in grid:
                b = bias.clone()
                b[c] = v
                f = score_logp(docs, logp, b)[0]
                if f > best + 1e-6:
                    best, bias = f, b
    return bias, best


def predictions(docs, logp, bias=None):
    out = {}
    for d in docs:
        lp = logp[d["id"]] + (bias if bias is not None else 0)
        out[d["id"]] = [LABELS[i] for i in lp.argmax(-1).tolist()]
    return out


def validate(test_docs, pred):
    ids = {d["id"] for d in test_docs}
    assert set(pred) == ids, f"ids differ: missing {ids - set(pred)}, extra {set(pred) - ids}"
    for d in test_docs:
        p = pred[d["id"]]
        assert isinstance(p, list) and len(p) == len(d["sents"]), f"{d['id']}: {len(p)} vs {len(d['sents'])}"
        assert all(l in L2I for l in p), d["id"]
    return f"ok: {len(ids)} docs, {sum(len(d['sents']) for d in test_docs)} sentences"


def write_submission(pred, out_dir):
    pj = os.path.join(out_dir, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")


# --------------------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------------------
@torch.no_grad()
def doc_logp(model, docs, args, device, amp):
    model.eval()
    wins = eval_windows(docs, args.max_len, args.max_sents)
    wins.sort(key=lambda w: -sum(len(x) for x in docs[w[0]]["ids"][w[1]:w[2]]))
    acc = {d["id"]: torch.zeros(len(d["ids"]), NC) for d in docs}
    cnt = {d["id"]: torch.zeros(len(d["ids"]), 1) for d in docs}
    for k in range(0, len(wins), args.eval_batch):
        chunk = wins[k:k + args.eval_batch]
        b = collate(docs, chunk)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(b["ids"].to(device), b["valid"].to(device), b["sid"].to(device), b["n_sent"])
        lp = F.log_softmax(logits.float(), -1).cpu()
        for j, (di, s, e) in enumerate(chunk):
            did = docs[di]["id"]
            acc[did][s:e] += lp[j, :e - s]
            cnt[did][s:e] += 1
    return {k: acc[k] / cnt[k].clamp(min=1) for k in acc}


def class_weights(docs, mode):
    if mode == "none":
        return None
    c = Counter(l for d in docs for l in d["labels"])
    n = sum(c.values())
    w = torch.tensor([(n / (NC * max(c.get(i, 0), 1))) ** 0.5 for i in range(NC)])
    w = w.clamp(max=10.0)
    return w / (w * torch.tensor([c.get(i, 0) for i in range(NC)])).sum() * n


def load_teacher(dirs):
    tot = None
    for d in dirs:
        t = torch.load(os.path.join(d, "train_logp.pt"))
        tot = t if tot is None else {k: tot[k] + t[k] for k in tot}
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

    data = load_phase3(args.data)
    if args.limit_docs:
        for k in data:
            data[k] = data[k][:args.limit_docs] if k != "test" else data[k]
    tok = Tok(args.tokenizer)
    for k in data:
        encode_docs(tok, data[k], args.max_sent_tokens)
    train_docs = data["train"] + (data["dev"] if args.full else [])
    dev_docs = None if args.full else data["dev"]
    teacher = load_teacher(args.teacher) if args.teacher else None
    model, peft = build(args)
    model.to(device)
    n_tr = C.count_trainable(model)
    ntok = sum(len(x) for d in train_docs for x in d["ids"])
    log(json.dumps({"arch": model.backbone.arch, "layers_kept": len(model.backbone.layers_kept), **peft,
                    "trainable_total": n_tr, "train_docs": len(train_docs), "train_sents":
                    sum(len(d["ids"]) for d in train_docs), "train_tokens": ntok, "device": str(device),
                    "amp": str(amp)}))
    groups = C.param_groups(model, args.lr, args.lr_head, args.lr_adapter, args.weight_decay)
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train_windows(train_docs, args.max_len, args.max_sents, random.Random(0)))
                                / (args.batch_size * args.accum))
    sched = C.warmup_linear(opt, args.epochs * steps_per_epoch, args.warmup)
    stepper = C.AmpStep(opt, amp)
    params = [p for p in model.parameters() if p.requires_grad]
    cw = class_weights(train_docs, args.class_weight)
    cw = cw.to(device) if cw is not None else None
    best, best_state, hist = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        wins = train_windows(train_docs, args.max_len, args.max_sents, rng)
        rng.shuffle(wins)
        t0, tot, nb = time.time(), 0.0, 0
        for k in range(0, len(wins), args.batch_size):
            b = collate(train_docs, wins[k:k + args.batch_size], teacher)
            lab = b["labels"].to(device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = model(b["ids"].to(device), b["valid"].to(device), b["sid"].to(device), b["n_sent"])
            loss = F.cross_entropy(logits.float().reshape(-1, NC), lab.reshape(-1), weight=cw, ignore_index=-100)
            if teacher is not None:
                kd = C.kd_loss(logits, b["teacher"].to(device), args.kd_T, mask=lab != -100)
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            (loss / args.accum).backward() if stepper.scaler is None else stepper.scaler.scale(loss / args.accum).backward()
            nb += 1
            tot += loss.item()
            if nb % args.accum == 0 or k + args.batch_size >= len(wins):
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
        row = {"epoch": ep, "loss": round(tot / max(nb, 1), 4), "sec": round(time.time() - t0, 1)}
        if dev_docs is not None:
            f1, _ = score_logp(dev_docs, doc_logp(model, dev_docs, args, device, amp))
            row["dev_f1"] = round(f1, 4)
            if f1 > best:
                best, best_state = f1, C.trainable_state(model)
        hist.append(row)
        log(json.dumps(row))
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    run = {"args": vars(args), "peft": peft, "trainable": n_tr, "history": hist,
           "arch": model.backbone.arch, "layers_kept": model.backbone.layers_kept}
    # outputs
    if dev_docs is not None:
        dlp = doc_logp(model, dev_docs, args, device, amp)
        torch.save(dlp, os.path.join(args.out, "dev_logp.pt"))
        f1, per = score_logp(dev_docs, dlp)
        bias, f1b = tune_bias(dev_docs, dlp)
        torch.save(bias, os.path.join(args.out, "bias.pt"))
        run.update(dev_f1=round(f1, 4), dev_per_label=per, dev_f1_bias_tuned=round(f1b, 4), bias=bias.tolist())
        log(f"dev Macro-F1 {f1:.4f} (bias-tuned on dev itself, optimistic: {f1b:.4f})")
    tlp = doc_logp(model, data["test"], args, device, amp)
    torch.save(tlp, os.path.join(args.out, "test_logp.pt"))
    pred = predictions(data["test"], tlp)
    log(validate(data["test"], pred))
    write_submission(pred, args.out)
    if args.save_train_logp:
        allp = doc_logp(model, data["train"] + data["dev"], args, device, amp)
        torch.save(allp, os.path.join(args.out, "train_logp.pt"))
    if args.save_model:
        C.save_trainable(model, os.path.join(args.out, "model.pt"), run)
    run["test_label_counts"] = dict(Counter(l for v in pred.values() for l in v))
    C.write_json(run, os.path.join(args.out, "run.json"))
    log(f"wrote {args.out}/submission.zip")


def ensemble(args):
    data = load_phase3(args.data)
    test_lp, dev_lp = None, None
    for r in args.runs:
        t = torch.load(os.path.join(r, "test_logp.pt"))
        test_lp = t if test_lp is None else {k: test_lp[k] + t[k] for k in t}
        dp = os.path.join(r, "dev_logp.pt")
        if dev_lp is not False and os.path.exists(dp):
            d = torch.load(dp)
            dev_lp = d if dev_lp is None else {k: dev_lp[k] + d[k] for k in d if k in dev_lp}
        else:
            dev_lp = False
    n = len(args.runs)
    test_lp = {k: v / n for k, v in test_lp.items()}
    os.makedirs(args.out, exist_ok=True)
    info = {"runs": args.runs}
    bias = None
    if dev_lp:
        dev_lp = {k: v / n for k, v in dev_lp.items()}
        dev_docs = [d for d in data["dev"] if d["id"] in dev_lp]
        info["dev_docs"] = len(dev_docs)
        info["dev_f1"] = round(score_logp(dev_docs, dev_lp)[0], 4)
        if args.tune_bias:
            bias, info["dev_f1_bias_tuned"] = tune_bias(dev_docs, dev_lp)
    if args.bias_from:
        bias = torch.load(os.path.join(args.bias_from, "bias.pt"))
    info["bias"] = None if bias is None else bias.tolist()
    pred = predictions(data["test"], test_lp, bias)
    print(validate(data["test"], pred))
    write_submission(pred, args.out)
    torch.save(test_lp, os.path.join(args.out, "test_logp.pt"))
    C.write_json(info, os.path.join(args.out, "run.json"))
    print(json.dumps(info))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--ckpt", required=True, help="Phase 1 export (model_best.pt) of any arch")
    t.add_argument("--code_dir", default=None, help="folder holding cs781_p1.py")
    t.add_argument("--method", default="full", choices=C.METHODS)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=float, default=32)
    t.add_argument("--targets", default=",".join(C.DEFAULT_TARGETS))
    t.add_argument("--train_norms", action="store_true")
    t.add_argument("--bidir", action="store_true", help="gpt: drop the causal mask")
    t.add_argument("--keep_layers", type=int, default=0, help="student: keep this many evenly spaced layers")
    t.add_argument("--context", default="lstm", choices=["lstm", "none"])
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--eval_batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5, help="backbone LR (full / norms)")
    t.add_argument("--lr_adapter", type=float, default=5e-4, help="LoRA / LORMA LR")
    t.add_argument("--lr_head", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--warmup", type=float, default=0.06)
    t.add_argument("--max_len", type=int, default=1024)
    t.add_argument("--max_sents", type=int, default=64)
    t.add_argument("--max_sent_tokens", type=int, default=128)
    t.add_argument("--class_weight", default="sqrt", choices=["sqrt", "none"])
    t.add_argument("--teacher", action="append", help="run dir(s) with train_logp.pt (distillation)")
    t.add_argument("--kd_alpha", type=float, default=0.5)
    t.add_argument("--kd_T", type=float, default=2.0)
    t.add_argument("--full", action="store_true", help="train on train+dev, no dev selection")
    t.add_argument("--save_train_logp", action="store_true", help="save log-probs on train+dev (teacher)")
    t.add_argument("--save_model", action="store_true")
    t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--limit_docs", type=int, default=0, help="smoke tests only")
    e = sub.add_parser("ensemble")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tune_bias", action="store_true")
    e.add_argument("--bias_from", default=None)
    e.add_argument("runs", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("pred")
    v = sub.add_parser("validate")
    v.add_argument("--data", required=True)
    v.add_argument("pred")
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "ensemble":
        ensemble(args)
    elif args.cmd == "score":
        data = load_phase3(args.data)
        p = args.pred
        if os.path.isdir(p):
            lp = torch.load(os.path.join(p, "dev_logp.pt"))
            print(score_logp([d for d in data["dev"] if d["id"] in lp], lp))
        else:
            pred = json.load(open(p))
            gold, pp = [], []
            for d in data["dev"]:
                gold += d["labels"]
                pp += [L2I[l] for l in pred[d["id"]]]
            print(macro_f1(gold, pp))
    else:
        data = load_phase3(args.data)
        print(validate(data["test"], json.load(open(args.pred))))


if __name__ == "__main__":
    main()
