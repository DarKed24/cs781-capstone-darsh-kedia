#!/usr/bin/env python3
"""CS781 Capstone, Phase 5: Legal Statute Identification, LSI (Darsh Kedia, 250311).

Multi-label: each document (Indian court judgment, statute mentions masked as <SECTION>/<ACT>)
gets one or more of 100 statute ids (0..99). Metric: Macro-F1 over the 100 labels
(evaluate_phase5.py; Micro-F1 and subset accuracy are tie-breakers).

Data (measured 2026-10-19): train 42,750 docs, dev 10,181, test 13,019; no id overlap between
splits; a few hundred exact-duplicate texts (mostly one-line orders such as "ORDER <ENTITY>, J.").
Every doc has >= 1 label, mean 3.7 (train) / 3.9 (dev); the rarest label still has 272 train /
68 dev docs. Median 0.8k tokens, mean 2.0k, 90th percentile 4.4k (Phase 1 tokenizer).
Hygiene: train docs whose exact text also appears in dev are dropped (keeps dev honest); train is
de-duplicated by text; dev is never trained on unless --full.

Model
  * The document is cut into chunks of `chunk_len` tokens: the first `head_chunks` chunks plus
    the last `tail_chunks` chunks (not overlapping the head). Default 3 + 1 x 512 = up to 2,048
    tokens, which is the whole document for ~70% of docs.
  * Each chunk -> Phase 1 backbone (any arch, any technique from ft_core) -> mean pooling ->
    + learned chunk-position embedding -> BiGRU over chunks -> [attention pooling ; max pooling]
    -> linear layer with 100 logits (binary relevance, one sigmoid per statute).
  * Loss: BCE with per-label pos_weight = sqrt(#neg / #pos) (helps rare labels for Macro-F1).
  * Decision rule: a global or per-label probability threshold tuned on dev (per-label is used
    only when it wins on a cross-half check, see `choose_thresholds`), and every document gets at
    least its top-scoring label (every gold set is non-empty).
  * Distillation (multi-label): the student loss is (1 - a) * BCE + a * T^2 * BCE(sigmoid(s/T),
    sigmoid(t/T)) on teacher logits saved with --save_train_logits [Hinton et al. 2015].

CLI
  prep      python p5_lsi.py prep --data DIR --tokenizer tok.json --cache CACHE.pt
  train     python p5_lsi.py train --data DIR --cache CACHE.pt --out RUN --ckpt model_best.pt
            [--method full|lora|lorma_plus|lorma_pi|norms|head] [--epochs 2] [--docs_per_epoch 16000]
            [--keep_layers 6 --teacher RUN_T --kd_alpha 0.5 --kd_T 2] [--full] [--save_train_logits N]
            Writes RUN/{prediction.json, submission.zip, test_prob.pt, dev_prob.pt, run.json}.
  ensemble  python p5_lsi.py ensemble --data DIR --out OUT RUN1 RUN2 ...   (mean of probabilities)
  score     python p5_lsi.py score --data DIR RUN_DIR_or_prediction.json   (dev Macro-F1)
  validate  python p5_lsi.py validate --data DIR prediction.json
  baseline  python p5_lsi.py baseline --data DIR --out OUT   TF-IDF + one-vs-rest logistic
            regression. REFERENCE ONLY (not a Phase 1 model, so never submit it).

References: Paul et al. 2022, LeSICiN (AAAI) and Vats et al. 2023 (ILSI / Indian LSI datasets);
Chalkidis et al. 2019 (hierarchical encoders for long legal documents); Hinton et al. 2015 (KD);
Hu et al. 2022 (LoRA); Bihany, Patel, Modi 2025 (LoRMA).
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
import zlib
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_core as C

PAD = 0
NL = 100
GRID = [round(0.02 * k, 2) for k in range(1, 48)]   # probability thresholds 0.02 .. 0.94


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


def _labels(r):
    return sorted({int(x) for x in r["labels"]})


def load_raw(data, need_train=True):
    """train (de-duplicated, dev texts removed), dev, test."""
    test = [{"id": str(r["id"]), "text": r["text"]} for r in json.load(open(find(data, "test.json"), encoding="utf-8"))]
    tids = {r["id"] for r in test}
    dev = [{"id": str(r["id"]), "text": r["text"], "labels": _labels(r)}
           for r in json.load(open(find(data, "dev.json"), encoding="utf-8")) if str(r["id"]) not in tids]
    dev_texts = {r["text"] for r in dev}
    train, info = [], {}
    if need_train:
        seen, n_raw, n_devdup = set(), 0, 0
        for r in json.load(open(find(data, "train.json"), encoding="utf-8")):
            n_raw += 1
            if r["text"] in dev_texts:
                n_devdup += 1
                continue
            if r["text"] in seen or str(r["id"]) in tids:
                continue
            seen.add(r["text"])
            train.append({"id": str(r["id"]), "text": r["text"], "labels": _labels(r)})
        info.update(train_raw=n_raw, train_dropped_dev_text=n_devdup)
    info.update(train=len(train), dev=len(dev), test=len(test))
    return {"train": train, "dev": dev, "test": test}, info


def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()


def prep(args):
    """Tokenise all splits once. Keeps the first `keep_head` and last `keep_tail` tokens."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    data, info = load_raw(args.data, need_train=not args.no_train)
    out = {"info": info, "keep_head": args.keep_head, "keep_tail": args.keep_tail,
           "tokenizer": os.path.abspath(args.tokenizer), "keep_all": args.keep_all,
           # tokens that carry the masked statute markers <SECTION> / <ACT> (for --select cue)
           "cue_ids": sorted({tok.token_to_id(t) for t in ("SECTION", "ACT") if tok.token_to_id(t) is not None})}
    t0 = time.time()
    for split, docs in data.items():
        recs = {}
        for k in range(0, len(docs), 512):
            part = docs[k:k + 512]
            encs = tok.encode_batch([clean_text(d["text"]) for d in part], add_special_tokens=False)
            for d, e in zip(part, encs):
                ids = e.ids or [1]   # unk for an empty doc
                recs[d["id"]] = {"head": torch.tensor(ids[:args.keep_head], dtype=torch.int16),
                                 "tail": torch.tensor(ids[-args.keep_tail:], dtype=torch.int16),
                                 "n_tok": len(ids), "labels": d.get("labels")}
                if args.keep_all:
                    recs[d["id"]]["all"] = torch.tensor(ids[:args.keep_all], dtype=torch.int16)
        out[split] = recs
        print(f"prep {split}: {len(recs)} docs, {time.time() - t0:.0f}s", flush=True)
    lens = sorted(r["n_tok"] for r in out["dev"].values())
    out["info"]["dev_median_tokens"] = lens[len(lens) // 2]
    torch.save(out, args.cache)
    print(json.dumps(out["info"]))


def chunks_of(rec, chunk_len, head_chunks, tail_chunks):
    """First `head_chunks` chunks, then up to `tail_chunks` chunks aligned to the document end
    that do not overlap the head. Positions: head chunk i -> i, tail chunks -> head_chunks + k
    with k = 0 for the last chunk."""
    n_tok = rec["n_tok"]
    head = rec["head"].tolist()[:chunk_len * head_chunks]
    chunks = [head[i:i + chunk_len] for i in range(0, len(head), chunk_len)]
    pos = list(range(len(chunks)))
    covered = len(head)
    if tail_chunks and n_tok > covered:
        rest = min(n_tok - covered, chunk_len * tail_chunks)
        tail = rec["tail"].tolist()[-rest:] if rest <= len(rec["tail"]) else rec["tail"].tolist()
        tc, end = [], len(tail)
        while end > 0 and len(tc) < tail_chunks:
            s = max(0, end - chunk_len)
            tc.append(tail[s:end])
            end = s
        tc = tc[::-1]
        chunks += tc
        pos += [head_chunks + k for k in range(len(tc) - 1, -1, -1)]
    return chunks, pos


def cue_chunks(rec, chunk_len, n, cue_ids):
    """Evidence-guided context selection (--select cue): split the whole document into
    consecutive chunks and keep the first chunk (parties, court, charge) plus the n - 1 chunks
    with the most masked statute markers (<SECTION>, <ACT>); ties go to the earlier chunk.
    Kept chunks stay in document order. Motivation (dev, 2026-10-25): for the 30% of docs longer
    than 2,048 tokens, the fixed first-3 + last-1 window misses 32% of the statute markers, and
    for 27% of those docs it misses most of them."""
    ids = rec["all"]
    chunks = [ids[k:k + chunk_len] for k in range(0, len(ids), chunk_len)]
    if len(chunks) <= n:
        keep = list(range(len(chunks)))
    else:
        cue = torch.zeros(len(ids), dtype=torch.bool)
        for c in cue_ids:
            cue |= ids == c
        score = [int(cue[k:k + chunk_len].sum()) for k in range(0, len(ids), chunk_len)]
        rest = sorted(range(1, len(chunks)), key=lambda k: (-score[k], k))[:n - 1]
        keep = sorted([0] + rest)
    return [chunks[k].tolist() for k in keep], list(range(len(keep)))


def multi_hot(labels):
    y = torch.zeros(NL)
    if labels:
        y[torch.tensor(labels, dtype=torch.long)] = 1.0
    return y


def collate(recs, ids, args, teacher=None):
    chunks, pos, owner, ys, tl = [], [], [], [], []
    for b, i in enumerate(ids):
        if getattr(args, "select", "headtail") == "cue":
            ch, ps = cue_chunks(recs[i], args.chunk_len, args.head_chunks + args.tail_chunks, args.cue_ids)
        else:
            ch, ps = chunks_of(recs[i], args.chunk_len, args.head_chunks, args.tail_chunks)
        chunks += ch
        pos += ps
        owner += [b] * len(ch)
        ys.append(multi_hot(recs[i]["labels"]))
        if teacher is not None:
            tl.append(teacher[i])
    T = max(len(c) for c in chunks)
    x = torch.full((len(chunks), T), PAD, dtype=torch.long)
    valid = torch.zeros((len(chunks), T), dtype=torch.bool)
    for k, c in enumerate(chunks):
        x[k, :len(c)] = torch.tensor(c, dtype=torch.long)
        valid[k, :len(c)] = True
    out = {"ids": x, "valid": valid, "pos": torch.tensor(pos), "owner": torch.tensor(owner),
           "n_docs": len(ids), "y": torch.stack(ys)}
    if teacher is not None:
        out["teacher"] = torch.stack(tl).float()
    return out


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class DocHead(nn.Module):
    """Chunk vectors [N, H] (+ positions, owners) -> doc logits [B, NL]."""

    def __init__(self, H, max_pos, hidden=384, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(max_pos, H)
        nn.init.zeros_(self.pos.weight)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(H, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.out = nn.Linear(4 * hidden, NL)

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
        att = (a[..., None] * h).sum(1)
        mx = h.masked_fill(~mask[..., None], -1e4).max(1).values
        return self.out(self.drop(torch.cat([att, mx], -1)))


class LSIModel(nn.Module):
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
    return LSIModel(bb, args.head_chunks + args.tail_chunks, args.dropout), peft


def kd_bce(student_logits, teacher_logits, T=2.0):
    """Binary (per-label) distillation: BCE between tempered sigmoids, scaled by T^2."""
    t = torch.sigmoid(teacher_logits.float() / T)
    return F.binary_cross_entropy_with_logits(student_logits.float() / T, t) * T * T


# --------------------------------------------------------------------------------------
# Metrics, thresholds, output
# --------------------------------------------------------------------------------------
def f1_per_label(Y, P):
    """Y, P: bool tensors [n, NL]. F1 per label; 0 when a label has no gold and no prediction
    (sklearn zero_division=0, as evaluate_phase5.py)."""
    tp = (Y & P).sum(0).double()
    fp = (~Y & P).sum(0).double()
    fn = (Y & ~P).sum(0).double()
    den = 2 * tp + fp + fn
    return torch.where(den > 0, 2 * tp / den.clamp(min=1), torch.zeros_like(den))


def macro_f1(Y, P):
    """f1_score(average='macro', zero_division=0) over all NL labels, as evaluate_phase5.py
    (its label count is max gold id + 1 = 100 on dev)."""
    return float(f1_per_label(Y, P).mean())


def micro_f1(Y, P):
    tp = float((Y & P).sum())
    den = 2 * tp + float((~Y & P).sum()) + float((Y & ~P).sum())
    return 2 * tp / den if den else 0.0


def decide(prob, thr, min_one=True):
    """prob [n, NL]; thr scalar or [NL] tensor. Adds the argmax label to docs with none."""
    P = prob > (thr if torch.is_tensor(thr) else float(thr))
    if min_one:
        empty = ~P.any(1)
        if empty.any():
            P[empty, prob[empty].argmax(1)] = True
    return P


def tune_global(Y, prob, min_one=True):
    return max((macro_f1(Y, decide(prob, t, min_one)), -abs(t - 0.5), t) for t in GRID)[2]


def tune_per_label(Y, prob):
    """Per-label threshold maximising that label's F1 (min_one is not applied while tuning)."""
    best = torch.full((NL,), 0.5)
    best_f = torch.full((NL,), -1.0)
    for t in GRID:
        f = f1_per_label(Y, prob > t).float()
        upd = f > best_f + 1e-9
        best[upd], best_f[upd] = t, f[upd]
    return best


def dev_matrix(recs, prob_dict):
    ids = [i for i in recs if i in prob_dict]
    Y = torch.stack([multi_hot(recs[i]["labels"]) for i in ids]).bool()
    prob = torch.stack([prob_dict[i].float() for i in ids])
    return ids, Y, prob


def choose_thresholds(recs, prob_dict):
    """Tunes a global and a per-label threshold. Their honest value is estimated by tuning on
    one half of dev (fixed split by crc32(id)) and scoring on the other half, both ways. The
    per-label rule is used only if it beats the global one there by > 0.003. Final thresholds
    are re-tuned on all of dev."""
    ids, Y, prob = dev_matrix(recs, prob_dict)
    half = torch.tensor([zlib.crc32(i.encode()) % 2 for i in ids])
    cross = {"global": [], "per_label": []}
    for a in (0, 1):
        A, B = half == a, half != a
        g = tune_global(Y[A], prob[A])
        pl = tune_per_label(Y[A], prob[A])
        cross["global"].append(macro_f1(Y[B], decide(prob[B], g)))
        cross["per_label"].append(macro_f1(Y[B], decide(prob[B], pl)))
    cg, cp = sum(cross["global"]) / 2, sum(cross["per_label"]) / 2
    g = tune_global(Y, prob)
    pl = tune_per_label(Y, prob)
    use_pl = cp > cg + 0.003
    thr = pl if use_pl else g
    P = decide(prob, thr)
    return {"rule": "per_label" if use_pl else "global", "thr": thr.tolist() if use_pl else g,
            "global_thr": g, "dev_f1_at_0.5": round(macro_f1(Y, decide(prob, 0.5)), 4),
            "dev_f1_cross_global": round(cg, 4), "dev_f1_cross_per_label": round(cp, 4),
            "dev_f1_honest": round(max(cg, cp) if use_pl else cg, 4),
            "dev_f1_tuned": round(macro_f1(Y, P), 4), "dev_micro_tuned": round(micro_f1(Y, P), 4),
            "dev_labels_per_doc_pred": round(float(P.sum(1).float().mean()), 3),
            "dev_labels_per_doc_gold": round(float(Y.sum(1).float().mean()), 3)}


def quick_dev_f1(recs, prob_dict):
    """Epoch selection: Macro-F1 at the best global threshold."""
    _, Y, prob = dev_matrix(recs, prob_dict)
    return macro_f1(Y, decide(prob, tune_global(Y, prob)))


def predictions(test_ids, prob_dict, thr):
    prob = torch.stack([prob_dict[i].float() for i in test_ids])
    thr = torch.tensor(thr) if isinstance(thr, list) else thr
    P = decide(prob, thr)
    return [{"id": i, "pred": [int(j) for j in torch.nonzero(P[k]).flatten().tolist()]} for k, i in enumerate(test_ids)]


def validate(test_ids, pred):
    ids = [p["id"] for p in pred]
    assert isinstance(pred, list) and pred, "prediction must be a non-empty list"
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert set(ids) == set(test_ids), f"ids differ: missing {len(set(test_ids) - set(ids))}, extra {len(set(ids) - set(test_ids))}"
    n = 0
    for p in pred:
        assert isinstance(p["id"], str) and isinstance(p["pred"], list), p
        assert all(type(x) is int and 0 <= x < NL for x in p["pred"]), p
        assert len(p["pred"]) == len(set(p["pred"])), p
        n += len(p["pred"])
    return f"ok: {len(ids)} docs, {n / len(ids):.2f} labels per doc, {sum(not p['pred'] for p in pred)} empty"


def write_submission(pred, out_dir):
    pj = os.path.join(out_dir, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(pred, f)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")


# --------------------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------------------
def doc_tokens(rec, args):
    return min(rec["n_tok"], args.chunk_len * (args.head_chunks + args.tail_chunks))


def batches_by_length(recs, ids, args, batch_docs):
    """Inference batches, longest docs first, each holding ~batch_docs x max doc tokens."""
    budget = batch_docs * args.chunk_len * (args.head_chunks + args.tail_chunks)
    order = sorted(ids, key=lambda i: -doc_tokens(recs[i], args))
    out, cur = [], []
    for i in order:   # descending, so the first doc of a batch is its longest
        if cur and (doc_tokens(recs[cur[0]], args) * (len(cur) + 1) > budget or len(cur) >= 64):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def train_batches(recs, ids, args, rng):
    """Length-bucketed batches with ~batch_docs * max_tokens tokens each: short docs are packed
    more densely (most docs are far below the 2,048-token cap). Batch order is shuffled."""
    budget = args.batch_docs * args.chunk_len * (args.head_chunks + args.tail_chunks)
    order = sorted(ids, key=lambda i: (doc_tokens(recs[i], args), rng.random()))
    out, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, doc_tokens(recs[i], args))
        if cur and (m * (len(cur) + 1) > budget or len(cur) >= args.max_batch_docs):
            out.append(cur)
            cur, m = [], doc_tokens(recs[i], args)
        cur.append(i)
        cur_max = m
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


@torch.no_grad()
def doc_logits(model, recs, ids, args, device, amp):
    model.eval()
    out = {}
    for chunk in batches_by_length(recs, ids, args, args.eval_batch):
        b = collate(recs, chunk, args)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(b, device)
        lg = logits.float().cpu()
        for j, i in enumerate(chunk):
            out[i] = lg[j]
    return out


def to_prob(logits):
    return {k: torch.sigmoid(v).half() for k, v in logits.items()}


@torch.no_grad()
def chunk_features(model, recs, ids, args, device, amp, log):
    """Frozen backbone: chunk vectors per doc, computed once (method head)."""
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
        if n % 50 == 0:
            log(f"  features {n + 1}/{len(bl)} batches, {time.time() - t0:.0f}s")
    return feats


def head_inputs(feats, ids, device):
    v = torch.cat([feats[i][0] for i in ids]).float().to(device)
    pos = torch.cat([feats[i][1] for i in ids]).to(device)
    owner = torch.cat([torch.full((len(feats[i][1]),), k) for k, i in enumerate(ids)]).to(device)
    return v, pos, owner


@torch.no_grad()
def head_logits(model, feats, ids, device, bs=256):
    model.eval()
    out = {}
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        v, pos, owner = head_inputs(feats, chunk, device)
        lg = model.head(v, pos, owner, len(chunk)).float().cpu()
        for j, i in enumerate(chunk):
            out[i] = lg[j]
    return out


def load_teacher(dirs):
    tot = None
    for d in dirs:
        t = torch.load(os.path.join(d, "train_logits.pt"))
        tot = t if tot is None else {k: tot[k] + t[k] for k in tot if k in t}
    return {k: v.float() / len(dirs) for k, v in tot.items()}


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
    if args.select == "cue":
        assert cache.get("keep_all"), "--select cue needs a cache made with prep --keep_all N"
        args.cue_ids = cache["cue_ids"]
    tr, dv, te = cache["train"], cache["dev"], cache["test"]
    assert not set(dv) & set(te), "dev must not contain test docs"
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
    sel_ids = random.Random(7).sample(dev_ids, min(args.dev_select_docs, len(dev_ids))) if args.dev_select_docs else dev_ids
    test_ids = [r["id"] for r in json.load(open(find(args.data, "test.json"), encoding="utf-8"))]
    assert set(test_ids) == set(te), "cache does not match test.json"
    if args.limit_docs:   # smoke test: the zip is NOT a valid submission
        test_ids = test_ids[:args.limit_docs]
    model, peft = build(args)
    model.to(device)
    n_tr = C.count_trainable(model)
    per_epoch = min(args.docs_per_epoch or len(pool), len(pool))
    # per-label positive weights sqrt(neg/pos), clipped
    pos_cnt = torch.zeros(NL)
    for i in pool:
        pos_cnt += multi_hot(tr[i]["labels"])
    pw = ((len(pool) - pos_cnt) / pos_cnt.clamp(min=1)).sqrt().clamp(1.0, args.max_pos_weight).to(device)
    if args.no_pos_weight:
        pw = None
    log(json.dumps({"arch": model.backbone.arch, "layers_kept": len(model.backbone.layers_kept), **peft,
                    "trainable_total": n_tr, "train_pool": len(pool), "docs_per_epoch": per_epoch,
                    "dev_docs": len(dev_ids), "dev_select_docs": len(sel_ids), "device": str(device),
                    "amp": str(amp), "max_tokens_per_doc": args.chunk_len * (args.head_chunks + args.tail_chunks),
                    "pos_weight_range": [round(float(pw.min()), 2), round(float(pw.max()), 2)] if pw is not None else None}))
    best, best_state, hist = -1.0, None, []
    feats = None
    allrecs = {**tr, **dv, **te}
    if args.method == "head":
        need = set(rng.sample(pool, per_epoch)) | set(dev_ids) | set(test_ids)
        pool = [i for i in pool if i in need]
        log(f"method head: caching frozen chunk features for {len(need)} docs")
        feats = chunk_features(model, allrecs, sorted(need), args, device, amp, log)
    groups = C.param_groups(model, args.lr, args.lr_head, args.lr_adapter, args.weight_decay)
    opt = torch.optim.AdamW(groups)
    stepper = C.AmpStep(opt, amp)
    params = [p for p in model.parameters() if p.requires_grad]

    def epoch_batches():
        ids = rng.sample(pool, per_epoch)
        if feats is not None:
            rng.shuffle(ids)
            return [ids[k:k + args.batch_docs] for k in range(0, len(ids), args.batch_docs)]
        return train_batches(tr, ids, args, rng)

    est = len(epoch_batches())
    steps = args.epochs * math.ceil(est / args.accum)
    sched = C.warmup_linear(opt, steps, args.warmup)

    def logits_for(ids):
        if feats is not None:
            return head_logits(model, feats, ids, device)
        return doc_logits(model, allrecs, ids, args, device, amp)

    for ep in range(1, args.epochs + 1):
        model.train()
        bl = epoch_batches()
        t0, tot, nb = time.time(), 0.0, 0
        for chunk in bl:
            tl = None
            if feats is not None:
                v, pos, owner = head_inputs(feats, chunk, device)
                logits = model.head(v, pos, owner, len(chunk))
                y = torch.stack([multi_hot(tr[i]["labels"]) for i in chunk]).to(device)
                if teacher is not None:
                    tl = torch.stack([teacher[i] for i in chunk]).to(device)
            else:
                b = collate(tr, chunk, args, teacher)
                y = b["y"].to(device)
                tl = b["teacher"].to(device) if teacher is not None else None
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    logits = model(b, device)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pw)
            if tl is not None:
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd_bce(logits, tl, args.kd_T)
            (stepper.scaler.scale(loss / args.accum) if stepper.scaler else loss / args.accum).backward()
            nb += 1
            tot += loss.item()
            if nb % args.accum == 0 or nb == len(bl):
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
                log(f"  ep {ep} batch {nb}/{len(bl)} loss {tot / nb:.4f} {time.time() - t0:.0f}s")
        row = {"epoch": ep, "loss": round(tot / max(nb, 1), 4), "sec": round(time.time() - t0, 1)}
        if sel_ids:
            f1 = quick_dev_f1(dv, to_prob(logits_for(sel_ids)))
            row["dev_sel_f1"] = round(f1, 4)
            if f1 > best:
                best, best_state = f1, C.trainable_state(model)
        hist.append(row)
        log(json.dumps(row))
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    run = {"args": vars(args), "peft": peft, "trainable": n_tr, "history": hist, "arch": model.backbone.arch,
           "layers_kept": model.backbone.layers_kept, "data": cache["info"]}
    thr = 0.5
    if dev_ids:
        t0 = time.time()
        dprob = to_prob(logits_for(dev_ids))
        torch.save(dprob, os.path.join(args.out, "dev_prob.pt"))
        ch = choose_thresholds(dv, dprob)
        thr = ch["thr"]
        run.update({k: v for k, v in ch.items() if k != "thr"}, threshold=thr)
        log(f"dev Macro-F1: honest (cross-half) {ch['dev_f1_honest']}, tuned {ch['dev_f1_tuned']} "
            f"[{ch['rule']}], at 0.5 {ch['dev_f1_at_0.5']} ({time.time() - t0:.0f}s)")
    t0 = time.time()
    tprob = to_prob(logits_for(test_ids))
    torch.save(tprob, os.path.join(args.out, "test_prob.pt"))
    pred = predictions(test_ids, tprob, thr)
    log(validate(test_ids, pred) + f" ({time.time() - t0:.0f}s)")
    write_submission(pred, args.out)
    if args.save_train_logits:
        ids = sorted(cache["train"])
        ids = random.Random(1234).sample(ids, min(args.save_train_logits, len(ids)))
        if feats is not None:
            ids = [i for i in ids if i in feats]
        log(f"teacher: logits on {len(ids)} train docs")
        tl = logits_for(ids)
        torch.save({k: v.half() for k, v in tl.items()}, os.path.join(args.out, "train_logits.pt"))
    if args.save_model:
        C.save_trainable(model, os.path.join(args.out, "model.pt"), run)
    C.write_json(run, os.path.join(args.out, "run.json"))
    log(f"wrote {args.out}/submission.zip")


def dev_labels(data):
    data_raw, _ = load_raw(data, need_train=False)
    return {r["id"]: {"labels": r["labels"]} for r in data_raw["dev"]}, [r["id"] for r in data_raw["test"]]


def load_probs(run, name):
    p = os.path.join(run, name)
    return torch.load(p) if os.path.exists(p) else None


def ensemble(args):
    dv, test_ids = dev_labels(args.data)
    w = [float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(args.runs)
    tp, dp = None, None
    for r, wi in zip(args.runs, w):
        t = load_probs(r, "test_prob.pt")
        tp = {k: wi * t[k].float() for k in t} if tp is None else {k: tp[k] + wi * t[k].float() for k in tp}
        d = load_probs(r, "dev_prob.pt")
        if dp is not False and d is not None:
            dp = {k: wi * d[k].float() for k in d} if dp is None else {k: dp[k] + wi * d[k].float() for k in dp if k in d}
        else:
            dp = False
    s = sum(w)
    tp = {k: (v / s).half() for k, v in tp.items()}
    os.makedirs(args.out, exist_ok=True)
    info, thr = {"runs": args.runs, "weights": w}, 0.5
    if dp:
        dp = {k: (v / s).half() for k, v in dp.items()}
        ch = choose_thresholds(dv, dp)
        thr = ch["thr"]
        info.update({k: v for k, v in ch.items() if k != "thr"}, threshold=thr)
        torch.save(dp, os.path.join(args.out, "dev_prob.pt"))
    pred = predictions(test_ids, tp, thr)
    print(validate(test_ids, pred))
    write_submission(pred, args.out)
    torch.save(tp, os.path.join(args.out, "test_prob.pt"))
    C.write_json(info, os.path.join(args.out, "run.json"))
    print(json.dumps({k: v for k, v in info.items() if k != "threshold"}))


# --------------------------------------------------------------------------------------
# TF-IDF reference baseline (CPU; NOT a Phase 1 model, reference only)
# --------------------------------------------------------------------------------------
def baseline(args):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    data, info = load_raw(args.data)
    sl = slice(args.char_start, args.char_end)
    T = lambda s: [clean_text(r["text"])[sl] for r in s]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=400000, sublinear_tf=True,
                          dtype=np.float32, token_pattern=r"(?u)<\w+>|\b\w\w+\b")
    Xtr = vec.fit_transform(T(data["train"]))
    Xdv, Xte = vec.transform(T(data["dev"])), vec.transform(T(data["test"]))
    for m in (Xtr, Xdv, Xte):
        m.sort_indices()
    ytr = np.stack([multi_hot(r["labels"]).numpy() for r in data["train"]]).astype(np.uint8)
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=args.C, class_weight="balanced"),
                              n_jobs=args.jobs).fit(Xtr, ytr)
    os.makedirs(args.out, exist_ok=True)
    dprob = {r["id"]: torch.tensor(p).half() for r, p in zip(data["dev"], clf.predict_proba(Xdv))}
    tprob = {r["id"]: torch.tensor(p).half() for r, p in zip(data["test"], clf.predict_proba(Xte))}
    recs = {r["id"]: r for r in data["dev"]}
    ch = choose_thresholds(recs, dprob)
    torch.save(dprob, os.path.join(args.out, "dev_prob.pt"))
    torch.save(tprob, os.path.join(args.out, "test_prob.pt"))
    pred = predictions([r["id"] for r in data["test"]], tprob, ch["thr"])
    print(validate([r["id"] for r in data["test"]], pred))
    write_submission(pred, args.out)
    run = {"model": "tfidf_ovr_lr (REFERENCE ONLY, do not submit)", "C": args.C, "chars": [args.char_start, args.char_end],
           "features": Xtr.shape[1], "data": info, **{k: v for k, v in ch.items() if k != "thr"}, "threshold": ch["thr"]}
    C.write_json(run, os.path.join(args.out, "run.json"))
    print(json.dumps({k: v for k, v in run.items() if k != "threshold"}))


def score_file(data, path):
    dv, _ = dev_labels(data)
    if os.path.isdir(path):
        ch = choose_thresholds(dv, torch.load(os.path.join(path, "dev_prob.pt")))
        print(json.dumps({k: v for k, v in ch.items() if k != "thr"}))
        return
    pred = {p["id"]: p["pred"] for p in json.load(open(path))}
    ids = [i for i in dv if i in pred]
    Y = torch.stack([multi_hot(dv[i]["labels"]) for i in ids]).bool()
    P = torch.stack([multi_hot(pred[i]) for i in ids]).bool()
    print(f"dev docs scored: {len(ids)}  Macro-F1 {macro_f1(Y, P):.6f}  Micro-F1 {micro_f1(Y, P):.6f}  "
          f"subset acc {float((Y == P).all(1).float().mean()):.6f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--keep_head", type=int, default=3072)
    p.add_argument("--keep_tail", type=int, default=1024)
    p.add_argument("--keep_all", type=int, default=0, help="also keep the first N tokens of every doc (for --select cue)")
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
    t.add_argument("--head_chunks", type=int, default=3, help="chunks from the start of the document")
    t.add_argument("--tail_chunks", type=int, default=1, help="chunks aligned to the end (not overlapping the head)")
    t.add_argument("--select", default="headtail", choices=["headtail", "cue"],
                   help="cue: first chunk + the chunks with the most <SECTION>/<ACT> markers (needs prep --keep_all)")
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--docs_per_epoch", type=int, default=16000, help="fresh random sample each epoch (0 = all)")
    t.add_argument("--batch_docs", type=int, default=4, help="token budget per batch = batch_docs x max doc tokens")
    t.add_argument("--max_batch_docs", type=int, default=32)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--eval_batch", type=int, default=8)
    t.add_argument("--dev_select_docs", type=int, default=2500, help="dev docs used for per-epoch selection (0 = all)")
    t.add_argument("--lr", type=float, default=3e-5, help="backbone LR (full / norms)")
    t.add_argument("--lr_adapter", type=float, default=5e-4, help="LoRA / LORMA LR")
    t.add_argument("--lr_head", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--warmup", type=float, default=0.06)
    t.add_argument("--max_pos_weight", type=float, default=10.0)
    t.add_argument("--no_pos_weight", action="store_true")
    t.add_argument("--teacher", action="append", help="run dir(s) with train_logits.pt (distillation)")
    t.add_argument("--kd_alpha", type=float, default=0.5)
    t.add_argument("--kd_T", type=float, default=2.0)
    t.add_argument("--full", action="store_true", help="train on train+dev, no dev selection, threshold 0.5")
    t.add_argument("--save_train_logits", type=int, default=0, help="teacher: logits on this many train docs")
    t.add_argument("--save_model", action="store_true")
    t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--log_every", type=int, default=500)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--limit_docs", type=int, default=0, help="smoke tests only")
    e = sub.add_parser("ensemble")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--weights", default="")
    e.add_argument("runs", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("pred")
    v = sub.add_parser("validate")
    v.add_argument("--data", required=True)
    v.add_argument("pred")
    b = sub.add_parser("baseline")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--C", type=float, default=4.0)
    b.add_argument("--char_start", type=int, default=None)
    b.add_argument("--char_end", type=int, default=40000)
    b.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "prep":
        prep(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "ensemble":
        ensemble(args)
    elif args.cmd == "score":
        score_file(args.data, args.pred)
    elif args.cmd == "baseline":
        baseline(args)
    else:
        _, test_ids = dev_labels(args.data)
        print(validate(test_ids, json.load(open(args.pred))))


if __name__ == "__main__":
    main()
