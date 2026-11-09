#!/usr/bin/env python3
"""CS781 Phase 5 (LSI): label-structure post-processing on saved probabilities (Darsh Kedia, 250311).

Why: a binary-relevance head scores each of the 100 statutes independently, but statutes come in
packages (e.g. a murder conviction cites the penal-code section for murder together with the
common-intention and procedure sections). The masked text (<SECTION>, <ACT>) hides the statute
names, so the label structure is information the encoder cannot see directly. This script tests
three ways of putting it back, on top of any finished run or ensemble (CPU only, no retraining):

  cooc      co-occurrence prior from TRAIN labels: q = (1 - lam) p + lam * (p @ M) / sum(p),
            M[k, j] = P(statute j | statute k) counted on train. lam tuned on dev.
  stack     second-stage "label-interaction" stacker: for each statute j a logistic regression on
            the logits of all 100 first-stage scores (learns which statutes raise or suppress j;
            a learned, discriminative version of the prior; cf. stacked binary relevance,
            Godbole & Sarawagi 2004). Fit on dev with cross-fitting.
  topk      cardinality decoding: predict the top-k labels with k = clip(round(c * sum(p)), 1, 15)
            (sum of calibrated probabilities = expected label count), c tuned on dev, instead of a
            probability threshold.
  resid     residual label-interaction layer (our variant): s_j = a_j logit(p_j) + sum_k W_jk logit(p_k)
            + b_j, starting exactly at the first-stage model, cross-label weights W L2-penalised.
            Unlike `stack` it cannot do worse than "base" without dev evidence, which matters for
            rare statutes (68 dev docs for the rarest).
  resid+topk  both.

Honest evaluation (the ablation numbers): dev is split in two halves by crc32(id) (the same
split as p5_lsi.choose_thresholds). Every parameter (lam, c, stacker weights, thresholds) is fitted
on one half and scored on the other, both ways, and averaged. Thresholds for the stacker are tuned
on out-of-fold stacker scores inside the fitting half, so nothing is scored on data it was fitted on.
For the test submission, parameters are re-fitted on all of dev the same way.

  python p5_post.py --data DIR --out OUT RUN1 [RUN2 ...]     (several runs are averaged first)
Writes OUT/<variant>/{prediction.json, submission.zip, run.json} and OUT/ablation.json.
"""
import argparse
import json
import os
import sys
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_lsi as P5

NL = P5.NL
LAMS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
CS = [round(0.6 + 0.05 * k, 2) for k in range(17)]   # 0.6 .. 1.4


def load_avg(runs, name):
    tot = None
    for r in runs:
        d = torch.load(os.path.join(r, name))
        tot = {k: v.float() for k, v in d.items()} if tot is None else {k: tot[k] + d[k].float() for k in tot}
    return {k: v / len(runs) for k, v in tot.items()}


def cooc_matrix(data):
    raw, _ = P5.load_raw(data)
    Y = torch.stack([P5.multi_hot(r["labels"]) for r in raw["train"]])
    co = Y.T @ Y                                     # co[k, j] = #docs with k and j
    return co / co.diag().clamp(min=1)[:, None]      # P(j | k), diagonal 1


def apply_cooc(Pm, M, lam):
    prior = (Pm @ M) / Pm.sum(1, keepdim=True).clamp(min=1e-6)
    return ((1 - lam) * Pm + lam * prior.clamp(max=1.0)).clamp(0, 1)


def topk_decide(Pm, c):
    k = (c * Pm.sum(1)).round().clamp(1, 15).long()
    order = Pm.argsort(1, descending=True)
    out = torch.zeros_like(Pm, dtype=torch.bool)
    ranks = torch.arange(NL)[None, :] < k[:, None]
    out.scatter_(1, order, ranks)
    return out


def logit(Pm):
    p = Pm.clamp(1e-4, 1 - 1e-4)
    return (p / (1 - p)).log()


def fit_stack(Y, Pm, C=0.5):
    from sklearn.linear_model import LogisticRegression
    X = logit(Pm).numpy()
    W = torch.zeros(NL, NL)
    b = torch.zeros(NL)
    for j in range(NL):
        y = Y[:, j].numpy()
        if y.min() == y.max():
            b[j] = -10.0 if y.max() == 0 else 10.0
            continue
        m = LogisticRegression(C=C, max_iter=300).fit(X, y)
        W[j] = torch.tensor(m.coef_[0], dtype=torch.float32)
        b[j] = float(m.intercept_[0])
    return W, b


def apply_stack(params, Pm):
    W, b = params
    return torch.sigmoid(logit(Pm) @ W.T + b)


def fit_resid(Y, Pm, l2=1e-3, steps=300):
    """Residual label-interaction layer: s_j = a_j * logit(p_j) + sum_k W_jk logit(p_k) + b_j with
    W (zero diagonal) L2-penalised, a_j = 1 and W = 0 at the start, so it begins exactly at the
    first-stage model and moves only as far as dev evidence supports. All labels fitted jointly."""
    X = logit(Pm)
    Yf = Y.float()
    W = torch.zeros(NL, NL, requires_grad=True)
    a = torch.ones(NL, requires_grad=True)
    b = torch.zeros(NL, requires_grad=True)
    off = 1 - torch.eye(NL)
    pos = Yf.sum(0)   # same sqrt(neg/pos) weighting as the first stage, so scores stay on its scale
    pw = ((len(Yf) - pos) / pos.clamp(min=1)).sqrt().clamp(1.0, 10.0)
    opt = torch.optim.Adam([W, a, b], lr=0.01)
    for _ in range(steps):
        s = X * a + X @ (W * off).T + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, Yf, pos_weight=pw) + l2 * (W * off).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (W * off).detach(), a.detach(), b.detach()


def apply_resid(params, Pm):
    W, a, b = params
    X = logit(Pm)
    return torch.sigmoid(X * a + X @ W.T + b)


def halves(ids):
    return torch.tensor([zlib.crc32(i.encode()) % 2 for i in ids])


def bagged(Y, Pm, fit, apply):
    """Two models on the two inner folds of the fitting set. Returns their out-of-fold scores
    (used to tune the threshold) and a transform that averages both models, so the threshold is
    tuned on the same kind of scores it is applied to."""
    f = torch.arange(len(Y)) % 2
    out = torch.zeros_like(Pm)
    fits = []
    for a in (0, 1):
        tr, te = f != a, f == a
        fits.append(fit(Y[tr], Pm[tr]))
        out[te] = apply(fits[-1], Pm[te])
    return out, (lambda Q: sum(apply(p, Q) for p in fits) / len(fits))


def fit_variant(name, Y, Pm, M):
    """Returns (transform(Pm) -> scores, decide(scores) -> bool) fitted on (Y, Pm)."""
    if name == "base":
        thr = P5.tune_global(Y, Pm)
        return (lambda Q: Q), (lambda Q: P5.decide(Q, thr)), {"thr": thr}
    if name == "cooc":
        best = max((P5.macro_f1(Y, P5.decide(apply_cooc(Pm, M, l), P5.tune_global(Y, apply_cooc(Pm, M, l)))), -l, l)
                   for l in LAMS)
        lam = best[2]
        thr = P5.tune_global(Y, apply_cooc(Pm, M, lam))
        return (lambda Q: apply_cooc(Q, M, lam)), (lambda Q: P5.decide(Q, thr)), {"lam": lam, "thr": thr}
    if name == "topk":
        c = max((P5.macro_f1(Y, topk_decide(Pm, c)), -abs(c - 1), c) for c in CS)[2]
        return (lambda Q: Q), (lambda Q: topk_decide(Q, c)), {"c": c}
    if name.split("+")[0] in ("resid", "stack"):
        fit, apply = (fit_resid, apply_resid) if name.startswith("resid") else (fit_stack, apply_stack)
        oof, tf = bagged(Y, Pm, fit, apply)
        if "+topk" not in name:
            thr = P5.tune_global(Y, oof)
            return tf, (lambda Q: P5.decide(Q, thr)), {"thr": thr}
        c = max((P5.macro_f1(Y, topk_decide(oof, c)), -abs(c - 1), c) for c in CS)[2]
        return tf, (lambda Q: topk_decide(Q, c)), {"c": c}
    raise ValueError(name)


VARIANTS = ["base", "cooc", "topk", "resid", "resid+topk", "stack"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    dv, test_ids = P5.dev_labels(a.data)
    ids, Y, Pm = P5.dev_matrix(dv, load_avg(a.runs, "dev_prob.pt"))
    tp = load_avg(a.runs, "test_prob.pt")
    Pt = torch.stack([tp[i] for i in test_ids])
    M = cooc_matrix(a.data)
    h = halves(ids)
    os.makedirs(a.out, exist_ok=True)
    table = {}
    for name in a.variants.split(","):
        mac, mic, card = [], [], []
        for s in (0, 1):
            A, B = h == s, h != s
            tf, dec, _ = fit_variant(name, Y[A], Pm[A], M)
            PB = dec(tf(Pm[B]))
            mac.append(P5.macro_f1(Y[B], PB))
            mic.append(P5.micro_f1(Y[B], PB))
            card.append(float(PB.sum(1).float().mean()))
        tf, dec, params = fit_variant(name, Y, Pm, M)
        PT = dec(tf(Pt))
        pred = [{"id": i, "pred": [int(j) for j in torch.nonzero(PT[k]).flatten().tolist()]} for k, i in enumerate(test_ids)]
        d = os.path.join(a.out, name.replace("+", "_"))
        os.makedirs(d, exist_ok=True)
        msg = P5.validate(test_ids, pred)
        P5.write_submission(pred, d)
        table[name] = {"dev_f1_honest": round(sum(mac) / 2, 4), "dev_micro_honest": round(sum(mic) / 2, 4),
                       "labels_per_doc": round(sum(card) / 2, 2), "params_full_dev": params}
        json.dump({"runs": a.runs, "variant": name, **table[name]}, open(os.path.join(d, "run.json"), "w"), indent=1)
        print(f"{name:11s} honest Macro-F1 {table[name]['dev_f1_honest']:.4f}  Micro {table[name]['dev_micro_honest']:.4f}  "
              f"labels/doc {table[name]['labels_per_doc']:.2f}  {params}  test: {msg}", flush=True)
    json.dump({"runs": a.runs, "gold_labels_per_doc": round(float(Y.sum(1).float().mean()), 2), "table": table},
              open(os.path.join(a.out, "ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
