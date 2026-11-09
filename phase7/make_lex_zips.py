#!/usr/bin/env python3
"""Builds the CPU lexical Phase 7 submissions (an ablation ladder) and scores each config on dev.

  python make_lex_zips.py DATA_DIR OUT_DIR [name ...]
Dev uses the train labels for the citation prior; test uses train+dev labels.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune"))
import p7_lex as L

T = dict(k3=1000, k1=6.0, b=1.0, max_df=0.3)          # tuned BM25 (on train, checked on dev)
CONFIGS = {
    "bm25_plain":          {},
    "bm25_tuned":          T,
    "bm25_ctx":            {**T, "ctx": 1, "ctx_w": 0.7},
    "bm25_ctx_date":       {**T, "ctx": 1, "ctx_w": 0.7, "temporal": 1},
    "bm25bi_ctx_date":     {**T, "ctx": 1, "ctx_w": 0.7, "temporal": 1, "ngram": 2, "min_df": 2},
    "bm25bb_ctx_date":     {**T, "ctx": 1, "ctx_w": 0.7, "temporal": 1, "ngram": 2, "ngram_min": 2, "min_df": 2},
    "bm25tri_ctx_date":    {**T, "ctx": 1, "ctx_w": 0.7, "temporal": 1, "ngram": 3, "min_df": 2},
    "bm25bb_ctx_date_pri": {**T, "ctx": 1, "ctx_w": 0.7, "temporal": 1, "ngram": 2, "ngram_min": 2, "min_df": 2, "prior": 0.3},
}

if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(CONFIGS)
    tq, _ = L.load(data, "train"); dq, dc = L.load(data, "dev"); eq, ec = L.load(data, "test")
    pr_dev, pr_test = L.prior_from(tq), L.prior_from(tq, dq)
    for n in names:
        cfg = CONFIGS[n]; t = time.time()
        f, p, r = L.micro_f1(dq, L.retrieve(dq, dc, cfg, pr_dev))
        pred = L.retrieve(eq, ec, cfg, pr_test)
        assert len(pred) == len(eq) and all(len(set(v)) == 7 and all(x in ec for x in v) for v in pred.values())
        d = os.path.join(out, n); L.write_zip(pred, d)
        json.dump({"cfg": {**L.DEFAULT, **cfg}, "dev": {"f1": f, "p": p, "r": r}}, open(f"{d}/run.json", "w"), indent=1)
        print(f"{n:22s} dev F1 {f:.4f} P {p:.4f} R {r:.4f}  ({time.time()-t:.0f}s)", flush=True)
