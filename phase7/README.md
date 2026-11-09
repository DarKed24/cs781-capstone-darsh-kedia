# Phase 7: Prior Case Retrieval (PCR)

**Task.** For each query judgment return exactly 7 ranked candidate ids from the split's pool. Metric:
Micro-F1@7. Train 827 queries / 4,320 candidates, dev 118 / 1,023, test 237 / 1,727; ~9k tokens per document.

**Best submission: Codabench 0.4566**, `bm25bb_ctx_date`: bigram BM25 tuned for document-length queries
+ citation-context query + temporal feasibility filter.

## Files
- `make_lex_zips.py`: builds the 8-rung lexical ablation ladder, scores each on dev, writes the test zips (CPU, minutes).
  `python make_lex_zips.py DATA_DIR OUT_DIR [config ...]` with `DATA_DIR` = `data/Phase_7_PCR`.
- `p7_lex.py`: BM25 (sparse matrix form) with switches for every idea; strips the `CITATION`/`CITATOR INFO` header.
- `p7_dense.py` + `...ipynb`: passage-level MaxSim bi-encoder on Phase 1 models with
  distantly supervised (TF-IDF) alignment, InfoNCE, and z-score fusion with BM25. Ran on Kaggle; dev-only results.
- The CITATOR INFO data finding (header lists later citing cases; 99% precise on dev) is stripped from all inputs and never used.

## Ablation ladder (dev Micro-F1@7 → Codabench)
| config | dev | Codabench |
|---|---|---|
| bm25_plain (defaults, unigrams) | 0.2012 | 0.2262 |
| + long-document tuning (k1 6, b 1, k3 → ∞, max_df 0.3) | 0.3735 | 0.3662 |
| + citation-context query (w 0.7) | 0.3902 | 0.3804 |
| + temporal filter | 0.4169 | 0.4217 |
| unigrams + bigrams | 0.4591 | 0.4513 |
| 1-3 grams | 0.4625 | 0.4560 |
| **bigrams only** | 0.4658 | **0.4566** |
| + citation prior 0.3 | 0.4669 | 0.4507 |

Dense (dev only, not submitted): zero-shot unilm 0.220 → contrastively fine-tuned 0.325-0.329 (all archs);
random alignment 0.301; best fusion with BM25 0.4691 (+0.003).
