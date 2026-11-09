# Phase 4: Court Judgment Prediction (CJPE)

**Task.** Predict the binary outcome (appeal accepted / rejected) of an Indian Supreme Court judgment whose
decision has been removed (ILDC). Metric: Macro-F1.

**Best submission: Codabench 0.7539**, `kd_unilm6`: a 6-layer (67M) unilm student distilled from the full fine-tuned unilm.

Code: `p4_cjpe.py` (`prep`, `train`, `ensemble`, `score`, `validate`); notebook
`CS781-CP-Phase4-Darsh-Kedia-250311.ipynb`. The round-2 notebook
was built but never run.

## Data hygiene (data finding)
`Test/test.json` (1,517 docs) is entirely inside `Train/dev.json`, with labels; the other dev docs appear twice.
`p4_cjpe.py` removes the test ids from dev and de-duplicates it (`dev_clean`, 994 docs, balanced). Train is
de-duplicated (37,387 records → 31,677 texts, 59% label 0).

## Method
- Text clean-up: ILDC has line breaks inside words ("t\nhe"); a break between letters is joined when the joined word is more frequent than the rarer fragment.
- Hierarchical tail encoder: the last 4 × 512 tokens, chunks aligned to the end of the document, each chunk
  mean-pooled from the Phase 1 backbone, + position-from-end embedding, BiGRU + attention pooling, 2-way head.
  Class-balanced CE; decision threshold tuned on dev_clean. Why the tail: TF-IDF + LR on dev_clean scores
  0.655 on the last 5k chars vs 0.576 on the first 5k (0.665 whole doc).
- 12k freshly sampled train docs per epoch; two jobs at a time on T4 x2.

## Results (dev_clean Macro-F1 at tuned threshold; Codabench = test)
| run | dev | Codabench |
|---|---|---|
| **kd_unilm6** (6-layer student) | 0.7686 | **0.7539** |
| ens_greedy (forward selection on dev) | 0.7826 | 0.7442 |
| ens_noencdec / ens_unilm / ens_all | 0.7635 / 0.7655 / 0.7605 | 0.7429 / 0.7421 / 0.7376 |
| unilm_full (teacher) | 0.7736 | 0.7383 |
| gpt_full | 0.7512 | 0.7350 |
| retnet_lorma | 0.7442 | 0.7330 |
| ens_archs | 0.7219 | 0.7224 |
| unilm_lora | 0.7287 | 0.7031 |
| gpt_lorma | 0.6908 | 0.7020 |
| unilm_lorma_pi | 0.7203 | 0.6837 |
| unilm_lorma | 0.6704 | 0.6657 |
| encdec_lorma | 0.6214 | 0.6096 |
| unilm_head (frozen) | 0.6358 | 0.5962 |
| unilm_frozen_probe (CPU, 200M unilm) | 0.604 | 0.5652 |
Kaggle: 263 min on T4 x2, 10 jobs, none failed.
