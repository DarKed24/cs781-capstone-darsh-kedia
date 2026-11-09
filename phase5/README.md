# Phase 5: Legal Statute Identification (LSI)

**Task.** Multi-label: assign one or more of 100 statute ids to each judgment (statute names masked as
`<SECTION>` / `<ACT>`). Metric: Macro-F1 over the 100 labels. Train 42,750 / dev 10,181 / test 13,019 docs.

**Best submission: Codabench 0.3385**, `ens_greedy` (8-member ensemble chosen on dev). TF-IDF + logistic regression
reference (not a Phase 1 model, never submitted): 0.405 dev.

Code: `../finetune/p5_lsi.py` (`prep`, `train`, `ensemble`, `baseline`), `../finetune/p5_post.py` (label-structure
post-processing), notebook `../finetune/kaggle_p5/CS781-CP-Phase5-Darsh-Kedia-250311.ipynb`. The round-2 notebook
(cue-based chunk selection) was built but not run. `NOTES.md` has the full method, ablations and failures.

## Method
First 3 × 512 + last 512 tokens (the whole document for ~70% of docs), each chunk mean-pooled from the backbone,
+ chunk-position embedding, BiGRU, [attention; max] pooling, 100 sigmoids. BCE with pos_weight = sqrt(neg/pos)
clipped at 10. Thresholds (global or per-label) chosen by an honest cross-half protocol on dev (halves fixed by
crc32(id)); every document gets at least its top label.

## Codabench ladder (16 scored; dev honest in brackets)
ens_greedy **0.3385** (0.3365), ens_all 0.3352 (0.3322), ens_noencdec 0.3340 (0.3315), ens_top3 0.3328 (0.3308),
ens_archs 0.3297 (0.3262), unilm_lora 0.3273 (0.3252), unilm_lorma 0.3269 (0.3216), gpt_lorma 0.3218 (0.3210),
unilm_lorma_pi 0.3211 (0.3183), retnet_lorma 0.3188 (0.3138), gpt_full 0.3173 (0.3122), kd_unilm6 / unilm_full
0.3118 / 0.3121 (0.3097 / 0.3070; the two uploads were mixed up), unilm_head 0.2771 (0.2764), encdec_lorma 0.2656
(0.2596), unilm_frozen_probe 0.2313 (0.232).
