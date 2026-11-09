# Phase 5: Legal Statute Identification (LSI)

**Task.** Multi-label: assign one or more of 100 statute ids to each judgment (statute names masked as
`<SECTION>` / `<ACT>`). Metric: Macro-F1 over the 100 labels. Train 42,750 / dev 10,181 / test 13,019 docs. Each epoch samples 14k fresh training documents.

**My best submission: 0.3385 on Codabench**, `ens_greedy`, an ensemble of 8 runs picked on dev. None of my
fine-tuned models beat a plain TF-IDF + logistic regression reference (0.405 dev), which I used only as a bar and never
submitted. My guess is that the models only read 2k tokens of each document and saw about one pass over train, while the
offence words that decide the statute are exactly what bag-of-words picks up.

Code: `p5_lsi.py` (`prep`, `train`, `ensemble`, `baseline`), `p5_post.py` (label-structure
post-processing), notebook `CS781-CP-Phase5-Darsh-Kedia-250311.ipynb`. The round-2 notebook
(cue-based chunk selection) was built but not run.

## Method
First 3 × 512 + last 512 tokens (the whole document for ~70% of docs), each chunk mean-pooled from the backbone,
+ chunk-position embedding, BiGRU, [attention; max] pooling, 100 sigmoids. BCE with pos_weight = sqrt(neg/pos)
clipped at 10. To pick thresholds (one global or one per label) without fooling myself, I tune on one half of dev and score on the
other half (halves fixed by crc32(id)), both ways. Every document gets at least its top label.

## All my submissions (16 scored; dev = cross-half Macro-F1)
| zip | backbone | method | dev | Codabench |
|---|---|---|---|---|
| ens_greedy | 8 runs chosen by forward selection on dev | mean of probabilities, per-label thresholds | 0.3365 | **0.3385** |
| ens_all | all trained runs | mean | 0.3322 | 0.3352 |
| ens_noencdec | all runs except encdec | mean | 0.3315 | 0.3340 |
| ens_top3 | 3 best runs on dev | mean | 0.3308 | 0.3328 |
| ens_archs | one run per architecture | mean | 0.3262 | 0.3297 |
| unilm_lora | unilm 400M | LoRA r16, 3 epochs | 0.3252 | 0.3273 |
| unilm_lorma | unilm 400M | LoRMA+ r16 | 0.3216 | 0.3269 |
| gpt_lorma | gpt | LoRMA+ r16 | 0.3210 | 0.3218 |
| unilm_lorma_pi | unilm 400M | LoRMA-pi r16 | 0.3183 | 0.3211 |
| retnet_lorma | retnet | LoRMA+ r16 | 0.3138 | 0.3188 |
| gpt_full | gpt | full fine-tuning, 2 epochs | 0.3122 | 0.3173 |
| unilm_full | unilm 400M | full fine-tuning (KD teacher) | 0.3070 | 0.3121 or 0.3118 |
| kd_unilm6 | 6-layer unilm student (67M) | distilled from unilm_full | 0.3097 | 0.3118 or 0.3121 |
| unilm_head | unilm 400M frozen | head only | 0.2764 | 0.2771 |
| encdec_lorma | encdec encoder | LoRMA+ r16 | 0.2596 | 0.2656 |
| unilm_frozen_probe | unilm 200M frozen, first 512 tokens, CPU | head only | 0.232 | 0.2313 |

The unilm_full and kd_unilm6 uploads were swapped by mistake, so which of the two scores belongs to which is unknown.
Every fine-tuned model stays below the TF-IDF reference (0.405); PEFT beats full fine-tuning here.
