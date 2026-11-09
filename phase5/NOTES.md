# Phase 5 (Legal Statute Identification): method, ablations and failures

Phase 5 notes (Darsh Kedia, 250311). They cover the reasoning, not just the leaderboard. Numbers marked *measured* come from runs.

## 1. What the data says, and what follows from it
- There are 100 statutes, and every document has at least one (3.7 on average in train). The rarest statute has 272 train docs and 68 dev docs. Macro-F1 weights each statute equally, so rare statutes matter as much as section 302. *Measured.*
- **Statute names are masked, but where they were cited is still visible.** Every citation becomes `<SECTION>` or `<ACT>` ("convicted under `<SECTION>` read with `<SECTION>` of the `<ACT>`"). The model therefore has to infer the statute from the facts and charge around each citation. It cannot read the statute's name. The median document has 8 markers, and 9.5% have none. *Measured on dev.*
- **Documents are long and uneven.** The median is 0.8k tokens, the mean 2.0k and the 90th percentile 4.4k. A fixed window of first 3 x 512 + last 512 tokens covers the whole document for 70% of docs. For the other 30%, it contains only **60% of the statute markers**, and for 27% of those long docs it misses most of them. *Measured on dev.* This observation drives the main original idea (section 3).
- Dev and test share no ids, so there is no Phase 4-style leak. There are about 300 exact-duplicate texts, mostly one-line orders such as "ORDER `<ENTITY>`, J.". The 210 train docs whose text also appears in dev were dropped. *Measured.*

## 2. Base method
- Each chunk is encoded by the Phase 1 backbone and mean-pooled. A chunk-position embedding is added. A BiGRU runs over the chunks, then attention pooling and max pooling, then 100 sigmoids (binary relevance).
  - Max pooling is there because a statute can be signalled by a single chunk.
- Loss: BCE with pos_weight = sqrt(neg/pos), clipped at 10. This pushes recall on rare statutes, which is what Macro-F1 rewards.
- Decision rule: a global threshold, or one threshold per label. Per-label is used only when it wins an honest cross-half check on dev. Every document gets at least its top label, since no gold set is empty.
  - On the TF-IDF reference, per-label thresholds did *not* beat a global one (0.4025 vs 0.4050, cross-half). With 68 dev positives for the rarest statute, 100 separate thresholds overfit. *Measured.*
- **How dev is used honestly.** Every tuned quantity (threshold, post-processing parameters) is fitted on one half of dev and scored on the other, both ways. The halves are fixed by crc32(id). That cross-half score is `dev_f1_honest`; `dev_f1_tuned` is the optimistic same-data number.
- Fine-tuning techniques: full fine-tuning, LoRA, LORMA+ and LORMA-pi; knowledge distillation into a 6-layer student; a frozen probe. The plan covers all four architectures.

## 3. Original idea 1: evidence-guided context selection (`--select cue`)
- **Idea.** Spend the fixed 4 x 512-token budget where the statutes are cited. Keep the first chunk (parties, court, charge), then add the 3 chunks with the most `<SECTION>`/`<ACT>` markers, in document order. This is a cheap, label-free relevance signal: a judgment discusses a statute where it cites it.
  - This differs from the usual head/tail truncation for long legal documents. It is also different from attention over all chunks, which costs more compute.
- **Effect on input coverage.** For long dev docs, the share of markers inside the window rises from 0.595 (head+tail) to 0.830 (cue). *Measured.*
- **Ablation.** Round 2 (`kaggle_p5/CS781-CP-Phase5-round2-Darsh-Kedia-250311.ipynb`) trains two pairs that differ only in the context-selection rule:
  - `unilm_full_cue` vs `unilm_full`
  - `unilm_lorma_cue` vs `unilm_lorma`

  It prints `ABLATION ...` lines with dev Macro-F1 for each pair. Report the gain overall and, ideally, split it by document length: short docs are identical under both rules, so any gain must come from the long ones. *Pending.*
- Caveat to state: the selection uses no labels, so it adds no leak, but it does assume the marker positions survive the anonymisation.

## 4. Original idea 2: putting statute structure back (`p5_post.py`)
Statutes come in packages: a murder conviction cites the murder section together with the common-intention section and procedure sections. Binary relevance scores each statute alone. Three ways of adding the structure back were tried on saved probabilities, all on CPU with no retraining:
- `cooc`: a co-occurrence prior from train labels, q = (1-lam)·p + lam·(p·M)/Σp, where M[k,j] = P(j | k).
- `resid`: a residual label-interaction layer, s_j = a_j·logit(p_j) + Σ_k W_jk·logit(p_k) + b_j. It starts exactly at the base model; W is L2-penalised and has a zero diagonal. It is trained with the same pos-weighted BCE and bagged over two inner folds.
- `topk`: cardinality decoding, predicting the top-k labels with k = round(c·Σp).

Honest cross-half dev Macro-F1, *measured*:

| first stage | base | cooc | topk | resid | resid+topk |
|---|---|---|---|---|---|
| TF-IDF reference (10k dev) | 0.4049 | 0.4045 | 0.3953 | **0.4071** | 0.3879 |
| frozen unilm probe (4k dev) | 0.2317 | **0.2322** | 0.2246 | 0.2258 | 0.2212 |

Micro-F1 rises with cooc and resid (+0.004 to +0.007 on TF-IDF), but Macro-F1 barely moves. Rerun on the fine-tuned ensemble; round 2 does this automatically. *Pending.*

## 5. Failures, and why (keep these in the report)
- **Plain stacked binary relevance hurt.** This was a per-label logistic regression on all 100 first-stage logits: 0.2066 vs 0.2317 on the probe. With 2k fitting docs it had 101 weights per label, and the rare labels have about 30 positives. It overfit and was miscalibrated: the threshold fell to 0.14 and it predicted 9.8 labels per doc against 3.9 gold. The fix was to make it residual (start at the base model and penalise only the cross-label weights) and to train it with the same pos-weighting as the first stage. *Measured.*
- **An unweighted residual layer also hurt** (0.2139 on the probe). Fitting with plain BCE recalibrates the scores towards the true base rates, so one global threshold no longer suits both frequent and rare statutes. Adding pos_weight recovered most of the loss (0.2258). *Measured.*
- **Inconsistent thresholds.** The first version tuned thresholds on inner-fold models and then applied a model refitted on all the data, so the score distributions differed. The fix was bagging: average the inner-fold models and tune on their out-of-fold scores. *Measured.*
- **Top-k cardinality decoding always lost** (-0.01 to -0.02). Macro-F1 rewards recall on rare labels, and a per-doc label budget takes slots away from them. *Measured.*
- **The frozen probe is weak** (0.232 vs TF-IDF 0.405). It read only the first 512 tokens with frozen features, which says the backbone needs fine-tuning for this task. *Measured.*
- Pending from the Kaggle runs: does PEFT lose to full fine-tuning again, as in Phase 4? Does encdec stay the weakest? Do dev-selected ensembles hold up on test?

## 6. Final ladder (Codabench, 16 zips scored)
| run | dev honest | Codabench |
|---|---|---|
| TF-IDF reference (never submitted) | 0.405 | n/a |
| **ens_greedy** (8 members) | 0.3365 | **0.3385** (best) |
| ens_all | 0.3322 | 0.3352 |
| ens_noencdec | 0.3315 | 0.3340 |
| ens_top3 | 0.3308 | 0.3328 |
| ens_archs | 0.3262 | 0.3297 |
| unilm_lora | 0.3252 | 0.3273 |
| unilm_lorma | 0.3216 | 0.3269 |
| gpt_lorma | 0.3210 | 0.3218 |
| unilm_lorma_pi | 0.3183 | 0.3211 |
| retnet_lorma | 0.3138 | 0.3188 |
| gpt_full | 0.3122 | 0.3173 |
| kd_unilm6 / unilm_full | 0.3097 / 0.3070 | 0.3118 and 0.3121 (the two were mixed up on upload, so which is which is unknown) |
| unilm_head | 0.2764 | 0.2771 |
| encdec_lorma | 0.2596 | 0.2656 |
| unilm_frozen_probe (CPU) | 0.232 | 0.2313 |
| round 2 cue runs | not run (time) | n/a |

- **Codabench tracked dev honest within +0.000 to +0.006.** The order is the same, apart from ties.
  - This is the closest dev/test match of any phase, which supports the cross-half protocol.
  - Unlike Phase 4, the dev-chosen ensemble kept its gain on test: 0.3385 against 0.3273 for the best single.

## 7. Main Kaggle run (2026-10-25/26, 3.96 h on T4 x2, 10 runs, none failed). *Measured*, dev honest Macro-F1
ens_greedy 0.3365 (8 members), ens_all 0.3322, ens_noencdec 0.3315, ens_top3 0.3308, ens_archs 0.3262; unilm_lora 0.3252, unilm_lorma 0.3216, gpt_lorma 0.3210, unilm_lorma_pi 0.3183, retnet_lorma 0.3138, gpt_full 0.3122, kd_unilm6 0.3097, unilm_full 0.3070, unilm_head 0.2764, encdec_lorma 0.2596.
- **Every fine-tuned model is below the TF-IDF reference (0.405).** The prep estimate of 0.38 to 0.48 was wrong. Likely reasons, which are inferences rather than tested:
  - (a) TF-IDF reads up to 40k chars of every doc, while the encoders see at most 2,048 tokens (about 8k chars).
  - (b) Each run saw 28k to 42k training docs, about one pass over train, so it is probably undertrained for 100 labels.
  - (c) The decisive cues are offence words and phrases ("dowry", "anticipatory bail", "cheque"), which bag-of-ngrams captures directly. A 110M model pretrained from scratch on 400M tokens represents these less sharply.
- **PEFT beat full fine-tuning here** (LoRA/LORMA 0.318-0.325 vs full 0.307-0.312). This matches Phase 3 and is the reverse of Phase 4.
- The KD 6-layer student (67M) matches its full-FT teacher (0.3097 vs 0.3070) at about half the inference cost.
- encdec is the weakest architecture again.
- **Per-label thresholds won the cross-half check for every fine-tuned run** (+0.01), unlike on TF-IDF.
- **Every run over-predicts:** 5.4 to 6.8 labels per doc against 3.9 gold. That is the cost of the pos-weighted loss.
- **Ensembling helps:** best single 0.3252, 8-model greedy ensemble 0.3365.

## 8. Status of round 2 (2026-10-27)
Round 2 was not run. The report presents evidence-guided context selection (`--select cue`) as a proposed method with a measured input-side effect only: marker coverage on long docs rose from 0.60 to 0.83. Its effect on Macro-F1 is untested; it is listed as future work. The label-structure results in section 4 come from the TF-IDF and probe probabilities only.
