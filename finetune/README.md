# Shared fine-tuning toolkit for Phases 3 to 7 (CS781 capstone, Darsh Kedia 250311)

Built on 2026-10-11 from what worked in Phase 2: fine-tuning the Phase 1 unilm for 4 epochs scored 0.5189 on Codabench. The best from-scratch model scored 0.4403.

## What exists

| File | What it is |
|---|---|
| `ft_core.py` | Task-independent code. `Backbone` loads any Phase 1 export (gpt, unilm, encdec, retnet) through `cs781_p1.load_model` and returns token states. `keep_layers=k` keeps k evenly spaced layers to make a distillation student. `apply_peft` sets up the fine-tuning technique. The file also has `kd_loss`, parameter groups, the LR schedule, AMP (fp16 on T4, bf16 on Ampere and newer) and a function that saves only the trained weights. |
| `p3_rr.py` | Phase 3 task: data loading, sentence packing, the model, training, prediction, ensembling, a validator and the zip writer. CLI commands: `train`, `ensemble`, `score`, `validate`. |
| `build_p3_notebook.py` | Builds the self-contained Kaggle notebook from `cs781_p1.py`, `ft_core.py` and `p3_rr.py`. Rebuild it after editing any of them. |
| `kaggle_p3/CS781-CP-Phase3-Darsh-Kedia-250311.ipynb` | The Phase 3 Kaggle notebook (132 KB). |
| `p4_cjpe.py` | Phase 4 task (judgment prediction): data loading with de-duplication and test-leak removal, text clean-up, tokenisation cache (`prep`), hierarchical tail-chunk model, training, threshold tuning, ensembling, validator and zip writer. CLI commands: `prep`, `train`, `ensemble`, `score`, `validate`. Added 2026-10-15; no shared file was changed. |
| `build_p4_notebook.py`, `kaggle_p4/CS781-CP-Phase4-Darsh-Kedia-250311.ipynb` | Builder and Kaggle notebook for Phase 4 (138 KB). |
| `tests/test_ft.py` | CPU checks. Each adapter starts out identical to the base layer. Merged weights match the forward pass. Outputs are padding-invariant for every arch × method pair (4 × 6). Run it with `python tests/test_ft.py DIR_WITH_TINY_CKPTS`. |

### Fine-tuning techniques (`--method`)
| method | update | trainable params (unilm, plus the 3.6M head) |
|---|---|---|
| `full` | all weights | 110M |
| `lora` [Hu et al. 2022] | W + (α/r)·BA on wq, wk, wv, wo | 1.2M at r=16 |
| `lorma_plus` [Bihany, Patel, Modi 2025] | (I + (α/r)·BA)·W, computed as h + s·B(A·h) with h = Wx. It starts exactly at Phase 1 because B = 0. | 1.2M at r=16 |
| `lorma_pi` [same paper] | π(BA)·W, where π cyclically shifts row i by i places. It is initialised so that π(BA) = I. | 1.2M at r=16 |
| `norms` | only the RMSNorm and retention-norm gains | ~21k |
| `head` | frozen backbone (linear-probe baseline) | 0 |

Distillation: train a teacher with `--save_train_logp`. Then train a student with `--keep_layers 6 --teacher RUN_T`. The loss is (1−α)·CE + α·T²·KL at α=0.5, T=2. The 6-layer unilm student has 67M parameters against the teacher's 110M.

**Paper check still needed.** I couldn't open the LoRMA paper from this container (arXiv is blocked), so both variants were written from the paper's abstract and a summary. The equations are in the `LoRMALinear` docstring. Before the viva, check two details against the PDF: the α/r scaling, and that the `lorma_pi` permutation is a cyclic shift of row i by i.

### How Phase 3 is modelled
- The backbone reads windows of up to 1024 tokens made of consecutive whole sentences. Each sentence is prefixed with `<sep>` and cut to 128 tokens (75% from the head, 25% from the tail). This way every sentence is encoded together with its neighbours.
- Each sentence's tokens are mean-pooled. A BiLSTM then runs over the sentence vectors, followed by a linear layer with 13 outputs. The BiLSTM gives gpt and retnet the right-hand context that their causal attention lacks.
- The loss is cross-entropy weighted by inverse-sqrt class frequency, to suit Macro-F1.
- Training windows start at random offsets each epoch. At test time the windows overlap by 50% and each sentence's log-probs are averaged.
- Model selection uses dev Macro-F1, computed over all 13 labels. It matches the official `evaluate_phase3.py` exactly (0.486984 on a test prediction). The dev set is only 10 documents, so the score is noisy.
- `--full` trains on train+dev for a fixed number of epochs.
- `ensemble` averages log-probs across runs. `--tune_bias` learns per-label offsets on dev. That dev score is optimistic, so only use the offsets if the gain is clear.

## Phase 3 on Kaggle
1. Upload the notebook. Choose Accelerator **GPU T4 x2**, turn Internet on, and attach the **`HF` secret** under Add-ons > Secrets. A new notebook has no secrets attached, and that is why the last Phase 2 reruns died after 15 s.
2. Use **Add Input** to attach the Phase 1 outputs: the unilm run (`darshkedia/notebook1792eb1390`) and the gpt run. Also attach encdec and retnet once they exist. The notebook finds every `model_best.pt`, reads its arch from the file, and finds `tokenizer.json`.
3. Run all cells, or Save & Run All. `QUICK = True` runs 1 epoch per job as a 10-minute end-to-end check.
4. The notebook runs two jobs at a time, one per T4. Each run writes `/kaggle/working/p3/<run>/submission.zip`. Three ensembles are added, and a summary table of dev scores is printed. `p3_results.zip` collects every zip and log.

Plan (11 runs; jobs whose arch has no checkpoint attached are skipped):
`unilm_full`, `unilm_lorma`, `unilm_lora`, `gpt_full`, `unilm_full_all` (train+dev, teacher), `unilm_lorma_pi`, `kd_unilm6` (6-layer student), `unilm_lorma_all`, `encdec_full`, `retnet_full`, `gpt_lorma`, plus ensembles `ens_dev3`, `ens_all`, `ens_archs`.
This plan covers the course rules for this phase: 4 techniques (full, LoRA, LORMA ×2) plus distillation, and every architecture that has a checkpoint.

**Time on T4.** Phase 3 train is 80 docs, 16.8k sentences and 677k tokens, which makes 748 windows per epoch. Phase 1 trained at about 13k tokens/s per T4, so full fine-tuning should take roughly 1 to 1.5 min per epoch. That puts a full run at about 12 to 15 min including dev and test inference, and LoRA/LORMA runs a little less. The whole plan on unilm + gpt should take about 1 to 1.5 h of wall time on T4 x2. These numbers are estimated from the Phase 1 throughput, not measured on Phase 3 yet. The CPU smoke runs passed for all 4 archs and all 6 methods, using tiny models and a real unilm checkpoint.

## What each later phase still needs
The approach and the time per run on one T4 are estimates. Token counts use 4.6 chars/token.

| Phase | Data (measured) | Head to add | Est. T4 time |
|---|---|---|---|
| 4 CJPE (binary, Macro-F1) | train ~33k docs (737 MB, on HF only), dev 3.5k, test 1.5k, ~22k chars (≈4.9k tokens) per doc | Chunk the document into ≤1024-token pieces, keeping the last 2-4 chunks, because the reasoning near the end carries the signal. Mean-pool each chunk, run attention or a BiLSTM over the chunks, then a binary head. Sample the train set. | ~45-90 min per epoch on the last 2k tokens of every doc. Subsample to ~10k docs for PEFT comparisons. |
| 5 LSI (multi-label, Macro-F1) | train 42.7k, dev 10.2k, test 13k, ~7k chars (≈1.5k tokens) per doc | Same chunk encoder, with a sigmoid head over the statute ids. Use BCE with positive weights, and tune one threshold per label (or a global one) on dev. | ~50-60 min per epoch at 1024 tokens. |
| 6 Summ (BERTScore) | train 7k (doc ~25k chars, ≈5.4k tokens), test 100 | Abstractive: encdec (when trained) or a gpt/unilm prefix-LM generating from the first or most important 768 tokens. Extractive fallback: a sentence scorer (reuse the Phase 3 head) choosing the sentences with the best ROUGE against the summary. | ~15-20 min per epoch with 1024 in / 256 out. Generating the 100 test summaries takes a few minutes. |
| 7 PCR (Micro-F1@7) | train 827 queries / 4.3k candidates, dev 118/1k, test 237/1.7k, ~37k chars per doc | Bi-encoder: embed documents chunk by chunk, trained contrastively on query↔relevant pairs with in-batch negatives. Fuse with BM25 and output the top 7. | Embedding every candidate takes ~15 min, and contrastive training ~30-60 min. |

The common pieces are already in place for all of these: `Backbone`, `apply_peft`, `kd_loss`, the AMP and LR helpers, and the run-dir conventions (`test_logp.pt`, `run.json`, `submission.zip`). Each phase only needs its own `pN_*.py` and a notebook builder copied from `build_p3_notebook.py`.

## Phase 1 work that raises every later score
- **Train encdec and retnet.** The rules require all four architectures, and Phase 6 needs a real decoder (encdec).
- **Continue pretraining gpt and unilm.** Both were still improving at 200M tokens: gpt validation loss was still falling at the end. The course suggests 400 to 700M tokens. `cs781_p1.py` can already extend a finished run with a fresh warm-up and cosine cycle (`lr_at(..., start=...)`). Another 200M tokens of unilm is about 2.2 h on T4 x2, and it lifts every downstream phase at once.
- **Domain-adaptive MLM is an option.** A short unilm MLM pass on the task texts (the Phase 4/5/6 train text is in-domain and allowed) before each phase's fine-tune.

## Phase 3 results (Codabench, pooled CL+IT Macro-F1)
| # | zip | model | dev | Codabench |
|---|---|---|---|---|
| 1 | `capstone/phase3/submissions/unilm_frozen_probe` | 200M unilm frozen (method `head`), BiLSTM sentence head, 3 epochs on CPU, train only | 0.335 | **0.3514** (2026-10-15) |

| 2 | Kaggle `p3/unilm_full` | 400M unilm, full fine-tune, 8 ep, train only | 0.4613 | |
| 3 | `p3/unilm_lorma` | unilm LORMA+ r16, 12 ep | 0.4696 | |
| 4 | `p3/unilm_lora` | unilm LoRA r16, 12 ep | 0.4656 | |
| 5 | `p3/gpt_full` | gpt full fine-tune, 8 ep | 0.4521 | 0.4673 (2026-10-16) |
| 6 | `p3/unilm_lorma_pi` | unilm LORMA-pi r16, 12 ep | 0.4719 | |
| 7 | `p3/kd_unilm6` | 6-layer unilm student (67M), KD from unilm_full_all | 0.4565 | 0.4518 (2026-10-16) |
| 8 | `p3/gpt_lorma` | gpt LORMA+ r16, 12 ep | 0.4695 | 0.4707 (2026-10-16) |
| 9 | `p3/unilm_full_all` | as 2, train+dev | n/a | 0.4774 (2026-10-16) |
| 10 | `p3/unilm_lorma_all` | as 3, train+dev | n/a | 0.4810 (2026-10-16) |
| 11 | `p3/ens_dev3` | mean of 2,3,4 | 0.4802 | 0.4899 (2026-10-15) |
| 12 | `p3/ens_archs` | mean of 2,5 | 0.4828 | **0.4872** (2026-10-15) |
| 13 | `p3/ens_all` | mean of 9,10,2,3,7 | n/a | 0.5149 (2026-10-16) |
| 14 | Kaggle `ens_all9` (separate CPU notebook) | mean of all 9 trained runs (2-10) | n/a | **0.5234** (2026-10-16, best) |
| 15 | Kaggle `encdec_full` (encdec 200M tokens, max_len 768) | encdec full fine-tune, 8 ep | 0.3585 | |
| 16 | Kaggle `encdec_lorma` | encdec LORMA+ r16, 12 ep | 0.3668 | |
| 17 | Kaggle `ens_all11` | mean of ens_all9 members + encdec_full + encdec_lorma | n/a | 0.5080 (2026-10-16; weak members pulled it below ens_all9) |

**Phase 3 final (closed 2026-10-16):** 11 zips scored on Codabench, best `ens_all9` 0.5234. Last submission was `ens_all11`. Never submitted: `ens_all9w`, `ens_all9b`, `encdec_full`, `encdec_lorma` (rows 15-16 have dev scores only), plus singles `unilm_lorma_pi`, `unilm_lorma`, `unilm_lora`, `unilm_full`.

Kaggle run 2026-10-15: 48 min on T4 x2, 9 runs + 3 ensembles, none failed. Full fine-tuning ~58 s/epoch, PEFT ~55 s, the 6-layer student ~33 s.

encdec (pretrained 200M tokens, val ppl 18.8, weakest generator) also gives the weakest Phase 3 features: dev 0.36-0.37 against 0.45-0.47 for unilm/gpt.

For the report: ensemble size was the strongest lever on this 10-doc test set (single best 0.4774, 2-3 models 0.487-0.490, 5 models 0.5149, 9 models 0.5234); the choice of fine-tuning method moved dev by at most 0.02.

Leaderboard context (2026-10-16): the Phase 3 leader had 0.6778 against our best 0.4899 (Phase 2: leader 0.7 vs ours 0.5189).

Codabench came in 0.016 above dev on this phase, so dev is a usable guide. It was not on Phase 2, where Codabench ran 0.04-0.09 below fold2.

## Phase 4 (CJPE), added 2026-10-15
- **Data.** Train has 37,387 records but 31,677 unique texts (de-duplicated). **`Test/test.json` (1,517 docs) is a subset of `Train/dev.json` with labels**, and the other dev docs appear twice each. `p4_cjpe.py` removes the test ids from dev and de-duplicates, leaving `dev_clean` = 994 docs (497 per label). Never train on the full dev.json. Train is 59% label 0; dev and test are balanced, so the loss is class-balanced and the decision threshold is tuned on dev_clean.
- **Long documents.** Median 3.4k tokens on dev/test, 2.4k on train. The model reads the last 4 x 512 tokens, cut into chunks aligned to the end of the document (option `--head_chunks` adds chunks from the start). Each chunk is mean-pooled, a learned position-from-end embedding is added, and a BiGRU with attention pooling over chunks feeds a 2-way head. Why the tail: a TF-IDF + logistic regression check on dev_clean scored 0.655 on the last 5k chars vs 0.576 on the first 5k chars (0.665 on the whole doc), matching ILDC (Malik et al. 2021).
- **Text clean-up.** ILDC text has line breaks inside words ("t\nhe"). A break between letters is joined when the joined word is more frequent than the rarer fragment; other breaks become spaces.
- **`--method head`** caches frozen chunk features once and trains only the head, so the probe is cheap.
- **Kaggle plan** (updated 2026-10-19 after the Phase 3 results; two at a time on T4 x2, 12k fresh train docs per epoch), in priority order: `unilm_lorma`, `unilm_full` (also the KD teacher), `retnet_lorma`, `unilm_lora`, `gpt_lorma`, `kd_unilm6`, `unilm_lorma_pi`, `gpt_full`, `encdec_lorma`, `unilm_head`. Jobs whose arch has no checkpoint attached are skipped, and no job starts after `MAX_HOURS = 9`. Ensembles: `ens_greedy` (forward selection on dev_clean), `ens_noencdec`, `ens_all`, `ens_unilm`, `ens_archs`. Estimated 4 to 6 h (not measured).
- Local scorer `macro_f1` matches `evaluate_phase4.py` exactly (0.760563 on a synthetic dev prediction).

### Phase 4 results
| # | zip | model | dev_clean | Codabench |
|---|---|---|---|---|
| 1 | `capstone/phase4/submissions/unilm_frozen_probe` | 200M unilm frozen (method `head`), last 4 x 512 tokens, GRU head, 3k train docs, 30 epochs on CPU | 0.557 at threshold 0, 0.604 at tuned threshold (optimistic) | **0.5652** (2026-10-19) |

## Phase 5 (LSI), added 2026-10-19
- **Data.** Train 42,750 docs, dev 10,181, test 13,019; 100 statute ids (0-99), every doc has >= 1 (mean 3.7 train / 3.9 dev); rarest label 272 train / 68 dev docs. Statute names are masked (`<SECTION>`, `<ACT>`). Median 0.8k tokens, mean 2.0k, p90 4.4k. **No test leak**: dev and test share no ids. ~300 exact-duplicate texts (mostly one-line orders); `p5_lsi.py` drops the 210 train docs whose text is also in dev and de-duplicates train (42,234 left).
- **Scorer.** `evaluate_phase5.py` macro-averages over max(gold id)+1 = 100 labels with zero_division=0, and rejects any id outside that range. Local `macro_f1`/`micro_f1` match sklearn exactly (random tests); the TF-IDF dev-half prediction scored 0.406403 with the official script.
- **Model.** First 3 x 512 tokens + last 512 (whole doc for ~70% of docs), each chunk mean-pooled from the backbone, + chunk position embedding, BiGRU, attention + max pooling, 100 sigmoid outputs. BCE with pos_weight sqrt(neg/pos) clipped at 10. Distillation is per-label tempered-sigmoid BCE on the teacher's saved train logits.
- **Thresholds.** `choose_thresholds` tunes a global and a per-label threshold; it scores each on dev by tuning on one half (crc32 of id) and testing on the other, uses per-label only if it wins by > 0.003, then re-tunes on all of dev. Every doc gets at least its top label. `dev_f1_honest` is the cross-half number; `dev_f1_tuned` is optimistic.
- **TF-IDF reference** (`p5_lsi.py baseline`, word 1-2 grams, one-vs-rest LR, C=4, balanced; NOT a Phase 1 model, never submit): dev Macro-F1 0.405 honest (0.4035 at threshold 0.5), Micro-F1 0.50. This is the bar.
- **Kaggle plan** (`build_p5_notebook.py` -> `kaggle_p5/CS781-CP-Phase5-Darsh-Kedia-250311.ipynb`, 14k fresh train docs per epoch, two jobs at a time, no new job after `MAX_HOURS = 3.8`; reordered 2026-10-25 after Phase 4 showed full FT and KD winning on test): `unilm_full` (KD teacher), `unilm_lorma`, `retnet_lorma`, `gpt_full`, `kd_unilm6`, `unilm_lora`, `encdec_lorma`, `gpt_lorma`, `unilm_lorma_pi`, `unilm_head`; ensembles `ens_greedy`, `ens_noencdec`, `ens_all`, `ens_top3`, `ens_archs`. The first 7 jobs cover the phase rules (all four archs; full, LoRA, LORMA; KD). Estimated 4-5 h (not measured). `p5_results.zip` also carries every run's fp16 dev/test probabilities, so ensembles can be rebuilt on CPU later.

- **Round 2, added 2026-10-25** (originality and methodology): `p5_lsi.py --select cue` picks the first chunk plus the 3 chunks with the most `<SECTION>`/`<ACT>` markers (prep with `--keep_all 16384`); it raises marker coverage on long dev docs from 0.60 to 0.83. `build_p5r2_notebook.py` -> `kaggle_p5/CS781-CP-Phase5-round2-Darsh-Kedia-250311.ipynb` trains `unilm_full_cue` and `unilm_lorma_cue` (the ablation pairs for the main run's `unilm_full`/`unilm_lorma`), pulls in the main run if attached, ensembles, and runs `p5_post.py` (label co-occurrence prior, residual label-interaction layer, top-k decoding; honest cross-half dev ablation). Method, ablations and failures: `capstone/phase5/REPORT_NOTES.md`.

### Phase 5 results
| # | zip | model | dev honest | Codabench |
|---|---|---|---|---|
| 1 | `capstone/phase5/submissions/unilm_frozen_probe` | 200M unilm frozen (method `head`), first 512 tokens only, GRU head, 6k train docs, 40 epochs on CPU, thresholds on 4k dev docs | 0.232 (0.224 at 0.5) | |

Kaggle round 1 (2026-10-19, all four archs, 263 min on T4 x2, 10/10 jobs ok). dev_clean Macro-F1 at the tuned threshold:
| run | dev_f1_tuned | min |
|---|---|---|
| unilm_full (also KD teacher) | 0.7736 | 54.5 |
| kd_unilm6 (6-layer student) | 0.7686 | 24.8 |
| gpt_full | 0.7512 | 41.8 |
| retnet_lorma | 0.7442 | 81.8 |
| unilm_lora | 0.7287 | 64.5 |
| unilm_lorma_pi | 0.7203 | 69.8 |
| gpt_lorma | 0.6908 | 57.5 |
| unilm_lorma | 0.6704 | 61.3 |
| unilm_head (frozen) | 0.6358 | 12.0 |
| encdec_lorma | 0.6214 | 31.5 |
| ens_greedy (unilm_full, kd_unilm6, gpt_full, unilm_lorma_pi; chosen on dev, optimistic) | 0.7826 | |
| ens_unilm / ens_noencdec / ens_all / ens_archs | 0.7655 / 0.7635 / 0.7605 / 0.7219 | |
Unlike Phase 3, full fine-tuning and the distilled student beat LoRA/LORMA here (31.7k train docs is enough data for full FT). All beat the TF-IDF reference (0.665) except encdec and the frozen probe.
Codabench, round 1 (2026-10-19; dev_clean tuned in brackets): kd_unilm6 **0.7539** (0.7686), ens_greedy 0.7442 (0.7826), ens_noencdec 0.7429 (0.7635), ens_unilm 0.7421 (0.7655), unilm_full 0.7383 (0.7736), ens_all 0.7376 (0.7605), gpt_full 0.7350 (0.7512), retnet_lorma 0.7330 (0.7442), ens_archs 0.7224 (0.7219), unilm_lora 0.7031 (0.7287), gpt_lorma 0.7020 (0.6908), unilm_lorma_pi 0.6837 (0.7203), unilm_lorma 0.6657 (0.6704), encdec_lorma 0.6096 (0.6214), unilm_head 0.5962 (0.6358). The distilled 6-layer student beat its teacher (+0.016) and every ensemble on test; dev-chosen ensembles lost most of their dev gain.
**PHASE 4 FINAL (closed 2026-10-21): best kd_unilm6 0.7539, 16 scored submissions. Round 2 was never run.** Round 2 notebook, kept as report material only (`kaggle_p4/CS781-CP-Phase4-round2-Darsh-Kedia-250311.ipynb`, `build_p4_notebook.py --round2`, needs the round-1 output attached): KD students from the round-1 unilm_full teacher (unilm seeds 7/11, 3-epoch, two-teacher with a new unilm_full_s2, gpt and retnet 6-layer students), unilm_full_s2, retnet_full; ensembles include `ens_kd` and the round-1 runs as `r1_*`.
