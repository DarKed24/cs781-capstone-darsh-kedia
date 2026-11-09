# Phase 2 (L-NER) working folder: state as of 2026-10-07 05:40 UTC

## Files
- `lner_utils.py`: loading, word offsets, span/BIO conversion, windowing, strict scorer, prediction.json validator, writer and zipper. CLI: `roundtrip`, `score`, `validate`, `zip`.
- `baseline_rules.py`: a non-neural baseline. It builds a gazetteer from the training gold spans and adds regexes for DATE, CASENO and STAT. `dev DATA_DIR` trains on fold1 and scores on fold2. `test DATA_DIR OUT_DIR` trains on fold1 and fold2, then writes and validates `prediction.json` for fold3 and zips it as `submission.zip`.
- The official scorer is at `../starter-kit/evaluation_script/Evaluation Script/evaluate_phase2.py`. Run it with `python3 -I`.

## What has been tested
- `strict_scores` now averages over all 12 labels, so a label with no gold spans and no predictions scores 0. This matches the official `evaluate_phase2.py`. On synthetic data, both gave Macro-F1 0.459004.
- The validator matches the Codabench schema:
  - The document IDs must be exactly those in fold3, with none missing and none extra.
  - Each entry is `[start, end, label]`, optionally with the entity text as a fourth field. The fourth field must be a string.
  - Offsets must be integers within the text, labels must be from the official list, and duplicate spans are rejected.
  - The leading/trailing whitespace check was removed because the official scorer does not have it.
- `baseline_rules.py` ran end to end only on made-up documents: dev scoring, test prediction, validation and a zip with prediction.json at its root all worked. **It has not been run on the real data.**

## Not done yet (at this point the data was not yet reachable: huggingface.co returned 403)
- Download Phase 2 into its own empty directory under `data/`, inspect the real format, and check that `load_texts` and `load_spans` parse it.
- Run `python3 -I lner_utils.py roundtrip fold1-text.json fold1-gold.json` to count how many gold spans fall on word boundaries. Fix `_WORD_RE` for anything that does not.
- Tune the baseline on fold2. Then produce and submit the first fold3 submission.
- Build a pluggable PyTorch token-tagging pipeline. The interface should be a tokenizer that returns `(ids, char_offsets)` and an encoder that maps `input_ids` to `[B, T, H]`. Add a BIO head, use `windows()` for long documents, and decode with `bio_to_spans`. Any small from-scratch model should drop in.
- Phase 1 was missed and is being redone from scratch; the Phase 1 tokenizer must expose character offsets for this tagger (see `subword_offsets`).
- PyPI works from the container, which allows `pip install torch`.

## Update 2026-10-07 ~06:10 UTC
- Hugging Face is reachable, but the dataset is **gated**: file downloads return 401 without a token. Needs either the 6 files uploaded to the project, or an `HF_TOKEN` env var in the cloud environment (only new sessions see it).
- `lner_model.py`: tokenizer-agnostic neural tagger. `BPETokenizer` wraps a tokenizers `tokenizer.json` (the Phase 1 format: byte-level BPE, 32k). It encodes word by word with leading whitespace so every punctuation mark is a token boundary (GPT-2 style regex otherwise merges ".," and gold spans like "Cr.P.C." before a comma end mid-token), and trims whitespace from offsets itself (trim_offsets does not trim a sequence's first token). Encoders: `transformer` (4L/256d, 3.6M params with a 2k vocab) and `bilstm`, both from scratch; `CallableEncoder` + `load_phase1_encoder` wrap a Phase 1 model. Sliding windows 512/384 with averaged log-probs, BIO-constrained Viterbi decoding. CLI: `tokenizer`, `train [--full]`, `predict`.
- Tested on synthetic data only: 0 misaligned spans, perfect round trip, train -> predict -> validated prediction.json -> zip works.
- `kaggle/CS781-CP-Phase2-Darsh-Kedia-250311.ipynb`: self-contained Kaggle notebook (writes the three .py files, downloads data with Kaggle secret `HF`, stand-in tokenizer or Phase 1 tokenizer, rules baseline zip, neural dev run, full run, fold3 zip, Phase 1 hook). Rebuild it after editing the .py files with `python3 kaggle/build_notebook.py kaggle/CS781-CP-Phase2-Darsh-Kedia-250311.ipynb`.

## Real data (2026-10-07 ~06:20 UTC)
- Data copied from `../data/Phase_2_LNER` into `data/`.
- 35 docs per fold, ~1.1M chars per fold, docs 10k-114k chars (one fold3 doc is empty text: output `[]`). fold1 4120 spans, fold2 4273. No overlapping spans; 4th field always equals the text slice; 16 gold spans have edge whitespace (unpredictable with trimmed offsets).
- Label counts fold1+2: DATE 1575, STAT 1260, PREC 1183, COURT 1013, AUTH 1003, CASENO 760, APP 407, RESP 381, JUDGE 245, A.COUNSEL 195, WIT 187, R.COUNSEL 184.
- Stand-in BPE (32k, trained on fold1+2 text) round trip: 27/4120 and 34/4273 spans off token boundaries (mostly possessives "High Court's", "Constitutional"); ceiling Macro-F1 0.994.
- Rules baseline fold2 (built on fold1): Macro-F1 0.254. Fold3 zip: `submissions/rules_v1/submission.zip` (validated).
- `phase1_encoder()` in lner_model.py loads the four Phase 1 archs via `cs781_p1.load_model`; tested with tiny random exports of all four (train step + padding invariance).
- CPU training here: transformer ~2.2 min/epoch, BiLSTM ~4 min/epoch (2 threads each). Logs in `runs/logs/`.

## Neural results on fold2 (trained on fold1, CPU, stand-in tokenizer)
- transformer 4L/256d from scratch: best Macro-F1 0.117 (25 epochs). Too little labelled data for a from-scratch transformer.
- BiLSTM 2x384: 0.386 at epoch 24-25, still rising slowly (official scorer agrees: 0.386412). Per label: COURT .80, DATE .88, STAT .62, JUDGE .48, R.COUNSEL .37, CASENO .36, AUTH .30, A.COUNSEL .24, PREC .24, WIT .15, APP .11, RESP .10.
- Hybrid (BiLSTM, but rules for DATE and COURT): 0.392.
- Submissions: `submissions/rules_v1` (rules, 0.254), `submissions/bilstm_fold1` (BiLSTM trained on fold1 only, 0.386). `runs/bilstm_full` (fold1+fold2, 25 epochs) is training for the next one.
- `submissions/hybrid_v1`: BiLSTM (fold1) with DATE and COURT from the rules (gazetteer built on fold1+2). Fold2 analogue scored 0.392.
- The first fold1+2 BiLSTM run finished 25 epochs but crashed saving (cwd on the shared folder vanished; paths are now resolved up front). Rerun writes to /home/user/p2runs (container-local) and gets copied to `runs/bilstm_full` when done.
- `submissions/bilstm_full`: BiLSTM trained on fold1+fold2 for 25 epochs (no dev score, since fold2 is in training; expected at or above 0.386). Weights in `runs/bilstm_full`.
- Four valid fold3 zips so far: rules_v1, bilstm_fold1, hybrid_v1, bilstm_full. Next: Phase 1 checkpoints (unilm, encdec) through `phase1_encoder`, then LORMA, another PEFT, and distillation.

## 2026-10-11 (deadline day)
- Codabench (fold3) scores: rules 0.2873, bilstm_fold1 0.3437, hybrid_v1 0.3445, bilstm_full 0.3793, hybrid_full 0.3789.
- New encoder `charbilstm`: BiLSTM over [token emb ; char-CNN of the token's characters] (char table built per vocab id from the tokenizer). Fold1-only run reached fold2 0.346 by epoch 8 (stopped early to free CPU).
- Word2vec CBOW (256d, window 5, 3 epochs) on the stand-in BPE ids of the first 300 MB of Phase 1 train_supreme_court.json (12,473 docs, 63M tokens): `runs/emb256.npy`, script `runs/w2v.py`. `--init_emb` initialises the token embedding.
- charbilstm + word2vec, fold1 only: **fold2 0.491** (best epoch 18 of 20). Big gain over 0.386.
- New zips: `charbilstm_full`, `charw2v_fold1`, `charw2v_full` (expected best). `predict_ensemble` averages log-probs over seeds; seeds 43 and 44 of charw2v_full are training for an ensemble zip.
- Zips `charw2v_ens2` (seeds 42+43) and `charw2v_ens3` (42+43+44) built 09:35 UTC 2026-10-11; all models in `runs/charw2v_full*`. 10 distinct valid zips total.
- Phase 1 fine-tune CLI: `lner_model.py train DATA OUT --tokenizer <p1 tokenizer.json> --encoder phase1 --phase1_ckpt <model_best.pt>` (+ `predict DATA OUT`). Smoke-tested only.
- Final Phase 2 Codabench scores (fold3): rules 0.2873, bilstm_fold1 0.3437, hybrid_v1 0.3445, bilstm_full 0.3793, hybrid_full 0.3789, charbilstm_full 0.4028, charw2v_fold1 0.3974, charw2v_full 0.4256, charw2v_ens2 0.4367, **charw2v_ens3 0.4403 (best)**.
- A 5-seed ensemble (seeds 45, 46) was abandoned at 17:01 UTC: CPU epochs slowed to 11-14 min and the runs could not finish before the 18:29 UTC deadline.
- `ens4_mixed` (17:28 UTC 10-11): inference-only ensemble of charw2v seeds 42/43/44 + charbilstm_full. Codabench 0.4390 (below ens3 0.4403).
- `kaggle_unilm/`: Kaggle GPU notebook (17:32 UTC 10-11) fine-tuning Phase 1 unilm (`--encoder phase1`) on fold1+fold2. Checked on CPU here: unilm model_best.pt loads, Phase 1 tokenizer misaligns 27/4120 gold spans, forward pass OK. Never trained end to end yet; starting point for Phase 3.
- **Zip 12, unilm fine-tune (Kaggle, 4 epochs, fold1+fold2, `kaggle_unilm` notebook): Codabench 0.5189, new best.** Phase 1 unilm (200M tokens) as bidirectional encoder, encoder LR 5e-5, head LR 1e-3, batch 8. The `kaggle_unilm` notebook and `cs781_phase2_unilm.zip` bundle are the base for Phase 3. Suggested follow-up: 6 epochs.

## Final Phase 2 results (Codabench, fold3 strict Macro-F1), closed 2026-10-11 18:29 UTC
| # | zip | model | score |
|---|---|---|---|
| 1 | rules_v1 | gazetteer + regex | 0.2873 |
| 2 | bilstm_fold1 | BiLSTM from scratch, fold1 | 0.3437 |
| 3 | hybrid_v1 | 2 + rules for DATE/COURT | 0.3445 |
| 4 | bilstm_full | BiLSTM, fold1+2 | 0.3793 |
| 5 | hybrid_full | 4 + rules for DATE/COURT | 0.3789 |
| 6 | charbilstm_full | char-CNN BiLSTM, fold1+2 | 0.4028 |
| 7 | charw2v_fold1 | 6 + word2vec init, fold1 | 0.3974 |
| 8 | charw2v_full | 6 + word2vec init, fold1+2 | 0.4256 |
| 9 | charw2v_ens2 | 2-seed ensemble of 8 | 0.4367 |
| 10 | charw2v_ens3 | 3-seed ensemble of 8 | 0.4403 |
| 11 | ens4_mixed | 10 + model 6 | 0.4390 |
| 12 | unilm_full (Kaggle) | Phase 1 unilm fine-tuned, 4 epochs, fold1+2 | **0.5189** |

Lessons: fine-tuning the Phase 1 unilm beat the best from-scratch ensemble by +0.08 on the first try, so later phases should start from the Phase 1 models (kaggle_unilm notebook). The 6-epoch rerun failed in 15 s because the new Kaggle notebook had no HF secret attached: always attach the `HF` secret (and internet) to every new Kaggle notebook. Local fold2 overestimated Codabench by ~0.04-0.05 for the BiLSTMs.
