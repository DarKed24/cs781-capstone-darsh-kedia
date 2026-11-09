# Phase 2: Legal Named Entity Recognition (L-NER)

**Task.** 12 entity types (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO),
character-offset spans per document. Metric: strict span Macro-F1 over the 12 labels (`evaluate_phase2.py`). Data: fold1 and fold2 labelled
(35 docs each, ~4.2k spans per fold), fold3 = test.

**My best submission: 0.5189 on Codabench**, the Phase 1 `unilm` (in bidirectional mode) fine-tuned as a BIO tagger on fold1+fold2.

I started with rules and small taggers trained from scratch, since the Phase 1 models weren't ready yet. Once the
unilm checkpoint existed, fine-tuning it beat my best from-scratch ensemble by about 0.08 on the first try.

## Method
- `lner_model.py` works with any tokenizer. `BPETokenizer` wraps the Phase 1 `tokenizer.json` and encodes word by
  word with a leading space, so every punctuation mark starts a new token. Without this, a gold span like
  "Cr.P.C." followed by a comma ends in the middle of a token. Offsets have whitespace trimmed. In the round trip
  only 27 of 4,120 gold spans don't line up with token boundaries (mostly possessives), so the ceiling is 0.994.
- Long documents (10k-114k chars) are tagged in sliding windows of 512 tokens with stride 384; log-probs are
  averaged in the overlaps and decoded with a BIO-constrained Viterbi, then mapped back to character offsets.
- Encoders: `transformer` (4L/256d from scratch), `bilstm`, `charbilstm` (BiLSTM over [BPE embedding; char-CNN]),
  optional word2vec CBOW initialisation (`w2v.py`, 256d, trained on Phase 1 Supreme Court text), and `phase1`
  (any Phase 1 checkpoint through `cs781_p1.load_model`; unilm uses mode 1, bidirectional).
- `baseline_rules.py`: gazetteer from training spans + regexes for DATE / CASENO / STAT.
- `lner_utils.py`: span↔BIO conversion, strict scorer (matches the official script to 6 decimals), validator, zipper.

## How to run
- Best model on Kaggle: `kaggle_unilm/CS781-CP-Phase2-unilm-Darsh-Kedia-250311.ipynb` (attach the Phase 1 unilm output and the `HF` secret).
- CPU: see the root README (`lner_model.py train|predict`, `baseline_rules.py dev|test`).

## All my submissions (fold3 strict Macro-F1, 12 scored)
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
| 12 | unilm_full | Phase 1 unilm (200M tokens) fine-tuned, 4 epochs, fold1+2 | **0.5189** |

Local fold2 (train on fold1) overestimated Codabench by ~0.04-0.05 for the BiLSTMs.
