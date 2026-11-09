# Phase 3: Rhetorical Role labelling

**Task.** Label every sentence of a judgment with one of 13 rhetorical roles (Corporate Law and IT domains).
Metric: Macro-F1 over all 13 labels, pooled. Train 80 docs / 16.8k sentences, dev 10 docs, test 10 docs.

**Best submission: Codabench 0.5234**, `ens_all9` (mean log-probs of 9 fine-tuned runs).

Code: `p3_rr.py` + `ft_core.py`; notebook `CS781-CP-Phase3-Darsh-Kedia-250311.ipynb`.

## Method
Consecutive whole sentences are packed into windows of up to 1,024 tokens (each sentence prefixed with `<sep>`,
cut to 128 tokens, 75% head / 25% tail), encoded by a Phase 1 backbone, mean-pooled per sentence, passed through
a BiLSTM over sentence vectors (document context, and right-hand context for the causal archs) and a 13-way
linear layer. Loss: cross-entropy weighted by inverse-sqrt class frequency. Windows overlap 50% at inference.
Training windows start at random offsets each epoch. Model selection by dev Macro-F1 (matches `evaluate_phase3.py`).

## Results (dev = 10 docs; Codabench = test)
| run | arch / method | dev | Codabench |
|---|---|---|---|
| unilm_frozen_probe | 200M unilm frozen, CPU | 0.335 | 0.3514 |
| unilm_full | unilm 400M, full FT, 8 ep | 0.4613 | |
| unilm_lora | LoRA r16, 12 ep | 0.4656 | |
| unilm_lorma | LoRMA+ r16, 12 ep | 0.4696 | |
| unilm_lorma_pi | LoRMA-π r16, 12 ep | 0.4719 | |
| gpt_full | gpt full FT | 0.4521 | 0.4673 |
| gpt_lorma | gpt LoRMA+ | 0.4695 | 0.4707 |
| kd_unilm6 | 6-layer student (67M), KD from unilm_full_all | 0.4565 | 0.4518 |
| encdec_full / encdec_lorma | encdec 200M | 0.3585 / 0.3668 | |
| unilm_full_all / unilm_lorma_all | train+dev | n/a | 0.4774 / 0.4810 |
| ens_archs / ens_dev3 | 2-3 model means | 0.4828 / 0.4802 | 0.4872 / 0.4899 |
| ens_all | 5 models | n/a | 0.5149 |
| **ens_all9** | all 9 unilm/gpt runs | n/a | **0.5234** |
| ens_all11 | 9 + encdec runs | n/a | 0.5080 |
Kaggle: 48 min on T4 x2, 9 runs + 3 ensembles.
