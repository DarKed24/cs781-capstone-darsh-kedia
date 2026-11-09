# Phase 3: Rhetorical Role labelling

**Task.** Label every sentence of a judgment with one of 13 rhetorical roles (Corporate Law and IT domains).
Metric: Macro-F1 over all 13 labels, pooled. Train 80 docs / 16.8k sentences, dev 10 docs, test 10 docs.

**My best submission: 0.5234 on Codabench**, `ens_all9`, which averages the log-probs of 9 fine-tuned runs.

The test set is only 10 documents, so single runs are noisy. The thing that helped most was averaging more
models, much more than which fine-tuning method I used.

Code: `p3_rr.py` + `ft_core.py`; notebook `CS781-CP-Phase3-Darsh-Kedia-250311.ipynb`.

## Method
I pack consecutive whole sentences into windows of up to 1,024 tokens. Each sentence starts with `<sep>` and is cut
to 128 tokens (75% from its start, 25% from its end). The Phase 1 backbone encodes the window, I mean-pool the tokens
of each sentence, and a BiLSTM runs over the sentence vectors. That gives every sentence its neighbours as context,
and gives gpt and retnet the right-hand context they don't have otherwise. A 13-way linear layer is on top. The loss
is cross-entropy weighted by inverse-sqrt class frequency because the metric is Macro-F1. Training windows start at a
random offset each epoch; at test time windows overlap by 50% and I average each sentence's log-probs. I pick the
best epoch by dev Macro-F1 (my scorer matches `evaluate_phase3.py`).

The 9 training runs come from the notebook. I built the ensembles `ens_all9` and `ens_all11`, and the encdec runs, in a
later session from the saved log-probs, with the same code:
`python p3_rr.py ensemble --data DATA_DIR --out OUT RUN_DIR1 RUN_DIR2 ...` (one run directory per member).
The KD teacher for `kd_unilm6` (`unilm_full_all`) was trained on train+dev, so the student's dev score isn't a
clean estimate; its test score is. There is no retnet run in this phase.

## All runs (dev = 10 docs; Codabench = test)
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
