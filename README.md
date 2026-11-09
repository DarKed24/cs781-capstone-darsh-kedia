# CS781 Capstone: In-House Legal Intelligence System

**Darsh Kedia, roll 250311, IIT Kanpur** (Codabench user `d_250311`). Course CS781 (Large Language Models), 2026.

This repository holds the code for all seven phases of the CS781 legal NLP capstone. Four ~110-125M
parameter language models were pretrained from scratch on the official Indian legal corpus (Phase 1).
They were then fine-tuned for six downstream tasks (Phases 2-7) using full fine-tuning, LoRA, two LoRMA
variants, knowledge distillation and ensembling. No pretrained weights, external corpora or external
LLM APIs are used anywhere in the pipeline. The data is the gated Hugging Face dataset
`Exploration-Lab/CS-781-Capstone`.

Model checkpoints, data files and submission zips are not in the repository (size and licence). Every
result can be rebuilt from the notebooks below on a free Kaggle T4 x2 session.

## Results at a glance

| Phase | Task | Metric | Best Codabench | Best submission | Scored zips |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | ROUGE-L / BLEU | not submitted (deadline missed); local pseudo-test ROUGE-L 0.288 | unilm, 400M tokens | 0 |
| 2 | Legal NER (L-NER) | strict Macro-F1 | **0.5189** | unilm fine-tuned (Phase 1 encoder + BIO head) | 12 |
| 3 | Rhetorical roles | Macro-F1 | **0.5234** | 9-model ensemble (unilm/gpt; full, LoRA, LoRMA, KD) | 11 |
| 4 | Judgment prediction (CJPE) | Macro-F1 | **0.7539** | 6-layer unilm student distilled from full-FT unilm | 16 |
| 5 | Statute identification (LSI) | Macro-F1 | **0.3385** | greedy 8-model ensemble | 16 |
| 6 | Summarisation | BERTScore F1 | **0.8522** | TF-IDF centroid extractive (best generator: gpt_full_ext 0.8330) | 18 |
| 7 | Prior case retrieval (PCR) | Micro-F1@7 | **0.4566** | bigram BM25 + citation-context query + temporal filter | 8 |

Full ladders (every scored zip with its dev score) are in each phase's README.

## Repository layout

```
phase1/      Phase 1 pretraining: model zoo (cs781_p1.py), tokenizer, Kaggle notebooks, tests
phase2/      Phase 2 L-NER: tokenizer-agnostic BIO tagger, rules baseline, word2vec, notebooks
finetune/    Shared fine-tuning toolkit for Phases 3-7 + task modules + Kaggle notebooks
  ft_core.py         Backbone loader for all 4 archs, LoRA / LoRMA+ / LoRMA-pi / norms / head, KD loss, AMP, LR schedule
  p3_rr.py           Phase 3 rhetorical roles         kaggle_p3/  notebook
  p4_cjpe.py         Phase 4 judgment prediction      kaggle_p4/  notebooks (round 1 run; round 2 not run)
  p5_lsi.py          Phase 5 statute identification  kaggle_p5/  notebooks (round 1 run; round 2 not run)
  p5_post.py         Phase 5 label-structure post-processing (CPU)
  p6_summ.py         Phase 6 generators (fine-tuned Phase 1 LMs)   kaggle_p6/
  p7_lex.py          Phase 7 lexical retrieval (BM25 + our additions)
  p7_dense.py        Phase 7 passage-level bi-encoder + fusion      kaggle_p7/
  build_pN_notebook.py   regenerate each self-contained notebook from the .py files
  tests/test_ft.py       CPU unit tests (adapter identity at init, merge equality, padding invariance)
phase3/ ... phase7/   per-phase README (method, results, how to run) and CPU-only scripts (Phases 6, 7)
requirements.txt
```

## The Phase 1 foundation models

All four share one byte-level BPE tokenizer (32,000 tokens, trained by us on the Phase 1 training
text; `phase1/tokenizer.json`), a 1,024-token context and one data pipeline. Training data: 420M tokens
prepared once, mixed 35% Supreme Court, 35% High Courts, 15% tribunals, 15% acts.

| key | family | params | design | tokens | val loss (ppl) | tail loss* | pseudo ROUGE-L |
|---|---|---|---|---|---|---|---|
| `gpt` | decoder-only | 109.5M | 12 layers, d 768, 12 heads, RoPE, RMSNorm, SwiGLU, tied embeddings | 200M | 2.203 (9.05) | **2.139** | 0.277 |
| `unilm` | hybrid encoder | 110.1M | one stack trained with causal LM and masked LM (mode embeddings) | 400M | 2.297 (9.94) | 2.234 | **0.288** |
| `encdec` | encoder-decoder | 123.7M | 6+6 layers, prefix-LM + T5 span corruption | 200M | 2.936 (18.84) | 2.936 | 0.178 |
| `retnet` | efficient (RetNet) | 116.6M | softmax-free multi-scale retention, O(1) recurrent decoding | 200M | 2.341 (10.39) | 2.313 | 0.280 |

\*tail loss = loss on the last 256 tokens given the first 768: the one number comparable across all four.

## How to run

Everything that needs a GPU runs as a **self-contained Kaggle notebook** (each notebook writes its own
`.py` files with `%%writefile`, so nothing has to be uploaded besides the notebook). Common setup:

1. Kaggle: *New notebook* > *File* > *Import notebook*, accelerator **GPU T4 x2**, Internet **on**.
2. Add-ons > Secrets: add a secret named **`HF`** holding a Hugging Face read token that has accepted the
   dataset's terms (`Exploration-Lab/CS-781-Capstone` is gated). Attach it to every new notebook.
3. For Phases 2-7, *Add Input* > *Your Work* and attach the Phase 1 output notebooks (one per architecture).
   The notebooks find every `model_best.pt` and `tokenizer.json` under `/kaggle/input` automatically and
   read each checkpoint's architecture from the file.
4. *Save Version* > *Save & Run All*. Each run writes `submission.zip` files plus a results bundle to
   `/kaggle/working`. Notebooks stop starting new jobs after a `MAX_HOURS` guard and are resumable.

| Phase | Notebook | Wall time (T4 x2) |
|---|---|---|
| 1 | `phase1/CS781-CP-Phase1-Darsh-Kedia-250311.ipynb` (set `STAGE` = `prepare_train`, `train`, `generate`, `pseudo_eval`; `ARCH` = `gpt`/`unilm`; separate encdec/retnet copies) | ~2.2 h per arch per 200M tokens (retnet 3.1 h, encdec 1.2 h) |
| 2 | `phase2/kaggle_unilm/CS781-CP-Phase2-unilm-Darsh-Kedia-250311.ipynb` (best); `phase2/kaggle/...` from-scratch models | ~30 min |
| 3 | `finetune/kaggle_p3/CS781-CP-Phase3-Darsh-Kedia-250311.ipynb` | 48 min |
| 4 | `finetune/kaggle_p4/CS781-CP-Phase4-Darsh-Kedia-250311.ipynb` | 4.4 h |
| 5 | `finetune/kaggle_p5/CS781-CP-Phase5-Darsh-Kedia-250311.ipynb` | 4.0 h |
| 6 | `finetune/kaggle_p6/CS781-CP-Phase6-Darsh-Kedia-250311.ipynb` (generators); `phase6/make_ladder.py` (CPU extractive, the best zips) | 85 min / ~10 min CPU |
| 7 | `phase7/make_lex_zips.py` (CPU, the submitted zips); `finetune/kaggle_p7/...` (dense + fusion) | ~10 min CPU / 64 min |

The CPU entry points, for working locally after downloading the data (`huggingface-cli download
Exploration-Lab/CS-781-Capstone --repo-type dataset --local-dir data`):

```bash
pip install -r requirements.txt
# Phase 1 unit tests (param counts, cached decoding == full forward for all 4 archs)
python phase1/tests/test_models.py
# Phase 2 rules baseline and from-scratch tagger
python phase2/baseline_rules.py dev data/Phase_2_LNER
python phase2/lner_model.py train data/Phase_2_LNER runs/charw2v --tokenizer phase1/tokenizer.json --encoder charbilstm --init_emb emb256.npy --full
# Phase 2 best model: Phase 1 unilm as the encoder (GPU recommended)
python phase2/lner_model.py train data/Phase_2_LNER runs/unilm --tokenizer phase1/tokenizer.json --encoder phase1 --phase1_ckpt unilm/model_best.pt --full --epochs 4
# Phase 3-5 fine-tuning outside Kaggle (any Phase 1 checkpoint, any method)
python finetune/p3_rr.py train --data data/Phase_3_RR --out runs/p3/unilm_lorma \
       --tokenizer phase1/tokenizer.json --ckpt unilm/model_best.pt --method lorma_plus --epochs 12
python finetune/p4_cjpe.py prep  ...   # then train / ensemble / score / validate, see --help
python finetune/p5_lsi.py  prep  ...   # then train / ensemble / baseline, see --help
# Phase 6 extractive ladder (CPU) - writes phase6/submissions/<name>/submission.zip
CS781_ROOT=. python phase6/make_ladder.py
# Phase 7 lexical ladder (CPU) - scores every config on dev and writes the test zips
python phase7/make_lex_zips.py data/Phase_7_PCR out/p7
```

Every task module has a `validate` or built-in validator that checks the submission schema before
zipping, and every local scorer was checked against the official `evaluate_phaseN.py` to 6 decimals.

## Fine-tuning techniques (Phases 2-7)

| method | what is trained | trainable (unilm) |
|---|---|---|
| `full` | all weights | 110M |
| `lora` (Hu et al. 2022) | W + (α/r)·BA on q, k, v, o | 1.2M at r = 16 |
| `lorma_plus` (Bihany, Patel, Modi 2025) | (I + (α/r)·BA)·W, multiplicative, starts at the Phase 1 model | 1.2M |
| `lorma_pi` (same paper) | π(BA)·W with a cyclic-shift permutation, initialised so π(BA) = I | 1.2M |
| `head` | frozen backbone, task head only (probe baseline) | 0 |
| KD | 6-layer student (67M) from a full-FT teacher, (1-α)·CE + α·T²·KL, α 0.5, T 2 | 67M |

All four Phase 1 architectures are fine-tuned in Phases 3, 4 and 5; LoRMA and distillation appear in
Phases 3, 4, 5 (and LoRMA in 6 and 7).

## Data findings (reported, not exploited)

- **Phase 4:** the released `Test/test.json` (1,517 docs) is entirely contained in `Train/dev.json`,
  labels included. `p4_cjpe.py` removes every test id from dev before training or tuning (994 clean dev docs).
- **Phase 7:** older Supreme Court judgments keep a `CITATOR INFO` header listing later cases that cite
  them; matching a query's own citation against it recovers gold links at 99% precision on dev. Both
  header fields are stripped from every document before modelling (`p7_lex.strip_meta`), and no submission used it.

## Use of AI assistance

AI coding assistance was used as permitted by the course (from 7 October 2026); its extent is stated in the LLM usage
declaration of the report. No external LLM API is called anywhere in the pipeline. References for every borrowed idea are
in the module docstrings and in the report's bibliography.
