# CS781 Capstone: In-House Legal Intelligence System

Darsh Kedia (250311), IIT Kanpur. Codabench username `d_250311`.

This is my code for all seven phases of the CS781 capstone. In Phase 1 I pretrained four small language models
(about 110-125M parameters each) from scratch on the official Indian legal corpus. Phases 2-7 then fine-tune
those models for the downstream tasks, using full fine-tuning, LoRA, two LoRMA variants, knowledge
distillation and ensembles. I didn't use any pretrained weights, outside corpora or LLM APIs anywhere in the
pipeline. All data comes from the gated HF dataset `Exploration-Lab/CS-781-Capstone`.

Checkpoints, data and submission zips aren't in the repo because of their size. Everything can be rebuilt from
the notebooks on a free Kaggle T4 x2 session.

## Results

| Phase | Task | Metric | Best Codabench | Best submission | Scored zips |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | ROUGE-L / BLEU | not submitted (I missed the deadline); local pseudo-test ROUGE-L 0.288 | unilm, 400M tokens | 0 |
| 2 | Legal NER (L-NER) | strict Macro-F1 | **0.5189** | Phase 1 unilm fine-tuned with a BIO head | 12 |
| 3 | Rhetorical roles | Macro-F1 | **0.5234** | ensemble of 9 runs (unilm/gpt; full, LoRA, LoRMA, KD) | 11 |
| 4 | Judgment prediction (CJPE) | Macro-F1 | **0.7539** | 6-layer unilm student distilled from the full-FT unilm | 16 |
| 5 | Statute identification (LSI) | Macro-F1 | **0.3385** | greedy ensemble of 8 runs | 16 |
| 6 | Summarisation | BERTScore F1 | **0.8522** | TF-IDF centroid extraction (best generator: gpt_full_ext 0.8330) | 18 |
| 7 | Prior case retrieval (PCR) | Micro-F1@7 | **0.4566** | bigram BM25 + citation-context query + date filter | 8 |

Each phase folder has its own README with every zip I submitted, what it was, and its dev and test score.

## Layout

There is one folder per phase. Each has the code for the best submission (and the other submissions), the
Kaggle notebook(s) and a README.

```
phase1/   cs781_p1.py (tokenizer, data prep, 4 architectures, training, generation, pseudo-eval),
          tokenizer.json, meta.json, Kaggle notebooks (gpt/unilm, encdec, retnet), tests/
phase2/   lner_model.py (BIO tagger, any encoder incl. Phase 1), lner_utils.py, baseline_rules.py, w2v.py,
          kaggle/ (from-scratch models), kaggle_unilm/ (best: Phase 1 unilm fine-tuned)
phase3/   p3_rr.py, ft_core.py, notebook, tests/
phase4/   p4_cjpe.py, ft_core.py, notebook (+ round-2 notebook, never run)
phase5/   p5_lsi.py, p5_post.py, ft_core.py, notebook (+ round-2 notebook, never run)
phase6/   p6_extract.py, make_ladder.py, make_fusion.py, eval_*.py (extractive, best), p6_summ.py, ft_core.py, notebook
phase7/   p7_lex.py, make_lex_zips.py (lexical, best), p7_dense.py, ft_core.py, notebook (dense + fusion)
tools/    report_figures.py (the data figure in the report)
```

`ft_core.py` is the same file in phases 3-7. It loads any of my Phase 1 checkpoints (gpt, unilm, encdec, retnet;
it picks up `cs781_p1.py` from `../phase1`). It has full fine-tuning, LoRA, LoRMA+, LoRMA-pi, norm-only and
frozen-head training, plus the distillation loss, mixed precision and the LR schedule.

## Phase 1 models

All four use the same byte-level BPE tokenizer (32k vocab, which I trained on the Phase 1 text;
`phase1/tokenizer.json`), a 1024-token context and the same data. I prepared 420M tokens once, mixed 35% Supreme
Court, 35% High Courts, 15% tribunals and 15% acts.

| key | family | params | design | tokens | val loss (ppl) | tail loss* | pseudo ROUGE-L |
|---|---|---|---|---|---|---|---|
| `gpt` | decoder-only | 109.5M | 12 layers, d 768, 12 heads, RoPE, RMSNorm, SwiGLU, tied embeddings | 200M | 2.203 (9.05) | **2.139** | 0.277 |
| `unilm` | hybrid encoder | 110.1M | one stack trained on both causal LM and masked LM (mode embeddings) | 400M | 2.297 (9.94) | 2.234 | **0.288** |
| `encdec` | encoder-decoder | 123.7M | 6+6 layers, prefix-LM + T5 span corruption | 200M | 2.936 (18.84) | 2.936 | 0.178 |
| `retnet` | efficient (RetNet) | 116.6M | multi-scale retention instead of softmax attention, O(1) recurrent decoding | 200M | 2.341 (10.39) | 2.313 | 0.280 |

\*tail loss = loss on the last 256 tokens given the first 768. It's the one number I could compare across all
four, since plain val loss means different things for each objective.

## Running it

Everything that needs a GPU is a self-contained Kaggle notebook. Each notebook writes the same `.py` files that
are in this repo with `%%writefile`, so you only need to upload the notebook.

1. On Kaggle: New notebook > File > Import notebook. Set the accelerator to GPU T4 x2 and turn Internet on.
2. Add-ons > Secrets: add a secret called `HF` with a Hugging Face read token that has access to the (gated)
   dataset. Every new notebook needs it attached.
3. For Phases 2-7, Add Input > Your Work and attach the Phase 1 output notebooks (one per architecture). The
   notebooks find every `model_best.pt` and `tokenizer.json` under `/kaggle/input` and read the architecture
   from the checkpoint.
4. Save Version > Save & Run All. Every run writes its `submission.zip` and a results bundle to
   `/kaggle/working`. There's a `MAX_HOURS` guard so the notebook stops starting new jobs before Kaggle's limit.

| Phase | What to run | Time |
|---|---|---|
| 1 | `phase1/CS781-CP-Phase1-Darsh-Kedia-250311.ipynb` (`STAGE` = `prepare_train`, `train`, `generate`, `pseudo_eval`; `ARCH` = `gpt` / `unilm`; encdec and retnet have their own copies) | ~2.2 h per arch per 200M tokens (retnet 3.1 h, encdec 1.2 h) |
| 2 | `phase2/kaggle_unilm/CS781-CP-Phase2-unilm-Darsh-Kedia-250311.ipynb` (best); `phase2/kaggle/...` for the from-scratch models | ~30 min |
| 3 | `phase3/CS781-CP-Phase3-Darsh-Kedia-250311.ipynb` | 48 min |
| 4 | `phase4/CS781-CP-Phase4-Darsh-Kedia-250311.ipynb` | 4.4 h |
| 5 | `phase5/CS781-CP-Phase5-Darsh-Kedia-250311.ipynb` | 4.0 h |
| 6 | `phase6/make_ladder.py` on CPU (extractive, the best zips); `phase6/CS781-CP-Phase6-Darsh-Kedia-250311.ipynb` (generators) | ~10 min CPU / 85 min |
| 7 | `phase7/make_lex_zips.py` on CPU (the submitted zips); `phase7/CS781-CP-Phase7-...ipynb` (dense + fusion) | ~10 min CPU / 64 min |

To run things locally, first download the data
(`huggingface-cli download Exploration-Lab/CS-781-Capstone --repo-type dataset --local-dir data`):

```bash
pip install -r requirements.txt
python phase1/tests/test_models.py
python phase2/baseline_rules.py dev data/Phase_2_LNER
python phase2/lner_model.py train data/Phase_2_LNER runs/charw2v --tokenizer phase1/tokenizer.json --encoder charbilstm --init_emb emb256.npy --full
python phase2/lner_model.py train data/Phase_2_LNER runs/unilm --tokenizer phase1/tokenizer.json --encoder phase1 --phase1_ckpt unilm/model_best.pt --full --epochs 4
python phase3/p3_rr.py train --data data/Phase_3_RR --out runs/p3/unilm_lorma \
       --tokenizer phase1/tokenizer.json --ckpt unilm/model_best.pt --method lorma_plus --epochs 12
python phase4/p4_cjpe.py prep ...    # then train / ensemble / score / validate (see --help)
python phase5/p5_lsi.py prep ...     # then train / ensemble / baseline (see --help)
CS781_ROOT=. python phase6/make_ladder.py
python phase7/make_lex_zips.py data/Phase_7_PCR out/p7
```

Before uploading anything I ran every zip through a validator that checks the submission format. I also checked
each of my local scorers against the official `evaluate_phaseN.py` (same numbers to 6 decimals).

## Fine-tuning methods

| method | what gets trained | trainable params (unilm) |
|---|---|---|
| `full` | everything | 110M |
| `lora` (Hu et al. 2022) | W + (α/r)·BA on q, k, v, o | 1.2M at r = 16 |
| `lorma_plus` (Bihany, Patel, Modi 2025) | (I + (α/r)·BA)·W, multiplicative, starts exactly at the Phase 1 model | 1.2M |
| `lorma_pi` (same paper) | π(BA)·W with the cyclic-shift permutation, initialised so π(BA) = I | 1.2M |
| `head` | frozen backbone, only the task head (probe baseline) | 0 |
| KD | 6-layer student (67M, layers 0, 2, 4, 7, 9, 11 of the Phase 1 model) trained from a full-FT teacher, (1-α)·CE + α·T²·KL with α 0.5, T 2 | 67M |

All four Phase 1 architectures are fine-tuned in Phases 4 and 5 (Phase 3 used unilm, gpt and encdec; Phase 6
used unilm, gpt and retnet). LoRMA and distillation are used in Phases 3, 4 and 5, and LoRMA also in 6 and 7.
The distilled student keeps 6 evenly spaced layers (0, 2, 4, 7, 9, 11) of the Phase 1 model and starts from
the Phase 1 weights, not from the fine-tuned teacher.

## Two data issues I found (and didn't use)

- **Phase 4:** the released `Test/test.json` (1,517 docs) is completely inside `Train/dev.json`, labels included.
  `p4_cjpe.py` drops every test id from dev before training or tuning, which leaves 994 clean dev docs.
- **Phase 7:** older Supreme Court judgments still have a `CITATOR INFO` header that lists the later cases
  citing them. Matching a query's own citation against it finds gold links with 99% precision on dev. I strip
  both header fields from every document before doing anything (`p7_lex.strip_meta`), and no submission used it.

## References

The papers each method is based on are listed at the top of each module and in my report's bibliography.
