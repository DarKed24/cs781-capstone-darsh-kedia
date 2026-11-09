# Phase 1 restart plan (written 2026-10-07)

Phase 1's Codabench window is closed, so Phase 1 now exists to give Phases 2 to 7 their four
foundation models and to cover Phase 1 in the report and viva. Phase 2 closes **2026-10-11,
11:59 PM IST**, so the goal is a first usable checkpoint tonight and all four by 10-09.

## What "four architectures" means here

All four are about 110 to 125M parameters (inside the 50 to 300M rule), share one 32k
tokenizer, one data pipeline and one notebook, and differ in architecture, not hyperparameters:

| key | architecture (PDF category) | params | why it is different | strength later |
|---|---|---|---|---|
| `gpt` | Decoder-only Transformer | 109.5M | causal softmax attention (RoPE, RMSNorm, SwiGLU) | generation (P6), safest baseline |
| `unilm` | Encoder/hybrid adapted for LM | 110.1M | one stack trained both causally and bidirectionally (MLM), mode embeddings, MLM head | bidirectional features for NER and RR (P2, P3) |
| `encdec` | Encoder-decoder Transformer | 123.7M | bidirectional encoder + cross-attending decoder, prefix-LM + T5 span corruption | classification (encoder) and summarisation (P4 to P6) |
| `retnet` | Efficient Transformer (RetNet) | 116.6M | no softmax, multi-scale decay retention, O(1) memory per generated token | long documents (P4, P7) |

The tokenizer is our own byte-level BPE (vocab 32,000) trained on training documents from all
four sources. Context length is 1024 tokens, enough for a 512-word prompt plus a 128-word
continuation.

## Budget

| item | value |
|---|---|
| tokens per architecture | 200M (the PDF recommends 400 to 700M; see "weak points") |
| tokens prepared once | 420M, mix 35% Supreme Court, 35% High Courts, 15% tribunals, 15% acts |
| batch | 128k tokens per optimiser step, so about 1,530 steps per run |
| speed (estimate, check in the smoke run) | 25k to 35k tokens/s on Kaggle T4 x2 for `gpt`/`unilm`; `encdec` faster per input token; `retnet` likely 20 to 30% slower (pure PyTorch, no fused kernels) |
| GPU time per run | about 2 to 3 hours, plus about 10 minutes to generate 1,000 test continuations |
| total | about 10 to 13 of the 30 weekly Kaggle GPU hours, leaving the rest for Phase 2 |
| data preparation | one CPU session (no GPU quota), roughly 20 to 40 minutes |

Every run checkpoints every 100 steps and stops itself at `MAX_HOURS = 10.5`, so a cut
session only loses minutes: attach the output as input and run again, it resumes exactly
(tested: an interrupted run resumed to bit-identical validation loss).

## Order of work

1. **Today, 10-07:** `STAGE="smoke"` on GPU (about 5 minutes), then `STAGE="prepare"` on CPU.
   Save the output as a dataset. Start `ARCH="gpt"`; first usable checkpoint tonight.
2. **10-08:** `unilm` then `encdec`. Phase 2 starts fine-tuning on `gpt` and `unilm` as soon as
   each lands. If the account allows two GPU sessions at once, run two architectures in parallel.
3. **10-09:** `retnet`. Phase 2 runs its LORMA / other PEFT / distillation experiments across all
   four and spreads its six submissions over 10-09 to 10-11.
4. **After Phase 2 (if quota allows):** continue the best one or two models towards 400M+ tokens
   by resuming with a larger `TRAIN_TOKENS`, then regenerate the Phase 1 predictions.

## What will be weak (be upfront about it in the report)

- **Undertrained models.** 200M tokens is about 2 tokens per parameter against roughly 20 for
  compute-optimal training. Expect high perplexity and repetitive, generic continuations;
  Phase 1 BERTScore will probably sit near the starter baseline.
- **One seed, short budget.** Architecture comparisons mostly measure how fast each one learns,
  not its capacity. Use `val_tail_loss` (loss on the last 256 tokens given the first 768) as the
  one number comparable across all four; plain validation loss is not comparable for `encdec`.
- **`encdec` learns from fewer tokens.** Only the decoder side carries loss (25% of each window
  in prefix mode), so its token efficiency is lower.
- **`gpt` is a poor fit for NER and RR.** Causal attention never sees right context; Phases 2 and
  3 will lean on `unilm` and the `encdec` encoder, and the report should say why.
- **`retnet` is efficient at generation, not training,** because it runs in plain PyTorch.
- **Data sample is the head of each file,** not a uniform random sample, because the files are
  streamed rather than downloaded. If the files are ordered by date or court, the sample is skewed.
- **No Phase 1 leaderboard score.** The window is closed; local pseudo-test ROUGE-L / BLEU-4
  (and optionally BERTScore) on held-out documents stand in for it in the report.

## Files

- `CS781-CP-Phase1-Darsh-Kedia-250311.ipynb`: the Kaggle notebook (all stages, all four archs).
- `cs781_p1.py`: the module the notebook writes and runs (also the import for Phases 2 to 7:
  `load_model`, `DecoderLM.hidden`, `EncDecLM.encode`, `RetNetLM.hidden`).
- `build_notebook.py`: regenerates the notebook from `cs781_p1.py` after edits.
- `tests/test_models.py`: parameter counts, cached/recurrent decoding equals full forward for all
  four archs, span corruption and MLM batch checks.

## Results (Kaggle 2x T4, fp16, shared data and tokenizer, seed 42)

| arch | params | tokens | GPU time | tok/s | val loss (ppl) | tail loss | pseudo ROUGE-L | pseudo BLEU-4 |
|---|---|---|---|---|---|---|---|---|
| `gpt` | 109.5M | 200M | 2.16 h | 25.8k | 2.203 (9.05) | **2.139** | 0.277 | 0.183 |
| `unilm` | 110.1M | 200M | 2.17 h | 25.8k | 2.545 (12.74) | 2.508 | 0.268 | 0.171 |
| `unilm` (continued) | 110.1M | 400M | +2.26 h | 25k | 2.297 (9.94) | 2.234 | **0.288** | **0.196** |
| `encdec` | 123.7M | 200M | 1.2 h | 47.8k | 2.936 (18.84)* | 2.936 | 0.178 | 0.054 |
| `retnet` | 116.6M | 200M | 3.06 h | 18.3k | 2.341 (10.39) | 2.313 | 0.280 | 0.188 |

`tail` (loss on the last 256 tokens given the first 768) is the cross-architecture number.
*encdec's validation runs prefix mode only, so its val loss equals its tail loss.

- At equal tokens (200M), `gpt` continues text best, `retnet` is close behind with no softmax
  attention, `unilm` pays for spending half its tokens on MLM, and `encdec` is weakest because its
  decoder gets loss on only 256 of every 1024 tokens and only half its batches are prefix-LM.
- Continuing `unilm` to 400M (fresh lr cycle, re-warm to 3e-4) cut its tail loss by 0.27 and made it
  the best pseudo-test generator; every curve was still falling at the end.
- `encdec` trains fastest per token (shorter decoder), `retnet` slowest (pure PyTorch retention).
- Pseudo-test BLEU/ROUGE are likely optimistic (possible near-duplicate documents across sources).
