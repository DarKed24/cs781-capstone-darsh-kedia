# Phase 1: legal language model pretraining from scratch

I trained four models of about 110-125M parameters here, with one shared tokenizer, one data pipeline and one
resumable training script. I missed the Phase 1 Codabench deadline (6 Oct), so I started again from scratch on
7 Oct and built these mainly as the base for Phases 2-7. Since I couldn't submit, I evaluated them on a local
pseudo-test instead: 200 held-out documents, a 512-word prompt and the real next 128 words, scored with
ROUGE-L and BLEU-4.

## Files
| file | purpose |
|---|---|
| `cs781_p1.py` | everything for Phase 1: tokenizer training, data prep, the four architectures, multi-GPU training, generation, pseudo-eval, and `load_model()`, which all later phases use |
| `CS781-CP-Phase1-Darsh-Kedia-250311.ipynb` | Kaggle notebook (gpt and unilm; `ARCH`, `STAGE` cells) |
| `CS781-CP-Phase1-encdec-...ipynb`, `...-retnet-...ipynb` | same notebook with `ARCH` preset |
| `tokenizer.json` | the trained 32k byte-level BPE tokenizer (pad 0, unk 1, `<s>` 2, `</s>` 3, `<mask>` 4, `<sep>` 5, 64 `<extra_id_k>`) |
| `meta.json` | token counts per source of the prepared data |
| `tests/test_models.py` | checks parameter counts, that cached / recurrent decoding gives the same output as a full forward pass for all four archs, and the span corruption and MLM batches |

## Architectures (all d_model 768, context 1024, tied embeddings, RMSNorm, pre-norm)
- **gpt**: 12 layers, 12 heads, RoPE, SwiGLU FFN (2048). Causal LM. 109.5M.
- **unilm**: the gpt stack plus a 2-row mode embedding; each batch is either causal LM (mode 0) or masked LM with bidirectional attention (mode 1, 15% masking, MLM transform head). 110.1M.
- **encdec**: 6 encoder + 6 decoder layers; batches alternate prefix-LM continuation (768 tokens in, 256 out) and T5 span corruption. 123.7M.
- **retnet**: 12 layers, 6 heads of multi-scale retention (per-head decay γ = 1 − 2^(−5−h)), group norm, swish gate; parallel form for training, recurrent form for O(1)-memory generation. 116.6M.

## Data and training
- I mixed the sources 35% Supreme Court, 35% High Courts, 15% tribunals and 15% acts, since most later tasks are on judgments. 420M tokens are prepared once (`STAGE="prepare"`), and every 100th document of each source is held out for validation.
- AdamW (β 0.9/0.95, wd 0.1), peak LR 6e-4, 100-step warm-up, cosine to 6e-5, 131,072 tokens per step (2 GPUs × micro-batch 8 × 1024 × accumulation 8), dropout 0, fp16 autocast, grad clip 1.0, checkpoint every 100 steps, auto-stop at `MAX_HOURS`, exact resume. If you resume a finished run with a larger `TRAIN_TOKENS`, it warms up again to half the peak (3e-4) and does another cosine cycle. I used this to take unilm from 200M to 400M tokens.
- Decoding for the continuation task: temperature 0.7, top-k 50, top-p 0.9, repetition penalty 1.1, no-repeat 4-gram, up to 224 new tokens cut to the 128-word target.

## How to run (Kaggle, 2x T4)
1. `STAGE="prepare_train"`, `ARCH="gpt"`: trains the tokenizer, writes the token shards and trains gpt (~2.4 h).
2. For the other archs attach the first notebook's output (for its `data/`) and set `ARCH`, `STAGE="train"`.
3. `STAGE="generate"` writes the Codabench `prediction.json`; `STAGE="pseudo_eval"` scores the held-out pseudo-test.

## Results (seed 42)

At 200M tokens gpt learns the most per token. unilm spends half its tokens on MLM, so it needed the extra 200M
tokens to catch up, and encdec only gets decoder loss on 256 of every 1024 tokens, so it ends up the weakest.
All the loss curves were still going down when training stopped.
| arch | params | tokens | GPU h | tok/s | val loss (ppl) | tail loss | pseudo ROUGE-L | BLEU-4 |
|---|---|---|---|---|---|---|---|---|
| gpt | 109.5M | 200M | 2.16 | 25.8k | 2.203 (9.05) | **2.139** | 0.277 | 0.183 |
| unilm | 110.1M | 200M | 2.17 | 25.8k | 2.545 (12.74) | 2.508 | 0.268 | 0.171 |
| unilm (continued) | 110.1M | 400M | +2.26 | 25k | 2.297 (9.94) | 2.234 | **0.288** | **0.196** |
| encdec | 123.7M | 200M | 1.2 | 47.8k | 2.936 (18.84) | 2.936 | 0.178 | 0.054 |
| retnet | 116.6M | 200M | 3.06 | 18.3k | 2.341 (10.39) | 2.313 | 0.280 | 0.188 |
