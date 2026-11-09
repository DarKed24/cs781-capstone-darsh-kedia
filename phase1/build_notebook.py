#!/usr/bin/env python3
"""Builds CS781-CP-Phase1-Darsh-Kedia-250311.ipynb from cs781_p1.py (keeps them in sync)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "cs781_p1.py")).read()
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# CS781 Capstone, Phase 1: Legal Language Model Pretraining

**Darsh Kedia, 250311.** One notebook trains all four Phase 1 architectures from random
initialisation on the `Exploration-Lab/CS-781-Capstone` Phase 1 corpus, using our own 32k
byte-level BPE tokenizer.

| key | architecture | params | what makes it different |
|---|---|---|---|
| `gpt` | Decoder-only Transformer | 109.5M | causal softmax attention, RoPE, RMSNorm, SwiGLU |
| `encdec` | Encoder-decoder Transformer | 123.7M | bidirectional encoder + cross-attending decoder; prefix-LM + T5 span corruption |
| `unilm` | Hybrid encoder (UniLM) | 110.1M | one stack trained both causally and bidirectionally (MLM), mode embeddings, BERT-style MLM head |
| `retnet` | Retentive Network (efficient) | 116.6M | no softmax: multi-scale exponential-decay retention, O(1) memory per generated token |

**How to run (Kaggle, GPU T4 x2, Internet on, secret `HF` = your Hugging Face token):**
1. `STAGE = "smoke"` once (about 5 minutes): tiny model, tiny data, whole pipeline end to end.
   Short on time? `STAGE = "prepare_train"` with `ARCH = "gpt"` does steps 2 and 3 in one GPU
   session (costs about 30 extra GPU minutes); its output then serves as the data input for the rest.
2. `STAGE = "prepare"` once, on a CPU session (no GPU quota): trains the tokenizer and writes
   the token bins. *Save Version*, then add that notebook's output as an input dataset.
3. `STAGE = "train"` with `ARCH` set to each of `gpt`, `unilm`, `encdec`, `retnet` (one *Save
   Version* run per architecture). Each run checkpoints every 100 steps and stops cleanly at
   `MAX_HOURS`; if a run is cut, add its output as input and run again, it resumes by itself.
   The run ends by writing `prediction.json` and `submission_<arch>.zip`.
""")

md("## 1. Setup: packages, Hugging Face login, GPU check")
code("""
!pip -q install ijson tokenizers huggingface_hub

import os, json, glob, subprocess, sys
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF").strip().strip("<>")
except Exception as e:
    print("Kaggle secret not available:", e, "| using HF_TOKEN from the environment if set")
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("Hugging Face login successful.")

import torch
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPU)])
""")

md("""
## 2. The Phase 1 code

Everything lives in one module, written to disk so that `torchrun` can launch it on both T4s.
It contains the tokenizer training, the streaming data preparation, the four architectures,
the training loop (mixed precision, DDP, resumable checkpoints), generation with KV caches
(recurrent state for RetNet), and local evaluation. References are cited at the top of the file.
""")
code("%%writefile cs781_p1.py\n" + SRC)

md("""
## 3. Configuration

Change `STAGE` and `ARCH` here; everything else has working defaults.
`SEED = 42` everywhere. The data mix is a deliberate choice to keep in the report: the test
prompts read like Supreme Court / High Court judgments, so those get 70% of the tokens.
""")
code("""
STAGE = "smoke"        # smoke | prepare | prepare_train | train | generate | pseudo_eval
ARCH = "gpt"           # gpt | unilm | encdec | retnet
RUN_NAME = ARCH        # change to keep several runs of one architecture apart
SEED = 42
MAX_HOURS = 10.5       # stop + checkpoint before Kaggle's 12h session limit
TRAIN_TOKENS = 200_000_000
WORK = "/kaggle/working/cs781_phase1" if os.path.exists("/kaggle") else os.path.abspath("cs781_phase1")


def find_input(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else ""


# Prepared data from an earlier "prepare" run, attached as an input dataset
meta = find_input("data/meta.json") or find_input("meta.json")
DATA_DIR = os.path.dirname(meta) if meta and STAGE not in ("smoke", "prepare", "prepare_train") else f"{WORK}/data"
# A checkpoint from an earlier, interrupted session of this run
RESUME_FROM = find_input(f"{RUN_NAME}/last.pt") if STAGE == "train" else ""
# For STAGE = "generate" in a fresh session: the trained model attached as input
GEN_CKPT = find_input(f"{RUN_NAME}/model_best.pt") if STAGE in ("generate", "pseudo_eval") else ""

cfg = dict(
    run_name=RUN_NAME + ("_smoke" if STAGE == "smoke" else ""), arch=ARCH, seed=SEED,
    smoke=STAGE == "smoke", out_dir=WORK,
    local_data_dir=os.environ.get("CS781_LOCAL_DATA", ""),  # e.g. on Colab with downloaded files
    data_dir=f"{WORK}/smoke_data" if STAGE == "smoke" else DATA_DIR,
    train_tokens=TRAIN_TOKENS, max_hours=MAX_HOURS, resume_from=RESUME_FROM,
    mix={"supreme_court": 0.35, "high_courts": 0.35, "tribunals": 0.15, "acts": 0.15},
    prepare_train_tokens=420_000_000, tokenizer_vocab=32000,
    seq_len=1024, micro_batch=8, batch_tokens=131072, lr=6e-4, warmup_steps=100,
    gen_checkpoint=GEN_CKPT or "best", temperature=0.7, top_k=50, top_p=0.9,
    repetition_penalty=1.1, no_repeat_ngram=4,
    hf_push_repo="",   # e.g. "darshkedia/cs781-phase1" to also back up checkpoints privately
)
os.makedirs(WORK, exist_ok=True)
with open("run.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(json.dumps(cfg, indent=2))
""")

md("""
## 4. Data preparation (tokenizer + token bins)

Streams each training file straight from the Hub (no 19 GB download): documents are parsed one
at a time, every 100th document goes to validation, and tokens are written as `uint16` bins.
The tokenizer is a byte-level BPE (vocab 32,000) trained only on training documents, with
special tokens `<pad> <unk> <s> </s> <mask> <sep>` and 64 sentinels for span corruption.
Validation documents with at least 640 words also give a local pseudo-test set
(512-word prompt, 128-word reference), mirroring the official task.
""")
code("""
if STAGE in ("smoke", "prepare", "prepare_train"):
    subprocess.run([sys.executable, "cs781_p1.py", "prepare", "--config", "run.json"], check=True)
    print(open(os.path.join(cfg["data_dir"], "meta.json")).read())
""")

md("""
## 5. Training

Mixed precision (fp16 + loss scaling on T4, bf16 on Ampere), AdamW (0.9, 0.95, wd 0.1),
cosine schedule with 100 warm-up steps, 128k tokens per optimiser step, gradient clipping 1.0,
context 1024. `val_tail_loss` is the loss on the last 256 tokens of a 1024-token window given
the first 768; it is the one validation number that is comparable across all four architectures
(the encoder-decoder only predicts those 256 tokens).
""")
code("""
if STAGE in ("smoke", "train", "prepare_train"):
    launcher = (["torchrun", "--standalone", f"--nproc_per_node={N_GPU}"] if N_GPU > 1
                else [sys.executable])
    subprocess.run(launcher + ["cs781_p1.py", "train", "--config", "run.json"], check=True)
""")
code("""
from IPython.display import Image, display
run_dir = os.path.join(WORK, cfg["run_name"])
if os.path.exists(f"{run_dir}/loss_curve.png"):
    display(Image(f"{run_dir}/loss_curve.png"))
if os.path.exists(f"{run_dir}/history.json"):
    hist = json.load(open(f"{run_dir}/history.json"))
    evals = [h for h in hist if "val_loss" in h]
    for h in evals[-5:]:
        print(f"step {h['step']:6d}  val_loss {h['val_loss']:.4f}  ppl {h['val_ppl']:.2f}  tail {h['val_tail_loss']:.4f}")
    speed = [h["tok_per_s"] for h in hist if "tok_per_s" in h]
    if speed:
        print("median training speed:", int(sorted(speed)[len(speed) // 2]), "tokens/s")
""")

md("""
## 6. Generate the Phase 1 test continuations

Loads the best checkpoint (lowest validation loss), keeps the last 800 prompt tokens
(768 for the encoder-decoder), and samples until each continuation has 128 words
(temperature 0.7, top-k 50, top-p 0.9, repetition penalty 1.1, no repeated 4-grams;
special tokens are never sampled). Writes `prediction.json` with only the continuation and
zips it as `submission_<arch>.zip` with `prediction.json` at the zip root.
""")
code("""
if STAGE in ("smoke", "train", "prepare_train", "generate"):
    subprocess.run([sys.executable, "cs781_p1.py", "generate", "--config", "run.json"], check=True)
    preds = json.load(open(f"{run_dir}/prediction.json"))
    k = next(iter(preds))
    print(len(preds), "predictions; example", k, "->", preds[k]["text"][:400])
    assert all(len(v["text"].split()) > 0 for v in preds.values())
""")

md("""
## 7. Local evaluation on held-out validation documents

ROUGE-L F1 and BLEU-4 exactly as in `evaluate_phase1.py`. BERTScore (the primary metric) needs
`bert-score` and downloads `roberta-large` as an evaluation-only scorer; it is off by default.
""")
code("""
USE_BERTSCORE = False
if STAGE in ("smoke", "train", "prepare_train", "pseudo_eval"):
    if USE_BERTSCORE:
        !pip -q install bert-score
    sys.path.insert(0, ".")
    import importlib, cs781_p1 as P
    importlib.reload(P)
    res = P.pseudo_eval(P.load_run_config("run.json"), use_bertscore=USE_BERTSCORE)
    print(res)
""")

md("""
## 8. Parameter counts of the four architectures (Phase 1 rule: 50M to 300M)
""")
code("""
sys.path.insert(0, ".")
import cs781_p1 as P
for a in ["gpt", "unilm", "encdec", "retnet"]:
    with torch.device("meta"):
        m = P.build_model(P.RunConfig(arch=a).model_config(32000))
    print(f"{a:7s} {P.count_params(m)/1e6:7.2f}M")
""")

md("""
## 9. Artifacts

`/kaggle/working/cs781_phase1/<run>/` holds `model_best.pt` and `model_final.pt` (fp16 weights +
config, load with `cs781_p1.load_model`), `last.pt` (full training state for resuming),
`history.json`, `loss_curve.png`, `prediction.json`, `submission_<run>.zip`, `pseudo_eval.json`.
`data/tokenizer.json` is the tokenizer every later phase must use.
Phases 2 to 7 get features from `DecoderLM.hidden(...)` (`unilm`: `mode=1, causal=False` for
bidirectional features), `EncDecLM.encode(...)` and `RetNetLM.hidden(...)`.
""")
code("""
for p in sorted(glob.glob(f"{WORK}/**/*", recursive=True)):
    if os.path.isfile(p) and not p.endswith(".bin"):
        print(f"{os.path.getsize(p)/1e6:9.2f} MB  {p}")
""")

md("""
## 10. Results and analysis

| arch | params | tokens seen | best val loss | val ppl | val tail loss | tokens/s (2xT4) | pseudo ROUGE-L | pseudo BLEU-4 | Codabench BERTScore |
|---|---|---|---|---|---|---|---|---|---|
| gpt | 109.5M | | | | | | | | |
| unilm | 110.1M | | | | | | | | |
| encdec | 123.7M | | | | | | | | |
| retnet | 116.6M | | | | | | | | |

*Fill in after the runs; discuss the comparable tail loss, speed, generation quality, and what
the gaps say about each architecture under a ~200M-token budget.*
""")

md("""
## 11. LLM usage declaration

*Draft, edit before submitting.* Following the course's updated LLM policy (LLMs allowed for code
generation and analysis; no external LLM APIs in the pipeline), an LLM assistant was used
to help write the data pipeline, model, training and generation code in this notebook. No LLM or
LLM API is called anywhere in training, inference or evaluation, and no pretrained weights are
used: every model here is trained from random initialisation on the provided Phase 1 corpus.
All design choices, experiments and conclusions were reviewed and are owned by me.
""")

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
    "nbformat": 4, "nbformat_minor": 4}
out = os.path.join(HERE, "CS781-CP-Phase1-Darsh-Kedia-250311.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, len(cells), "cells")
