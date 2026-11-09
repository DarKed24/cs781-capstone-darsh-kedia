#!/usr/bin/env python3
"""Builds the self-contained Kaggle notebook for Phase 6 (summarisation) from the .py files in this folder.

  python build_p6_notebook.py [OUT.ipynb]
Rebuild after editing ft_core.py, p6_summ.py or ../phase1/cs781_p1.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_p6", "CS781-CP-Phase6-Darsh-Kedia-250311.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def writefile(path, name):
    return code(f"%%writefile {name}\n" + open(path, encoding="utf-8").read())



INTRO = """
# CS781 Phase 6: abstractive summarisation from the Phase 1 models (Darsh Kedia, 250311)

**Before running (Kaggle):** Accelerator = GPU T4 x2, Internet on, **Add-ons > Secrets: attach `HF`**,
and **Add Input** the Phase 1 notebook outputs (`cs781_phase1/<arch>/model_best.pt`, ONE per arch:
unilm 400M, gpt, retnet; encdec is not used here). One of them must hold `cs781_phase1/data/tokenizer.json`.
Then **Save Version > Save & Run All**. Expected wall time about 2-2.5 h (MAX_HOURS stops new jobs).

Each job fine-tunes a Phase 1 LM as a decoder-only summariser: `[source 576 tok] <sep> [summary 447 tok]`,
loss on the summary only, greedy decoding with no-repeat-4-gram. Each job writes a Codabench-ready
`submission.zip` under `/kaggle/working/p6/<run>/` as soon as it finishes, plus BERTScore/ROUGE on 100
held-out train documents (never trained on).

Original idea, ablated: **extract-then-generate**. The source is either the header-skipped lead of the
judgment (`head`) or ~470 words picked by a supervised extractive sentence scorer (`ext`, see
`p6_extract.py`). Same model, same budget, only the source changes.
"""

CONFIG = r'''
import glob, os, torch
WORK = "/kaggle/working"
OUT = f"{WORK}/p6"
CACHE = f"{WORK}/p6_cache.pt"
QUICK = False          # True = 64 train docs, short generation: end-to-end check (~10 min)
MAX_HOURS = 1.45       # no new job starts after this (each job is ~35-45 min)
TRAIN_MIN = 28         # training-time cap per job (minutes); the LR schedule is shrunk to fit
CKPT = {}
for p in sorted(glob.glob("/kaggle/input/**/model_best.pt", recursive=True)):
    arch = torch.load(p, map_location="cpu", weights_only=False)["model_config"]["arch"]
    CKPT.setdefault(arch, p)
toks = sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
TOKENIZER = toks[0] if toks else None
print("checkpoints:", CKPT); print("tokenizer:", TOKENIZER)
assert TOKENIZER and ("unilm" in CKPT or "gpt" in CKPT), "attach the Phase 1 outputs"
'''

PLAN = r'''
# (name, arch, extra args). Priority order; two jobs run at a time (one per T4).
Q = " --n_train 64 --n_eval 10 --epochs 1 --quick" if QUICK else ""
COMMON = f"--max_train_min {TRAIN_MIN} --epochs 2 --batch 4 --accum 4" + Q
PLAN = [
    ("unilm_full_ext",  "unilm",  "--method full --src ext"),
    ("gpt_full_ext",    "gpt",    "--method full --src ext"),
    ("unilm_full_head", "unilm",  "--method full --src head"),        # ablation: no extraction
    ("unilm_lorma_ext", "unilm",  "--method lorma_plus --rank 16 --alpha 32 --src ext"),
    ("retnet_full_ext", "retnet", "--method full --src ext"),
    ("gpt_full_head",   "gpt",    "--method full --src head"),        # ablation on a second arch
    ("unilm_lora_ext",  "unilm",  "--method lora --rank 16 --alpha 32 --src ext"),
]
PLAN = [p for p in PLAN if p[1] in CKPT]
for p in PLAN: print(p)
'''

DATA = r'''
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
tokn = UserSecretsClient().get_secret("HF").strip().strip("<>")
root = snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                         allow_patterns=["Phase_6_Summ/*"], token=tokn, local_dir=f"{WORK}/hf")
DATA = f"{root}/Phase_6_Summ"
print(sorted(glob.glob(f"{DATA}/*/*")))
import subprocess   # fits the extractive scorer, builds head/ext sources, tokenises (~5-8 min, CPU)
subprocess.run(["python", "p6_summ.py", "prep", "--data", DATA, "--tokenizer", TOKENIZER, "--cache", CACHE], check=True)
# pre-download the BERTScore model once so the two parallel jobs do not race
import bert_score; bert_score.score(["a test"], ["a test"], lang="en", device="cpu")
'''

RUN = r'''
import subprocess, time, json
os.makedirs(OUT, exist_ok=True)
n_gpu = max(1, torch.cuda.device_count())
pending, running, done, failed = list(PLAN), {}, set(), set()
t_start = time.time()
while pending or running:
    for g in range(n_gpu):
        if g in running or not pending:
            continue
        if time.time() - t_start > MAX_HOURS * 3600:
            print("time budget reached, skipping:", [p[0] for p in pending]); pending.clear(); break
        job = pending.pop(0); name, arch, extra = job
        cmd = (f"python p6_summ.py train --cache {CACHE} --out {OUT}/{name} --ckpt {CKPT[arch]} "
               f"--code_dir . --tokenizer {TOKENIZER} {COMMON} {extra}")
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        log = open(f"{OUT}/{name}/stdout.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        running[g] = (job, subprocess.Popen(cmd.split(), stdout=log, stderr=subprocess.STDOUT, env=env), time.time())
        print(f"[{(time.time()-t_start)/60:5.1f} min] GPU{g} start {name}: {extra}", flush=True)
    for g, (job, proc, t0) in list(running.items()):
        if proc.poll() is not None:
            del running[g]
            ok = proc.returncode == 0 and os.path.exists(f"{OUT}/{job[0]}/submission.zip")
            (done if ok else failed).add(job[0])
            tail = open(f"{OUT}/{job[0]}/stdout.log").read().strip().splitlines()[-3:]
            print(f"[{(time.time()-t_start)/60:5.1f} min] GPU{g} {'done' if ok else 'FAILED'} {job[0]} "
                  f"({(time.time()-t0)/60:.1f} min)\n    " + "\n    ".join(tail), flush=True)
    time.sleep(15)
print("done:", sorted(done), "failed:", sorted(failed))
'''

TABLE = r'''
import json, pandas as pd, zipfile
rows = []
for d in sorted(glob.glob(f"{OUT}/*/run.json")):
    r = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
    rows.append({"run": name, "arch": r["arch"], "method": r["method"], "src": r["src"],
                 "trainable_M": round(r["peft"]["backbone_trainable"] / 1e6, 2), "steps": r["steps"],
                 "train_min": round(r["train_min"], 1), "final_loss": round(r["history"][-1]["loss"], 4),
                 **{f"hold_{k}": round(v, 4) for k, v in r["hold"].items()}})
df = pd.DataFrame(rows).sort_values("hold_bertscore", ascending=False)
print(df.to_string(index=False)); df.to_csv(f"{OUT}/summary.csv", index=False)
with zipfile.ZipFile(f"{WORK}/p6_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f, os.path.relpath(f, WORK))
print("p6_results.zip", os.path.getsize(f"{WORK}/p6_results.zip"), "bytes")
from IPython.display import FileLink; display(FileLink("p6_results.zip"))
'''

cells = [
    md(INTRO),
    code(CONFIG),
    code("!pip -q install tokenizers huggingface_hub bert-score rouge-score"),
    writefile(os.path.join(HERE, "..", "phase1", "cs781_p1.py"), "cs781_p1.py"),
    writefile(os.path.join(HERE, "ft_core.py"), "ft_core.py"),
    writefile(os.path.join(HERE, "..", "phase6", "p6_extract.py"), "p6_extract.py"),
    writefile(os.path.join(HERE, "p6_summ.py"), "p6_summ.py"),
    code(PLAN),
    code(DATA),
    md("Runs the plan. Per-run logs: `/kaggle/working/p6/<run>/stdout.log`; each finished run already has its `submission.zip`."),
    code(RUN),
    code(TABLE),
    md("Download `p6_results.zip` (every submission zip, held-out scores, logs) and upload the zips to Codabench Phase 6, "
       "best held-out BERTScore first.\n\n## LLM usage declaration\nLLMs were used for code generation; no external LLM API is called in this pipeline."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(OUT, os.path.getsize(OUT), "bytes")
