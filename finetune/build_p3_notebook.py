#!/usr/bin/env python3
"""Builds the self-contained Kaggle notebook for Phase 3 from the .py files in this folder.

  python build_p3_notebook.py [OUT.ipynb]
Rebuild after editing ft_core.py, p3_rr.py or ../phase1/cs781_p1.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_p3", "CS781-CP-Phase3-Darsh-Kedia-250311.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def writefile(path, name):
    return code(f"%%writefile {name}\n" + open(path, encoding="utf-8").read())


INTRO = """
# CS781 Phase 3: Rhetorical Role labelling from the Phase 1 models (Darsh Kedia, 250311)

**Before running (Kaggle):** Accelerator = GPU T4 x2, Internet on, **Add-ons > Secrets: attach `HF`**
(Hugging Face token), and **Add Input** the Phase 1 notebook outputs that hold the checkpoints:
the unilm run (`cs781_phase1/unilm/model_best.pt`), the gpt run (`cs781_phase1/gpt/model_best.pt`),
and encdec / retnet when they exist. One of them must also hold `cs781_phase1/data/tokenizer.json`.
Paths and architectures are found automatically.

What it does: runs the experiment plan below two jobs at a time (one per T4), each job writing a
Codabench-ready `submission.zip` under `/kaggle/working/p3/<run>/`, then builds ensembles and prints a
table of dev Macro-F1 per run. Upload the zips to Codabench Phase 3 (best dev first).

Methods (see `ft_core.py`): full fine-tuning, LoRA, LORMA (`lorma_plus`, `lorma_pi`), norm-only tuning,
and knowledge distillation into a 6-layer student initialised from the teacher's layers.
"""

CONFIG = r'''
# ---- configuration ----------------------------------------------------------------
import glob, os, torch
WORK = "/kaggle/working"
OUT = f"{WORK}/p3"
EPOCHS = 8                 # full fine-tuning epochs (PEFT runs use EPOCHS + 4)
QUICK = False              # True = 1 epoch per run, to check the pipeline end to end (~10 min)

# find the Phase 1 checkpoints and the shared tokenizer among the attached inputs
CKPT = {}
for p in sorted(glob.glob("/kaggle/input/**/model_best.pt", recursive=True)):
    arch = torch.load(p, map_location="cpu", weights_only=False)["model_config"]["arch"]
    CKPT.setdefault(arch, p)
toks = sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
TOKENIZER = toks[0] if toks else None
print("checkpoints:", CKPT)
print("tokenizer:", TOKENIZER)
assert "unilm" in CKPT and TOKENIZER, "attach the Phase 1 unilm output (model_best.pt + data/tokenizer.json)"
'''

PLAN = r'''
# ---- experiment plan: (name, arch, extra args, runs after) ----------------------------
# Runs without --full train on train and pick the best epoch on dev (dev Macro-F1 is reported);
# --full runs train on train+dev for a fixed number of epochs (no dev score, usually a bit better).
E, EP = (1, 1) if QUICK else (EPOCHS, EPOCHS + 4)
PLAN = [
    ("unilm_full",      "unilm", f"--method full --epochs {E}", []),
    ("unilm_lorma",     "unilm", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("unilm_lora",      "unilm", f"--method lora --rank 16 --alpha 32 --epochs {EP}", []),
    ("gpt_full",        "gpt",   f"--method full --epochs {E}", []),
    ("unilm_full_all",  "unilm", f"--method full --epochs {E} --full --save_train_logp", []),
    ("unilm_lorma_pi",  "unilm", f"--method lorma_pi --rank 16 --epochs {EP}", []),
    ("kd_unilm6",       "unilm", f"--method full --keep_layers 6 --epochs {E} --teacher {OUT}/unilm_full_all", ["unilm_full_all"]),
    ("unilm_lorma_all", "unilm", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP} --full", []),
    ("encdec_full",     "encdec", f"--method full --epochs {E}", []),
    ("retnet_full",     "retnet", f"--method full --epochs {E}", []),
    ("gpt_lorma",       "gpt",   f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
]
PLAN = [p for p in PLAN if p[1] in CKPT]
for p in PLAN: print(p[0], p[1], p[2])
'''

DATA = r'''
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
tokn = UserSecretsClient().get_secret("HF").strip().strip("<>")
root = snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                         allow_patterns=["Phase_3_RR/*"], token=tokn, local_dir=f"{WORK}/hf")
DATA = f"{root}/Phase_3_RR"
print(sorted(glob.glob(f"{DATA}/*/*")))
'''

RUN = r'''
# ---- run the plan two jobs at a time, one per GPU -----------------------------------
import subprocess, time, json
os.makedirs(OUT, exist_ok=True)
n_gpu = max(1, torch.cuda.device_count())
pending, running, done, failed = list(PLAN), {}, set(), set()
t_start = time.time()
while pending or running:
    for g in range(n_gpu):
        if g in running:
            continue
        job = next((p for p in pending if all(d in done for d in p[3])), None)
        if job is None:
            continue
        pending.remove(job)
        name, arch, extra, _ = job
        cmd = (f"python p3_rr.py train --data {DATA} --out {OUT}/{name} --tokenizer {TOKENIZER} "
               f"--ckpt {CKPT[arch]} --code_dir . {extra}")
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
    # jobs whose teacher failed can never start
    for p in list(pending):
        if any(d in failed for d in p[3]):
            pending.remove(p); failed.add(p[0]); print("skipped", p[0], "(dependency failed)")
    time.sleep(10)
print("done:", sorted(done), "failed:", sorted(failed))
'''

ENSEMBLE = r'''
# ---- ensembles (average of per-sentence log-probs) ---------------------------------
import subprocess
def ens(name, runs, extra=""):
    runs = [f"{OUT}/{r}" for r in runs if os.path.exists(f"{OUT}/{r}/test_logp.pt")]
    if len(runs) >= 2:
        r = subprocess.run(["python", "p3_rr.py", "ensemble", "--data", DATA, "--out", f"{OUT}/{name}", *extra.split(), *runs],
                           capture_output=True, text=True)
        print(name, (r.stdout + r.stderr).strip().splitlines()[-1])
ens("ens_dev3", ["unilm_full", "unilm_lorma", "unilm_lora"])            # dev-scored members
ens("ens_all", ["unilm_full_all", "unilm_lorma_all", "unilm_full", "unilm_lorma", "kd_unilm6"])
ens("ens_archs", ["unilm_full", "gpt_full", "encdec_full", "retnet_full"])
'''

TABLE = r'''
# ---- summary: dev Macro-F1 per run (runs trained with --full have none) ---------------
import json, pandas as pd
rows = []
for d in sorted(glob.glob(f"{OUT}/*/run.json")):
    r = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
    rows.append({"run": name, "arch": r.get("arch"), "method": r.get("peft", {}).get("method"),
                 "trainable_M": round(r.get("trainable", 0) / 1e6, 2),
                 "dev_f1": r.get("dev_f1"), "dev_f1_bias(optimistic)": r.get("dev_f1_bias_tuned"),
                 "epoch_sec": (r.get("history") or [{}])[-1].get("sec"),
                 "zip": os.path.exists(f"{OUT}/{name}/submission.zip")})
df = pd.DataFrame(rows).sort_values("dev_f1", ascending=False, na_position="last")
print(df.to_string(index=False))
df.to_csv(f"{OUT}/summary.csv", index=False)
# keep the output small and downloadable: zips + logs + run.json (+ teacher log-probs)
import zipfile
with zipfile.ZipFile(f"{WORK}/p3_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f, os.path.relpath(f, WORK))
print("p3_results.zip", os.path.getsize(f"{WORK}/p3_results.zip"), "bytes")
'''

cells = [
    md(INTRO),
    code(CONFIG),
    code("!pip -q install tokenizers huggingface_hub"),
    writefile(os.path.join(HERE, "..", "phase1", "cs781_p1.py"), "cs781_p1.py"),
    writefile(os.path.join(HERE, "ft_core.py"), "ft_core.py"),
    writefile(os.path.join(HERE, "p3_rr.py"), "p3_rr.py"),
    code(PLAN),
    code(DATA),
    md("Runs the plan (about 1 to 1.5 h on T4 x2 with unilm + gpt attached). Per-run logs: `/kaggle/working/p3/<run>/stdout.log`."),
    code(RUN),
    code(ENSEMBLE),
    code(TABLE),
    md("Download `p3_results.zip` (all submission zips, logs and run.json files) or the single zips from `/kaggle/working/p3/<run>/submission.zip`, and upload to Codabench Phase 3.\n\n"
       "## LLM usage declaration\nLLMs were used for code generation; no external LLM API is called in this pipeline."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(OUT, os.path.getsize(OUT), "bytes")
