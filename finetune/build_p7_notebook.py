#!/usr/bin/env python3
"""Builds the self-contained Kaggle notebook for Phase 7 (prior case retrieval) from the .py files here.

  python build_p7_notebook.py [OUT.ipynb]
Rebuild after editing ft_core.py, p7_lex.py, p7_dense.py or ../phase1/cs781_p1.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_p7", "CS781-CP-Phase7-Darsh-Kedia-250311.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def writefile(path, name):
    return code(f"%%writefile {name}\n" + open(path, encoding="utf-8").read())


INTRO = """
# CS781 Phase 7: prior case retrieval with a Phase 1 bi-encoder + BM25 (Darsh Kedia, 250311)

**Before running (Kaggle):** Accelerator = GPU T4 x2, Internet on, **Add-ons > Secrets: attach `HF`**,
and **Add Input** the Phase 1 notebook outputs (`cs781_phase1/<arch>/model_best.pt`, ONE per arch:
unilm 400M, gpt, retnet; encdec is not used). One of them must hold `cs781_phase1/data/tokenizer.json`.
Then **Save Version > Save & Run All**. Expected wall time about 1.5-2 h (MAX_HOURS stops new jobs).

1. CPU prep (~10-20 min): the lexical stage from `p7_lex.py` (bigram BM25 + citation-context query + temporal
   filter, dev Micro-F1@7 0.466), tokenised citation-context passages for queries and evenly spaced chunks for
   candidates, and distantly supervised (passage, chunk) training pairs.
2. Each GPU job trains a passage-level bi-encoder on a Phase 1 LM with InfoNCE, scores dev/test with MaxSim, tunes the
   fusion weight on dev and writes a Codabench-ready `submission.zip` under `/kaggle/working/p7/<run>/`.
3. A final CPU cell fuses all dense runs with BM25 (weight tuned on dev) into `p7/ens/fuse_all/submission.zip`.

Ablations: `unilm_frozen` (no training, zero-shot Phase 1 features) and `unilm_full_randalign`
(positives are a random passage and chunk instead of the TF-IDF-aligned pair).
"""

CONFIG = r'''
import glob, os, torch
WORK = "/kaggle/working"
OUT = f"{WORK}/p7"
CACHE = f"{WORK}/p7_cache.pt"
QUICK = False          # True = 256 training pairs, 1 epoch: end-to-end check
MAX_HOURS = 1.6        # no new job starts after this
TRAIN_MIN = 20         # training-time cap per job (minutes)
CKPT = {}
for p in sorted(glob.glob("/kaggle/input/**/model_best.pt", recursive=True)):
    arch = torch.load(p, map_location="cpu", weights_only=False)["model_config"]["arch"]
    CKPT.setdefault(arch, p)
toks = sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
TOKENIZER = toks[0] if toks else None
print("checkpoints:", CKPT); print("tokenizer:", TOKENIZER)
assert TOKENIZER and "unilm" in CKPT, "attach the Phase 1 outputs (unilm 400M at least)"
'''

PLAN = r'''
# (name, arch, extra args). Priority order; two jobs run at a time (one per T4).
Q = " --quick --epochs 1" if QUICK else ""
COMMON = f"--max_train_min {TRAIN_MIN} --batch 32 --grad_ckpt" + Q
PLAN = [
    ("unilm_full",           "unilm",  "--method full --epochs 4"),
    ("unilm_lorma",          "unilm",  "--method lorma_plus --rank 16 --alpha 32 --epochs 4"),
    ("unilm_frozen",         "unilm",  "--method head --epochs 0"),                    # ablation: no fine-tuning
    ("unilm_full_randalign", "unilm",  "--method full --epochs 4 --align random"),     # ablation: no alignment
    ("gpt_full",             "gpt",    "--method full --epochs 4"),
    ("retnet_lorma",         "retnet", "--method lorma_plus --rank 16 --alpha 32 --epochs 4"),
]
PLAN = [p for p in PLAN if p[1] in CKPT]
if QUICK: PLAN = PLAN[:2]
for p in PLAN: print(p)
'''

DATA = r'''
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
tokn = UserSecretsClient().get_secret("HF").strip().strip("<>")
root = snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                         allow_patterns=["Phase_7_PCR/*"], token=tokn, local_dir=f"{WORK}/hf")
DATA = f"{root}/Phase_7_PCR"
print(sorted(glob.glob(f"{DATA}/*/*")))
import subprocess, time
t = time.time()
subprocess.run(["python", "p7_dense.py", "prep", "--data", DATA, "--tokenizer", TOKENIZER, "--cache", CACHE], check=True)
print(f"prep {(time.time()-t)/60:.1f} min")
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
        cmd = (f"python p7_dense.py train --cache {CACHE} --out {OUT}/{name} --ckpt {CKPT[arch]} "
               f"--code_dir . {COMMON} {extra}")
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

FUSE = r'''
# Ensembles (CPU): BM25 + mean of z-scored dense runs, weight tuned on dev
trained = [d for d in sorted(glob.glob(f"{OUT}/*")) if os.path.exists(f"{d}/dense_dev.npy")]
ft = [d for d in trained if not d.endswith(("unilm_frozen", "unilm_full_randalign"))]
for name, runs in [("fuse_all", ft), ("fuse_unilm", [d for d in ft if "/unilm_" in d]), ("fuse_every", trained)]:
    if runs:
        subprocess.run(["python", "p7_dense.py", "fuse", "--cache", CACHE, "--runs", ",".join(runs),
                        "--out", f"{OUT}/ens", "--name", name], check=True)
'''

TABLE = r'''
import json, pandas as pd, zipfile
rows = []
for d in sorted(glob.glob(f"{OUT}/*/run.json")):
    r = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
    rows.append({"run": name, "arch": r["arch"], "method": r["method"], "align": r.get("align"),
                 "trainable_M": round(r["peft"]["backbone_trainable"] / 1e6, 2), "steps": r["steps"],
                 "train_min": round(r["train_min"], 1),
                 "final_loss": round(r["history"][-1]["loss"], 4) if r["history"] else None,
                 "dev_lex": round(r["dev_lex_f1"], 4), "dev_dense": round(r["dev_dense_f1"], 4),
                 "best_w": r["best_w"], "dev_fused": round(r["dev_fused"][str(r["best_w"])], 4)})
df = pd.DataFrame(rows).sort_values("dev_fused", ascending=False)
print(df.to_string(index=False)); df.to_csv(f"{OUT}/summary.csv", index=False)
for f in sorted(glob.glob(f"{OUT}/ens/fuse_*.json")):
    r = json.load(open(f)); print(os.path.basename(f), "lex", round(r["lex_only"], 4), "best_w", r.get("best_w"),
                                 "fused", round(max(r.get("dev_fused", {"0": r["lex_only"]}).values()), 4))
with zipfile.ZipFile(f"{WORK}/p7_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f, os.path.relpath(f, WORK))
print("p7_results.zip", os.path.getsize(f"{WORK}/p7_results.zip"), "bytes")
from IPython.display import FileLink; display(FileLink("p7_results.zip"))
'''

cells = [
    md(INTRO),
    code(CONFIG),
    code("!pip -q install tokenizers huggingface_hub scikit-learn"),
    writefile(os.path.join(HERE, "..", "phase1", "cs781_p1.py"), "cs781_p1.py"),
    writefile(os.path.join(HERE, "ft_core.py"), "ft_core.py"),
    writefile(os.path.join(HERE, "p7_lex.py"), "p7_lex.py"),
    writefile(os.path.join(HERE, "p7_dense.py"), "p7_dense.py"),
    code(PLAN),
    code(DATA),
    md("Runs the plan. Per-run logs: `/kaggle/working/p7/<run>/stdout.log`; each finished run already has its `submission.zip`."),
    code(RUN),
    code(FUSE),
    code(TABLE),
    md("Download `p7_results.zip` (every submission zip, dense score matrices, dev scores, logs) and upload the zips to "
       "Codabench Phase 7, best dev first.\n\n## LLM usage declaration\nLLMs were used for code generation; no external LLM API is called in this pipeline."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(OUT, os.path.getsize(OUT), "bytes")
