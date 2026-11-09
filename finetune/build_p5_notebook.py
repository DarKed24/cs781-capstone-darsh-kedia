#!/usr/bin/env python3
"""Builds the self-contained Kaggle notebook for Phase 5 (LSI) from the .py files in this folder.

  python build_p5_notebook.py [OUT.ipynb]
Rebuild after editing ft_core.py, p5_lsi.py or ../phase1/cs781_p1.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_p5", "CS781-CP-Phase5-Darsh-Kedia-250311.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def writefile(path, name):
    return code(f"%%writefile {name}\n" + open(path, encoding="utf-8").read())


INTRO = """
# CS781 Phase 5: Legal Statute Identification (LSI) from the Phase 1 models (Darsh Kedia, 250311)

**Before running (Kaggle):** Accelerator = GPU T4 x2, Internet on, **Add-ons > Secrets: attach `HF`**
(Hugging Face token), and **Add Input** the four Phase 1 notebook outputs that hold the checkpoints
(`cs781_phase1/<arch>/model_best.pt`; attach only ONE output per architecture: unilm 400M, gpt, encdec,
retnet). One of them must also hold `cs781_phase1/data/tokenizer.json`. Paths and architectures are
found automatically.

What it does: tokenises the Phase 5 data once, then runs the experiment plan two jobs at a time (one per
T4). Each job writes a Codabench-ready `submission.zip` under `/kaggle/working/p5/<run>/`; ensembles are
added and a table of dev Macro-F1 per run is printed. Upload the zips to Codabench Phase 5, best dev first.

Task: multi-label, 100 statute ids, Macro-F1. Each document is read as its first 3 x 512 tokens plus its
last 512 tokens (the whole document for ~70% of docs); every chunk is encoded by the Phase 1 backbone and
mean-pooled, a BiGRU over the chunks with attention + max pooling feeds 100 sigmoid outputs. Loss: BCE
with sqrt(neg/pos) positive weights. Decision: a global or per-label threshold tuned on dev (per-label
only when it wins a cross-half check on dev), and every document gets at least its top label.

Dev hygiene: dev and test share no ids; train docs whose text also appears in dev are dropped. No run
trains on dev, so `dev_f1_honest` (thresholds tuned on one half of dev, scored on the other) is an honest
held-out estimate. `dev_f1_tuned` is tuned and scored on the same docs (slightly optimistic).

Methods (see `ft_core.py`): full fine-tuning, LoRA, LORMA (`lorma_plus`, `lorma_pi`), a frozen-backbone
probe, and knowledge distillation into a 6-layer student initialised from the teacher's layers.
"""

CONFIG = r'''
# ---- configuration ----------------------------------------------------------------
import glob, os, torch
WORK = "/kaggle/working"
OUT = f"{WORK}/p5"
CACHE = f"{WORK}/p5_cache.pt"
DOCS_PER_EPOCH = 14000     # fresh random sample of the ~42.2k train docs each epoch
QUICK = False              # True = 600 docs, 1 epoch per run: end-to-end check (~30 min, test inference dominates)
MAX_HOURS = 3.8            # no new job starts after this; keeps the whole run near 4.5-5 h of GPU quota

CKPT = {}
for p in sorted(glob.glob("/kaggle/input/**/model_best.pt", recursive=True)):
    arch = torch.load(p, map_location="cpu", weights_only=False)["model_config"]["arch"]
    CKPT.setdefault(arch, p)
toks = sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
TOKENIZER = toks[0] if toks else None
print("checkpoints:", CKPT)
print("tokenizer:", TOKENIZER)
assert "unilm" in CKPT and TOKENIZER, "attach the Phase 1 unilm output (model_best.pt + data/tokenizer.json)"
missing = [a for a in ("gpt", "unilm", "encdec", "retnet") if a not in CKPT]
print("missing archs (the course wants all four in this phase):", missing or "none")
'''

PLAN = r'''
# ---- experiment plan: (name, arch, extra args, runs after) ----------------------------
# Every run trains on train only and keeps the epoch with the best Macro-F1 on 2,500 dev docs.
# Priority order: the first 7 jobs cover the course rules for this phase (all four archs; full FT,
# LoRA and LORMA; distillation). If time runs short, the last jobs are the ones skipped.
N = 600 if QUICK else DOCS_PER_EPOCH
E, EP = (1, 1) if QUICK else (2, 3)
COMMON = f"--docs_per_epoch {N}" + (" --dev_select_docs 300" if QUICK else "")
PLAN = [   # Phase 4 lesson (test scores): full FT and the 6-layer KD student beat PEFT, so they go first
    ("unilm_full",     "unilm",  f"--method full --epochs {E} --save_train_logits {2 * N}", []),   # also the KD teacher
    ("unilm_lorma",    "unilm",  f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("retnet_lorma",   "retnet", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("gpt_full",       "gpt",    f"--method full --epochs {E}", []),
    ("kd_unilm6",      "unilm",  f"--method full --keep_layers 6 --epochs {E} --teacher {OUT}/unilm_full", ["unilm_full"]),
    ("unilm_lora",     "unilm",  f"--method lora --rank 16 --alpha 32 --epochs {EP}", []),
    ("encdec_lorma",   "encdec", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("gpt_lorma",      "gpt",    f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("unilm_lorma_pi", "unilm",  f"--method lorma_pi --rank 16 --epochs {EP}", []),
    ("unilm_head",     "unilm",  f"--method head --epochs 20 --batch_docs 64 --accum 1", []),
]
PLAN = [p for p in PLAN if p[1] in CKPT]
for p in PLAN: print(p[0], p[1], p[2])
'''

DATA = r'''
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
tokn = UserSecretsClient().get_secret("HF").strip().strip("<>")
root = snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                         allow_patterns=["Phase_5_LSI/*"], token=tokn, local_dir=f"{WORK}/hf")
DATA = f"{root}/Phase_5_LSI"
print(sorted(glob.glob(f"{DATA}/*/*")))
# tokenise once (~66k docs; about 3 min)
import subprocess
subprocess.run(["python", "p5_lsi.py", "prep", "--data", DATA, "--tokenizer", TOKENIZER, "--cache", CACHE], check=True)
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
        if time.time() - t_start > MAX_HOURS * 3600:
            print("time budget reached, skipping:", [p[0] for p in pending]); pending.clear(); break
        pending.remove(job)
        name, arch, extra, _ = job
        cmd = (f"python p5_lsi.py train --data {DATA} --cache {CACHE} --out {OUT}/{name} "
               f"--ckpt {CKPT[arch]} --code_dir . {COMMON} {extra}")
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
    for p in list(pending):
        if any(d in failed for d in p[3]):
            pending.remove(p); failed.add(p[0]); print("skipped", p[0], "(dependency failed)")
    time.sleep(15)
print("done:", sorted(done), "failed:", sorted(failed))
'''

ENSEMBLE = r'''
# ---- ensembles (mean of per-document probabilities; thresholds re-tuned on dev) -------
import subprocess, sys
sys.path.insert(0, ".")
import p5_lsi as P5
def ens(name, runs):
    runs = [f"{OUT}/{r}" for r in runs if os.path.exists(f"{OUT}/{r}/test_prob.pt")]
    if len(runs) >= 2:
        r = subprocess.run(["python", "p5_lsi.py", "ensemble", "--data", DATA, "--out", f"{OUT}/{name}", *runs],
                           capture_output=True, text=True)
        print(name, (r.stdout + r.stderr).strip().splitlines()[-1])
# Phase 3 lesson: more members helped most, weak archs (encdec) pulled the mean down. Greedy forward
# selection on dev (10k docs, so only mildly optimistic), scored at the best global threshold.
dvl, _ = P5.dev_labels(DATA)
cands = {os.path.basename(os.path.dirname(f)): torch.load(f) for f in glob.glob(f"{OUT}/*/dev_prob.pt")
         if not os.path.basename(os.path.dirname(f)).startswith("ens_")}
def dev_of(names):
    lp = {k: sum(cands[n][k].float() for n in names) / len(names) for k in cands[names[0]]}
    return P5.quick_dev_f1(dvl, lp)
ranked = sorted(cands, key=lambda n: -dev_of([n]))
print("single-run dev (global thr):", {n: round(dev_of([n]), 4) for n in ranked})
sel = ranked[:1]
while len(sel) < len(ranked):
    best = max((dev_of(sel + [n]), n) for n in ranked if n not in sel)
    if best[0] < dev_of(sel) - 0.002:   # keep adding while dev does not drop
        break
    sel.append(best[1])
print("greedy selection:", sel, round(dev_of(sel), 4), "(a single run: use its own zip)" if len(sel) == 1 else "")
ens("ens_greedy", sel)
ens("ens_noencdec", [n for n in ranked if not n.startswith("encdec") and n != "unilm_head"])
ens("ens_all", ranked)
ens("ens_top3", ranked[:3])
ens("ens_archs", [n for n in ["unilm_lorma", "gpt_lorma", "retnet_lorma", "encdec_lorma"] if n in cands])
'''

TABLE = r'''
# ---- summary: dev Macro-F1 per run --------------------------------------------------
import json, pandas as pd
rows = []
for d in sorted(glob.glob(f"{OUT}/*/run.json")):
    r = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
    rows.append({"run": name, "arch": r.get("arch"), "method": r.get("peft", {}).get("method"),
                 "trainable_M": round(r.get("trainable", 0) / 1e6, 2),
                 "dev_honest": r.get("dev_f1_honest"), "dev_tuned": r.get("dev_f1_tuned"),
                 "dev@0.5": r.get("dev_f1_at_0.5"), "rule": r.get("rule"), "micro": r.get("dev_micro_tuned"),
                 "labels/doc": r.get("dev_labels_per_doc_pred"),
                 "epoch_min": round(((r.get("history") or [{}])[-1].get("sec") or 0) / 60, 1),
                 "zip": os.path.exists(f"{OUT}/{name}/submission.zip")})
df = pd.DataFrame(rows).sort_values("dev_honest", ascending=False, na_position="last")
print(df.to_string(index=False))
df.to_csv(f"{OUT}/summary.csv", index=False)
import zipfile
with zipfile.ZipFile(f"{WORK}/p5_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f, os.path.relpath(f, WORK))
    for f in glob.glob(f"{OUT}/*/*_prob.pt"):   # small (fp16), lets ensembles be rebuilt on CPU later
        z.write(f, os.path.relpath(f, WORK))
print("p5_results.zip", os.path.getsize(f"{WORK}/p5_results.zip"), "bytes")
'''

cells = [
    md(INTRO),
    code(CONFIG),
    code("!pip -q install tokenizers huggingface_hub"),
    writefile(os.path.join(HERE, "..", "phase1", "cs781_p1.py"), "cs781_p1.py"),
    writefile(os.path.join(HERE, "ft_core.py"), "ft_core.py"),
    writefile(os.path.join(HERE, "p5_lsi.py"), "p5_lsi.py"),
    code(PLAN),
    code(DATA),
    md("Runs the plan (estimated 4 to 5 h on T4 x2 with all four archs attached; not measured yet). "
       "No new job starts after MAX_HOURS. Per-run logs: `/kaggle/working/p5/<run>/stdout.log`."),
    code(RUN),
    code(ENSEMBLE),
    code(TABLE),
    md("Download `p5_results.zip` (all submission zips, logs, run.json files and the dev/test probabilities) "
       "or single zips from `/kaggle/working/p5/<run>/submission.zip`, and upload to Codabench Phase 5.\n\n"
       "## LLM usage declaration\nLLMs were used for code generation; no external LLM API is called in this pipeline."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(OUT, os.path.getsize(OUT), "bytes")
