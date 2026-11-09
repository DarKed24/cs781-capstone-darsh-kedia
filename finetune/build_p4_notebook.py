#!/usr/bin/env python3
"""Builds the self-contained Kaggle notebook for Phase 4 (CJPE) from the .py files in this folder.

  python build_p4_notebook.py [OUT.ipynb] [--round2]
Rebuild after editing ft_core.py, p4_cjpe.py or ../phase1/cs781_p1.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND = 2 if "--round2" in sys.argv else 1
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = _args[0] if _args else os.path.join(HERE, "kaggle_p4", "CS781-CP-Phase4-Darsh-Kedia-250311.ipynb" if ROUND == 1
                                          else "CS781-CP-Phase4-round2-Darsh-Kedia-250311.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def writefile(path, name):
    return code(f"%%writefile {name}\n" + open(path, encoding="utf-8").read())


INTRO = """
# CS781 Phase 4: Court Judgment Prediction (CJPE) from the Phase 1 models (Darsh Kedia, 250311)

**Before running (Kaggle):** Accelerator = GPU T4 x2, Internet on, **Add-ons > Secrets: attach `HF`**
(Hugging Face token), and **Add Input** the Phase 1 notebook outputs that hold the checkpoints:
the unilm 400M run (`cs781_phase1/unilm/model_best.pt`), the gpt run (`cs781_phase1/gpt/model_best.pt`),
and encdec / retnet when they exist (attach only ONE output per architecture). One of them must also
hold `cs781_phase1/data/tokenizer.json`. Paths and architectures are found automatically.

What it does: tokenises the Phase 4 data once, then runs the experiment plan two jobs at a time (one per
T4). Each job writes a Codabench-ready `submission.zip` under `/kaggle/working/p4/<run>/`; ensembles are
added and a table of dev Macro-F1 per run is printed. Upload the zips to Codabench Phase 4, best dev first.

Long documents: each judgment is read as its last 4 x 512 tokens (chunks aligned to the end of the
document, where the reasoning before the removed decision sits); each chunk is encoded by the Phase 1
backbone and mean-pooled, and a BiGRU with attention pooling over the chunks gives the prediction.
The decision threshold is tuned on dev for Macro-F1.

Dev note: the released `Test/test.json` is a subset of `Train/dev.json`. The code removes every test id
from dev (994 unique dev docs remain) and never trains on dev, so dev scores are honest held-out scores.

Methods (see `ft_core.py`): full fine-tuning, LoRA, LORMA (`lorma_plus`, `lorma_pi`), a frozen-backbone
probe, and knowledge distillation into a 6-layer student initialised from the teacher's layers.
"""

CONFIG = r'''
# ---- configuration ----------------------------------------------------------------
import glob, os, torch
WORK = "/kaggle/working"
OUT = f"{WORK}/p4"
CACHE = f"{WORK}/p4_cache.pt"
DOCS_PER_EPOCH = 12000     # fresh random sample of the ~31.7k train docs each epoch
QUICK = False              # True = 600 docs, 1 epoch per run: end-to-end check (~15-20 min)
MAX_HOURS = 9.0           # no new job starts after this (Kaggle stops a commit at 12 h and the outputs would be lost)

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
ROUND = %d
# ---- experiment plan: (name, arch, extra args, runs after) ----------------------------
# Every run trains on train only and keeps the epoch with the best dev_clean Macro-F1.
N = 600 if QUICK else DOCS_PER_EPOCH
E, EP = (1, 1) if QUICK else (2, 3)
COMMON = f"--docs_per_epoch {N}"
PLAN = [   # ordered by priority: if time runs short, the last jobs are the ones skipped
    ("unilm_lorma",    "unilm",  f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("unilm_full",     "unilm",  f"--method full --epochs {E} --save_train_logp {N}", []),   # also the KD teacher
    ("retnet_lorma",   "retnet", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("unilm_lora",     "unilm",  f"--method lora --rank 16 --alpha 32 --epochs {EP}", []),
    ("gpt_lorma",      "gpt",    f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("kd_unilm6",      "unilm",  f"--method full --keep_layers 6 --epochs {E} --teacher {OUT}/unilm_full", ["unilm_full"]),
    ("unilm_lorma_pi", "unilm",  f"--method lorma_pi --rank 16 --epochs {EP}", []),
    ("gpt_full",       "gpt",    f"--method full --epochs {E}", []),
    ("encdec_lorma",   "encdec", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP}", []),
    ("unilm_head",     "unilm",  f"--method head --epochs 30 --batch_docs 32 --accum 1", []),
]
if ROUND == 2:   # round 1 on Codabench: the 6-layer distilled student was best (0.7539), ensembles did not beat it
    t1 = sorted(glob.glob("/kaggle/input/**/p4/unilm_full/train_logp.pt", recursive=True))
    assert t1, "round 2: attach the round-1 Phase 4 notebook output (needs p4/unilm_full/train_logp.pt)"
    R1T = os.path.dirname(t1[0])          # round-1 unilm_full = teacher
    PLAN = [
        ("unilm_full_s2",  "unilm",  f"--method full --epochs {E} --seed 7 --save_train_logp {N}", []),
        ("kd_unilm6_s2",   "unilm",  f"--method full --keep_layers 6 --epochs {E} --seed 7 --teacher {R1T}", []),
        ("kd_unilm6_s3",   "unilm",  f"--method full --keep_layers 6 --epochs {E} --seed 11 --teacher {R1T}", []),
        ("kd_gpt6",        "gpt",    f"--method full --keep_layers 6 --epochs {E} --seed 7 --teacher {R1T}", []),
        ("kd_retnet6",     "retnet", f"--method full --keep_layers 6 --epochs {E} --seed 7 --teacher {R1T}", []),
        ("kd_unilm6_2t",   "unilm",  f"--method full --keep_layers 6 --epochs {E} --seed 13 --teacher {R1T} --teacher {OUT}/unilm_full_s2", ["unilm_full_s2"]),
        ("kd_unilm6_3ep",  "unilm",  f"--method full --keep_layers 6 --epochs 3 --seed 17 --teacher {R1T}", []),
        ("retnet_full",    "retnet", f"--method full --epochs {E}", []),
    ]
PLAN = [p for p in PLAN if p[1] in CKPT]
for p in PLAN: print(p[0], p[1], p[2])
'''

DATA = r'''
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
tokn = UserSecretsClient().get_secret("HF").strip().strip("<>")
root = snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                         allow_patterns=["Phase_4_CJPE/*"], token=tokn, local_dir=f"{WORK}/hf")
DATA = f"{root}/Phase_4_CJPE"
print(sorted(glob.glob(f"{DATA}/*/*")))
# tokenise once (train ~31.7k docs after de-duplication; a few minutes on Kaggle's CPUs)
import subprocess
subprocess.run(["python", "p4_cjpe.py", "prep", "--data", DATA, "--tokenizer", TOKENIZER, "--cache", CACHE], check=True)
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
        cmd = (f"python p4_cjpe.py train --data {DATA} --cache {CACHE} --out {OUT}/{name} "
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
# ---- ensembles (average of per-document log-probs; threshold re-tuned on dev) ---------
import subprocess
def ens(name, runs):
    runs = [DIRS.get(r, f"{OUT}/{r}") for r in runs]
    runs = [r for r in runs if os.path.exists(f"{r}/test_logp.pt")]
    if len(runs) >= 2:
        r = subprocess.run(["python", "p4_cjpe.py", "ensemble", "--data", DATA, "--out", f"{OUT}/{name}", *runs],
                           capture_output=True, text=True)
        print(name, (r.stdout + r.stderr).strip().splitlines()[-1])
# Phase 3 lesson: more members helped most, weak archs (encdec) pulled the mean down. With 994 dev docs,
# a greedy forward selection on dev is a reasonable guide (slightly optimistic: it is chosen on dev).
import sys, torch
sys.path.insert(0, ".")
import p4_cjpe as P4
dvl, _ = P4.dev_labels(DATA)
# runs of this notebook, plus (round 2) the runs of an attached earlier Phase 4 output, prefixed r1_
DIRS = {}
for f in sorted(glob.glob("/kaggle/input/**/p4/*/dev_logp.pt", recursive=True)):
    d = os.path.dirname(f)
    if not os.path.basename(d).startswith("ens_") and os.path.exists(f"{d}/test_logp.pt"):
        DIRS.setdefault("r1_" + os.path.basename(d), d)
for f in glob.glob(f"{OUT}/*/dev_logp.pt"):
    d = os.path.dirname(f)
    if not os.path.basename(d).startswith("ens_"):
        DIRS[os.path.basename(d)] = d
cands = {n: torch.load(f"{d}/dev_logp.pt") for n, d in DIRS.items()}
def dev_of(names):
    lp = {k: sum(cands[n][k] for n in names) / len(names) for k in cands[names[0]]}
    return P4.tune_thr(dvl, lp)[1]
ranked = sorted(cands, key=lambda n: -dev_of([n]))
print("single-run dev (tuned):", {n: round(dev_of([n]), 4) for n in ranked})
sel = ranked[:1]
for _ in range(len(ranked)):
    best = max((dev_of(sel + [n]), n) for n in ranked if n not in sel) if len(sel) < len(ranked) else (0, None)
    if best[1] is None or best[0] < dev_of(sel) - 0.002:   # keep adding while dev does not drop
        break
    sel.append(best[1])
print("greedy selection:", sel, round(dev_of(sel), 4))
ens("ens_greedy", sel)
base = lambda n: n[3:] if n.startswith("r1_") else n
ens("ens_noencdec", [n for n in ranked if not base(n).startswith("encdec")])
ens("ens_all", ranked)
ens("ens_unilm", [n for n in ranked if base(n).startswith(("unilm", "kd_unilm")) and base(n) != "unilm_head"])
ens("ens_kd", [n for n in ranked if base(n).startswith("kd_")])
ens("ens_top6", ranked[:6])
ens("ens_top10", ranked[:10])
ens("ens_archs", [n for n in ["unilm_lorma", "gpt_lorma", "retnet_lorma", "encdec_lorma"] if n in cands])
'''

TABLE = r'''
# ---- summary: dev_clean Macro-F1 per run --------------------------------------------
import json, pandas as pd
rows = []
for d in sorted(glob.glob(f"{OUT}/*/run.json")):
    r = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
    rows.append({"run": name, "arch": r.get("arch"), "method": r.get("peft", {}).get("method"),
                 "trainable_M": round(r.get("trainable", 0) / 1e6, 2),
                 "dev_f1@0": r.get("dev_f1"), "dev_f1_tuned": r.get("dev_f1_tuned"), "thr": r.get("threshold"),
                 "epoch_min": round(((r.get("history") or [{}])[-1].get("sec") or 0) / 60, 1),
                 "zip": os.path.exists(f"{OUT}/{name}/submission.zip")})
df = pd.DataFrame(rows).sort_values("dev_f1_tuned", ascending=False, na_position="last")
print(df.to_string(index=False))
df.to_csv(f"{OUT}/summary.csv", index=False)
import zipfile
with zipfile.ZipFile(f"{WORK}/p4_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f, os.path.relpath(f, WORK))
print("p4_results.zip", os.path.getsize(f"{WORK}/p4_results.zip"), "bytes")
'''

ROUND2_NOTE = """
## Round 2
Round 1 on Codabench: the 6-layer distilled unilm student scored best (0.7539), above its teacher and every
ensemble. Round 2 therefore trains more distilled students from the round-1 teacher (new seeds, gpt and retnet
students, a two-teacher student, a 3-epoch student), a second unilm teacher, and retnet full fine-tuning;
`ens_kd` averages all students.
**Also attach the round-1 Phase 4 notebook output** (Add Input > Your Work): its `p4/*/dev_logp.pt` and
`test_logp.pt` files are picked up as `r1_*` members, so the ensembles combine both rounds.
"""

cells = [
    md(INTRO + (ROUND2_NOTE if ROUND == 2 else "")),
    code(CONFIG),
    code("!pip -q install tokenizers huggingface_hub"),
    writefile(os.path.join(HERE, "..", "phase1", "cs781_p1.py"), "cs781_p1.py"),
    writefile(os.path.join(HERE, "ft_core.py"), "ft_core.py"),
    writefile(os.path.join(HERE, "p4_cjpe.py"), "p4_cjpe.py"),
    code(PLAN % ROUND),
    code(DATA),
    md("Runs the plan (estimated 4 to 6 h on T4 x2 with all four archs attached; not measured yet). "
       "No new job starts after MAX_HOURS. Per-run logs: `/kaggle/working/p4/<run>/stdout.log`."),
    code(RUN),
    code(ENSEMBLE),
    code(TABLE),
    md("Download `p4_results.zip` (all submission zips, logs and run.json files) or single zips from "
       "`/kaggle/working/p4/<run>/submission.zip`, and upload to Codabench Phase 4.\n\n"
       "## LLM usage declaration\nLLMs were used for code generation; no external LLM API is called in this pipeline."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(OUT, os.path.getsize(OUT), "bytes")
