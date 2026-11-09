#!/usr/bin/env python3
"""Builds the Phase 5 round-2 Kaggle notebook: the evidence-guided context-selection ablation.

  python build_p5r2_notebook.py [OUT.ipynb]
It reuses build_p5_notebook.py (same cells and code) with a different intro, plan, prep and
ensemble cell. Round 2 trains the same recipes as two main-run jobs with `--select cue`
(first chunk + the 3 chunks with the most <SECTION>/<ACT> markers) instead of the fixed first-3 +
last-1 window, so each pair differs only in which part of the document the model reads. If the
main Phase 5 notebook's output is attached as an input, its runs are copied in and combined.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_p5", "CS781-CP-Phase5-round2-Darsh-Kedia-250311.ipynb")
src = open(os.path.join(HERE, "build_p5_notebook.py"), encoding="utf-8").read()


def swap(start, end, new):
    global src
    a = src.index(start)
    b = src.index(end, a)
    src = src[:a] + new + src[b:]


swap('INTRO = """', '\nCONFIG = ', '''INTRO = """
# CS781 Phase 5, round 2: evidence-guided context selection (Darsh Kedia, 250311)

**Ablation.** Statute names are masked in the text, but the masks stay: every place a statute is cited
reads `<SECTION>` or `<ACT>`. For the 30% of documents longer than 2,048 tokens, the main notebook's fixed
window (first 3 x 512 + last 512 tokens) contains only 60% of these markers. Round 2 keeps the first chunk
and then the 3 chunks with the most markers (`--select cue`), which raises that to 83% on dev. Same
4 x 512-token budget, same model, same training recipe; only the choice of context differs.
Pairs: `unilm_full_cue` vs `unilm_full`, `unilm_lorma_cue` vs `unilm_lorma` (main notebook).

**Before running (Kaggle):** GPU T4 x2, Internet on, **Add-ons > Secrets: attach `HF`**, and **Add Input**
the unilm 400M Phase 1 output (`notebookb7c597f318`, holds model_best.pt; the tokenizer comes from the gpt
output `notebook35d98d4ec2`, attach it too). Optional but recommended: also **Add Input** the finished
main Phase 5 notebook (Your Work), so its runs are copied in, ensembled with round 2, and the label-structure
post-processing (`p5_post.py`) is run on the best ensemble. About 1.5 h.
"""
''')
src = src.replace('OUT = f"{WORK}/p5"', 'OUT = f"{WORK}/p5r2"').replace('CACHE = f"{WORK}/p5_cache.pt"', 'CACHE = f"{WORK}/p5_cache_all.pt"')
swap('PLAN = [   #', 'PLAN = [p for p in PLAN', '''PLAN = [   # same recipes as the main notebook, plus --select cue
    ("unilm_full_cue",  "unilm", f"--method full --epochs {E} --select cue", []),
    ("unilm_lorma_cue", "unilm", f"--method lorma_plus --rank 16 --alpha 32 --epochs {EP} --select cue", []),
]
''')
src = src.replace('"--tokenizer", TOKENIZER, "--cache", CACHE], check=True)',
                  '"--tokenizer", TOKENIZER, "--cache", CACHE, "--keep_all", "16384"], check=True)')
swap("ENSEMBLE = r'''", "\nTABLE = ", """ENSEMBLE = r'''
# ---- bring in the main run (if attached), ensembles, label-structure post-processing ----
import shutil, subprocess, sys
sys.path.insert(0, ".")
import p5_lsi as P5
for f in glob.glob("/kaggle/input/**/p5/*/dev_prob.pt", recursive=True):
    d = os.path.dirname(f); name = os.path.basename(d)
    if not os.path.exists(f"{OUT}/{name}") and os.path.exists(f"{d}/test_prob.pt"):
        shutil.copytree(d, f"{OUT}/{name}", ignore=shutil.ignore_patterns("*.zip", "train_logits.pt", "model.pt"))
        print("main run copied:", name)
def ens(name, runs):
    runs = [f"{OUT}/{r}" for r in runs if os.path.exists(f"{OUT}/{r}/test_prob.pt")]
    if len(runs) >= 2:
        r = subprocess.run(["python", "p5_lsi.py", "ensemble", "--data", DATA, "--out", f"{OUT}/{name}", *runs],
                           capture_output=True, text=True)
        print(name, (r.stdout + r.stderr).strip().splitlines()[-1])
dvl, _ = P5.dev_labels(DATA)
cands = {os.path.basename(os.path.dirname(f)): torch.load(f) for f in glob.glob(f"{OUT}/*/dev_prob.pt")
         if not os.path.basename(os.path.dirname(f)).startswith(("ens_", "post"))}
def dev_of(names):
    lp = {k: sum(cands[n][k].float() for n in names) / len(names) for k in cands[names[0]]}
    return P5.quick_dev_f1(dvl, lp)
ranked = sorted(cands, key=lambda n: -dev_of([n]))
print("single-run dev (global thr):", {n: round(dev_of([n]), 4) for n in ranked})
for a, b in [("unilm_full", "unilm_full_cue"), ("unilm_lorma", "unilm_lorma_cue")]:
    if a in cands and b in cands:
        print(f"ABLATION {b} vs {a}: {dev_of([b]):.4f} vs {dev_of([a]):.4f}  (pair ensemble {dev_of([a, b]):.4f})")
sel = ranked[:1]
while len(sel) < len(ranked):
    best = max((dev_of(sel + [n]), n) for n in ranked if n not in sel)
    if best[0] < dev_of(sel) - 0.002:
        break
    sel.append(best[1])
print("greedy selection:", sel, round(dev_of(sel), 4))
ens("ens_greedy2", sel)
ens("ens_cue_pairs", [n for n in ["unilm_full", "unilm_full_cue", "unilm_lorma", "unilm_lorma_cue"] if n in cands])
ens("ens_all2", [n for n in ranked if not n.startswith("encdec") and n != "unilm_head"])
best_ens = f"{OUT}/ens_greedy2" if os.path.exists(f"{OUT}/ens_greedy2/test_prob.pt") else f"{OUT}/{ranked[0]}"
r = subprocess.run(["python", "p5_post.py", "--data", DATA, "--out", f"{OUT}/post", "--variants", "base,cooc,resid", best_ens],
                   capture_output=True, text=True)
print("post-processing on", os.path.basename(best_ens)); print("\\n".join(l for l in (r.stdout + r.stderr).splitlines() if l.split(" ")[0] in ("base", "cooc", "resid") or "Error" in l))
'''
""")
src = src.replace('writefile(os.path.join(HERE, "p5_lsi.py"), "p5_lsi.py"),',
                  'writefile(os.path.join(HERE, "p5_lsi.py"), "p5_lsi.py"),\n    writefile(os.path.join(HERE, "p5_post.py"), "p5_post.py"),')
src = src.replace('"Runs the plan (estimated 4 to 5 h on T4 x2 with all four archs attached; not measured yet). "',
                  '"Runs the two round-2 jobs in parallel (estimated 70-80 min on T4 x2; not measured). "')
src = src.replace('/kaggle/working/p5/<run>', '/kaggle/working/p5r2/<run>').replace('p5_results.zip', 'p5r2_results.zip')
src = src.replace('OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kaggle_p5", "CS781-CP-Phase5-Darsh-Kedia-250311.ipynb")',
                  f'OUT = {OUT!r}')
assert src.count("--select cue") >= 2 and "p5_post.py" in src
exec(compile(src, "build_p5_notebook.py(round2)", "exec"), {"__file__": os.path.join(HERE, "build_p5_notebook.py"), "__name__": "__main__"})
