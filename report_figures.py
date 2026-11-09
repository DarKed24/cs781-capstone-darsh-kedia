#!/usr/bin/env python3
"""Builds the data-analysis figure and statistics used in the report."""
import json, sys, os, collections, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
load = lambda p: json.load(open(os.path.join(D, p)))
stats = {}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "figure.dpi": 150})
fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.4))

cnt = collections.Counter()
for f in ["Phase_2_LNER/Train/fold1-gold.json", "Phase_2_LNER/Train/fold2-gold.json"]:
    for spans in load(f).values():
        for s in spans: cnt[s[2]] += 1
labs = [l for l, _ in cnt.most_common()]
ax[0, 0].barh(labs[::-1], [cnt[l] for l in labs[::-1]], color="#4C72B0")
ax[0, 0].set_title("(a) P2: entity spans per label"); stats["p2_counts"] = dict(cnt)

rc = {}
for dom in ["CL", "IT"]:
    c = collections.Counter(l for d in load(f"Phase_3_RR/Train/{dom}_train.json") for l in d["labels"])
    rc[dom] = c
roles = [r for r, _ in (rc["CL"] + rc["IT"]).most_common()]
y = np.arange(len(roles))
for i, dom in enumerate(["CL", "IT"]):
    tot = sum(rc[dom].values())
    ax[0, 1].barh(y + (0.2 if i else -0.2), [100 * rc[dom][r] / tot for r in roles], height=0.4, label=dom)
ax[0, 1].set_yticks(y, roles); ax[0, 1].invert_yaxis(); ax[0, 1].legend(frameon=False)
ax[0, 1].set_title("(b) P3: % of sentences per role"); ax[0, 1].set_xlabel("%")
stats["p3"] = {k: dict(v) for k, v in rc.items()}

tr = load("Phase_5_LSI/Train/train.json")
lc = collections.Counter(l for d in tr for l in d["labels"])
freq = sorted(lc.values(), reverse=True)
ax[0, 2].bar(range(len(freq)), freq, color="#55A868", width=1.0)
ax[0, 2].set_yscale("log"); ax[0, 2].set_xlabel("statute rank"); ax[0, 2].set_title("(c) P5: docs per statute")
nl = [len(d["labels"]) for d in tr]
stats["p5"] = dict(n_labels=len(lc), max=freq[0], min=freq[-1], mean_labels=st.mean(nl), median_labels=st.median(nl),
                   share_1=sum(x == 1 for x in nl) / len(nl), share_ge6=sum(x >= 6 for x in nl) / len(nl))

s6 = load("Phase_6_Summ/Train/train.json")
dw = np.array([len(x["document"].split()) for x in s6]); sw = np.array([len(x["summary"].split()) for x in s6])
ax[1, 0].scatter(dw, sw, s=1.5, alpha=0.3, color="#C44E52", rasterized=True)
ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("document words"); ax[1, 0].set_ylabel("summary words")
ax[1, 0].axhline(400, ls="--", lw=0.8, color="k"); ax[1, 0].text(dw.min() * 1.2, 430, "BERTScore window", fontsize=6.5)
ax[1, 0].set_title("(d) P6: document vs summary length")
ov = []
for x in s6[:2000]:
    dv = set(x["document"].lower().split()); sv = x["summary"].lower().split()
    ov.append(sum(w in dv for w in sv) / max(1, len(sv)))
stats["p6"] = dict(n=len(s6), doc_median=float(np.median(dw)), sum_median=float(np.median(sw)),
                   ratio_median=float(np.median(dw / np.maximum(sw, 1))), unigram_overlap=float(np.mean(ov)))

for split, f, col in [("train", "Phase_7_PCR/Train/train.json", "#8172B2"), ("dev", "Phase_7_PCR/Dev/dev.json", "#CCB974")]:
    g = [len(eval(v["relevant_candidates"])) if isinstance(v["relevant_candidates"], str) else len(v["relevant_candidates"]) for v in load(f).values()]
    ax[1, 1].hist(np.clip(g, 0, 30), bins=range(0, 32), alpha=0.6, label=f"{split} (mean {np.mean(g):.1f})", color=col, density=True)
    stats[f"p7_{split}"] = dict(n=len(g), mean=float(np.mean(g)), median=float(np.median(g)), max=int(max(g)))
ax[1, 1].axvline(7, ls="--", lw=0.8, color="k"); ax[1, 1].legend(frameon=False, fontsize=6.5)
ax[1, 1].set_xlabel("gold cases per query (clip 30)"); ax[1, 1].set_title("(e) P7: gold per query (k = 7)")

meta = load("../phase1/meta.json") if os.path.exists(os.path.join(D, "../phase1/meta.json")) else None
if meta:
    src = list(meta["sources"]); toks = [meta["sources"][s]["train_tokens"] / 1e6 for s in src]
    cpt = [meta["sources"][s]["chars_per_token"] for s in src]
    ax[1, 2].bar(["SC", "HC", "Trib.", "Acts"][:len(src)], toks, color="#64B5CD")
    for i, (t, c) in enumerate(zip(toks, cpt)): ax[1, 2].text(i, t + 3, f"{c:.2f}", ha="center", fontsize=6.5)
    ax[1, 2].set_ylabel("M tokens"); ax[1, 2].set_ylim(0, 170); ax[1, 2].set_title("(f) P1: tokens per source\n(label: chars/token)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "data_analysis.pdf"))
json.dump(stats, open(os.path.join(OUT, "data_stats.json"), "w"), indent=1, default=float)
print(json.dumps(stats, indent=1, default=float))
