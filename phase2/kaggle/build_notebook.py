import json, sys
import os; P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}
def wf(name): return code(f"%%writefile {name}\n" + open(P + name).read())

cells = [
md("""
# CS781 Capstone, Phase 2: Legal Named Entity Recognition
**Darsh Kedia, 250311.** Codabench username `d_250311`.

Task: tag 12 entity types (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO) with character offsets in Indian court judgments. Metric: strict Macro-F1 over all 12 labels.

Pipeline: byte-level BPE tokenizer with character offsets, encoder, linear BIO head, sliding windows with averaged log-probs, BIO-constrained Viterbi decoding, spans, `prediction.json`, `submission.zip`.

Everything is trained from random initialisation on the official data only. No pretrained weights, no external corpora, no LLM APIs. Once a Phase 1 checkpoint exists, set `PHASE1_TOKENIZER` / `PHASE1_CKPT` below and use the hook in section 7.

Run on Kaggle with a GPU (T4 or P100). Needs the Kaggle secret `HF` holding a Hugging Face token that has accepted the dataset's terms.
"""),
code("""
# ---- Config ----
SEED = 42
PHASE1_TOKENIZER = None   # e.g. "/kaggle/input/datasets/darshkedia/cs781-phase1/tokenizer.json"
PHASE1_CKPT = None        # e.g. "/kaggle/input/datasets/darshkedia/cs781-phase1/model_final.pt"\nPHASE1_CODE_DIR = None    # folder holding cs781_p1.py from the Phase 1 notebook
ENCODER = "bilstm"        # "bilstm" (fold2 0.386 on CPU) or "transformer" (0.117: too little data from scratch)
DEV_EPOCHS = 30
RUN_NAME = f"{ENCODER}_scratch"
WORK = "/kaggle/working"
"""),
code("!pip -q install tokenizers huggingface_hub"),
md("## 1. Helper modules\n`lner_utils.py`: loading, span/BIO conversion, windows, strict scorer (matches the official `evaluate_phase2.py`), validator, zipper. `lner_model.py`: tokenizer, encoders, tagger, training and inference. `baseline_rules.py`: non-neural gazetteer + regex baseline."),
wf("lner_utils.py"), wf("lner_model.py"), wf("baseline_rules.py"),
md("## 2. Data"),
code("""
import os, json, sys
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
HF_TOKEN = UserSecretsClient().get_secret("HF")
root = snapshot_download("Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                         allow_patterns=["Phase_2_LNER/*"], token=HF_TOKEN, local_dir=f"{WORK}/hf")
DATA = f"{WORK}/p2data"
os.makedirs(DATA, exist_ok=True)
for sub in ("Train", "Test"):
    d = os.path.join(root, "Phase_2_LNER", sub)
    for f in os.listdir(d):
        os.replace(os.path.join(d, f), os.path.join(DATA, f))
print(sorted(os.listdir(DATA)))
"""),
code("""
import importlib, lner_utils as U, lner_model as M, baseline_rules as R
from collections import Counter
t1, g1 = U.load_texts(f"{DATA}/fold1-text.json"), U.load_spans(f"{DATA}/fold1-gold.json")
t2, g2 = U.load_texts(f"{DATA}/fold2-text.json"), U.load_spans(f"{DATA}/fold2-gold.json")
t3 = U.load_texts(f"{DATA}/fold3-text.json")
print("labels.txt:", U.load_labels(f"{DATA}/labels.txt"))
for name, t, g in [("fold1", t1, g1), ("fold2", t2, g2), ("fold3", t3, None)]:
    chars = sum(map(len, t.values()))
    print(name, "docs", len(t), "chars", chars, "spans", sum(map(len, g.values())) if g else "-")
print("label counts fold1+2:", Counter(l for g in (g1, g2) for v in g.values() for *_, l in v).most_common())
"""),
md("## 3. Tokenizer\nThe Phase 1 tokenizer if available, otherwise a stand-in byte-level BPE with the same settings trained on fold1+fold2 text."),
code("""
TOK = PHASE1_TOKENIZER or M.train_bpe(list(t1.values()) + list(t2.values()), f"{WORK}/tokenizer_standin.json", 32000)
tok = M.BPETokenizer(TOK)
# Round trip: gold -> token BIO -> spans. This is the ceiling for any tagger on this tokenization.
for name, t, g in [("fold1", t1, g1), ("fold2", t2, g2)]:
    enc, (mis, tot) = M.encode_docs(tok, t, g)
    back = {d: M.decode_tags(o, [x if x != M.IGNORE else 0 for x in tg]) for d, (_, o, tg) in enc.items()}
    print(name, f"spans not on token boundaries {mis}/{tot}", "round-trip macro F1 %.4f" % U.strict_scores(g, back)[0]["F1"])
"""),
md("## 4. Non-neural baseline (gazetteer + regex)\nA quick, always-valid submission and a reference point. Not a Phase 1 experiment."),
code("""
pred = R.predict(t2, R.build_gazetteer(t1, g1))
U._print_scores(*U.strict_scores(g2, pred))
pred3 = R.predict(t3, R.build_gazetteer({**t1, **t2}, {**g1, **g2}))
os.makedirs(f"{WORK}/rules", exist_ok=True)
U.make_zip(U.write_predictions(pred3, f"{WORK}/rules/prediction.json", texts=t3), f"{WORK}/submission_rules.zip")
"""),
md("## 5. Neural tagger: train on fold1, select epoch on fold2"),
code("""
import copy, torch
cfg = copy.deepcopy(M.DEFAULT_CFG)
cfg["encoder"] = ENCODER
if ENCODER == "bilstm":
    cfg["encoder_args"] = dict(M.BILSTM_ARGS); cfg["lr"] = 2e-3
cfg["epochs"] = DEV_EPOCHS
model, hist = M.train(cfg, tok, t1, g1, t2, g2)
best_ep = max(hist, key=lambda r: r["dev_f1"])["epoch"]
print("best epoch", best_ep, "dev F1", max(r["dev_f1"] for r in hist))
dev_pred = M.predict_docs(model, M.encode_docs(tok, t2)[0], cfg["max_len"], cfg["stride"], next(model.parameters()).device)
U._print_scores(*U.strict_scores(g2, dev_pred))
"""),
md("## 6. Final model on fold1+fold2, predict fold3, validate, zip"),
code("""
cfg_full = dict(cfg, epochs=best_ep)
model_full, _ = M.train(cfg_full, tok, {**t1, **t2}, {**g1, **g2})
dev = next(model_full.parameters()).device
pred3 = M.predict_docs(model_full, M.encode_docs(tok, t3)[0], cfg["max_len"], cfg["stride"], dev)
out = f"{WORK}/{RUN_NAME}"; os.makedirs(out, exist_ok=True)
torch.save(model_full.state_dict(), f"{out}/model.pt")
json.dump({"cfg": dict(cfg_full, tokenizer=TOK), "dev_history": hist}, open(f"{out}/run.json", "w"), indent=1)
path = U.write_predictions(pred3, f"{out}/prediction.json", texts=t3)   # raises if invalid
print(U.make_zip(path, f"{WORK}/submission_{RUN_NAME}.zip"))
"""),
md("""
## 7. Hook: fine-tune a Phase 1 model
`M.phase1_encoder` loads any of the four Phase 1 exports with `cs781_p1.load_model` and picks the feature extractor per architecture: `unilm` bidirectional mode, the `encdec` encoder, causal `gpt` and `retnet` hidden states. Set `PHASE1_TOKENIZER` to the Phase 1 tokenizer.json too. Full fine-tuning here; LORMA and the other PEFT and distillation experiments build on the same `M.train(..., model=...)` call.
"""),
code("""
if PHASE1_CKPT:
    # cs781_p1.py comes from the Phase 1 notebook output (attach it as a Kaggle input).
    enc = M.phase1_encoder(PHASE1_CKPT, p1_dir=PHASE1_CODE_DIR)
    print("Phase 1 arch:", enc.arch)   # NER should prefer unilm or encdec (bidirectional)
    p1_model = M.Tagger(enc); p1_model.pad_id = tok.pad_id
    cfg_p1 = dict(cfg, encoder_lr=5e-5, lr=1e-3, epochs=8, batch_size=8)
    p1_model, hist_p1 = M.train(cfg_p1, tok, t1, g1, t2, g2, model=p1_model)
    print("best dev F1", max(r["dev_f1"] for r in hist_p1))
"""),
md("""
## LLM usage declaration
LLMs were used for code generation and analysis, as permitted by the updated course policy. No external LLM API is called anywhere in this pipeline.

_Fill in: tools used, for which parts, and how outputs were checked._
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
