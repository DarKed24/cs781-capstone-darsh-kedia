"""Phase 2 (L-NER) utilities: data loading, char-offset <-> token BIO alignment,
strict span scoring, prediction.json validation and Codabench zip packaging.

Pure Python (no torch / transformers needed) so it works with any tokenizer:
a tokenizer only has to give (start_char, end_char) offsets for each token.

CLI:
  python lner_utils.py roundtrip TEXT.json GOLD.json   # ceiling F1 of word-level BIO alignment
  python lner_utils.py score GOLD.json PRED.json        # strict macro P/R/F1 (local held-out fold)
  python lner_utils.py validate TEST_TEXT.json PRED.json
  python lner_utils.py zip PRED.json [OUT.zip]          # prediction.json at zip root
"""

import json
import re
import sys
import zipfile
from collections import defaultdict

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
          "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_texts(path):
    """{doc_id: text}. Also accepts a list of {"id", "text"} records."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {str(d["id"]): d["text"] for d in data}
    return {str(k): v for k, v in data.items()}


def _norm_span(e):
    """Accept [s, e, label(, text)] or {"start","end","label"} style entries."""
    if isinstance(e, dict):
        s = e.get("start", e.get("start_char", e.get("begin")))
        t = e.get("end", e.get("end_char"))
        lab = e.get("label", e.get("type", e.get("entity")))
        return int(s), int(t), str(lab)
    return int(e[0]), int(e[1]), str(e[2])


def load_spans(path):
    """{doc_id: [(start, end, label), ...]} from a gold or prediction file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):  # list of {"id", "entities"/"spans"/"pred"}
        data = {str(d["id"]): d.get("entities", d.get("spans", d.get("pred", []))) for d in data}
    return {str(k): [_norm_span(e) for e in v] for k, v in data.items()}


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ----------------------------------------------------------------------------
# Pre-tokenization with character offsets
# ----------------------------------------------------------------------------

# Words, numbers and single punctuation marks each become a token. Splitting
# punctuation off matters: gold spans like "Section 302" or "(2003) 5 SCC 123"
# often end right before a comma/full stop.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def word_offsets(text):
    """[(start, end), ...] for words/punctuation in text (whitespace dropped)."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def subword_offsets(text, encode_word):
    """Offsets for a subword tokenizer that works word by word.

    encode_word(word) -> list of token strings whose concatenated *character*
    lengths equal len(word) (true for most BPE/WordPiece once markers such as
    '##' or '</w>' are stripped). Returns [(tok, start, end), ...].
    """
    out = []
    for s, e in word_offsets(text):
        pos = s
        for tok in encode_word(text[s:e]):
            ln = len(tok)
            out.append((tok, pos, pos + ln))
            pos += ln
        if pos != e:
            raise ValueError(f"subword lengths do not cover word {text[s:e]!r}")
    return out


# ----------------------------------------------------------------------------
# Spans <-> BIO tags
# ----------------------------------------------------------------------------

def spans_to_bio(offsets, spans):
    """Tag each token B-X / I-X / O. A token belongs to a span if it overlaps it.

    Returns (tags, misaligned) where misaligned lists spans whose edges do not
    fall on token boundaries; those can never be predicted exactly from tags.
    """
    tags = ["O"] * len(offsets)
    misaligned = []
    starts = {s for s, _ in offsets}
    ends = {e for _, e in offsets}
    for s, e, lab in sorted(spans):
        if s not in starts or e not in ends:
            misaligned.append((s, e, lab))
        first = True
        for i, (ts, te) in enumerate(offsets):
            if te <= s or ts >= e:
                continue
            if tags[i] != "O":  # overlapping gold spans: keep the earlier one
                continue
            tags[i] = ("B-" if first else "I-") + lab
            first = False
    return tags, misaligned


def bio_to_spans(offsets, tags):
    """Decode BIO tags to [(start, end, label)]. A stray I-X starts a new span."""
    spans = []
    cur = None
    for (ts, te), tag in zip(offsets, tags):
        if tag == "O":
            if cur:
                spans.append(tuple(cur)); cur = None
            continue
        pre, lab = tag.split("-", 1)
        if pre == "B" or cur is None or cur[2] != lab:
            if cur:
                spans.append(tuple(cur))
            cur = [ts, te, lab]
        else:
            cur[1] = te
    if cur:
        spans.append(tuple(cur))
    return spans


def windows(n_tokens, max_len, stride):
    """Sliding windows [(lo, hi)] covering n_tokens; overlap = max_len - stride.
    When predicting, keep each token's label from the window where it is
    furthest from an edge (or average logits over windows)."""
    if n_tokens <= max_len:
        return [(0, n_tokens)]
    out, lo = [], 0
    while True:
        hi = min(lo + max_len, n_tokens)
        out.append((lo, hi))
        if hi == n_tokens:
            return out
        lo += stride


# ----------------------------------------------------------------------------
# Strict scoring
# ----------------------------------------------------------------------------

def strict_scores(gold, pred, labels=LABELS):
    """Strict match on (doc, start, end, label). Macro = mean over `labels` of
    per-label P/R/F1 over all 12 labels, matching the official
    evaluate_phase2.py: a label with no gold and no predictions scores 0."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for doc in set(gold) | set(pred):
        g = set(gold.get(doc, []))
        p = set(pred.get(doc, []))
        for s in p & g: tp[s[2]] += 1
        for s in p - g: fp[s[2]] += 1
        for s in g - p: fn[s[2]] += 1
    per = {}
    for lab in labels:
        P = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        R = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[lab] = {"P": P, "R": R, "F1": F, "tp": tp[lab], "fp": fp[lab], "fn": fn[lab]}
    k = max(len(labels), 1)
    macro = {m: sum(v[m] for v in per.values()) / k for m in ("P", "R", "F1")}
    return macro, per


# ----------------------------------------------------------------------------
# Validation and packaging
# ----------------------------------------------------------------------------

def validate(pred_raw, texts, labels=LABELS):
    """Return a list of problems with a prediction dict (empty list = OK)."""
    errs = []
    if not isinstance(pred_raw, dict):
        return ["top level must be an object {doc_id: [[start, end, label], ...]}"]
    ids, want = set(pred_raw), set(texts)
    if want - ids:
        errs.append(f"missing {len(want - ids)} doc ids, e.g. {sorted(want - ids)[:5]}")
    if ids - want:
        errs.append(f"{len(ids - want)} unknown doc ids, e.g. {sorted(ids - want)[:5]}")
    lab_set = set(labels)
    for doc, spans in pred_raw.items():
        if not isinstance(spans, list):
            errs.append(f"{doc}: value must be a list"); continue
        n = len(texts.get(doc, ""))
        seen = set()
        for sp in spans:
            if not isinstance(sp, list) or len(sp) not in (3, 4):
                errs.append(f"{doc}: bad entry {sp!r}"); continue
            s, e, lab = sp[0], sp[1], sp[2]
            if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool):
                errs.append(f"{doc}: offsets must be ints {sp!r}"); continue
            if not 0 <= s < e <= n:
                errs.append(f"{doc}: offsets out of range {sp!r} (len {n})")
            if lab not in lab_set:
                errs.append(f"{doc}: unknown label {lab!r}")
            if len(sp) == 4 and not isinstance(sp[3], str):
                errs.append(f"{doc}: optional 4th field must be the entity text {sp!r}")
            if (s, e, lab) in seen:
                errs.append(f"{doc}: duplicate span {sp!r}")
            seen.add((s, e, lab))
    return errs


def make_zip(pred_path, out_path="submission.zip"):
    """Write prediction.json directly at the root of the archive."""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    return out_path


def write_predictions(pred, path="prediction.json", texts=None):
    """pred: {doc_id: [(s, e, label)]}. Writes [[s, e, label]] lists (sorted)."""
    out = {d: [[int(s), int(e), l] for s, e, l in sorted(set(v))] for d, v in pred.items()}
    if texts is not None:
        for d in texts:
            out.setdefault(d, [])
        errs = validate(out, texts)
        if errs:
            raise ValueError("invalid predictions:\n" + "\n".join(errs[:20]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _print_scores(macro, per):
    print(f"{'label':<10} {'P':>6} {'R':>6} {'F1':>6} {'tp':>6} {'fp':>6} {'fn':>6}")
    for lab, v in per.items():
        print(f"{lab:<10} {v['P']:6.3f} {v['R']:6.3f} {v['F1']:6.3f} {v['tp']:6d} {v['fp']:6d} {v['fn']:6d}")
    print(f"{'MACRO':<10} {macro['P']:6.3f} {macro['R']:6.3f} {macro['F1']:6.3f}")


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 1
    cmd = argv[1]
    if cmd == "roundtrip":
        texts, gold = load_texts(argv[2]), load_spans(argv[3])
        pred, bad, total = {}, 0, 0
        for d, spans in gold.items():
            offs = word_offsets(texts[d])
            tags, mis = spans_to_bio(offs, spans)
            pred[d] = bio_to_spans(offs, tags)
            bad += len(mis); total += len(spans)
        print(f"spans not on word boundaries: {bad}/{total}")
        _print_scores(*strict_scores(gold, pred))
    elif cmd == "score":
        _print_scores(*strict_scores(load_spans(argv[2]), load_spans(argv[3])))
    elif cmd == "validate":
        texts = load_texts(argv[2])
        with open(argv[3], encoding="utf-8") as f:
            errs = validate(json.load(f), texts)
        print("OK" if not errs else "\n".join(errs[:50]))
        return 1 if errs else 0
    elif cmd == "zip":
        print(make_zip(argv[2], argv[3] if len(argv) > 3 else "submission.zip"))
    else:
        print(__doc__); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
