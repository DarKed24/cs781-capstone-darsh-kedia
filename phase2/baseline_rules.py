"""Phase 2 baseline without any neural model: a gazetteer learned from the
training gold spans plus a few regexes. Pure Python (needs only lner_utils).

It exists to (1) validate the whole data -> prediction.json -> zip path on the
real files and (2) give a quick first Codabench submission. It is not a
"Phase 1 model" experiment and should be described as a non-neural baseline.

Usage:
  # dev run: learn from fold1, score on fold2 (prints strict macro P/R/F1)
  python baseline_rules.py dev  DATA_DIR
  # submission: learn from fold1+fold2, predict fold3, validate, zip
  python baseline_rules.py test DATA_DIR OUT_DIR

DATA_DIR holds fold1-text.json, fold1-gold.json, fold2-text.json,
fold2-gold.json, fold3-text.json (labels.txt optional).
"""

import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lner_utils as U  # noqa: E402

MIN_COUNT = 1        # gazetteer entry must be seen this many times
MIN_PURITY = 0.6     # share of the entry's majority label among its gold uses
MIN_LEN = 3          # ignore very short surface strings ("Sh", "Ld")

_MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|"
           r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?")
REGEXES = [
    ("DATE", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\.?\s+{_MONTHS},?\s+\d{{4}}\b")),
    ("DATE", re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b")),
    ("DATE", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2}\b")),
    ("CASENO", re.compile(r"\b(?:Criminal|Civil|Crl\.?|Writ|W\.P\.|S\.L\.P\.|SLP)[\w.() ]{0,40}?"
                          r"No\.?\s*\d+(?:\s*(?:/|of)\s*\d{2,4})?", re.I)),
    ("STAT", re.compile(r"\b(?:Indian Penal Code|Code of Criminal Procedure|Constitution of India|"
                        r"Code of Civil Procedure|Evidence Act|[A-Z][\w.,' ]{2,80}? Act,? (?:18|19|20)\d{2})\b")),
]


def build_gazetteer(texts, gold):
    """surface string -> majority label, kept only if frequent and unambiguous."""
    by_surface = defaultdict(Counter)
    for d, spans in gold.items():
        t = texts[d]
        for s, e, lab in spans:
            by_surface[t[s:e]][lab] += 1
    gaz = {}
    for surf, c in by_surface.items():
        lab, n = c.most_common(1)[0]
        if len(surf.strip()) >= MIN_LEN and sum(c.values()) >= MIN_COUNT and n / sum(c.values()) >= MIN_PURITY:
            gaz[surf] = lab
    return gaz


def _gazetteer_regex(gaz):
    # Longest first so "Supreme Court of India" wins over "Supreme Court".
    alts = sorted(gaz, key=len, reverse=True)
    if not alts:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")


def predict_doc(text, gaz, gaz_re):
    cands = []  # (start, end, label, priority)
    if gaz_re is not None:
        for m in gaz_re.finditer(text):
            cands.append((m.start(), m.end(), gaz[m.group(0)], 0))
    for lab, rx in REGEXES:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            while e > s and text[e - 1] in " ,;:": e -= 1
            cands.append((s, e, lab, 1))
    # Resolve overlaps: gazetteer before regex, then longer spans first.
    cands.sort(key=lambda x: (x[3], -(x[1] - x[0]), x[0]))
    taken, out = [], []
    for s, e, lab, _ in cands:
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e)); out.append((s, e, lab))
    return sorted(out)


def predict(texts, gaz):
    gaz_re = _gazetteer_regex(gaz)
    return {d: predict_doc(t, gaz, gaz_re) for d, t in texts.items()}


def main(argv):
    if len(argv) < 3 or argv[1] not in ("dev", "test"):
        print(__doc__); return 1
    data = argv[2]
    p = lambda name: os.path.join(data, name)
    t1, g1 = U.load_texts(p("fold1-text.json")), U.load_spans(p("fold1-gold.json"))
    t2, g2 = U.load_texts(p("fold2-text.json")), U.load_spans(p("fold2-gold.json"))
    if argv[1] == "dev":
        pred = predict(t2, build_gazetteer(t1, g1))
        U._print_scores(*U.strict_scores(g2, pred))
        return 0
    out = argv[3] if len(argv) > 3 else "."
    os.makedirs(out, exist_ok=True)
    t3 = U.load_texts(p("fold3-text.json"))
    pred = predict(t3, build_gazetteer({**t1, **t2}, {**g1, **g2}))
    path = U.write_predictions(pred, os.path.join(out, "prediction.json"), texts=t3)
    print(U.make_zip(path, os.path.join(out, "submission.zip")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
