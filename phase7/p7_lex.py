#!/usr/bin/env python3
"""CS781 Phase 7: prior case retrieval, lexical part (Darsh Kedia, 250311).

References:
  [1] Robertson & Zaragoza 2009, The Probabilistic Relevance Framework: BM25 and Beyond
"""
import argparse, json, math, os, re, sys, time, zipfile
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

K = 7
TAG = re.compile(r"<(NAME|ORG|CITATION)>")
TOK = re.compile(r"[a-z][a-z0-9]+|\d+[a-z]?")
STOP = set(ENGLISH_STOP_WORDS) | {"said", "learned", "court", "case", "section", "act", "also", "would"}
YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20[0-2]\d)(?!\d)")
DOJ = re.compile(r"DATE OF JUDGMENT:?\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")
META = re.compile(r"CITATION\s*:.*?(?=\bACT\s*:|\bHEADNOTE\s*:|$)", re.S)


def strip_meta(text):
    head, tail = text[:8000], text[8000:]
    return META.sub(" ", head, count=1) + tail

DEFAULT = dict(strip_meta=1, ngram=1, ngram_min=1, k1=1.2, b=0.75, k3=8.0, min_df=1, max_df=0.5,
               ctx=0, ctx_w=0.0, temporal=0, prior=0.0)

SPLITS = {"train": ("Train/train.json", "Train/train_candidates.json"),
          "dev": ("Dev/dev.json", "Dev/dev_candidates.json"),
          "test": ("Test/test.json", "Test/test_candidates.json")}


def load(data, split):
    qf, cf = SPLITS[split]
    q = json.load(open(os.path.join(data, qf), encoding="utf-8"))
    c = json.load(open(os.path.join(data, cf), encoding="utf-8"))
    return q, c


def analyzer(text):
    toks = [t for t in TOK.findall(TAG.sub(" ", text.lower())) if t not in STOP]
    return toks


def make_analyzer(n_max, n_min=1):
    def an(text):
        t = analyzer(text); out = []
        for n in range(n_min, n_max + 1):
            out += t if n == 1 else ["_".join(t[i:i + n]) for i in range(len(t) - n + 1)]
        return out
    return an


def query_year(text):
    m = DOJ.search(text[:3000])
    if m:
        return int(m.group(3))
    ys = [int(y) for y in YEAR.findall(text)]
    return max(ys) if ys else None


def min_decision_year(text):
    m = DOJ.search(text[:3000])
    if m:
        return int(m.group(3))
    ys = [int(y) for y in YEAR.findall(text)]
    return max(ys) if ys else None


def citation_context(sents, w):
    hit = [i for i, s in enumerate(sents) if "<CITATION>" in s]
    keep = sorted({j for i in hit for j in range(max(0, i - w), min(len(sents), i + w + 1))})
    return " ".join(sents[j] for j in keep)


class Index:
    def __init__(self, cands, ngram=1, k1=1.2, b=0.75, min_df=1, max_df=0.5, strip=1, ngram_min=1):
        self.ids = list(cands)
        self.pos = {c: i for i, c in enumerate(self.ids)}
        texts = [" ".join(cands[c]["text"]) for c in self.ids]
        if strip:
            texts = [strip_meta(t) for t in texts]
        self.years = np.array([min_decision_year(t) or 0 for t in texts])
        self.cv = CountVectorizer(analyzer=make_analyzer(ngram, ngram_min),
                                  min_df=min_df, max_df=max_df, dtype=np.float32)
        tf = self.cv.fit_transform(texts).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean()
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * dl[tf.row] / avg)
        w = tf.data * (k1 + 1) / (tf.data + norm) * self.idf[tf.col]
        self.W = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def score(self, qtexts, k3=8.0):
        qtf = self.cv.transform(qtexts).tocsr().astype(np.float32)
        qtf.data = qtf.data * (k3 + 1) / (qtf.data + k3)
        return np.asarray((qtf @ self.W.T).todense())


def zrow(S):
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True) + 1e-9
    return (S - mu) / sd


def retrieve(queries, cands, cfg, prior_counts=None, index=None, return_scores=False):
    cfg = {**DEFAULT, **cfg}
    idx = index or Index(cands, cfg["ngram"], cfg["k1"], cfg["b"], cfg["min_df"], cfg["max_df"], cfg["strip_meta"], cfg["ngram_min"])
    qids = list(queries)
    full = [" ".join(queries[q]["text"]) for q in qids]
    if cfg["strip_meta"]:
        full = [strip_meta(t) for t in full]
    S = zrow(idx.score(full, cfg["k3"]))
    if cfg["ctx"] and cfg["ctx_w"] > 0:
        ctx = [citation_context(queries[q]["text"], cfg["ctx"]) or full[i] for i, q in enumerate(qids)]
        S = (1 - cfg["ctx_w"]) * S + cfg["ctx_w"] * zrow(idx.score(ctx, cfg["k3"]))
    if cfg["prior"] and prior_counts:
        p = np.array([math.log1p(prior_counts.get(c, 0)) for c in idx.ids], dtype=np.float32)
        S = S + cfg["prior"] * p[None, :]
    if cfg["temporal"]:
        for i, q in enumerate(qids):
            y = query_year(full[i])
            if y:
                S[i, idx.years > y + cfg["temporal"] - 1] -= 1e3
    for i, q in enumerate(qids):
        if q in idx.pos:
            S[i, idx.pos[q]] = -1e9
    top = np.argsort(-S, axis=1)[:, :K]
    pred = {q: [idx.ids[j] for j in top[i]] for i, q in enumerate(qids)}
    return (pred, S, idx) if return_scores else pred


def micro_f1(queries, pred):
    tp = fp = fn = 0
    for q, v in queries.items():
        g = {x for x in v["relevant_candidates"] if x}; p = set(pred[q][:K])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp), tp / (tp + fn)


def prior_from(*query_sets):
    c = Counter()
    for qs in query_sets:
        for v in qs.values():
            c.update(x for x in v["relevant_candidates"] if x)
    return c


def cmd_eval(a):
    cfg = json.loads(a.cfg) if a.cfg else {}
    q, c = load(a.data, a.split)
    prior = prior_from(load(a.data, "train")[0]) if a.split != "train" else None
    t = time.time()
    pred = retrieve(q, c, cfg, prior)
    f, p, r = micro_f1(q, pred)
    print(f"{a.split} {json.dumps(cfg)} F1 {f:.4f} P {p:.4f} R {r:.4f} ({time.time()-t:.0f}s)")


def write_zip(pred, out):
    os.makedirs(out, exist_ok=True)
    pj = os.path.join(out, "prediction.json")
    json.dump(pred, open(pj, "w"), indent=1)
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")


def cmd_predict(a):
    cfg = json.loads(a.cfg) if a.cfg else {}
    q, c = load(a.data, "test")
    prior = prior_from(load(a.data, "train")[0], load(a.data, "dev")[0])
    pred = retrieve(q, c, cfg, prior)
    assert len(pred) == len(q) and all(len(set(v)) == K and all(x in c for x in v) for v in pred.values())
    write_zip(pred, a.out)
    json.dump({"cfg": {**DEFAULT, **cfg}}, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "predict"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"eval": cmd_eval, "predict": cmd_predict}[a.cmd](a)
