"""Static token embeddings (word2vec CBOW, Mikolov et al. 2013) trained on the
Phase 1 Supreme Court pretraining corpus (first ~300MB), on the stand-in BPE ids.
Course data only; static embeddings are allowed (capstone FAQ)."""
import json, sys, time, numpy as np
from tokenizers import Tokenizer
from gensim.models import Word2Vec
raw, tokpath, out = sys.argv[1:4]
tok = Tokenizer.from_file(tokpath)
docs = []
with open(raw, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip().rstrip(",")
        if not line.startswith("{"): continue
        try: r = json.loads(line)
        except ValueError: continue
        t = r.get("text")
        if isinstance(t, str) and t: docs.append(t)
print("docs", len(docs), flush=True)
t0 = time.time(); sents = []
for k in range(0, len(docs), 256):
    for e in tok.encode_batch(docs[k:k+256], add_special_tokens=False):
        ids = [str(i) for i in e.ids]
        for j in range(0, len(ids), 1000): sents.append(ids[j:j+1000])
del docs
print("tokens", sum(map(len, sents)), "sec", round(time.time()-t0), flush=True)
m = Word2Vec(sents, vector_size=256, window=5, min_count=1, sg=0, negative=10, workers=2, epochs=3, seed=42)
V = tok.get_vocab_size(); W = np.random.default_rng(42).normal(0, 0.1, (V, 256)).astype("float32"); hit = 0
for i in range(V):
    if str(i) in m.wv: W[i] = m.wv[str(i)]; hit += 1
W[0] = 0
np.save(out, W); print("saved", out, "coverage", hit, V, "sec", round(time.time()-t0), flush=True)
