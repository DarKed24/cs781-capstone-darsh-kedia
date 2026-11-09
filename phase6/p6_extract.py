"""Phase 6 CPU extractive summarisers (safety ladder + supervised sentence scorer).

References:
  [1] Mihalcea & Tarau 2004, TextRank
  [2] Nallapati et al. 2017, SummaRuNNer
"""
import json, re, sys, os, random, zipfile, math, argparse
import numpy as np

ROOT = os.environ.get('CS781_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TRAIN = f'{ROOT}/data/Phase_6_Summ/Train/train.json'
TEST = f'{ROOT}/data/Phase_6_Summ/Test/test.json'
OUT = f'{ROOT}/phase6/submissions'

ABBR = r'(?:s|ss|Art|Arts|No|Nos|Mr|Mrs|Ms|Dr|J|JJ|Sh|Smt|vs|v|i\.e|e\.g|viz|cl|Rs|para|paras|Ltd|Co|Pvt|Sec|O|r|R|U\.P|M\.P|A\.P|St|Govt|Hon|Ors|Anr|etc)'
_split = re.compile(r'(?<=[.?!;])\s+(?=[A-Z("\'])')

def sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    parts = _split.split(text)
    out, buf = [], ''
    for p in parts:
        buf = (buf + ' ' + p).strip() if buf else p
        last = buf.split()[-1] if buf.split() else ''
        if re.fullmatch(ABBR + r'\.', last) or re.fullmatch(r'[A-Z]\.', last) or len(buf.split()) < 4:
            continue
        out.append(buf); buf = ''
    if buf: out.append(buf)
    return out

HEADER_CUES = re.compile(r'(judgment of the court was delivered by|delivered the judgment|^.{0,80}\bJ\s*\.\s*:|ORDER|J U D G M E N T|JUDGMENT)', re.I)

def skip_header(sents):
    for i, s in enumerate(sents[:15]):
        m = re.search(r'(delivered by|delivered the following judgment)\s*[^:]{0,60}:\s*', s, re.I) or re.search(r'\bJ\s*\.\s*:\s*', s)
        if m:
            rest = s[m.end():].strip()
            return ([rest] if len(rest.split()) > 3 else []) + sents[i + 1:]
    return sents

def take_budget(sents, idx, W):
    out, n = [], 0
    for i in idx:
        w = len(sents[i].split())
        if n and n + w > W * 1.15: continue
        out.append(i); n += w
        if n >= W: break
    return out

def lead(doc, W, skip=False):
    s = sentences(doc)
    if skip: s = skip_header(s) or s
    return ' '.join(s[i] for i in take_budget(s, range(len(s)), W))

def textrank(doc, W, skip=True, centroid=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    s = sentences(doc)
    if skip: s = skip_header(s) or s
    if len(s) < 3: return ' '.join(s)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(s)
    if centroid:
        c = np.asarray(X.mean(0)).ravel(); sc = X @ c
    else:
        S = (X @ X.T).toarray(); np.fill_diagonal(S, 0)
        S = S / (S.sum(1, keepdims=True) + 1e-9)
        sc = np.ones(len(s)) / len(s)
        for _ in range(30): sc = 0.15 / len(s) + 0.85 * S.T @ sc
    order = list(np.argsort(-sc))
    keep = sorted(take_budget(s, order, W))
    return ' '.join(s[i] for i in keep)

def toks(t): return re.findall(r'[a-z0-9]+', t.lower())
STOP = set('the of and to a in that is was be by for on as it with this which or an not are at from had has have were he his its any been under said such they their there no but shall also court'.split())

def sent_targets(sents, ref):
    rset = {}
    for w in toks(ref):
        if w not in STOP: rset[w] = 1
    y = []
    for s in sents:
        t = [w for w in toks(s) if w not in STOP]
        y.append(sum(1 for w in t if w in rset) / (len(t) + 3))
    return np.array(y)

def sent_feats(sents, doc_vocab_tf=None):
    n = len(sents); rows = []
    for i, s in enumerate(sents):
        t = toks(s)
        rows.append(' '.join(t) + f' __pos{min(int(10*i/max(n,1)),9)} __abs{min(i,20)} __len{min(len(t)//10,8)}')
    return rows

class Scorer:
    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import SGDRegressor
        self.hv = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), alternate_sign=False, norm='l2')
        self.m = SGDRegressor(alpha=1e-6, max_iter=1, tol=None, random_state=0)
    def fit(self, docs, refs, epochs=3):
        data = []
        for d, r in zip(docs, refs):
            s = skip_header(sentences(d)) or sentences(d)
            data.append((sent_feats(s), sent_targets(s, r)))
        for ep in range(epochs):
            random.Random(ep).shuffle(data)
            for k in range(0, len(data), 64):
                F = [f for fs, _ in data[k:k+64] for f in fs]; Y = np.concatenate([y for _, y in data[k:k+64]])
                self.m.partial_fit(self.hv.transform(F), Y)
        return self
    def summarise(self, doc, W, order='doc'):
        s = skip_header(sentences(doc)) or sentences(doc)
        if len(s) < 3: return ' '.join(s)
        sc = self.m.predict(self.hv.transform(sent_feats(s)))
        keep = take_budget(s, list(np.argsort(-sc)), W)
        if order == 'doc': keep = sorted(keep)
        return ' '.join(s[i] for i in keep)

def oracle(doc, ref, W):
    s = skip_header(sentences(doc)) or sentences(doc)
    y = sent_targets(s, ref)
    return ' '.join(s[i] for i in sorted(take_budget(s, list(np.argsort(-y)), W)))

def write_zip(name, preds, meta=None):
    d = f'{OUT}/{name}'; os.makedirs(d, exist_ok=True)
    for p in preds: assert p['summary'].strip(), p['id']
    json.dump(preds, open(f'{d}/prediction.json', 'w'), ensure_ascii=False)
    with zipfile.ZipFile(f'{d}/submission.zip', 'w', zipfile.ZIP_DEFLATED) as z: z.write(f'{d}/prediction.json', 'prediction.json')
    if meta: json.dump(meta, open(f'{d}/run.json', 'w'), indent=1)
