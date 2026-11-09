import json, random, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p6_extract import *
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import rankdata
tr = json.load(open(TRAIN)); te = json.load(open(TEST))
random.Random(781).shuffle(tr); hold, fit = tr[:150], tr[150:]
sc = Scorer().fit([x['document'] for x in fit], [x['summary'] for x in fit])
def scores(doc):
    s = skip_header(sentences(doc)) or sentences(doc)
    if len(s) < 3: return s, None
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(s)
    c = np.asarray(X.mean(0)).ravel(); cen = X @ c
    S = (X @ X.T).toarray(); np.fill_diagonal(S, 0); S = S / (S.sum(1, keepdims=True) + 1e-9)
    tr_ = np.ones(len(s)) / len(s)
    for _ in range(30): tr_ = 0.15 / len(s) + 0.85 * S.T @ tr_
    sup = sc.m.predict(sc.hv.transform(sent_feats(s)))
    n = len(s); R = lambda v: rankdata(v) / n
    return s, {'tr': R(tr_), 'cen': R(cen), 'sup': R(sup)}
def fuse(doc, w, W=400):
    s, r = scores(doc)
    if r is None: return ' '.join(s)
    tot = sum(w[k] * r[k] for k in w)
    return ' '.join(s[i] for i in sorted(take_budget(s, list(np.argsort(-tot)), W)))
V = {'fuse3_400': {'tr': 1, 'cen': 1, 'sup': 1}, 'fuse_trcen_400': {'tr': 1, 'cen': 1}, 'fuse3_450': {'tr': 1, 'cen': 1, 'sup': 1}}
d = json.load(open('holdout_preds.json'))
for name, w in V.items():
    W = 450 if name.endswith('450') else 400
    write_zip(name, [{'id': x['id'], 'summary': fuse(x['document'], w, W)} for x in te], {'method': name, 'weights': w})
    d['preds'][name] = {x['id']: fuse(x['document'], w, W) for x in hold}
    print('done', name, flush=True)
d['preds'] = {k: v for k, v in d['preds'].items() if k.startswith('fuse')}
json.dump(d, open('holdout_preds_fusion.json', 'w'))
