import json, random, sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p6_extract import *
tr = json.load(open(TRAIN)); te = json.load(open(TEST))
random.Random(781).shuffle(tr)
hold, fit = tr[:150], tr[150:]
t=time.time(); sc = Scorer().fit([x['document'] for x in fit], [x['summary'] for x in fit]); print('fit', time.time()-t, flush=True)
METHODS = {
 'lead400': lambda d: lead(d, 400),
 'leadskip400': lambda d: lead(d, 400, skip=True),
 'leadskip250': lambda d: lead(d, 250, skip=True),
 'textrank400': lambda d: textrank(d, 400),
 'centroid400': lambda d: textrank(d, 400, centroid=True),
 'supsent400': lambda d: sc.summarise(d, 400),
 'supsent600': lambda d: sc.summarise(d, 600),
 'supsent400_rank': lambda d: sc.summarise(d, 400, order='rank'),
}
hp = {}
for name, f in METHODS.items():
    write_zip(name, [{'id': x['id'], 'summary': f(x['document'])} for x in te], {'method': name})
    hp[name] = {x['id']: f(x['document']) for x in hold}
    print('done', name, flush=True)
hp['oracle400'] = {x['id']: oracle(x['document'], x['summary'], 400) for x in hold}
json.dump({'truth': {x['id']: x['summary'] for x in hold}, 'preds': hp}, open('holdout_preds.json', 'w'))
