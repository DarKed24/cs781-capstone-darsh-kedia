import json, sys, bert_score, torch
from rouge_score import rouge_scorer
torch.set_num_threads(4)
d = json.load(open('holdout_preds.json')); T = d['truth']; ids = list(T)[:int(sys.argv[1]) if len(sys.argv)>1 else None]
refs = [T[k] for k in ids]; rs = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
res = {}
for name, P in d['preds'].items():
    preds = [P[k].strip() for k in ids]
    _,_,f = bert_score.score(preds, refs, lang='en', batch_size=16)
    r = [rs.score(a,b) for a,b in zip(refs,preds)]
    res[name] = dict(bertscore=float(f.mean()), rouge1=sum(x['rouge1'].fmeasure for x in r)/len(r), rougeL=sum(x['rougeL'].fmeasure for x in r)/len(r))
    print(name, res[name], flush=True)
    json.dump(res, open('holdout_scores.json','w'), indent=1)
