#!/usr/bin/env python3
"""CS781 Phase 7: prior case retrieval with a Phase 1 bi-encoder, fused with BM25 (Darsh Kedia, 250311).

Lexical stage: p7_lex.py (bigram BM25 + citation-context query + temporal filter), tuned on train/dev.
Dense stage (this file), on top of it:

  * PASSAGE-LEVEL BI-ENCODER. Judgments are ~9k tokens, far beyond the 1024-token Phase 1 context, and a
    single citation is usually about one point of law. So a query is represented by its
    CITATION-CONTEXT PASSAGES (the sentence holding each masked <CITATION> marker, +/-1 sentence), and a
    candidate by up to MAXC chunks of CLEN tokens spread evenly over the judgment.
    score(q, c) = mean over query passages of max over candidate chunks of cos(p, chunk)   (MaxSim,
    after ColBERT, Khattab & Zaharia 2020, at passage instead of token granularity).
  * DISTANTLY SUPERVISED ALIGNMENT (original idea). The labels say only "query q cites candidate c",
    not which passage cites it or which part of c is cited. For every gold (q, c) pair we pick the
    (passage, chunk) pair with the highest TF-IDF cosine as the positive; a hard negative chunk comes
    from the best-matching chunk of a high-BM25 candidate that q does not cite.
  * Contrastive InfoNCE with in-batch negatives (Karpukhin et al. 2020, DPR; van den Oord et al. 2018),
    mean pooling, temperature TAU, on any Phase 1 arch with any ft_core method (full, lora, lorma_plus,
    or head = frozen backbone, which is the zero-shot baseline with --epochs 0).
  * Fusion: z(BM25 stage) + w * z(dense), w chosen on dev.

  python p7_dense.py prep  --data DIR --tokenizer tok.json --cache c.pt        (CPU, ~10-15 min)
  python p7_dense.py train --cache c.pt --ckpt model_best.pt --code_dir . --out RUN --method full ...
  python p7_dense.py fuse  --cache c.pt --runs RUN1,RUN2 --out DIR             (CPU, writes zips)
"""
import argparse, json, math, os, random, sys, time, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_lex as L

PLEN, CLEN, MAXQ, MAXC = 192, 192, 24, 24
LEX_CFG = dict(k3=1000, k1=6.0, b=1.0, max_df=0.3, ctx=1, ctx_w=0.7, temporal=1,
               ngram=2, ngram_min=2, min_df=2)
K = 7


# ----------------------------------------------------------------------------------------------
# prep: tokenise passages/chunks, lexical scores, distant-supervision training pairs
# ----------------------------------------------------------------------------------------------
def query_passages(sents, tok):
    hit = [i for i, s in enumerate(sents) if "<CITATION>" in s]
    spans, last = [], -10
    for i in hit:                                    # merge markers in adjacent sentences
        lo, hi = max(0, i - 1), min(len(sents), i + 2)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
    texts = [" ".join(sents[a:b]) for a, b in spans]
    if not texts:                                    # no marker: fall back to even chunks
        return doc_chunks(" ".join(sents), tok, PLEN, MAXQ)
    if len(texts) > MAXQ:
        texts = [texts[round(i * (len(texts) - 1) / (MAXQ - 1))] for i in range(MAXQ)]
    out = []
    for e in tok.encode_batch(texts):
        ids = e.ids
        if len(ids) > PLEN:                          # keep the middle, where the marker is
            s = (len(ids) - PLEN) // 2; ids = ids[s:s + PLEN]
        out.append(ids)
    return out


def doc_chunks(text, tok, n, maxc):
    ids = tok.encode(text).ids
    ch = [ids[i:i + n] for i in range(0, max(1, len(ids)), n)]
    ch = [c for c in ch if len(c) >= 16] or ch[:1]
    if len(ch) > maxc:
        ch = [ch[round(i * (len(ch) - 1) / (maxc - 1))] for i in range(maxc)]
    return ch


def tfidf_rows(seqs, vocab=32000 + 64 + 6):
    """L2-normalised log-tf idf vectors over token ids (scipy CSR), for the alignment step."""
    from scipy import sparse
    rows, cols, vals = [], [], []
    for r, s in enumerate(seqs):
        u, c = np.unique(np.asarray(s, dtype=np.int64), return_counts=True)
        rows.append(np.full(len(u), r)); cols.append(u); vals.append(1 + np.log(c))
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(seqs), vocab), dtype=np.float32)
    return M


def prep(a):
    from tokenizers import Tokenizer
    from scipy import sparse
    tok = Tokenizer.from_file(a.tokenizer)
    out, t0 = {}, time.time()
    prior = {"train": None}
    tq = L.load(a.data, "train")[0]; dq = L.load(a.data, "dev")[0]
    prior["dev"], prior["test"] = L.prior_from(tq), L.prior_from(tq, dq)
    for split in ("train", "dev", "test"):
        q, c = L.load(a.data, split)
        pred, S, idx = L.retrieve(q, c, LEX_CFG, prior[split], return_scores=True)
        qids = list(q)
        qp = [query_passages([L.strip_meta(s) for s in q[k]["text"]], tok) for k in qids]
        cc = [doc_chunks(L.strip_meta(" ".join(c[k]["text"])), tok, CLEN, MAXC) for k in idx.ids]
        rec = {"qids": qids, "cids": idx.ids, "qpass": qp, "cchunks": cc,
               "lex": S.astype(np.float32), "lex_top": np.argsort(-S, 1)[:, :200].astype(np.int32)}
        if "relevant_candidates" in q[qids[0]]:
            rec["gold"] = [[x for x in q[k]["relevant_candidates"] if x] for k in qids]
            rec["lex_f1"] = L.micro_f1(q, pred)
        out[split] = rec
        print(split, len(qids), "queries", len(idx.ids), "cands", f"{time.time()-t0:.0f}s",
              rec.get("lex_f1", ""), flush=True)
    # distant supervision on train: best (passage, chunk) pair per gold link + a hard negative chunk
    tr = out["train"]; pos = {c: i for i, c in enumerate(tr["cids"])}
    flat_c = [ch for cs in tr["cchunks"] for ch in cs]
    off = np.cumsum([0] + [len(cs) for cs in tr["cchunks"]])
    Cm = tfidf_rows(flat_c)
    df = np.bincount(Cm.indices, minlength=Cm.shape[1]); idf = np.log((Cm.shape[0] + 1) / (df + 1)).astype(np.float32)
    def norm(M):
        M = M @ sparse.diags(idf); n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9
        return sparse.diags(1 / n) @ M
    Cm = norm(Cm).tocsr()
    pairs, rng = [], random.Random(781)
    for qi, gold in enumerate(tr["gold"]):
        P = norm(tfidf_rows(tr["qpass"][qi]))
        gs = set(gold)
        negs = [j for j in tr["lex_top"][qi][:30] if tr["cids"][j] not in gs and tr["cids"][j] != tr["qids"][qi]]
        for g in gold:
            ci = pos[g]; sim = (P @ Cm[off[ci]:off[ci + 1]].T).toarray()
            pi, cj = np.unravel_index(sim.argmax(), sim.shape)
            nj = rng.choice(negs[:15]) if negs else rng.randrange(len(tr["cids"]))
            ns = (P[pi] @ Cm[off[nj]:off[nj + 1]].T).toarray().ravel()
            pairs.append((qi, int(pi), ci, int(cj), int(nj), int(ns.argmax())))
    out["pairs"] = pairs
    print("training pairs", len(pairs), f"{time.time()-t0:.0f}s", flush=True)
    torch.save(out, a.cache)


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, backbone, dim=256):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(backbone.d_model, dim)

    def forward(self, ids, valid):
        h = self.backbone(ids, valid).float()
        m = valid.unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.proj(v), dim=-1)


def pad(seqs, dev):
    n = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(dev), (ids != 0).to(dev)


@torch.no_grad()
def embed(model, seqs, dev, amp, bs=128):
    model.eval(); out = []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; ids, va = pad([seqs[j] for j in idx], dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out.append(model(ids, va).float().cpu())
    E = torch.cat(out); R = torch.empty_like(E); R[torch.tensor(order)] = E
    return R


def maxsim_scores(model, rec, dev, amp):
    """[n_query, n_cand] mean-over-passages of max-over-chunks cosine."""
    flatq = [p for ps in rec["qpass"] for p in ps]; qn = [len(ps) for ps in rec["qpass"]]
    flatc = [c for cs in rec["cchunks"] for c in cs]; cn = [len(cs) for cs in rec["cchunks"]]
    Eq, Ec = embed(model, flatq, dev, amp).to(dev), embed(model, flatc, dev, amp).to(dev)
    nC = len(cn); cidx = torch.repeat_interleave(torch.arange(nC), torch.tensor(cn)).to(dev)
    S = torch.empty(len(qn), nC)
    qo = np.cumsum([0] + qn)
    for i in range(len(qn)):
        sim = Eq[qo[i]:qo[i + 1]] @ Ec.T                                   # [p, total chunks]
        best = torch.full((sim.shape[0], nC), -2.0, device=dev).scatter_reduce(
            1, cidx.expand(sim.shape[0], -1), sim, reduce="amax", include_self=True)
        S[i] = best.mean(0).cpu()
    return S.numpy()


def micro_f1(S, rec):
    tp = fp = fn = 0
    for i, q in enumerate(rec["qids"]):
        top = np.argsort(-S[i])[:K]; p = {rec["cids"][j] for j in top}; g = set(rec["gold"][i])
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    return 2 * tp / (2 * tp + fp + fn)


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def fuse_scores(lex, dense, w):
    # lex already carries the temporal (-1e3) and self (-1e9) penalties on top of its z-scores
    return lex + w * zrow(dense)


def train(a):
    import ft_core as C
    C.set_seed(a.seed)
    dev, amp = C.device_and_amp()
    data = torch.load(a.cache, weights_only=False)
    bb = C.Backbone(a.ckpt, a.code_dir, bidir=a.bidir, grad_ckpt=a.grad_ckpt)
    peft = C.apply_peft(bb, a.method, a.rank, a.alpha)
    model = Encoder(bb, a.dim).to(dev)
    tr, pairs = data["train"], data["pairs"]
    if a.align == "random":      # ablation of the distant-supervision alignment: any passage, any chunk
        r0 = random.Random(a.seed + 1)
        pairs = [(qi, r0.randrange(len(tr["qpass"][qi])), ci, r0.randrange(len(tr["cchunks"][ci])),
                  nj, r0.randrange(len(tr["cchunks"][nj]))) for qi, _, ci, _, nj, _ in pairs]
    if a.quick:
        pairs = pairs[:256]
    steps_ep = len(pairs) // a.batch
    total = max(1, steps_ep * a.epochs)
    hist, t0, step = [], time.time(), 0
    if a.epochs > 0:
        groups = C.param_groups(model, a.lr, a.lr_head, a.lr_adapter)
        opt = torch.optim.AdamW(groups); sch = C.warmup_linear(opt, total, 0.06)
        stepper = C.AmpStep(opt, amp)
        params = [p for p in model.parameters() if p.requires_grad]
        rng = random.Random(a.seed)
        for ep in range(a.epochs):
            model.train(); rng.shuffle(pairs); tot = 0.0
            for b in range(steps_ep):
                batch = pairs[b * a.batch:(b + 1) * a.batch]
                qs = [tr["qpass"][qi][pi] for qi, pi, *_ in batch]
                cs = [tr["cchunks"][ci][cj] for _, _, ci, cj, _, _ in batch]
                ns = [tr["cchunks"][nj][nk] for *_, nj, nk in batch]
                owner = torch.tensor([ci for _, _, ci, *_ in batch] + [nj for *_, nj, _ in batch], device=dev)
                gold_sets = [set(tr["gold"][qi]) for qi, *_ in batch]
                ids, va = pad(qs, dev); cids_, cva = pad(cs + ns, dev)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    eq = model(ids, va); ec = model(cids_, cva)
                logits = eq @ ec.T / a.tau                                  # [B, 2B]
                # mask in-batch "negatives" that are actually gold for this query (false negatives)
                fn = torch.tensor([[tr["cids"][o] in gs for o in owner.tolist()] for gs in gold_sets], device=dev)
                tgt = torch.arange(len(batch), device=dev)
                fn[tgt, tgt] = False
                logits = logits.masked_fill(fn, -1e4)
                loss = F.cross_entropy(logits, tgt)
                stepper(loss, params); sch.step(); step += 1; tot += loss.item()
                if (time.time() - t0) / 60 > a.max_train_min:
                    break
            hist.append({"epoch": ep, "loss": tot / max(1, b + 1), "min": (time.time() - t0) / 60})
            print(hist[-1], flush=True)
            if (time.time() - t0) / 60 > a.max_train_min:
                print("training-time cap reached"); break
    train_min = (time.time() - t0) / 60
    os.makedirs(a.out, exist_ok=True)
    res = {"arch": bb.arch, "method": a.method, "align": a.align, "peft": peft, "epochs": a.epochs, "steps": step,
           "train_min": train_min, "history": hist, "pairs": len(pairs)}
    for split in ("dev", "test"):
        S = maxsim_scores(model, data[split], dev, amp)
        np.save(f"{a.out}/dense_{split}.npy", S.astype(np.float32))
    Sd = np.load(f"{a.out}/dense_dev.npy"); rec = data["dev"]
    res["dev_dense_f1"] = micro_f1(Sd + np.where(rec["lex"] < -100, rec["lex"], 0), rec)   # dense + same filters
    res["dev_lex_f1"] = micro_f1(rec["lex"], rec)
    grid = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    res["dev_fused"] = {str(w): micro_f1(fuse_scores(rec["lex"], Sd, w), rec) for w in grid}
    res["best_w"] = max(grid, key=lambda w: res["dev_fused"][str(w)])
    St = fuse_scores(data["test"]["lex"], np.load(f"{a.out}/dense_test.npy"), res["best_w"])
    write_zip(St, data["test"], a.out)
    C.write_json(res, f"{a.out}/run.json")
    print(json.dumps({k: res[k] for k in ("dev_lex_f1", "dev_dense_f1", "best_w", "dev_fused", "train_min")}))


def write_zip(S, rec, out):
    pred = {}
    for i, q in enumerate(rec["qids"]):
        top = [j for j in np.argsort(-S[i]) if rec["cids"][j] != q][:K]
        pred[q] = [rec["cids"][j] for j in top]
    assert all(len(set(v)) == K for v in pred.values())
    os.makedirs(out, exist_ok=True)
    json.dump(pred, open(f"{out}/prediction.json", "w"), indent=1)
    with zipfile.ZipFile(f"{out}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{out}/prediction.json", "prediction.json")


def fuse(a):
    """Ensembles: z(BM25) + w * mean_r z(dense_r), w tuned on dev. Also writes the lexical-only zip."""
    data = torch.load(a.cache, weights_only=False)
    runs = [r for r in a.runs.split(",") if r and os.path.exists(f"{r}/dense_dev.npy")]
    dv, te = data["dev"], data["test"]
    report = {"lex_only": micro_f1(dv["lex"], dv), "runs": runs}
    write_zip(te["lex"], te, f"{a.out}/lex_only")
    if runs:
        Dd = np.mean([zrow(np.load(f"{r}/dense_dev.npy")) for r in runs], 0)
        Dt = np.mean([zrow(np.load(f"{r}/dense_test.npy")) for r in runs], 0)
        grid = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        report["dev_fused"] = {str(w): micro_f1(fuse_scores(dv["lex"], Dd, w), dv) for w in grid}
        w = max(grid, key=lambda w: report["dev_fused"][str(w)]); report["best_w"] = w
        write_zip(fuse_scores(te["lex"], Dt, w), te, f"{a.out}/{a.name}")
    json.dump(report, open(f"{a.out}/fuse_{a.name}.json", "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    import ft_core as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--data", required=True); p.add_argument("--tokenizer", required=True)
    p.add_argument("--cache", required=True)
    t = sp.add_parser("train")
    t.add_argument("--cache", required=True); t.add_argument("--out", required=True)
    t.add_argument("--ckpt", required=True); t.add_argument("--code_dir", default=None)
    t.add_argument("--tokenizer", default=None)
    t.add_argument("--method", default="full", choices=C.METHODS)
    t.add_argument("--rank", type=int, default=16); t.add_argument("--alpha", type=float, default=32)
    t.add_argument("--bidir", action="store_true"); t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--dim", type=int, default=256); t.add_argument("--tau", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=3); t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-5); t.add_argument("--lr_adapter", type=float, default=5e-4)
    t.add_argument("--lr_head", type=float, default=1e-3)
    t.add_argument("--max_train_min", type=float, default=30); t.add_argument("--seed", type=int, default=781)
    t.add_argument("--quick", action="store_true")
    t.add_argument("--align", default="tfidf", choices=["tfidf", "random"])
    f = sp.add_parser("fuse"); f.add_argument("--cache", required=True); f.add_argument("--runs", default="")
    f.add_argument("--out", required=True); f.add_argument("--name", default="fuse_all")
    a = ap.parse_args()
    {"prep": prep, "train": train, "fuse": fuse}[a.cmd](a)
