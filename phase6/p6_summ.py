#!/usr/bin/env python3
"""CS781 Phase 6: abstractive summarisation with the Phase 1 LMs (Darsh Kedia, 250311)."""
import argparse, json, math, os, random, sys, time, zipfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC_LEN = 576
TGT_LEN = 1024 - SRC_LEN - 1
SRC_WORDS = 470


def prep(a):
    import p6_extract as X
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    tr = json.load(open(f"{a.data}/Train/train.json")); te = json.load(open(f"{a.data}/Test/test.json"))
    random.Random(781).shuffle(tr)
    hold, fit = tr[:a.n_hold], tr[a.n_hold:]
    t = time.time()
    sc = X.Scorer().fit([x["document"] for x in fit], [x["summary"] for x in fit])
    print(f"extractive scorer fitted in {time.time()-t:.0f}s", flush=True)
    def enc(texts): return [e.ids for e in tok.encode_batch(texts)]
    out = {}
    for split, recs in (("train", fit), ("hold", hold), ("test", te)):
        head = [X.lead(r["document"], SRC_WORDS, skip=True) for r in recs]
        ext = [sc.summarise(r["document"], SRC_WORDS) for r in recs]
        out[split] = {"id": [r["id"] for r in recs], "head": enc(head), "ext": enc(ext),
                      "tgt": enc([r.get("summary", "") or "" for r in recs]),
                      "ref": [r.get("summary", "") for r in recs]}
        print(split, len(recs), f"{time.time()-t:.0f}s", flush=True)
    torch.save(out, a.cache)


def make_example(src, tgt, P1):
    src = src[:SRC_LEN]; tgt = tgt[:TGT_LEN]
    seq = src + [P1.SEP] + tgt
    lab = [-100] * (len(src) + 1) + tgt
    return seq[:-1], lab[1:]


@torch.no_grad()
def generate(model, P1, prompts, dev, dtype, max_new, batch=16, rep=1.15, ngram=4):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        ids, valid = P1.left_pad([prompts[i] for i in idx])
        ids, valid = ids.to(dev), valid.to(dev)
        blocker = P1.NgramBlocker(ngram, len(idx))
        gen = torch.empty((len(idx), 0), dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits, st = model.prefill(ids, valid)
            for t in range(max_new):
                lg = logits.float()
                lg[:, :P1.N_SPECIAL] = -float("inf")
                if gen.numel():
                    prev = torch.gather(lg, 1, gen)
                    lg.scatter_(1, gen, torch.where(prev > 0, prev / rep, prev * rep))
                for b in range(len(idx)):
                    ban = blocker.banned(b)
                    if ban: lg[b, list(ban)] = -float("inf")
                nxt = lg.argmax(-1)
                gen = torch.cat([gen, nxt[:, None]], 1)
                for b, k in enumerate(nxt.tolist()): blocker.add(b, k)
                if t < max_new - 1:
                    logits, st = model.step(nxt, st)
        for b, row in zip(idx, gen.tolist()): outs[b] = row
        print(f"[gen] {min(s+batch, len(order))}/{len(order)}", flush=True)
    return outs


def score(preds, refs, dev):
    from rouge_score import rouge_scorer
    import bert_score
    rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r = [rs.score(b, a) for a, b in zip(preds, refs)]
    _, _, f = bert_score.score(preds, refs, lang="en", batch_size=16, device=str(dev), verbose=False)
    return {"bertscore": float(f.mean()), "rouge1": sum(x["rouge1"].fmeasure for x in r) / len(r),
            "rougeL": sum(x["rougeL"].fmeasure for x in r) / len(r)}


def train(a):
    import ft_core as C
    P1 = C.import_p1(a.code_dir)
    from tokenizers import Tokenizer
    C.set_seed(a.seed)
    dev, dtype = C.device_and_amp()
    data = torch.load(a.cache)
    model, mcfg = P1.load_model(a.ckpt, "cpu")
    assert mcfg.arch in ("gpt", "unilm", "retnet"), "decoder-style archs only"
    if mcfg.arch == "unilm": model.mlm_transform = torch.nn.Identity()
    model.cfg.grad_ckpt = a.grad_ckpt
    peft = C.apply_peft(model, a.method, r=a.rank, alpha=a.alpha)
    model.to(dev)
    os.makedirs(a.out, exist_ok=True)
    tr = data["train"]
    n = min(a.n_train or len(tr["id"]), len(tr["id"]))
    exs = [make_example(tr[a.src][i], tr["tgt"][i], P1) for i in range(n)]
    params = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr or (3e-5 if a.method == "full" else 3e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01 if a.method == "full" else 0.0, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(n / (a.batch * a.accum))
    total = steps_per_epoch * a.epochs
    sched = C.warmup_linear(opt, total, 0.05)
    amp = C.AmpStep(opt, dtype)
    print(json.dumps({"arch": mcfg.arch, "peft": peft, "n_train": n, "steps": total, "lr": lr}), flush=True)
    t0 = time.time(); step = 0; hist = []; budget = a.max_train_min * 60
    model.train()
    done = False
    for ep in range(a.epochs):
        perm = list(range(n)); random.Random(a.seed + ep).shuffle(perm)
        run = 0.0; cnt = 0
        for k in range(0, n - a.batch + 1, a.batch):
            bx = [exs[i] for i in perm[k:k + a.batch]]
            L = max(len(x) for x, _ in bx)
            ids = torch.full((len(bx), L), P1.PAD, dtype=torch.long); lab = torch.full((len(bx), L), -100, dtype=torch.long)
            for j, (x, y) in enumerate(bx):
                ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                kw = {"mode": 0} if mcfg.arch == "unilm" else {}
                loss = model(ids.to(dev), lab.to(dev), **kw) / a.accum
            if amp.scaler: amp.scaler.scale(loss).backward()
            else: loss.backward()
            run += loss.item() * a.accum; cnt += 1
            if cnt % a.accum == 0:
                if amp.scaler: amp.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if amp.scaler: amp.scaler.step(opt); amp.scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step == 20:
                    per = (time.time() - t0) / 20
                    fit_steps = int(budget / per)
                    if fit_steps < total:
                        print(f"[budget] {per:.2f}s/step, schedule {total} -> {fit_steps} steps", flush=True)
                        total = fit_steps; sched = C.warmup_linear(opt, total, 0.05)
                        for _ in range(step): sched.step()
                if step % 50 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run/cnt:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
                if step >= total: done = True; break
        hist.append({"epoch": ep, "loss": run / max(cnt, 1), "min": (time.time() - t0) / 60})
        if done: break
    model.eval()
    tok = Tokenizer.from_file(a.tokenizer)
    res = {"arch": mcfg.arch, "method": a.method, "src": a.src, "peft": peft, "history": hist, "steps": step,
           "train_min": (time.time() - t0) / 60}
    def run_split(split, k=None):
        d = data[split]; m = len(d["id"]) if k is None else min(k, len(d["id"]))
        prompts = [d[a.src][i][:SRC_LEN] + [P1.SEP] for i in range(m)]
        rows = generate(model, P1, prompts, dev, dtype, TGT_LEN if not a.quick else 64, batch=a.gen_batch)
        txt = [s.strip() or "the" for s in tok.decode_batch(rows)]
        return d["id"][:m], txt, d["ref"][:m]
    ids, txt, refs = run_split("hold", a.n_eval)
    res["hold"] = score(txt, refs, dev); res["hold_n"] = len(ids)
    json.dump({i: t for i, t in zip(ids, txt)}, open(f"{a.out}/hold_pred.json", "w"))
    print("holdout", res["hold"], flush=True)
    ids, txt, _ = run_split("test")
    preds = [{"id": i, "summary": t} for i, t in zip(ids, txt)]
    json.dump(preds, open(f"{a.out}/prediction.json", "w"), ensure_ascii=False)
    with zipfile.ZipFile(f"{a.out}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{a.out}/prediction.json", "prediction.json")
    json.dump(res, open(f"{a.out}/run.json", "w"), indent=1)
    if a.save_weights:
        C.save_trainable(model, f"{a.out}/trainable.pt", {"args": vars(a)})
    print("done", json.dumps(res["hold"]), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--data"); p.add_argument("--tokenizer"); p.add_argument("--cache")
    p.add_argument("--n_hold", type=int, default=150)
    t = sub.add_parser("train")
    for k, v in dict(cache=None, ckpt=None, code_dir=".", out=None, tokenizer=None, method="full", src="ext").items():
        t.add_argument("--" + k, default=v)
    for k, v in dict(n_train=0, rank=16, alpha=32, epochs=2, batch=4, accum=4, seed=0, n_eval=100, gen_batch=25).items():
        t.add_argument("--" + k, type=int, default=v)
    t.add_argument("--lr", type=float, default=None); t.add_argument("--max_train_min", type=float, default=35)
    t.add_argument("--grad_ckpt", action="store_true"); t.add_argument("--quick", action="store_true")
    t.add_argument("--save_weights", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
