import sys, torch
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cs781_p1 as P
torch.manual_seed(0)

# 1. full-size parameter counts (meta device: no memory)
for arch in ["gpt", "encdec", "unilm", "retnet"]:
    cfg = P.RunConfig(arch=arch)
    with torch.device("meta"):
        m = P.build_model(cfg.model_config(32000))
    print(f"{arch:7s} {P.count_params(m)/1e6:7.2f}M params")

# 2. incremental decoding == full forward (left padding included)
def full_logits(m, arch, seq):
    ids = torch.tensor([seq])
    if arch == "retnet":
        h, _, _ = m.hidden(ids); return m.lm_head(h)[0, -1]
    return m.lm_head(m.hidden(ids, causal=True))[0, -1]

for arch in ["gpt", "unilm", "retnet"]:
    cfg = P.RunConfig(arch=arch, smoke=True).apply_smoke()
    mc = cfg.model_config(300); mc.vocab_size = 300
    m = P.build_model(mc).eval()
    seqs = [list(torch.randint(10, 300, (n,)).tolist()) for n in (7, 12)]
    ids, valid = P.left_pad(seqs)
    logits, st = m.prefill(ids, valid)
    extra = torch.randint(10, 300, (2, 5))
    errs = [max((logits[b] - full_logits(m, arch, seqs[b])).abs().max().item() for b in range(2))]
    for t in range(5):
        logits, st = m.step(extra[:, t], st)
        errs.append(max((logits[b] - full_logits(m, arch, seqs[b] + extra[b, :t+1].tolist())).abs().max().item() for b in range(2)))
    print(f"{arch:7s} incremental vs full max |dlogit| = {max(errs):.2e}")
    assert max(errs) < 1e-3

# encdec: cached decoding vs teacher-forced decode
cfg = P.RunConfig(arch="encdec", smoke=True).apply_smoke()
mc = cfg.model_config(300); m = P.build_model(mc).eval()
seqs = [torch.randint(10, 300, (n,)).tolist() for n in (9, 15)]
ids, valid = P.left_pad(seqs)
logits, st = m.prefill(ids, valid)
extra = torch.randint(10, 300, (2, 4)); errs = []
for t in range(4):
    logits, st = m.step(extra[:, t], st)
for b in range(2):
    mem = m.encode(torch.tensor([seqs[b]]))
    dec = torch.tensor([[P.SEP] + extra[b].tolist()])
    ref = m.decode(dec, mem, None)[0, -1]
    errs.append((logits[b] - ref).abs().max().item())
print(f"encdec  cached vs teacher-forced max |dlogit| = {max(errs):.2e}"); assert max(errs) < 1e-3

# 3. span corruption sanity
import numpy as np
rng = np.random.default_rng(0)
mask = P.random_spans_mask(768, 0.15, 3.0, rng); print("span mask density", mask.mean())
raw = torch.randint(100, 300, (2, mc.max_len + 1))
b, tag = P.make_batch("encdec", raw, np.random.default_rng(3), mc)
print("encdec batch", tag, {k: tuple(v.shape) for k, v in b.items()})
loss = m(**b); loss.backward(); print("encdec loss", loss.item())
for arch in ["unilm"]:
    mcu = P.RunConfig(arch=arch, smoke=True).apply_smoke().model_config(300); mu = P.build_model(mcu)
    for s in range(4):
        bb, tag = P.make_batch(arch, raw, np.random.default_rng(s), mcu, vocab_size=300)
        print(arch, tag, round(mu(**bb).item(), 3))
print("ALL MODEL TESTS PASSED")
