"""CPU checks for ft_core / p3_rr. Usage: python tests/test_ft.py SMOKE_DIR (holds tiny gpt/unilm/encdec/retnet .pt)."""
import os, sys, copy, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "phase1"))
import torch, torch.nn as nn
import ft_core as C, p3_rr as R

smoke = sys.argv[1]
torch.manual_seed(0)
base = nn.Linear(32, 48, bias=False)
x = torch.randn(5, 32)
for cls, kw in [(C.LoRALinear, {}), (C.LoRMALinear, {"variant": "plus"}), (C.LoRMALinear, {"variant": "pi"})]:
    m = cls(copy.deepcopy(base), r=4, alpha=8, **kw)
    assert torch.allclose(m(x), base(x), atol=1e-5), cls
    with torch.no_grad():
        m.B.normal_(0, 0.1); m.A.add_(0.1 * torch.randn_like(m.A))
    assert torch.allclose(m(x), x @ m.merged_weight().T, atol=1e-4), cls
    if kw.get("variant") == "pi":
        r = torch.linalg.matrix_rank(m.transform()).item()
        print("lorma_pi transform rank", r, "of", 48, "(r=4)")
print("adapters ok")

class A: pass
for arch in ["gpt", "unilm", "encdec", "retnet"]:
    for method in C.METHODS:
        a = A(); a.ckpt = f"{smoke}/{arch}.pt"; a.code_dir = None; a.keep_layers = 1 if method == "head" else 0
        a.bidir = False; a.grad_ckpt = False; a.method = method; a.rank = 4; a.alpha = 8
        a.targets = "wq,wk,wv,wo"; a.train_norms = False; a.context = "lstm"; a.dropout = 0.0
        model, peft = R.build(a)
        model.eval()
        docs = [{"id": "a", "ids": [[5, 10, 11, 12], [5, 13, 14]], "labels": [0, 1]},
                {"id": "b", "ids": [[5] + list(range(20, 40)), [5, 41, 42], [5, 43]], "labels": [0, 1, 2]}]
        b1 = R.collate(docs, [(0, 0, 2)])
        b2 = R.collate(docs, [(0, 0, 2), (1, 0, 3)])
        with torch.no_grad():
            o1 = model(b1["ids"], b1["valid"], b1["sid"], b1["n_sent"])[0, :2]
            o2 = model(b2["ids"], b2["valid"], b2["sid"], b2["n_sent"])[0, :2]
        assert torch.allclose(o1, o2, atol=1e-4), (arch, method, (o1 - o2).abs().max())
        tr = C.count_trainable(model)
        print(f"{arch:7s} {method:10s} trainable {tr:>9,d} wrapped {peft['wrapped_linears']}")
print("padding invariance ok for every arch x method")
