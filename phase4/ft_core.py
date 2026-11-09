#!/usr/bin/env python3
"""CS781 Capstone, Phases 2-7: shared fine-tuning toolkit (Darsh Kedia, 250311).

References:
  [1] Hu et al. 2022, LoRA
  [2] Bihany, Patel, Modi 2025, LoRMA (Findings of ACL 2025)
  [3] Zhao et al. 2024, Tuning LayerNorm in Attention; Ben Zaken et al. 2022, BitFit
  [4] Hinton, Vinyals, Dean 2015, Knowledge Distillation; Sanh et al. 2019, DistilBERT
"""
import json
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_amp():
    if not torch.cuda.is_available():
        return torch.device("cpu"), None
    dev = torch.device("cuda")
    return dev, (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)


def import_p1(code_dir=None):
    code_dir = code_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1")
    if code_dir not in sys.path:
        sys.path.append(code_dir)
    import cs781_p1
    return cs781_p1


def checkpoint_arch(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model_config"]["arch"]


def _spread(n_have, n_keep):
    if n_keep >= n_have:
        return list(range(n_have))
    if n_keep == 1:
        return [n_have - 1]
    return sorted({round(i * (n_have - 1) / (n_keep - 1)) for i in range(n_keep)})


class Backbone(nn.Module):

    def __init__(self, ckpt, code_dir=None, keep_layers=None, bidir=False, grad_ckpt=False,
                 keep_decoder=False):
        super().__init__()
        P1 = import_p1(code_dir)
        model, mcfg = P1.load_model(ckpt, "cpu")
        self.arch, self.mcfg, self.bidir = mcfg.arch, mcfg, bidir
        self.d_model = mcfg.d_model
        stack = "enc" if self.arch == "encdec" else "blocks"
        n_have = len(getattr(model, stack))
        self.layers_kept = _spread(n_have, keep_layers) if keep_layers else list(range(n_have))
        if keep_layers:
            setattr(model, stack, nn.ModuleList(getattr(model, stack)[i] for i in self.layers_kept))
        if self.arch == "encdec" and not keep_decoder:
            model.dec = nn.ModuleList()
        if self.arch == "unilm":
            model.mlm_transform = nn.Identity()
        model.cfg.grad_ckpt = grad_ckpt
        self.model = model
        self.source_ckpt = os.path.abspath(ckpt)

    def forward(self, ids, valid):
        m, full = self.model, bool(valid.all())
        mask = None if full else valid[:, None, None, :]
        if self.arch == "unilm":
            return m.hidden(ids, mask=mask, causal=False, mode=1)
        if self.arch == "gpt":
            if self.bidir:
                return m.hidden(ids, mask=mask, causal=False)
            return m.hidden(ids, causal=True)
        if self.arch == "encdec":
            return m.encode(ids, valid)
        if self.arch == "retnet":
            return m.hidden(ids, None if full else valid)[0]
        raise ValueError(self.arch)


class LoRALinear(nn.Module):
    adapter = True

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        out_f, in_f = base.weight.shape
        self.A = nn.Parameter(torch.empty(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype))

    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A


_ROLL_IDX = {}


def _roll_index(d, device):
    key = (d, str(device))
    if key not in _ROLL_IDX:
        a = torch.arange(d, device=device)
        _ROLL_IDX[key] = (a[None, :] - a[:, None]) % d
    return _ROLL_IDX[key]


class LoRMALinear(nn.Module):
    adapter = True

    def __init__(self, base: nn.Linear, r=8, alpha=16, variant="plus", dropout=0.0):
        super().__init__()
        self.base, self.r, self.variant = base, r, variant
        self.scale = alpha / r if variant == "plus" else 1.0
        d = base.weight.shape[0]
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if variant == "pi":
            with torch.no_grad():
                self.A[0].zero_()
                self.A[0, 0] = 1.0
                self.B[:, 0] = 1.0
        self.drop = nn.Dropout(dropout)

    def transform(self):
        M = self.B @ self.A
        if self.variant == "plus":
            return torch.eye(M.shape[0], device=M.device, dtype=M.dtype) + self.scale * M
        return M.gather(1, _roll_index(M.shape[0], M.device))

    def forward(self, x):
        h = self.base(x)
        if self.variant == "plus":
            return h + self.scale * F.linear(F.linear(self.drop(h), self.A.to(h.dtype)), self.B.to(h.dtype))
        return F.linear(self.drop(h), self.transform().to(h.dtype))

    def merged_weight(self):
        return self.transform() @ self.base.weight


NORM_KEYS = ("n1.weight", "n2.weight", "nc.weight", "norm.weight", "enc_norm.weight",
             "dec_norm.weight", "gn_weight")
DEFAULT_TARGETS = ("wq", "wk", "wv", "wo")
METHODS = ("full", "lora", "lorma_plus", "lorma_pi", "norms", "head")


def apply_peft(backbone, method, r=8, alpha=16, targets=DEFAULT_TARGETS, dropout=0.05,
               train_norms=False):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    for p in backbone.parameters():
        p.requires_grad = method == "full"
    n_wrapped = 0
    if method in ("lora", "lorma_plus", "lorma_pi"):
        for name, mod in list(backbone.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and cname in targets:
                    if method == "lora":
                        new = LoRALinear(child, r, alpha, dropout)
                    else:
                        new = LoRMALinear(child, r, alpha, "plus" if method == "lorma_plus" else "pi", dropout)
                    setattr(mod, cname, new)
                    n_wrapped += 1
        for n, p in backbone.named_parameters():
            if n.endswith((".A", ".B")):
                p.requires_grad = True
    if method == "norms" or (train_norms and method != "full"):
        for n, p in backbone.named_parameters():
            if n.endswith(NORM_KEYS) or n.endswith("mode_emb.weight"):
                p.requires_grad = True
    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in {id(p): p for p in backbone.parameters()}.values())
    return {"method": method, "wrapped_linears": n_wrapped, "backbone_trainable": n_train,
            "backbone_total": n_all}


def is_adapter_param(name):
    return name.endswith((".A", ".B"))


def param_groups(model, lr_backbone, lr_head, lr_adapter, weight_decay=0.01):
    groups = {"backbone": [], "adapter": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            groups["adapter" if is_adapter_param(n) else "backbone"].append(p)
        else:
            groups["head"].append(p)
    lrs = {"backbone": lr_backbone, "adapter": lr_adapter, "head": lr_head}
    return [{"params": ps, "lr": lrs[k], "weight_decay": 0.0 if k == "adapter" else weight_decay, "name": k}
            for k, ps in groups.items() if ps]


def warmup_linear(opt, total_steps, warmup_frac=0.06):
    warm = max(1, int(warmup_frac * total_steps))
    f = lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


class AmpStep:

    def __init__(self, opt, amp_dtype, clip=1.0):
        self.opt, self.clip = opt, clip
        self.scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    def __call__(self, loss, params):
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(params, self.clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(params, self.clip)
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)


def kd_loss(student_logits, teacher_logp, T=2.0, mask=None):
    t = F.softmax(teacher_logp.float() / T, -1)
    s = F.log_softmax(student_logits.float() / T, -1)
    kl = (t * (torch.log(t.clamp(min=1e-9)) - s)).sum(-1)
    if mask is not None:
        kl = kl[mask]
    return kl.mean() * T * T


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in keep}


def save_trainable(model, path, meta):
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in trainable_state(model).items()}
    torch.save({"meta": meta, "state_dict": sd}, path)


def load_trainable(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    res = model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                 for k, v in ck["state_dict"].items()}, strict=False)
    return ck["meta"], res


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
