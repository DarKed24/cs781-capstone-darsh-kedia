#!/usr/bin/env python3
"""CS781 Capstone, Phases 2-7: shared fine-tuning toolkit (Darsh Kedia, 250311).

Task-independent pieces that every phase reuses:

  Backbone        any Phase 1 architecture (gpt, unilm, encdec, retnet) as a feature
                  extractor  ids [B, T] (+ valid mask) -> hidden states [B, T, H].
                  Optional layer dropping (`keep_layers`) builds a smaller student
                  for knowledge distillation, initialised from the teacher's layers.
  apply_peft      fine-tuning technique on a backbone:
                    full        every weight trainable (classic fine-tuning)
                    lora        additive low-rank update   W' = W + (a/r) B A          [1]
                    lorma_plus  multiplicative update      W' = (I + (a/r) B A) W       [2]
                    lorma_pi    multiplicative update      W' = pi(B A) W               [2]
                    norms       only the normalisation gains are trained              [3]
                    head        frozen backbone, only the task head is trained (probe)
  kd_loss         Hinton-style distillation loss on softened distributions          [4]
  Trainer bits    AMP dtype, parameter groups, warm-up + linear decay schedule,
                  saving only the trainable weights.

References (cite in the report):
  [1] Hu et al. 2022, LoRA: Low-Rank Adaptation of Large Language Models.
  [2] Bihany, Patel, Modi 2025, LoRMA: Low-Rank Multiplicative Adaptation for LLMs
      (Findings of ACL 2025, arXiv 2506.07621). LoRMA_+ adds the identity to the
      scaled low-rank product (additive rank inflation); LoRMA_pi cyclically shifts
      the rows of B A (permutation rank inflation), initialised so pi(B A) = I.
      Both re-order the computation: h = W x first, then the d x d transform.
  [3] Zhao et al. 2024, Tuning LayerNorm in Attention (normalisation-only tuning);
      Ben Zaken et al. 2022, BitFit (tiny-subset tuning).
  [4] Hinton, Vinyals, Dean 2015, Distilling the Knowledge in a Neural Network;
      Sanh et al. 2019, DistilBERT (student initialised from every other teacher layer).
"""
import json
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_amp():
    """(device, autocast dtype or None). T4 (sm_75) has no fast bf16, so it gets fp16."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), None
    dev = torch.device("cuda")
    return dev, (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)


def import_p1(code_dir=None):
    if code_dir and code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import cs781_p1
    return cs781_p1


def checkpoint_arch(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model_config"]["arch"]


# --------------------------------------------------------------------------------------
# Backbone: Phase 1 model -> hidden states
# --------------------------------------------------------------------------------------
def _spread(n_have, n_keep):
    """Evenly spaced layer indices, always keeping the first and the last layer."""
    if n_keep >= n_have:
        return list(range(n_have))
    if n_keep == 1:
        return [n_have - 1]
    return sorted({round(i * (n_have - 1) / (n_keep - 1)) for i in range(n_keep)})


class Backbone(nn.Module):
    """Feature extractor over a Phase 1 export (model_best.pt / model_final.pt).

      gpt     DecoderLM.hidden, causal (bidir=True runs it without the causal mask)
      unilm   DecoderLM.hidden with mode=1, causal=False (bidirectional, as pretrained by MLM)
      encdec  EncDecLM.encode (bidirectional encoder; the decoder is dropped unless keep_decoder)
      retnet  RetNetLM.hidden (causal decay retention)

    Batches are right-padded with `valid` marking real tokens. Padding is masked for the
    bidirectional archs and cannot reach real tokens in the causal ones.
    """

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
            model.mlm_transform = nn.Identity()   # pretraining-only head
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


# --------------------------------------------------------------------------------------
# PEFT layers
# --------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = W x + (alpha/r) B A x, with B = 0 at init [1]."""
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
    """idx[i, j] = (j - i) mod d, so M.gather(1, idx) shifts row i of M right by i."""
    key = (d, str(device))
    if key not in _ROLL_IDX:
        a = torch.arange(d, device=device)
        _ROLL_IDX[key] = (a[None, :] - a[:, None]) % d
    return _ROLL_IDX[key]


class LoRMALinear(nn.Module):
    """Multiplicative low-rank adaptation [2]: W' = T W, applied as y = T (W x).

    variant "plus": T = I + (alpha/r) B A (additive rank inflation, rank(T) >= d - r).
                    B = 0 at init, so T = I and the model starts exactly at Phase 1.
    variant "pi":   T = pi(B A), pi = cyclic shift of row i by i positions (permutation rank
                    inflation). Init: B[:, 0] = 1, A[0, 0] = 1, so B A = 1 e_0^T and pi(B A) = I;
                    the other rank-1 terms start as LoRA does (A random, B = 0).
    A is r x d_out and B is d_out x r (T acts on the output space of W).
    """
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
        if self.variant == "plus":   # h + s B (A h): never builds the d x d matrix
            return h + self.scale * F.linear(F.linear(self.drop(h), self.A.to(h.dtype)), self.B.to(h.dtype))
        return F.linear(self.drop(h), self.transform().to(h.dtype))

    def merged_weight(self):
        return self.transform() @ self.base.weight


NORM_KEYS = ("n1.weight", "n2.weight", "nc.weight", "norm.weight", "enc_norm.weight",
             "dec_norm.weight", "gn_weight")
DEFAULT_TARGETS = ("wq", "wk", "wv", "wo")      # attention / retention projections in every arch
METHODS = ("full", "lora", "lorma_plus", "lorma_pi", "norms", "head")


def apply_peft(backbone, method, r=8, alpha=16, targets=DEFAULT_TARGETS, dropout=0.05,
               train_norms=False):
    """Set up `backbone` for one fine-tuning technique. Returns a short summary dict."""
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


# --------------------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------------------
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
    """loss.backward + clip + step, with a GradScaler when autocasting to fp16."""

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
    """KL(teacher_T || student_T) * T^2 [4]. teacher_logp are log-probs at T=1
    (log-softmax is a valid logit vector), re-softened here."""
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
    """Only the trained weights (adapters + head for PEFT; everything for full FT, in fp16)."""
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
