#!/usr/bin/env python3
"""CS781 Capstone, Phase 1: legal language model pretraining from scratch.

Darsh Kedia (250311). One module shared by all four Phase 1 architectures:

  gpt     Decoder-only Transformer (pre-norm, RoPE, RMSNorm, SwiGLU, tied embeddings)
  encdec  Encoder-decoder Transformer, trained on a 50/50 mix of prefix-LM continuation
          and T5 span corruption
  unilm   Hybrid encoder: one shared Transformer trained with both causal-LM and
          bidirectional masked-LM objectives (UniLM style), with mode embeddings
  retnet  Retentive Network: softmax-free multi-scale retention with a parallel form for
          training and an O(1)-per-token recurrent form for generation

Stages (CLI):  python cs781_p1.py {prepare|train|generate|pseudo_eval} --config run.json
Multi-GPU:     torchrun --standalone --nproc_per_node=2 cs781_p1.py train --config run.json

References (cite in the report as well):
  [1] Vaswani et al. 2017, Attention Is All You Need.
  [2] Radford et al. 2019, GPT-2 (init scheme, pre-norm decoder).
  [3] Su et al. 2021, RoFormer: rotary position embeddings (RoPE).
  [4] Zhang & Sennrich 2019, RMSNorm.   [5] Shazeer 2020, GLU variants (SwiGLU).
  [6] Raffel et al. 2020, T5 (span corruption, random_spans_noise_mask).
  [7] Dong et al. 2019, UniLM.   [8] Devlin et al. 2019, BERT (MLM 80/10/10, MLM head).
  [9] Sun et al. 2023, Retentive Network (multi-scale decay, parallel/recurrent forms).
  [10] Karpathy, nanoGPT (memmap token bins, training-loop structure).
  [11] Sennrich et al. 2016 BPE; HF `tokenizers` library for byte-level BPE training.
"""
import argparse
import dataclasses
import io
import json
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Special tokens (ids are fixed by their order here; the Phase 2-7 code relies on them)
# --------------------------------------------------------------------------------------
N_SENTINELS = 64
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>", "<sep>"] + [
    f"<extra_id_{i}>" for i in range(N_SENTINELS)]
PAD, UNK, BOS, EOS, MASK, SEP = 0, 1, 2, 3, 4, 5
SENTINEL0 = 6
N_SPECIAL = len(SPECIAL_TOKENS)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    arch: str = "gpt"            # gpt | encdec | unilm | retnet
    vocab_size: int = 32000
    d_model: int = 768
    n_layer: int = 12            # gpt / unilm / retnet
    n_head: int = 12
    d_ff: int = 2048
    max_len: int = 1024
    dropout: float = 0.0
    rope_base: float = 10000.0
    n_enc_layer: int = 6         # encdec
    n_dec_layer: int = 6
    enc_len: int = 768           # encdec: encoder context, decoder predicts max_len - enc_len
    grad_ckpt: bool = False

    @property
    def dec_len(self):
        return self.max_len - self.enc_len


# Full-size presets (~110-125M parameters each with a 32k vocabulary).
PRESETS = {
    "gpt": dict(arch="gpt", d_model=768, n_layer=12, n_head=12, d_ff=2048),
    "encdec": dict(arch="encdec", d_model=768, n_enc_layer=6, n_dec_layer=6, n_head=12, d_ff=2048),
    "unilm": dict(arch="unilm", d_model=768, n_layer=12, n_head=12, d_ff=2048),
    "retnet": dict(arch="retnet", d_model=768, n_layer=12, n_head=6, d_ff=2048),
}
SMOKE_MODEL = dict(d_model=64, n_layer=2, n_enc_layer=2, n_dec_layer=2, n_head=4, d_ff=128)

PHASE1_SOURCES = {
    "supreme_court": "Phase_1_Pretraining/Train/train_supreme_court.json",
    "high_courts": "Phase_1_Pretraining/Train/train_high_courts.json",
    "tribunals": "Phase_1_Pretraining/Train/train_tribunals.json",
    "acts": "Phase_1_Pretraining/Train/train_acts.json",
}


@dataclass
class RunConfig:
    run_name: str = "gpt"
    arch: str = "gpt"
    seed: int = 42
    smoke: bool = False
    out_dir: str = "/kaggle/working/cs781_phase1"
    data_dir: str = "/kaggle/working/cs781_phase1/data"   # tokenizer.json + token bins
    # ---- data source ----
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""     # if set, read the source files from here instead of HF
    sources: dict = field(default_factory=lambda: dict(PHASE1_SOURCES))
    test_file: str = "Phase_1_Pretraining/Test/test.json"
    mix: dict = field(default_factory=lambda: {
        "supreme_court": 0.35, "high_courts": 0.35, "tribunals": 0.15, "acts": 0.15})
    prepare_train_tokens: int = 420_000_000   # written once, shared by all four runs
    val_tokens: int = 4_000_000
    val_every_doc: int = 100     # every 100th document of each source goes to validation
    n_pseudo_test: int = 200     # held-out 512-word prompt / 128-word reference pairs
    tokenizer_vocab: int = 32000
    tokenizer_sample_chars: int = 250_000_000
    # ---- model ----
    model: dict = field(default_factory=dict)   # overrides on top of PRESETS[arch]
    # ---- training ----
    train_tokens: int = 200_000_000
    seq_len: int = 1024
    micro_batch: int = 8
    batch_tokens: int = 131072
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    extend_peak_frac: float = 0.5   # continued pretraining: re-warm to lr * this, then cosine again
    weight_decay: float = 0.1
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    save_every: int = 100
    max_hours: float = 11.0      # stop cleanly (and checkpoint) before Kaggle's 12h limit
    resume_from: str = ""        # path to a last.pt from a previous session (else out_dir/run)
    compile: bool = False
    hf_push_repo: str = ""       # optional private HF model repo for checkpoints
    # ---- generation ----
    gen_checkpoint: str = "best"   # best | final | path to a .pt
    gen_max_new_tokens: int = 224
    gen_target_words: int = 128
    gen_batch: int = 32
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram: int = 4
    max_test_examples: int = 0   # 0 = all

    def apply_smoke(self):
        """Tiny everything, so the whole pipeline runs in minutes (CPU or GPU)."""
        self.model = {**SMOKE_MODEL, **self.model}
        self.seq_len = 128
        self.prepare_train_tokens = min(self.prepare_train_tokens, 2_000_000)
        self.val_tokens = min(self.val_tokens, 100_000)
        self.val_every_doc = min(self.val_every_doc, 10)
        self.n_pseudo_test = min(self.n_pseudo_test, 8)
        self.tokenizer_vocab = min(self.tokenizer_vocab, 2000)
        self.tokenizer_sample_chars = min(self.tokenizer_sample_chars, 5_000_000)
        self.train_tokens = min(self.train_tokens, 120_000)
        self.micro_batch = 4
        self.batch_tokens = 2048
        self.warmup_steps = 5
        self.lr = 2e-3
        self.log_every, self.eval_every, self.eval_batches, self.save_every = 5, 20, 4, 20
        self.gen_max_new_tokens, self.gen_target_words, self.gen_batch = 48, 32, 8
        self.max_test_examples = self.max_test_examples or 16
        return self

    def model_config(self, vocab_size):
        kw = dict(PRESETS[self.arch])
        kw.update(self.model)
        kw["max_len"] = self.seq_len
        kw.setdefault("enc_len", self.seq_len * 3 // 4)
        kw["vocab_size"] = vocab_size
        return ModelConfig(**kw)

    @property
    def run_dir(self):
        return os.path.join(self.out_dir, self.run_name)


def load_run_config(path_or_dict):
    d = path_or_dict
    if isinstance(d, str):
        with open(d) as f:
            d = json.load(f)
    known = {f.name for f in dataclasses.fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in d.items() if k in known})
    if cfg.smoke:
        cfg.apply_smoke()
    return cfg


def log0(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------
# Reading the corpus (streams JSON arrays, JSON objects and JSON Lines without loading
# whole multi-GB files into memory)
# --------------------------------------------------------------------------------------
def record_text(rec):
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        t = rec.get("text")
        if isinstance(t, str):
            return t
        return "\n".join(v for v in rec.values() if isinstance(v, str))
    return ""


def iter_json_records(f):
    """Yields (key_or_index, record) from a binary file object holding a JSON array,
    a JSON object {id: record} or JSON Lines."""
    import ijson
    head = f.read(1 << 16)
    f.seek(0)
    stripped = head.lstrip()
    first = stripped[:1]
    if first == b"[":
        for i, rec in enumerate(ijson.items(f, "item", use_float=True)):
            yield i, rec
        return
    if first == b"{":
        nl = stripped.find(b"\n")
        line = stripped[:nl] if nl >= 0 else stripped
        try:
            json.loads(line)
            is_jsonl = nl >= 0 and len(stripped[nl:].strip()) > 0
        except ValueError:
            is_jsonl = False
        if not is_jsonl:
            for k, rec in ijson.kvitems(f, "", use_float=True):
                yield k, rec
            return
    for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
        line = line.strip()
        if line:
            yield i, json.loads(line)


def open_source(cfg, rel_path):
    """Opens a dataset file either from local_data_dir or straight from the HF Hub."""
    if cfg.local_data_dir:
        return open(os.path.join(cfg.local_data_dir, rel_path), "rb")
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    return fs.open(f"datasets/{cfg.hf_repo}/{rel_path}", "rb", block_size=8 << 20)


def iter_source_docs(cfg, rel_path):
    with open_source(cfg, rel_path) as f:
        for key, rec in iter_json_records(f):
            text = record_text(rec)
            if text and text.strip():
                yield key, text


def load_json_file(cfg, rel_path):
    with open_source(cfg, rel_path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Tokenizer: byte-level BPE trained on the Phase 1 corpus (own vocabulary, 32k)
# --------------------------------------------------------------------------------------
def train_tokenizer(texts, vocab_size, out_path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    for i, s in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(s) == i, (s, tok.token_to_id(s))
    tok.save(out_path)
    return tok


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


# --------------------------------------------------------------------------------------
# Stage 1: prepare (tokenizer + uint16 token bins + held-out pseudo-test set)
# --------------------------------------------------------------------------------------
def prepare(cfg):
    os.makedirs(cfg.data_dir, exist_ok=True)
    tok_path = os.path.join(cfg.data_dir, "tokenizer.json")
    t0 = time.time()
    total_mix = sum(cfg.mix.values())
    mix = {k: v / total_mix for k, v in cfg.mix.items()}

    if not os.path.exists(tok_path):
        def sample_texts():
            for src, frac in mix.items():
                budget, seen = int(cfg.tokenizer_sample_chars * frac), 0
                for idx, (_, text) in enumerate(iter_source_docs(cfg, cfg.sources[src])):
                    if idx % cfg.val_every_doc == 0:
                        continue   # never fit the tokenizer on validation documents
                    yield text
                    seen += len(text)
                    if seen >= budget:
                        break
                log0(f"[tokenizer] {src}: {seen/1e6:.1f}M chars sampled")
        log0(f"[tokenizer] training byte-level BPE, vocab {cfg.tokenizer_vocab}")
        train_tokenizer(sample_texts(), cfg.tokenizer_vocab, tok_path)
        log0(f"[tokenizer] saved {tok_path} ({time.time()-t0:.0f}s)")
    tok = load_tokenizer(tok_path)
    assert tok.get_vocab_size() < 65535

    meta = {"vocab_size": tok.get_vocab_size(), "sources": {}, "mix": mix}
    pseudo_prompts, pseudo_refs = {}, {}
    val_f = open(os.path.join(cfg.data_dir, "val.bin"), "wb")
    for src, frac in mix.items():
        quota = int(cfg.prepare_train_tokens * frac)
        vquota = int(cfg.val_tokens * frac)
        n_train = n_val = n_docs = n_pseudo = 0
        chars = 0
        pseudo_cap = max(1, round(cfg.n_pseudo_test * frac))
        train_f = open(os.path.join(cfg.data_dir, f"train_{src}.bin"), "wb")
        batch, batch_is_val = [], []

        def flush():
            nonlocal n_train, n_val, chars
            encs = tok.encode_batch(batch)
            for text, enc, is_val in zip(batch, encs, batch_is_val):
                ids = enc.ids + [EOS]
                arr = np.asarray(ids, dtype=np.uint16)
                if is_val:
                    if n_val < vquota:
                        arr = arr[:vquota - n_val]
                        arr.tofile(val_f)
                        n_val += len(arr)
                elif n_train < quota:
                    frac_kept = min(1.0, (quota - n_train) / len(arr))
                    arr = arr[:quota - n_train]
                    arr.tofile(train_f)
                    n_train += len(arr)
                    chars += int(len(text) * frac_kept)
            batch.clear()
            batch_is_val.clear()

        for idx, (key, text) in enumerate(iter_source_docs(cfg, cfg.sources[src])):
            is_val = idx % cfg.val_every_doc == 0
            if is_val:
                words = text.split()
                if len(words) >= 640 and n_pseudo < pseudo_cap:
                    n_pseudo += 1
                    pid = f"{src}_{key}"
                    pseudo_prompts[pid] = {"text": " ".join(words[:512])}
                    pseudo_refs[pid] = {"text": " ".join(words[512:640])}
            batch.append(text)
            batch_is_val.append(is_val)
            n_docs += 1
            if len(batch) >= 256:
                flush()
            if n_train >= quota and n_val >= vquota:
                break
        if batch:
            flush()
        train_f.close()
        meta["sources"][src] = {"train_tokens": n_train, "val_tokens": n_val, "docs_read": n_docs,
                                "chars_per_token": round(chars / max(n_train, 1), 3),
                                "quota": quota, "short": n_train < quota}
        log0(f"[prepare] {src}: {n_train/1e6:.1f}M train tokens, {n_val/1e6:.2f}M val, "
             f"{n_docs} docs, {chars/max(n_train,1):.2f} chars/token ({time.time()-t0:.0f}s)")
    val_f.close()
    with open(os.path.join(cfg.data_dir, "pseudo_test.json"), "w") as f:
        json.dump(pseudo_prompts, f, ensure_ascii=False)
    with open(os.path.join(cfg.data_dir, "pseudo_truth.json"), "w") as f:
        json.dump(pseudo_refs, f, ensure_ascii=False)
    meta["pseudo_test"] = len(pseudo_prompts)
    with open(os.path.join(cfg.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log0(json.dumps(meta, indent=2))
    return meta


class TokenData:
    """Random fixed-length windows from the uint16 bins (memory-mapped)."""

    def __init__(self, data_dir, split):
        if split == "train":
            paths = sorted(p for p in os.listdir(data_dir) if p.startswith("train_") and p.endswith(".bin"))
        else:
            paths = ["val.bin"]
        self.arrays = [np.memmap(os.path.join(data_dir, p), dtype=np.uint16, mode="r") for p in paths]
        self.arrays = [a for a in self.arrays if len(a) > 0]
        self.lens = np.array([len(a) for a in self.arrays], dtype=np.float64)
        assert len(self.arrays), f"no {split} tokens in {data_dir}"

    @property
    def n_tokens(self):
        return int(self.lens.sum())

    def sample(self, rng, batch, length):
        p = self.lens / self.lens.sum()
        out = np.empty((batch, length), dtype=np.int64)
        for b in range(batch):
            a = self.arrays[rng.choice(len(self.arrays), p=p)]
            start = rng.integers(0, len(a) - length)
            out[b] = a[start:start + length]
        return torch.from_numpy(out)


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def rope_tables(pos, head_dim, base):
    """pos: (B, T) long -> cos, sin of shape (B, 1, T, head_dim/2)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=pos.device).float() / head_dim))
    f = pos.float()[..., None] * inv
    return f.cos()[:, None], f.sin()[:, None]


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)


class Attention(nn.Module):
    """Multi-head softmax attention (self or cross) with an optional KV cache."""

    def __init__(self, d, n_head, dropout=0.0, cross=False):
        super().__init__()
        self.h, self.hd, self.cross, self.dropout = n_head, d // n_head, cross, dropout
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def _split(self, t):
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.hd).transpose(1, 2)

    def forward(self, x, rope=None, mask=None, causal=False, mem=None, cache=None):
        B, T, _ = x.shape
        q = self._split(self.wq(x))
        if self.cross:
            if cache is not None and "k" in cache:
                k, v = cache["k"], cache["v"]
            else:
                k, v = self._split(self.wk(mem)), self._split(self.wv(mem))
                if cache is not None:
                    cache["k"], cache["v"] = k, v
        else:
            k, v = self._split(self.wk(x)), self._split(self.wv(x))
            if rope is not None:
                q, k = apply_rope(q, *rope), apply_rope(k, *rope)
            if cache is not None:
                if "k" in cache:
                    k = torch.cat([cache["k"], k], 2)
                    v = torch.cat([cache["v"], v], 2)
                cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal and mask is None,
            dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__()
        d = cfg.d_model
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, cfg.n_head, cfg.dropout)
        self.has_cross = cross
        if cross:
            self.nc = RMSNorm(d)
            self.xattn = Attention(d, cfg.n_head, cfg.dropout, cross=True)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, mask=None, causal=False, mem=None, mem_mask=None, cache=None):
        sc = None if cache is None else cache.setdefault("self", {})
        x = x + self.drop(self.attn(self.n1(x), rope, mask, causal, cache=sc))
        if self.has_cross:
            cc = None if cache is None else cache.setdefault("cross", {})
            x = x + self.drop(self.xattn(self.nc(x), mask=mem_mask, mem=mem, cache=cc))
        return x + self.mlp(self.n2(x))


def run_blocks(blocks, x, grad_ckpt, *args, caches=None, **kw):
    for i, b in enumerate(blocks):
        if grad_ckpt and b.training and caches is None:
            x = torch.utils.checkpoint.checkpoint(lambda t, b=b: b(t, *args, **kw), x, use_reentrant=False)
        else:
            x = b(x, *args, cache=None if caches is None else caches[i], **kw)
    return x


def init_weights(model, n_residual_layers):
    """GPT-2 style init: N(0, 0.02), output projections scaled by 1/sqrt(2L) [2]."""
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            std = 0.02
            if name.endswith(("wo.weight", "w2.weight")):
                std = 0.02 / math.sqrt(2 * n_residual_layers)
            nn.init.normal_(p, 0.0, std)


def left_pad(seqs, pad=PAD):
    P = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), P), pad, dtype=torch.long)
    valid = torch.zeros((len(seqs), P), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, P - len(s):] = torch.tensor(s, dtype=torch.long)
            valid[i, P - len(s):] = True
    return ids, valid


def positions_from_valid(valid):
    return (valid.long().cumsum(1) - 1).clamp(min=0)


def ce(logits, labels, per_token=False):
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=-100, reduction="none" if per_token else "mean")
    return loss.view(labels.shape) if per_token else loss


# --------------------------------------------------------------------------------------
# Architecture A (gpt) and C (unilm): shared Transformer stack
# --------------------------------------------------------------------------------------
class DecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.unilm = cfg.arch == "unilm"
        if self.unilm:
            # mode 0 = causal (left-to-right) LM, mode 1 = bidirectional (MLM / encoder use)
            self.mode_emb = nn.Embedding(2, cfg.d_model)
            self.mlm_transform = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                                               RMSNorm(cfg.d_model))
        init_weights(self, cfg.n_layer)
        self.lm_head.weight = self.emb.weight   # tied

    def hidden(self, ids, pos=None, mask=None, causal=True, mode=0, caches=None):
        """Final hidden states. Phase 2-7 use this as the feature extractor
        (unilm: mode=1, causal=False gives bidirectional features)."""
        B, T = ids.shape
        x = self.emb(ids)
        if self.unilm:
            x = x + self.mode_emb.weight[mode]
        if pos is None:
            pos = torch.arange(T, device=ids.device)[None].expand(B, T)
        rope = rope_tables(pos, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base)
        x = run_blocks(self.blocks, x, self.cfg.grad_ckpt, rope, mask, causal, caches=caches)
        return self.norm(x)

    def forward(self, input_ids, labels, mode=0, per_token=False):
        if mode == 0:
            return ce(self.lm_head(self.hidden(input_ids, causal=True, mode=0)), labels, per_token)
        h = self.hidden(input_ids, causal=False, mode=1)
        sel = labels != -100
        return ce(self.lm_head(self.mlm_transform(h[sel])), labels[sel])

    @torch.no_grad()
    def prefill(self, ids, valid):
        B, P = ids.shape
        pos = positions_from_valid(valid)
        eye = torch.eye(P, dtype=torch.bool, device=ids.device)
        causal = torch.ones(P, P, dtype=torch.bool, device=ids.device).tril()
        mask = (causal & (valid[:, None, :] | eye))[:, None]
        caches = [{} for _ in self.blocks]
        h = self.hidden(ids, pos, mask, causal=False, mode=0, caches=caches)
        return self.lm_head(h[:, -1]), {"caches": caches, "valid": valid, "pos": pos[:, -1]}

    @torch.no_grad()
    def step(self, tok, st):
        st["pos"] = st["pos"] + 1
        st["valid"] = torch.cat([st["valid"], torch.ones_like(st["valid"][:, :1])], 1)
        mask = st["valid"][:, None, None, :]
        h = self.hidden(tok[:, None], st["pos"][:, None], mask, causal=False, mode=0, caches=st["caches"])
        return self.lm_head(h[:, -1]), st


# --------------------------------------------------------------------------------------
# Architecture B (encdec): encoder-decoder Transformer
# --------------------------------------------------------------------------------------
class EncDecLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.enc = nn.ModuleList(Block(cfg) for _ in range(cfg.n_enc_layer))
        self.enc_norm = RMSNorm(cfg.d_model)
        self.dec = nn.ModuleList(Block(cfg, cross=True) for _ in range(cfg.n_dec_layer))
        self.dec_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_weights(self, cfg.n_enc_layer + cfg.n_dec_layer)
        self.lm_head.weight = self.emb.weight

    def _rope(self, pos):
        return rope_tables(pos, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base)

    def encode(self, ids, valid=None):
        """Bidirectional encoder states (Phase 2-7 feature extractor)."""
        B, S = ids.shape
        if valid is None or bool(valid.all()):
            pos, mask = torch.arange(S, device=ids.device)[None].expand(B, S), None
        else:
            pos, mask = positions_from_valid(valid), valid[:, None, None, :]
        x = run_blocks(self.enc, self.emb(ids), self.cfg.grad_ckpt, self._rope(pos), mask, False)
        return self.enc_norm(x)

    def decode(self, dec_ids, mem, mem_valid, pos=None, caches=None):
        B, T = dec_ids.shape
        if pos is None:
            pos = torch.arange(T, device=dec_ids.device)[None].expand(B, T)
        mem_mask = None if mem_valid is None or bool(mem_valid.all()) else mem_valid[:, None, None, :]
        causal = caches is None or "self" not in caches[0]
        x = run_blocks(self.dec, self.emb(dec_ids), self.cfg.grad_ckpt, self._rope(pos), None,
                       causal and T > 1, mem=mem, mem_mask=mem_mask, caches=caches)
        return self.lm_head(self.dec_norm(x))

    def forward(self, enc_ids, enc_valid, dec_in, labels, per_token=False):
        mem = self.encode(enc_ids, enc_valid)
        return ce(self.decode(dec_in, mem, enc_valid), labels, per_token)

    @torch.no_grad()
    def prefill(self, ids, valid):
        mem = self.encode(ids, valid)
        caches = [{} for _ in self.dec]
        start = torch.full((ids.size(0), 1), SEP, dtype=torch.long, device=ids.device)
        pos = torch.zeros_like(start)
        logits = self.decode(start, mem, valid, pos, caches)
        return logits[:, -1], {"caches": caches, "mem": mem, "valid": valid, "pos": pos[:, 0]}

    @torch.no_grad()
    def step(self, tok, st):
        st["pos"] = st["pos"] + 1
        logits = self.decode(tok[:, None], st["mem"], st["valid"], st["pos"][:, None], st["caches"])
        return logits[:, -1], st


# --------------------------------------------------------------------------------------
# Architecture D (retnet): Retentive Network [9]
# --------------------------------------------------------------------------------------
class Retention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, H = cfg.d_model, cfg.n_head
        self.h, self.hd = H, d // H
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.gn_weight = nn.Parameter(torch.ones(d))
        # multi-scale decay, gamma_h = 1 - 2^(-5-h) as in RetNet
        self.register_buffer("log_gamma", torch.log(1 - 2.0 ** (-5.0 - torch.arange(H).float())),
                             persistent=False)
        self._mask_cache = {}

    def decay_mask(self, T, device):
        key = (T, str(device))
        if key not in self._mask_cache:
            i = torch.arange(T, device=device).float()
            diff = i[:, None] - i[None, :]
            D = torch.exp(self.log_gamma.to(device)[:, None, None] * diff.clamp(min=0))
            D = D * (diff >= 0)
            D = D / D.sum(-1, keepdim=True).sqrt()
            self._mask_cache = {key: D}
        return self._mask_cache[key]

    def _qkv(self, x, rope):
        B, T, _ = x.shape
        sp = lambda t: t.view(B, T, self.h, self.hd).transpose(1, 2)
        q, k, v = sp(self.wq(x)), sp(self.wk(x)), sp(self.wv(x))
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return q, k * self.hd ** -0.5, v

    def _out(self, x, o):
        # per-head normalisation (GroupNorm with one group per head). It is invariant to a
        # positive per-row scale, which is what makes the parallel form (normalised decay) and
        # the recurrent form (unnormalised state) agree. Dividing by the detached row max first
        # keeps the eps from breaking that invariance when activations are tiny.
        B, T = x.shape[:2]
        o = o.float()
        o = o / o.detach().abs().amax(-1, keepdim=True).clamp(min=1e-20)
        o = F.layer_norm(o, (self.hd,), eps=1e-6)
        o = o.transpose(1, 2).reshape(B, T, -1) * self.gn_weight.float()
        return self.wo(F.silu(self.wg(x)) * o.type_as(x))

    def forward(self, x, rope, valid=None, return_state=False):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, rope)
        if valid is not None:
            m = valid[:, None, :, None].to(k.dtype)
            k, v = k * m, v * m
        D = self.decay_mask(T, x.device)
        qk = (q @ k.transpose(-1, -2)) * D.to(q.dtype)
        qk = qk / qk.detach().abs().sum(-1, keepdim=True).clamp(min=1)
        o = qk @ v
        state = None
        if return_state:
            w = torch.exp(self.log_gamma[:, None] * (T - 1 - torch.arange(T, device=x.device)).float())
            state = torch.einsum("bhtk,bhtv,ht->bhkv", k.float(), v.float(), w)
        return self._out(x, o), state

    def step(self, x, rope, S):
        q, k, v = self._qkv(x, rope)
        S = torch.exp(self.log_gamma)[None, :, None, None] * S + k.float().transpose(-1, -2) @ v.float()
        return self._out(x, q.float() @ S), S


class RetBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.ret = Retention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, valid=None, return_state=False):
        y, st = self.ret(self.n1(x), rope, valid, return_state)
        x = x + self.drop(y)
        return x + self.mlp(self.n2(x)), st

    def step(self, x, rope, S):
        y, S = self.ret.step(self.n1(x), rope, S)
        x = x + y
        return x + self.mlp(self.n2(x)), S


class RetNetLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(RetBlock(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_weights(self, cfg.n_layer)
        self.lm_head.weight = self.emb.weight

    def _rope(self, pos):
        return rope_tables(pos, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base)

    def hidden(self, ids, valid=None, return_states=False):
        B, T = ids.shape
        pos = (torch.arange(T, device=ids.device)[None].expand(B, T) if valid is None
               else positions_from_valid(valid))
        rope = self._rope(pos)
        x, states = self.emb(ids), []
        for b in self.blocks:
            if self.cfg.grad_ckpt and self.training:
                x, st = torch.utils.checkpoint.checkpoint(b, x, rope, valid, False, use_reentrant=False)
            else:
                x, st = b(x, rope, valid, return_states)
            states.append(st)
        return self.norm(x), states, pos

    def forward(self, input_ids, labels, per_token=False):
        h, _, _ = self.hidden(input_ids)
        return ce(self.lm_head(h), labels, per_token)

    @torch.no_grad()
    def prefill(self, ids, valid):
        h, states, pos = self.hidden(ids, valid, return_states=True)
        return self.lm_head(h[:, -1]), {"S": states, "pos": pos[:, -1]}

    @torch.no_grad()
    def step(self, tok, st):
        st["pos"] = st["pos"] + 1
        rope = self._rope(st["pos"][:, None])
        x = self.emb(tok[:, None])
        for i, b in enumerate(self.blocks):
            x, st["S"][i] = b.step(x, rope, st["S"][i])
        return self.lm_head(self.norm(x)[:, -1]), st


def build_model(mcfg: ModelConfig):
    return {"gpt": DecoderLM, "unilm": DecoderLM, "encdec": EncDecLM, "retnet": RetNetLM}[mcfg.arch](mcfg)


def count_params(model):
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n


# --------------------------------------------------------------------------------------
# Objectives: turning raw token windows into per-architecture batches
# --------------------------------------------------------------------------------------
def random_spans_mask(length, density, mean_span, rng):
    """T5 random_spans_noise_mask [6]: boolean mask with ~density of tokens in spans."""
    num_noise = max(1, int(round(length * density)))
    num_spans = max(1, min(N_SENTINELS - 1, int(round(num_noise / mean_span))))
    num_nonnoise = length - num_noise

    def partition(n_items, n_segs):
        cuts = np.sort(rng.choice(np.arange(1, n_items), n_segs - 1, replace=False)) if n_segs > 1 else []
        return np.diff(np.concatenate([[0], cuts, [n_items]])).astype(int)

    noise, keep = partition(num_noise, num_spans), partition(num_nonnoise, num_spans)
    mask, pos = np.zeros(length, dtype=bool), 0
    for a, b in zip(keep, noise):
        pos += a
        mask[pos:pos + b] = True
        pos += b
    return mask


def make_batch(arch, raw, rng, mcfg, train=True, vocab_size=None):
    """raw: (B, max_len + 1) token windows. Returns (kwargs for model.forward, tag)."""
    T = mcfg.max_len
    if arch in ("gpt", "retnet"):
        return {"input_ids": raw[:, :T], "labels": raw[:, 1:T + 1]}, "clm"
    if arch == "unilm":
        if not train or rng.random() < 0.5:
            return {"input_ids": raw[:, :T], "labels": raw[:, 1:T + 1], "mode": 0}, "clm"
        x = raw[:, :T].clone()
        labels = torch.full_like(x, -100)
        sel = torch.from_numpy(rng.random(x.shape) < 0.15)
        labels[sel] = x[sel]
        r = torch.from_numpy(rng.random(x.shape))
        x[sel & (r < 0.8)] = MASK
        rand_pos = sel & (r >= 0.8) & (r < 0.9)
        x[rand_pos] = torch.from_numpy(
            rng.integers(N_SPECIAL, vocab_size or mcfg.vocab_size, size=int(rand_pos.sum()))).long()
        return {"input_ids": x, "labels": labels, "mode": 1}, "mlm"
    if arch == "encdec":
        E, Dl = mcfg.enc_len, mcfg.dec_len
        B = raw.size(0)
        if not train or rng.random() < 0.5:   # prefix-LM continuation (= the Phase 1 task)
            labels = raw[:, E:E + Dl]
            dec_in = torch.cat([torch.full((B, 1), SEP, dtype=torch.long), labels[:, :-1]], 1)
            return {"enc_ids": raw[:, :E], "enc_valid": torch.ones(B, E, dtype=torch.bool),
                    "dec_in": dec_in, "labels": labels}, "prefix"
        enc_ids = torch.full((B, E), PAD, dtype=torch.long)
        enc_valid = torch.zeros(B, E, dtype=torch.bool)
        labels = torch.full((B, Dl), -100, dtype=torch.long)
        dec_in = torch.full((B, Dl), PAD, dtype=torch.long)
        for b in range(B):
            toks = raw[b, :E].numpy()
            m = random_spans_mask(E, 0.15, 3.0, rng)
            enc, tgt, sid, prev = [], [], 0, False
            for t, is_noise in zip(toks, m):
                if is_noise:
                    if not prev:
                        enc.append(SENTINEL0 + sid)
                        tgt.append(SENTINEL0 + sid)
                        sid += 1
                    tgt.append(int(t))
                else:
                    enc.append(int(t))
                prev = is_noise
            tgt = (tgt + [EOS])[:Dl]
            enc_ids[b, :len(enc)] = torch.tensor(enc)
            enc_valid[b, :len(enc)] = True
            labels[b, :len(tgt)] = torch.tensor(tgt)
            dec_in[b, 0] = BOS
            dec_in[b, 1:len(tgt)] = torch.tensor(tgt[:-1])
        return {"enc_ids": enc_ids, "enc_valid": enc_valid, "dec_in": dec_in, "labels": labels}, "span"
    raise ValueError(arch)


# --------------------------------------------------------------------------------------
# Stage 2: training (single GPU, DDP via torchrun, or CPU for smoke tests)
# --------------------------------------------------------------------------------------
def setup_device():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return device, rank, world


def amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def lr_at(step, total, cfg, start=0):
    """Warm-up + cosine. start > 0 means a finished run is being extended: a fresh cycle from step
    start that re-warms to lr * extend_peak_frac and decays to the same floor by the new total."""
    peak = cfg.lr if start == 0 else cfg.lr * cfg.extend_peak_frac
    floor = cfg.lr * cfg.min_lr_frac
    step, total = step - start, total - start
    if step < cfg.warmup_steps:
        return floor * (start > 0) + (peak - floor * (start > 0)) * (step + 1) / cfg.warmup_steps
    prog = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * prog))


def to_dev(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, val, cfg, mcfg, device, dtype):
    """Validation loss on fixed windows. `tail` = loss on the last dec_len tokens given the
    preceding enc_len tokens, which is the one number comparable across all four archs."""
    model.eval()
    rng = np.random.default_rng(1234)
    tot = tail = 0.0
    for _ in range(cfg.eval_batches):
        raw = val.sample(rng, cfg.micro_batch, mcfg.max_len + 1)
        batch, _ = make_batch(cfg.arch, raw, rng, mcfg, train=False)
        batch = to_dev(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            per_tok = model(**batch, per_token=True)
        tot += per_tok.mean().item()
        if cfg.arch == "encdec":
            tail += per_tok.mean().item()
        else:
            E, Dl = mcfg.enc_len, mcfg.dec_len
            tail += per_tok[:, E - 1:E - 1 + Dl].mean().item()
    model.train()
    return tot / cfg.eval_batches, tail / cfg.eval_batches


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def export_weights(raw_model, mcfg, path, extra=None):
    sd = {k: v.detach().to("cpu", torch.float16 if v.is_floating_point() else v.dtype)
          for k, v in raw_model.state_dict().items()}
    save_atomic({"model_config": asdict(mcfg), "state_dict": sd, **(extra or {})}, path)


def load_model(path, device="cpu"):
    """Loads an exported model (model_final.pt / model_best.pt) or a training last.pt."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    model = build_model(mcfg)
    sd = ck.get("state_dict") or ck["model"]
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return model.to(device).eval(), mcfg


def hf_push(cfg, files):
    if not cfg.hf_push_repo or int(os.environ.get("RANK", "0")) != 0:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(cfg.hf_push_repo, private=True, exist_ok=True)
        for f in files:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"{cfg.run_name}/{os.path.basename(f)}",
                            repo_id=cfg.hf_push_repo)
    except Exception as e:   # never let an upload failure kill a training run
        log0(f"[hf_push] failed: {e}")


def train(cfg):
    device, rank, world = setup_device()
    seed_everything(cfg.seed + rank)
    dtype = amp_dtype(device)
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.data_dir, "meta.json")) as f:
        vocab = json.load(f)["vocab_size"]
    mcfg = cfg.model_config(vocab)
    model = build_model(mcfg).to(device)
    n_params = count_params(model)
    log0(f"[train] arch={cfg.arch} run={cfg.run_name} params={n_params/1e6:.2f}M "
         f"device={device} world={world} amp={dtype}")
    if not cfg.smoke:
        assert 50e6 <= n_params <= 300e6, "Phase 1 rule: 50M-300M parameters"
    train_data, val_data = TokenData(cfg.data_dir, "train"), TokenData(cfg.data_dir, "val")
    log0(f"[train] train bin tokens={train_data.n_tokens/1e6:.1f}M val={val_data.n_tokens/1e6:.2f}M")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, cfg.beta2), eps=1e-8,
                            fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    per_micro = cfg.micro_batch * cfg.seq_len * world
    accum = max(1, cfg.batch_tokens // per_micro)
    tokens_per_step = per_micro * accum
    total_steps = math.ceil(cfg.train_tokens / tokens_per_step)
    step, best_val, history = 0, float("inf"), []
    sched_start = 0

    resume = cfg.resume_from or os.path.join(cfg.run_dir, "last.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val, history = ck["step"], ck["best_val"], ck.get("history", [])
        log0(f"[train] resumed from {resume} at step {step}")
        sched_start = ck.get("sched_start", 0)
        if ck.get("total_steps") and ck["total_steps"] != total_steps:
            if total_steps > ck["total_steps"] and step >= ck["total_steps"]:
                sched_start = step   # extending a finished run: start a new lr cycle here
                log0(f"[train] extending finished run: {ck['total_steps']} -> {total_steps} steps, "
                     f"re-warming to lr {cfg.lr * cfg.extend_peak_frac:.2e}")
            else:
                log0(f"[train] warning: total_steps changed {ck['total_steps']} -> {total_steps}")

    raw_model = model
    if cfg.compile:
        model = torch.compile(model)
    if world > 1:
        # unilm's MLM head is unused on causal micro-batches, hence find_unused_parameters
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=cfg.arch == "unilm")
    log0(f"[train] accum={accum} tokens/step={tokens_per_step} total_steps={total_steps} "
         f"(~{total_steps*tokens_per_step/1e6:.0f}M tokens)")

    def checkpoint(final=False):
        if rank != 0:
            return
        state = {"model": raw_model.state_dict(), "optim": opt.state_dict(), "scaler": scaler.state_dict(),
                 "step": step, "best_val": best_val, "history": history, "total_steps": total_steps,
                 "sched_start": sched_start,
                 "model_config": asdict(mcfg), "run_config": asdict(cfg)}
        save_atomic(state, os.path.join(cfg.run_dir, "last.pt"))
        with open(os.path.join(cfg.run_dir, "history.json"), "w") as f:
            json.dump(history, f)
        files = [os.path.join(cfg.run_dir, "last.pt")]
        if final:
            export_weights(raw_model, mcfg, os.path.join(cfg.run_dir, "model_final.pt"),
                           {"step": step, "params": n_params})
            files.append(os.path.join(cfg.run_dir, "model_final.pt"))
        hf_push(cfg, files)

    t_start, t_log = time.time(), time.time()
    model.train()
    stopped_early = False
    while step < total_steps:
        lr = lr_at(step, total_steps, cfg, sched_start)
        for g in opt.param_groups:
            g["lr"] = lr
        losses = {}
        for micro in range(accum):
            rng = np.random.default_rng([cfg.seed, step, micro, rank])
            raw = train_data.sample(rng, cfg.micro_batch, mcfg.max_len + 1)
            batch, tag = make_batch(cfg.arch, raw, rng, mcfg, train=True, vocab_size=vocab)
            batch = to_dev(batch, device)
            sync = micro == accum - 1 or world == 1
            ctx = model.no_sync() if (world > 1 and not sync) else _nullctx()
            with ctx:
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    loss = model(**batch)
                scaler.scale(loss / accum).backward()
            losses.setdefault(tag, []).append(loss.item())
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == total_steps:
            dt = time.time() - t_log
            t_log = time.time()
            rec = {"step": step, "lr": lr, "gnorm": round(gnorm, 3),
                   "tokens": step * tokens_per_step,
                   "tok_per_s": round(cfg.log_every * tokens_per_step / max(dt, 1e-9)),
                   "elapsed_h": round((time.time() - t_start) / 3600, 3),
                   **{f"loss_{k}": round(float(np.mean(v)), 4) for k, v in losses.items()}}
            history.append(rec)
            log0(json.dumps(rec))
        if step % cfg.eval_every == 0 or step == total_steps:
            vl, vt = evaluate(raw_model, val_data, cfg, mcfg, device, dtype)
            history.append({"step": step, "val_loss": vl, "val_ppl": math.exp(min(vl, 20)),
                            "val_tail_loss": vt})
            log0(f"[eval] step {step}: val_loss={vl:.4f} ppl={math.exp(min(vl, 20)):.2f} tail={vt:.4f}")
            if vl < best_val and rank == 0:
                best_val = vl
                export_weights(raw_model, mcfg, os.path.join(cfg.run_dir, "model_best.pt"),
                               {"step": step, "val_loss": vl, "params": n_params})
        if step % cfg.save_every == 0:
            checkpoint()
        if (time.time() - t_start) / 3600 > cfg.max_hours:
            log0(f"[train] time budget {cfg.max_hours}h reached at step {step}; checkpointing. "
                 f"Re-run with resume_from pointing at last.pt to continue.")
            stopped_early = True
            break
    checkpoint(final=not stopped_early)
    if rank == 0:
        plot_history(history, os.path.join(cfg.run_dir, "loss_curve.png"), cfg.run_name)
    if world > 1:
        torch.distributed.destroy_process_group()
    return {"step": step, "best_val": best_val, "params": n_params, "finished": not stopped_early}


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def plot_history(history, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(9, 4.5))
    for key in ("loss_clm", "loss_mlm", "loss_prefix", "loss_span"):
        pts = [(h["step"], h[key]) for h in history if key in h]
        if pts:
            plt.plot(*zip(*pts), label=f"train {key[5:]}")
    pts = [(h["step"], h["val_loss"]) for h in history if "val_loss" in h]
    if pts:
        plt.plot(*zip(*pts), "o-", label="val")
    pts = [(h["step"], h["val_tail_loss"]) for h in history if "val_tail_loss" in h]
    if pts:
        plt.plot(*zip(*pts), "s--", label="val tail (comparable)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"Phase 1 {title}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------------------
# Stage 3: generation -> prediction.json -> submission zip
# --------------------------------------------------------------------------------------
class NgramBlocker:
    """Incremental no-repeat-ngram constraint over the generated tokens."""

    def __init__(self, n, batch):
        self.n, self.seen, self.hist = n, [dict() for _ in range(batch)], [[] for _ in range(batch)]

    def banned(self, b):
        if self.n <= 0 or len(self.hist[b]) < self.n - 1:
            return ()
        return self.seen[b].get(tuple(self.hist[b][len(self.hist[b]) - self.n + 1:]), ())

    def add(self, b, tok):
        h = self.hist[b]
        h.append(tok)
        if self.n > 0 and len(h) >= self.n:
            self.seen[b].setdefault(tuple(h[-self.n:-1]), set()).add(h[-1])


def sample_next(logits, generated, blocker, cfg, gen):
    logits = logits.float()
    logits[:, :N_SPECIAL] = -float("inf")    # no EOS/PAD/sentinels inside a continuation
    B = logits.size(0)
    if cfg.repetition_penalty != 1.0 and generated.numel():
        prev = torch.gather(logits, 1, generated)
        prev = torch.where(prev > 0, prev / cfg.repetition_penalty, prev * cfg.repetition_penalty)
        logits.scatter_(1, generated, prev)
    for b in range(B):
        ban = blocker.banned(b)
        if ban:
            logits[b, list(ban)] = -float("inf")
    if cfg.temperature <= 0:
        return logits.argmax(-1)
    logits = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(logits, min(cfg.top_k, logits.size(-1)), -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if 0 < cfg.top_p < 1:
        sl, si = torch.sort(logits, descending=True)
        cp = sl.softmax(-1).cumsum(-1)
        drop = cp - sl.softmax(-1) > cfg.top_p
        sl = sl.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(1)


@torch.no_grad()
def generate_continuations(model, mcfg, tok, prompts, cfg, device, dtype=None, seed=42):
    """prompts: list of str -> list of str (first gen_target_words words of the continuation)."""
    keep = mcfg.enc_len if mcfg.arch == "encdec" else max(mcfg.max_len - cfg.gen_max_new_tokens, mcfg.max_len // 2)
    encs = [e.ids[-keep:] for e in tok.encode_batch(prompts)]
    order = sorted(range(len(prompts)), key=lambda i: len(encs[i]))
    out = [""] * len(prompts)
    gen = torch.Generator(device=device).manual_seed(seed)
    for s in range(0, len(order), cfg.gen_batch):
        idx = order[s:s + cfg.gen_batch]
        ids, valid = left_pad([encs[i] for i in idx])
        ids, valid = ids.to(device), valid.to(device)
        blocker = NgramBlocker(cfg.no_repeat_ngram, len(idx))
        generated = torch.empty((len(idx), 0), dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits, st = model.prefill(ids, valid)
            for t in range(cfg.gen_max_new_tokens):
                nxt = sample_next(logits, generated, blocker, cfg, gen)
                generated = torch.cat([generated, nxt[:, None]], 1)
                for b, tkn in enumerate(nxt.tolist()):
                    blocker.add(b, tkn)
                if t >= cfg.gen_target_words and t % 16 == 0:
                    texts = tok.decode_batch(generated.tolist())
                    if all(len(x.split()) > cfg.gen_target_words for x in texts):
                        break
                if t < cfg.gen_max_new_tokens - 1:
                    logits, st = model.step(nxt, st)
        for b, text in zip(idx, tok.decode_batch(generated.tolist())):
            words = text.split()[:cfg.gen_target_words]
            out[b] = " ".join(words) if words else "the"
        log0(f"[generate] {min(s + cfg.gen_batch, len(order))}/{len(order)}")
    return out


def resolve_checkpoint(cfg):
    if cfg.gen_checkpoint in ("best", "final"):
        p = os.path.join(cfg.run_dir, f"model_{cfg.gen_checkpoint}.pt")
        if not os.path.exists(p):
            p = os.path.join(cfg.run_dir, "model_best.pt")
        return p
    return cfg.gen_checkpoint


def write_prediction(preds, path, zip_path):
    for k, v in preds.items():
        assert isinstance(v, dict) and isinstance(v.get("text"), str) and v["text"].strip(), k
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction.json")


def generate_stage(cfg, prompts_dict=None, out_name="prediction.json", zip_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(device)
    tok = load_tokenizer(os.path.join(cfg.data_dir, "tokenizer.json"))
    ckpt = resolve_checkpoint(cfg)
    model, mcfg = load_model(ckpt, device)
    log0(f"[generate] {ckpt}: {count_params(model)/1e6:.2f}M params, arch {mcfg.arch}")
    if prompts_dict is None:
        prompts_dict = load_json_file(cfg, cfg.test_file)
    items = list(prompts_dict.items())
    if cfg.max_test_examples:
        items = items[:cfg.max_test_examples]
    t0 = time.time()
    texts = generate_continuations(model, mcfg, tok, [record_text(r) for _, r in items], cfg, device, dtype, cfg.seed)
    preds = {str(k): {"text": t} for (k, _), t in zip(items, texts)}
    path = os.path.join(cfg.run_dir, out_name)
    zip_path = os.path.join(cfg.run_dir, zip_name or f"submission_{cfg.run_name}.zip")
    write_prediction(preds, path, zip_path)
    log0(f"[generate] wrote {len(preds)} predictions to {path} and {zip_path} ({time.time()-t0:.0f}s)")
    return preds


# Local metrics mirroring evaluate_phase1.py (ROUGE-L F1 and corpus BLEU-4 on words).
def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(p, r):
    pt, rt = p.split(), r.split()
    if not pt or not rt:
        return 0.0
    l = _lcs(pt, rt)
    if not l:
        return 0.0
    pp, rr = l / len(pt), l / len(rt)
    return 2 * pp * rr / (pp + rr)


def bleu4(preds, refs):
    from collections import Counter
    clipped, totals, pl, rl = [0] * 4, [0] * 4, 0, 0
    for p, r in zip(preds, refs):
        p, r = p.split(), r.split()
        pl, rl = pl + len(p), rl + len(r)
        for n in range(1, 5):
            pn = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
            rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            totals[n - 1] += sum(pn.values())
            clipped[n - 1] += sum(min(c, rn.get(g, 0)) for g, c in pn.items())
    prec = [c / t if t else 0.0 for c, t in zip(clipped, totals)]
    if min(prec) <= 0 or pl == 0:
        return 0.0
    bp = 1.0 if pl > rl else math.exp(1 - rl / pl)
    return bp * math.exp(sum(math.log(x) for x in prec) / 4)


def pseudo_eval(cfg, use_bertscore=False):
    """Scores the model on held-out validation documents (512-word prompt, 128-word ref)."""
    with open(os.path.join(cfg.data_dir, "pseudo_test.json")) as f:
        prompts = json.load(f)
    with open(os.path.join(cfg.data_dir, "pseudo_truth.json")) as f:
        truth = json.load(f)
    saved = cfg.max_test_examples
    cfg.max_test_examples = 0
    preds = generate_stage(cfg, prompts, out_name="pseudo_prediction.json", zip_name="pseudo_submission.zip")
    cfg.max_test_examples = saved
    ids = list(preds)
    res = {"n": len(ids),
           "rougeL_f1": float(np.mean([rouge_l(preds[i]["text"], truth[i]["text"]) for i in ids])),
           "bleu4": bleu4([preds[i]["text"] for i in ids], [truth[i]["text"] for i in ids])}
    if use_bertscore:
        import bert_score   # evaluation-only metric model, same as the official scorer
        _, _, f1 = bert_score.score([preds[i]["text"] for i in ids], [truth[i]["text"] for i in ids], lang="en")
        res["bertscore_f1"] = float(f1.mean())
    with open(os.path.join(cfg.run_dir, "pseudo_eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    log0(f"[pseudo_eval] {res}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "generate", "pseudo_eval"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_run_config(a.config)
    if a.stage == "prepare":
        prepare(cfg)
    elif a.stage == "train":
        train(cfg)
    elif a.stage == "generate":
        generate_stage(cfg)
    else:
        pseudo_eval(cfg)


if __name__ == "__main__":
    main()
