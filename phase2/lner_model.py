"""Phase 2 (L-NER) neural pipeline: tokenizer -> encoder -> BIO tagger."""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lner_utils as U

SEED = 42
TAGS = ["O"] + [f"{p}-{l}" for l in U.LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
IGNORE = -100

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>", "<sep>"] + [f"<extra_id_{i}>" for i in range(64)]


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_bpe(texts, out_path, vocab_size=32000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(list(texts), trainer=trainer)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tok.save(out_path)
    return out_path


class BPETokenizer:

    def __init__(self, path, split_punct=True):
        from tokenizers import Tokenizer
        self.path = path
        self.tok = Tokenizer.from_file(path)
        self.vocab_size = self.tok.get_vocab_size()
        self.pad_id = self.tok.token_to_id("<pad>") or 0
        self.split_punct = split_punct

    def encode(self, text):
        if not self.split_punct:
            enc = self.tok.encode(text, add_special_tokens=False)
            return enc.ids, enc.offsets
        pieces, starts, prev = [], [], 0
        for s, e in U.word_offsets(text):
            pieces.append(text[prev:e]); starts.append(prev); prev = e
        if not pieces:
            return [], []
        ids, offs = [], []
        for enc, base in zip(self.tok.encode_batch(pieces, add_special_tokens=False), starts):
            ids += enc.ids
            offs += [(base + a, base + b) for a, b in enc.offsets]
        out = []
        for a, b in offs:
            while a < b and text[a].isspace(): a += 1
            while b > a and text[b - 1].isspace(): b -= 1
            out.append((a, b))
        return ids, out


def token_tags(offsets, spans):
    keep = [i for i, (s, e) in enumerate(offsets) if e > s]
    tags, mis = U.spans_to_bio([offsets[i] for i in keep], spans)
    out = [IGNORE] * len(offsets)
    for i, t in zip(keep, tags):
        out[i] = TAG2ID[t]
    return out, len(mis)


def decode_tags(offsets, tag_ids):
    keep = [i for i, (s, e) in enumerate(offsets) if e > s]
    return U.bio_to_spans([offsets[i] for i in keep], [TAGS[tag_ids[i]] for i in keep])


def _allowed_transitions():
    n = len(TAGS)
    allowed = torch.ones(n, n, dtype=torch.bool)
    start = torch.ones(n, dtype=torch.bool)
    for j, t in enumerate(TAGS):
        if t.startswith("I-"):
            lab = t[2:]
            start[j] = False
            for i, u in enumerate(TAGS):
                allowed[i, j] = u in (f"B-{lab}", f"I-{lab}")
    return allowed, start


_ALLOWED, _START = _allowed_transitions()


def viterbi_bio(logp):
    T, n = logp.shape
    if T == 0:
        return []
    neg = torch.finfo(logp.dtype).min / 2
    trans = torch.where(_ALLOWED, torch.zeros(()), torch.full((), neg))
    score = torch.where(_START, logp[0], torch.full((), neg))
    back = torch.zeros(T, n, dtype=torch.long)
    for t in range(1, T):
        cand = score[:, None] + trans
        score, back[t] = cand.max(0)
        score = score + logp[t]
    best = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        best.append(int(back[t, best[-1]]))
    return best[::-1]


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, hidden=256, layers=4, heads=4, ff=1024, max_len=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.hidden_size = hidden
        self.tok = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.drop(self.tok(input_ids) + self.pos(pos)[None])
        x = self.layers(x, src_key_padding_mask=~attention_mask.bool())
        return self.norm(x)


class BiLSTMEncoder(nn.Module):

    def __init__(self, vocab_size, hidden=256, layers=2, emb=256, dropout=0.3, pad_id=0, **_):
        super().__init__()
        self.hidden_size = hidden
        self.tok = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb, hidden // 2, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, input_ids, attention_mask):
        x = self.drop(self.tok(input_ids))
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        return out


class CharBiLSTMEncoder(nn.Module):

    def __init__(self, vocab_size, tokenizer_path, hidden=384, layers=2, emb=256, char_emb=32,
                 char_filters=96, max_chars=16, dropout=0.3, pad_id=0, **_):
        super().__init__()
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        table = torch.zeros(vocab_size, max_chars, dtype=torch.long)
        for i in range(vocab_size):
            text = tok.decode([i], skip_special_tokens=False) or ""
            chars = [min(ord(c), 255) + 1 for c in text[:max_chars]]
            table[i, :len(chars)] = torch.tensor(chars, dtype=torch.long) if chars else table[i, :0]
        self.register_buffer("char_table", table, persistent=False)
        self.hidden_size = hidden
        self.tok = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.cemb = nn.Embedding(257, char_emb, padding_idx=0)
        self.conv = nn.Conv1d(char_emb, char_filters, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb + char_filters, hidden // 2, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        ch = self.char_table[input_ids].view(B * T, -1)
        c = self.conv(self.cemb(ch).transpose(1, 2))
        c = c.masked_fill((ch == 0)[:, None, :], -1e4).max(-1).values
        c = torch.where(c < -1e3, torch.zeros_like(c), c).view(B, T, -1)
        x = self.drop(torch.cat([self.tok(input_ids), c], -1))
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return out


class CallableEncoder(nn.Module):

    def __init__(self, module, hidden_size, fn):
        super().__init__()
        self.module, self.hidden_size, self.fn = module, hidden_size, fn

    def forward(self, input_ids, attention_mask):
        return self.fn(self.module, input_ids, attention_mask)


def load_phase1_encoder(module, ckpt_path, strip_prefix="", map_location="cpu"):
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if strip_prefix:
        sd = {k[len(strip_prefix):] if k.startswith(strip_prefix) else k: v for k, v in sd.items()}
    res = module.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt_path}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    return res


def phase1_encoder(ckpt_path, p1_dir=None, device="cpu"):
    if p1_dir:
        sys.path.insert(0, p1_dir)
    import cs781_p1 as P1
    model, mcfg = P1.load_model(ckpt_path, device)
    model.train()
    arch = mcfg.arch
    if arch == "gpt":
        fn = lambda m, ids, am: m.hidden(ids, causal=True, mode=0)
    elif arch == "unilm":
        fn = lambda m, ids, am: m.hidden(ids, mask=am.bool()[:, None, None, :], causal=False, mode=1)
    elif arch == "encdec":
        fn = lambda m, ids, am: m.encode(ids, am.bool())
    elif arch == "retnet":
        fn = lambda m, ids, am: m.hidden(ids, am.bool())[0]
    else:
        raise ValueError(f"unknown Phase 1 arch {arch!r}")
    enc = CallableEncoder(model, mcfg.d_model, fn)
    enc.arch = arch
    return enc


ENCODERS = {"transformer": TransformerEncoder, "bilstm": BiLSTMEncoder, "charbilstm": CharBiLSTMEncoder}


class Tagger(nn.Module):
    def __init__(self, encoder, n_tags=len(TAGS), dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.hidden_size, n_tags)

    def forward(self, input_ids, attention_mask):
        return self.head(self.drop(self.encoder(input_ids, attention_mask)))


def encode_docs(tokenizer, texts, gold=None):
    out, mis, total = {}, 0, 0
    for d, t in texts.items():
        ids, offs = tokenizer.encode(t)
        tags = None
        if gold is not None:
            spans = gold.get(d, [])
            tags, m = token_tags(offs, spans)
            mis += m; total += len(spans)
        out[d] = (ids, offs, tags)
    return out, (mis, total)


def make_windows(encoded, max_len, stride):
    ex = []
    for d, (ids, _, tags) in encoded.items():
        for lo, hi in U.windows(len(ids), max_len, stride):
            ex.append((ids[lo:hi], tags[lo:hi]))
    return ex


def collate(batch, pad_id):
    T = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    am = torch.zeros((len(batch), T), dtype=torch.long)
    for b, (i, t) in enumerate(batch):
        ids[b, :len(i)] = torch.tensor(i)
        lab[b, :len(t)] = torch.tensor(t)
        am[b, :len(i)] = 1
    return ids, am, lab


@torch.no_grad()
def predict_docs(model, encoded, max_len, stride, device, batch_size=16):
    return predict_ensemble([model], encoded, max_len, stride, device, batch_size)


@torch.no_grad()
def doc_logprobs(model, encoded, max_len, stride, device, batch_size=16):
    model.eval()
    jobs = []
    acc = {}
    for d, (ids, _, _) in encoded.items():
        acc[d] = (torch.zeros(len(ids), len(TAGS)), torch.zeros(len(ids), 1))
        jobs += [(d, lo, hi) for lo, hi in U.windows(len(ids), max_len, stride)]
    pad = getattr(model, "pad_id", 0)
    for k in range(0, len(jobs), batch_size):
        chunk = jobs[k:k + batch_size]
        batch = [(encoded[d][0][lo:hi], [IGNORE] * (hi - lo)) for d, lo, hi in chunk]
        ids, am, _ = collate(batch, pad)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
            logits = model(ids.to(device), am.to(device))
        logp = F.log_softmax(logits.float(), -1).cpu()
        for b, (d, lo, hi) in enumerate(chunk):
            s, c = acc[d]
            s[lo:hi] += logp[b, :hi - lo]
            c[lo:hi] += 1
    return {d: s / c.clamp(min=1) for d, (s, c) in acc.items()}


def predict_ensemble(models, encoded, max_len, stride, device, batch_size=16):
    total = None
    for m in models:
        lp = doc_logprobs(m, encoded, max_len, stride, device, batch_size)
        total = lp if total is None else {d: total[d] + lp[d] for d in total}
    return {d: decode_tags(encoded[d][1], viterbi_bio(total[d] / len(models))) for d in encoded}


def build_model(cfg, vocab_size, pad_id):
    if cfg["encoder"] == "phase1":
        enc = phase1_encoder(cfg["phase1_ckpt"], cfg.get("phase1_code_dir"))
    else:
        enc = ENCODERS[cfg["encoder"]](vocab_size, pad_id=pad_id, **cfg["encoder_args"])
    model = Tagger(enc, dropout=cfg.get("head_dropout", 0.1))
    model.pad_id = pad_id
    return model


def train(cfg, tokenizer, train_texts, train_gold, dev_texts=None, dev_gold=None, log=print, model=None):
    set_seed(cfg.get("seed", SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_train, (mis, tot) = encode_docs(tokenizer, train_texts, train_gold)
    log(f"train docs {len(enc_train)}, gold spans {tot}, not on token boundaries {mis}")
    examples = make_windows(enc_train, cfg["max_len"], cfg["stride"])
    enc_dev = encode_docs(tokenizer, dev_texts)[0] if dev_texts else None
    if model is None:
        model = build_model(cfg, tokenizer.vocab_size, tokenizer.pad_id)
    model.to(device)
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, windows {len(examples)}, device {device}")
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = [{"params": enc_params, "lr": cfg.get("encoder_lr", cfg["lr"])}, {"params": rest, "lr": cfg["lr"]}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.01))
    steps = cfg["epochs"] * math.ceil(len(examples) / cfg["batch_size"])
    warm = int(cfg.get("warmup", 0.06) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)) if s >= warm
        else (s + 1) / max(1, warm))
    weight = torch.ones(len(TAGS)); weight[0] = cfg.get("o_weight", 1.0)
    weight = weight.to(device)
    best, best_state, hist = -1.0, None, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); random.shuffle(examples); t0 = time.time(); tot_loss = 0.0
        for k in range(0, len(examples), cfg["batch_size"]):
            ids, am, lab = collate(examples[k:k + cfg["batch_size"]], tokenizer.pad_id)
            ids, am, lab = ids.to(device), am.to(device), lab.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                logits = model(ids, am)
            loss = F.cross_entropy(logits.float().view(-1, len(TAGS)), lab.view(-1), weight=weight, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot_loss += loss.item()
        row = {"epoch": ep, "loss": tot_loss / math.ceil(len(examples) / cfg["batch_size"]), "sec": time.time() - t0}
        if enc_dev is not None:
            macro, _ = U.strict_scores(dev_gold, predict_docs(model, enc_dev, cfg["max_len"], cfg["stride"], device))
            row["dev_f1"] = macro["F1"]
            if macro["F1"] > best:
                best = macro["F1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(row); log(json.dumps(row))
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


DEFAULT_CFG = {
    "encoder": "transformer",
    "encoder_args": {"hidden": 256, "layers": 4, "heads": 4, "ff": 1024, "max_len": 1024, "dropout": 0.1},
    "max_len": 512, "stride": 384, "batch_size": 16, "epochs": 20, "lr": 5e-4,
    "weight_decay": 0.01, "warmup": 0.06, "o_weight": 0.5, "seed": SEED,
}
BILSTM_ARGS = {"hidden": 384, "layers": 2, "emb": 256, "dropout": 0.3}


def _folds(data):
    p = lambda n: os.path.join(data, n)
    return (U.load_texts(p("fold1-text.json")), U.load_spans(p("fold1-gold.json")),
            U.load_texts(p("fold2-text.json")), U.load_spans(p("fold2-gold.json")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tokenizer", "train", "predict"])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--tokenizer")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--encoder", default="transformer", choices=list(ENCODERS) + ["phase1"])
    ap.add_argument("--phase1_ckpt", help="Phase 1 export (model_best.pt / model_final.pt); use with --encoder phase1")
    ap.add_argument("--phase1_code_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"),
                    help="folder holding cs781_p1.py")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--full", action="store_true", help="train on fold1+fold2 (no dev selection)")
    ap.add_argument("--init_emb", help=".npy [vocab, emb] to initialise the token embedding (e.g. word2vec)")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    a.data, a.out = os.path.abspath(a.data), os.path.abspath(a.out)
    if a.tokenizer:
        a.tokenizer = os.path.abspath(a.tokenizer)

    if a.cmd == "tokenizer":
        t1, _, t2, _ = _folds(a.data)
        texts = list(t1.values()) + list(t2.values())
        print(train_bpe(texts, a.out, a.vocab))
        return 0

    if a.cmd == "train":
        cfg = json.loads(json.dumps(DEFAULT_CFG))
        cfg["encoder"] = a.encoder
        if a.encoder in ("bilstm", "charbilstm"):
            cfg["encoder_args"] = dict(BILSTM_ARGS); cfg["lr"] = 2e-3
        if a.encoder == "charbilstm":
            cfg["encoder_args"]["tokenizer_path"] = a.tokenizer
        if a.encoder == "phase1":
            cfg.update(encoder_args={}, phase1_ckpt=os.path.abspath(a.phase1_ckpt),
                       phase1_code_dir=os.path.abspath(a.phase1_code_dir),
                       lr=1e-3, encoder_lr=5e-5, epochs=6, batch_size=8, o_weight=1.0)
        if a.batch_size: cfg["batch_size"] = a.batch_size
        if a.epochs: cfg["epochs"] = a.epochs
        if a.lr: cfg["lr"] = a.lr
        tok = BPETokenizer(a.tokenizer)
        t1, g1, t2, g2 = _folds(a.data)
        os.makedirs(a.out, exist_ok=True)
        if a.seed is not None: cfg["seed"] = a.seed
        model = None
        if a.init_emb:
            import numpy as np
            set_seed(cfg["seed"])
            model = build_model(cfg, tok.vocab_size, tok.pad_id)
            W = torch.from_numpy(np.load(a.init_emb)).float()
            model.encoder.tok.weight.data.copy_(W)
            cfg["init_emb"] = os.path.abspath(a.init_emb)
        if a.full:
            model, hist = train(cfg, tok, {**t1, **t2}, {**g1, **g2}, model=model)
        else:
            model, hist = train(cfg, tok, t1, g1, t2, g2, model=model)
        cfg["tokenizer"] = a.tokenizer
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
        json.dump({"cfg": cfg, "history": hist}, open(os.path.join(a.out, "run.json"), "w"), indent=1)
        return 0

    if a.cmd == "predict":
        run = json.load(open(os.path.join(a.out, "run.json")))
        cfg = run["cfg"]
        tok = BPETokenizer(a.tokenizer or cfg["tokenizer"])
        model = build_model(cfg, tok.vocab_size, tok.pad_id)
        model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location="cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        t3 = U.load_texts(os.path.join(a.data, "fold3-text.json"))
        pred = predict_docs(model, encode_docs(tok, t3)[0], cfg["max_len"], cfg["stride"], device)
        path = U.write_predictions(pred, os.path.join(a.out, "prediction.json"), texts=t3)
        print(U.make_zip(path, os.path.join(a.out, "submission.zip")))
        return 0


if __name__ == "__main__":
    sys.exit(main())
