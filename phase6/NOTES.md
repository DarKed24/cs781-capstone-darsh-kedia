# Phase 6 (abstractive summarisation): report notes

Deadline 2026-11-03 23:59 IST (18:29 UTC). Metric: BERTScore F1 (bert_score, lang='en' = roberta-large,
no baseline rescaling), ROUGE-1/L secondary. 7,030 train pairs (median document 3.1k words, median
summary 638 words), 100 test documents, no dev split: a fixed 150-document held-out slice of train
(seed 781 shuffle) is used for every local estimate and never trained on.

## Data observations that shaped the method
- Summaries are long (median 638 words) and headnote-like: facts, then the question, then the holding.
- The scorer's BERTScore uses roberta-large with a 512-token cap, so only the first ~400 words of a
  prediction (and of the reference) influence the primary metric. Length budgets are ~400 words;
  ROUGE (secondary) still sees everything.
- Judgments open with a cause title, case numbers and a counsel list, which never appear in the
  summary. `skip_header` drops everything before the first "delivered by ...:" / "J.:" cue.

## Stage 1: CPU extractive ladder (capstone/phase6/p6_extract.py, make_ladder.py)
| zip | method |
|---|---|
| lead400 | first 400 words |
| leadskip400 / leadskip250 | header-skipped lead, 400 / 250 words |
| textrank400 | TextRank over TF-IDF sentence cosine (Mihalcea & Tarau 2004), document order |
| centroid400 | TF-IDF centroid similarity |
| supsent400 / supsent600 | supervised sentence scorer, 400 / 600 words, document order |
| supsent400_rank | same, sentences ordered by predicted score instead of document order |

Supervised sentence scorer (original): each train sentence gets a target = smoothed precision of its
content words against the reference summary vocabulary (oracle-style labels, cf. SummaRuNNer,
Nallapati et al. 2017); a linear SGD regressor over hashed word uni+bigrams plus relative-position,
absolute-position and length buckets is trained on 6,880 train documents (3 epochs, ~2.5 min CPU).
Held-out (first 60 of the 150 held-out train docs, official scorer settings):

| method | BERTScore F1 | ROUGE-1 | ROUGE-L |
|---|---|---|---|
| oracle400 (upper bound, uses the reference) | 0.8774 | 0.642 | 0.461 |
| textrank400 | 0.8512 | 0.529 | 0.289 |
| centroid400 | 0.8503 | 0.529 | 0.290 |
| supsent400 | 0.8451 | 0.527 | 0.287 |
| supsent600 | 0.8450 | 0.553 | 0.305 |
| supsent400_rank | 0.8443 | 0.527 | 0.252 |
| leadskip400 | 0.8336 | 0.467 | 0.233 |
| lead400 | 0.8335 | 0.466 | 0.232 |
| leadskip250 | 0.8273 | 0.374 | 0.194 |
| untrained-ish gpt (8-doc CPU smoke run, 64 tokens) | 0.831 | 0.201 | 0.145 |

Findings: content selection matters (+0.018 BERTScore over the lead); header skipping changes almost
nothing (+0.0001) because the header is short relative to 400 words; the supervised scorer matches
TextRank on ROUGE but is 0.006 lower on BERTScore; document order beats score order (ROUGE-L -0.035,
BERTScore -0.001); 600 words adds ROUGE but no BERTScore (truncation at 512 tokens, as predicted);
the oracle shows a 0.026 headroom for better selection.

Rank fusion (mean of per-document sentence ranks from TextRank, centroid and the supervised scorer),
held-out 60 docs: fuse3_400 0.8518 / R1 0.540 / RL 0.293 (best held-out BERTScore and ROUGE of the
400-word methods), fuse_trcen_400 0.8505, fuse3_450 0.8508.

## Stage 2: fine-tuned Phase 1 LMs as generators (finetune/p6_summ.py, build_p6_notebook.py)
Decoder-only window `[source 576 tok] <sep> [summary 447 tok]`, loss on summary tokens only; greedy
decoding, no-repeat-4-gram, repetition penalty 1.15; training capped at ~28 min per run (schedule shrunk
to fit). Archs: unilm (400M tokens, causal mode 0), gpt, retnet. Encdec excluded: its decoder holds only
256 tokens (~200 words), and it was the weakest Phase 1 model.

Original idea: extract-then-generate. The source window holds ~470 words of a ~3.1k-word judgment.
`src=head` feeds the header-skipped lead; `src=ext` feeds the 470 words picked by the supervised
sentence scorer (document order). Ablation pairs: unilm_full_ext vs unilm_full_head, gpt_full_ext vs
gpt_full_head. PEFT: LORMA_+ and LoRA on unilm (r=16), vs full FT.

## Results
Codabench (2026-11-03 13:35 UTC, 8 scored, minimum met): centroid400 0.8522 BEST, textrank400 0.8519,
supsent400 0.8450, supsent600 0.8443, supsent400_rank 0.8437, leadskip400 0.8267, lead400 0.8264,
leadskip250 0.8214. Fusion (16:00 UTC): fuse_trcen_400 0.8520, fuse3_400 0.8513, fuse3_450 0.8499 (rank fusion did not beat
centroid400 alone on test; held-out gains of +0.0007 were noise). 11 scored. CB tracks held-out within 0.002 for selection methods; lead methods ~0.007 lower on test.

Kaggle run 2026-11-03 (T4 x2, 84.7 min total, all 7 jobs ok, ~17.5 min training = 860 steps of 16
examples = 2 epochs, each run then generates 100 held-out + 100 test summaries). Held-out (100 docs):

| run | arch | method | src | trainable M | final loss | BERTScore | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| unilm_full_head | unilm | full | head | 109.5 | 2.395 | 0.8342 | 0.426 | 0.217 |
| gpt_full_head | gpt | full | head | 109.5 | 2.306 | 0.8329 | 0.422 | 0.213 |
| gpt_full_ext | gpt | full | ext | 109.5 | 2.363 | 0.8318 | 0.438 | 0.217 |
| unilm_full_ext | unilm | full | ext | 109.5 | 2.457 | 0.8313 | 0.435 | 0.215 |
| unilm_lorma_ext | unilm | LORMA_+ r16 | ext | 1.18 | 2.680 | 0.8307 | 0.437 | 0.215 |
| unilm_lora_ext | unilm | LoRA r16 | ext | 1.18 | 2.673 | 0.8303 | 0.431 | 0.215 |
| retnet_full_ext | retnet | full | ext | 116.6 | 2.623 | 0.8266 | 0.415 | 0.203 |

Analysis:
- Every generator is below the extractive methods (best 0.834 vs TextRank 0.851, and even below the
  lead baseline's ROUGE): a 110M LM trained on 400M tokens, fine-tuned ~17 min, decoding 360 words
  greedily, produces fluent-looking but generic, partly repetitive legal text. Copying real sentences
  wins on a semantic-similarity metric.
- Extract-then-generate ablation (the original idea): `ext` raises ROUGE-1 (+0.009 unilm, +0.016 gpt:
  the generator copies more reference content words) but lowers BERTScore slightly (-0.003, -0.001).
  Plausible reason: the extracted source is a bag of non-adjacent sentences, so the model loses the
  narrative opening (parties, facts in order) that headnote summaries also start with; BERTScore over the
  first 512 tokens rewards that ordering. Effect sizes are within noise for 100 docs; honest verdict:
  no BERTScore gain, a small ROUGE gain.
- Codabench (test) for generators, 16:19 UTC: gpt_full_ext 0.8330, gpt_full_head 0.8304, retnet_full_ext
  0.8261. On test the ext source beats head for gpt (+0.0026), the reverse of the held-out sign (-0.001):
  the ext-vs-head difference is within noise on 100 docs either way; report it as inconclusive.
- Codabench 16:50 UTC: unilm_full_ext 0.8312, unilm_full_head 0.8301, unilm_lora_ext 0.8297,
  unilm_lorma_ext 0.8294. On TEST, ext beats head for both archs (unilm +0.0011, gpt +0.0026) while
  held-out showed the reverse (-0.003, -0.001); ROUGE-1 favours ext on held-out for both. Fair summary:
  extract-then-generate gives a consistent small ROUGE gain and a BERTScore effect within +-0.003 noise.
  LoRA vs LORMA_+ vs full FT on test: 0.8297 / 0.8294 / 0.8312 (PEFT within 0.002 of full FT with 1.1%
  of the parameters).

## Final Codabench ladder (18 scored, phase closes 2026-11-03 18:29 UTC)
centroid400 0.8522 BEST, fuse_trcen_400 0.8520, textrank400 0.8519, fuse3_400 0.8513, fuse3_450 0.8499,
supsent400 0.8450, supsent600 0.8443, supsent400_rank 0.8437, gpt_full_ext 0.8330, unilm_full_ext 0.8312,
gpt_full_head 0.8304, unilm_full_head 0.8301, unilm_lora_ext 0.8297, unilm_lorma_ext 0.8294,
leadskip400 0.8267, lead400 0.8264, retnet_full_ext 0.8261, leadskip250 0.8214.
- PEFT vs full FT: LORMA_+ (1.18M trainable, 1.1% of params) is within 0.0006 of full FT on the same
  source and slightly above LoRA at equal rank, consistent with Phase 3/5 (PEFT ~ full FT).
- Archs: unilm ~ gpt > retnet (retnet also 1.5x slower to train), matching the Phase 1 ranking.
- Training loss is still falling at the 17-minute cap; longer training and beam search are the obvious
  untested next steps.

## Honest expectations / analysis notes
- A 110M model pretrained on 400M tokens, fine-tuned for <30 min on 6.9k pairs, generating 360 words
  greedily: expect BERTScore close to the extractive zips; extractive text is fluent legal prose that
  matches the reference style, which BERTScore rewards.
