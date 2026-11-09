# Phase 7: Prior Case Retrieval, report notes (Darsh Kedia, 250311)

## Task and data
- For each query judgment, return exactly 7 candidate ids from that split's pool, ranked. Metric: **Micro-F1@7**
  (TP/FP/FN pooled over all queries; `evaluate_phase7.py`). Because exactly 7 are always returned, only the ranking matters.
- Splits (queries / candidates): train 827 / 4,320, dev 118 / 1,023, test 237 / 1,727. Documents average ~37-50k
  characters (~9k Phase 1 tokens), far beyond the 1,024-token context.
- Gold citations per query: train mean 6.4 (median ~4, max 52), dev mean 8.2 (max 92). Every gold id is in the pool.
- **Every query is also a member of its own candidate pool** (same id, identical text). It is never gold, so it is
  always removed before ranking (it would otherwise take rank 1 on every query).
- Citations inside the query are masked by the organisers as `<CITATION>`; names and organisations as `<NAME>`/`<ORG>`.
- Candidate pools overlap across splits (train∩test 740 candidates); queries do not overlap.

## Data finding: CITATOR INFO leak (not used in any submission)
Supreme Court documents (scraped with their headers) contain `CITATION: <this case's reporter citations> CITATOR INFO :
<later cases that cite this one, e.g. "R 1969 SC 840 (11)">`. A candidate's CITATOR INFO can therefore name the query
that cites it. Matching each query's own `YYYY AIR page` citation against candidates' CITATOR INFO lists:

| split | queries with an AIR cite | predicted links | correct | precision |
|---|---|---|---|---|
| train | 237 / 827 | 552 | 503 | 91% |
| dev | 35 / 118 | 145 | 143 | 99% |
| test | 66 / 237 | 196 | (unknown) | |

This contradicts the evident intent of masking citations, so both header fields are cut from queries and candidates
before any modelling (`p7_lex.strip_meta`, on by default). Stripping changed tuned BM25 by only +0.001 dev F1, so the
lexical model was not exploiting it through number tokens. No leak zip was built or submitted: it is reported only as
a data finding.

## Method 1: lexical retrieval with three task-specific ideas (CPU, `finetune/p7_lex.py`)
BM25 in sparse-matrix form (Robertson & Zaragoza 2009) so a whole judgment can be the query.
1. **Long-document BM25 tuning.** Defaults (k1 1.2, b 0.75, k3 8) are tuned for short queries. Here the query is a
   ~9k-token judgment: raw query tf (k3 → ∞), slow tf saturation (k1 6) and full length normalisation (b 1.0),
   and dropping terms in >30% of the pool.
2. **Citation-context query (original).** Sentences within ±1 of each `<CITATION>` marker describe the cited cases;
   their BM25 scores are z-normalised and fused with the whole-document query (weight 0.7 on the context query).
3. **Temporal feasibility filter (original).** A judgment can only cite earlier cases. Query year from its
   `DATE OF JUDGMENT` header; candidate year from its own header if present, otherwise the latest year it mentions
   (a lower bound on its decision date). Candidates later than the query are ranked last. On train it removes ~43% of
   each pool while losing ~8% of gold links (gold candidates that mention later years: appended notes, later editions).
   First version used the max year mentioned anywhere, which the CITATOR INFO header inflated: it killed 25% of gold
   and *hurt* dev F1 (0.331 → 0.309). Fixing the year source turned it into a +0.03-0.04 gain. (Worth a sentence in
   the report as a "what failed and why".)
4. **Word n-grams.** Bigrams capture legal phrases ("natural justice", "res judicata", statute names).
5. **Citation prior.** log(1 + times the candidate was cited by training queries) (train for dev, train+dev for
   test), added with weight 0.3. Tiny effect: few test candidates are frequently cited.

### Ablation ladder (dev Micro-F1@7; train in brackets where measured; tuned on train, checked on dev)
| zip | config | dev F1 | dev P | dev R | Codabench |
|---|---|---|---|---|---|
| bm25_plain | BM25 defaults, unigrams | 0.2012 (train 0.169) | 0.219 | 0.186 | 0.2262 |
| bm25_tuned | + long-doc tuning (k1 6, b 1, k3 ∞, max_df .3) | 0.3735 | 0.407 | 0.345 | 0.3662 |
| bm25_ctx | + citation-context query (w .7) | 0.3902 | 0.425 | 0.361 | 0.3804 |
| bm25_ctx_date | + temporal filter | 0.4169 (train 0.341) | 0.454 | 0.385 | 0.4217 |
| bm25bi_ctx_date | unigrams + bigrams | 0.4591 (train 0.379) | 0.500 | 0.425 | 0.4513 |
| bm25tri_ctx_date | 1-3 grams | 0.4625 (train 0.388) | 0.504 | 0.428 | 0.4560 |
| bm25bb_ctx_date | bigrams only | 0.4658 (train 0.385) | 0.507 | 0.431 | **0.4566** (best) |
| bm25bb_ctx_date_pri | + citation prior 0.3 | 0.4669 | 0.509 | 0.432 | 0.4507 |

Other sweep points (unigram, before bigrams): k3 8 → 1000 +0.05 dev; b 0.75 → 0.9 → 1.0 and k1 1.2 → 2 → 3 → 6 each
helped, k1 10 no further gain; context window ±2 slightly worse than ±1; context weight 0.5-0.85 all within 0.01.
Each piece is additive: the full lexical system more than doubles plain BM25 (0.20 → 0.47).

## Method 2: Phase 1 passage-level bi-encoder + fusion (Kaggle, `finetune/p7_dense.py`)
- Query = its citation-context passages (≤24, ≤192 tokens, merged when markers are adjacent); candidate = ≤24
  chunks of 192 tokens spread evenly over the judgment. score = mean over passages of max over chunks of cosine
  (MaxSim at passage granularity, after ColBERT).
- **Distantly supervised alignment (original).** Labels only say q cites c. For each gold pair the (passage, chunk)
  pair with the highest TF-IDF cosine is the positive; the hard negative is the best-matching chunk of a top-15 BM25
  candidate q does not cite. InfoNCE (τ 0.05) with in-batch negatives; in-batch "negatives" that are gold for the
  query are masked (false-negative masking).
- Runs: unilm full FT, unilm LORMA+, gpt full FT, retnet LORMA+; ablations unilm frozen (zero-shot) and unilm full with
  random alignment. Fusion z(BM25 stage) + w·z(dense), w tuned on dev; ensembles of dense runs.
- CPU prep measured here: 6.5 min (lexical stage for 3 splits + tokenisation + 5,308 aligned training pairs).
  Queries average 3.2-3.7 citation-context passages; candidates 18.5-19 chunks. CPU smoke test (gpt, LORMA+, 96 pairs) ran end to end.
- Expected before the run (stated to Darsh): dense alone well below BM25 (dev ~0.15-0.30); fusion +0.00 to +0.03
  over the lexical 0.466 (dev ~0.47-0.50). Reason: 110M models pretrained on 400M tokens, 5.3k distantly supervised pairs.
- Results (Kaggle run, T4 x2, 3,854 s total: prep 4.2 min, 6 jobs two at a time, 0 failures).
  **None of these zips were uploaded before the deadline: all rows are dev-only.** 4 epochs = 660 steps, batch 32.

| run | arch | method | trainable | align | train min | final loss | dev dense alone | best w | dev fused (lex 0.4658) |
|---|---|---|---|---|---|---|---|---|---|
| gpt_full | gpt | full | 109.5M | tfidf | 13.7 | 0.097 | 0.3291 | 0.2 | 0.4691 |
| unilm_lorma | unilm 400M | LORMA+ r16 | 1.18M | tfidf | 14.3 | 0.753 | 0.3268 | 0.1 | 0.4691 |
| unilm_full | unilm 400M | full | 109.5M | tfidf | 14.3 | 0.301 | 0.3257 | 0.2 | 0.4680 |
| retnet_lorma | retnet | LORMA+ r16 | 1.18M | tfidf | 17.8 | 0.473 | 0.3246 | 0.1 | 0.4680 |
| unilm_full_randalign (ablation) | unilm 400M | full | 109.5M | **random** | 14.5 | 1.122 | 0.3013 | 0.3 | 0.4691 |
| unilm_frozen (ablation) | unilm 400M | none (zero-shot) | 0 | n/a | 0 | n/a | 0.2201 | 0.1 | 0.4669 |
| fuse_all (4 fine-tuned runs) | | | | | | | | 0.1 | 0.4680 |
| fuse_unilm (unilm_full + unilm_lorma) | | | | | | | | 0.1 | 0.4691 |
| fuse_every (all 6) | | | | | | | | 0.2 | 0.4691 |

Analysis:
- **Contrastive fine-tuning works for the dense model:** zero-shot Phase 1 features 0.220 → 0.325-0.329 after
  14 min of InfoNCE (+0.11), on every architecture. All four fine-tuned archs land within 0.005 of each other.
- **Distant-supervision alignment helps the dense model:** TF-IDF-aligned positives 0.3257 vs random (passage, chunk)
  positives 0.3013 (+0.024, same arch/method/steps). The random-alignment loss stays much higher (1.12 vs 0.30).
- **LORMA+ matches full fine-tuning** with 1.1% of the weights (unilm 0.3268 vs 0.3257), consistent with Phases 3 and 5.
- **Fusion adds almost nothing:** best +0.003 dev F1 (0.4658 → 0.4691), i.e. ~3 more correct links out of 826
  predictions; weights > 0.5 hurt. Even the random-alignment run fuses to the same 0.4691, so differences between
  runs in the fused column are within noise (one link ≈ 0.0011 F1). Expected range before the run was 0 to +0.03 over
  the lexical stage: the result is at the bottom of it.
- Why the dense model stays far below BM25 (0.33 vs 0.47): 110M models pretrained on ~200-400M tokens; only 5.3k
  weakly aligned training pairs; a candidate is seen through ≤24 chunks of 192 tokens (~4.6k of ~9k tokens), while BM25
  sees the whole document and exact rare-phrase matches (the bigram gain shows how much exact phrasing matters).
  The dense errors overlap BM25's, so fusion has little complementary signal to add.
- Low gpt_full final loss (0.097) with no better dev than unilm (0.30) suggests overfitting of the training pairs.

## Codabench log
**FINAL (phase closed 2026-11-07 18:29 UTC): 8 scored, all lexical, best bm25bb_ctx_date 0.4566.** No CITATOR-leak
zip was ever built or submitted. The Kaggle dense runs (best dev fused 0.4691, +0.003 over lexical) were not submitted.
- 2026-11-07 ~05:55 UTC: all 8 lexical zips scored (minimum of 6 met). Best bm25bb_ctx_date 0.4566.
  Codabench tracks dev within ~0.01-0.02 on every rung, and every idea keeps its sign on test: long-doc tuning +0.14,
  citation-context query +0.014, temporal filter +0.041, bigrams +0.035 (uni+bi 0.4513, 1-3 grams 0.4560, bigram-only 0.4566).
  The citation prior is the one idea that flipped: +0.001 dev, -0.006 test (0.4566 -> 0.4507), so it is reported as no gain.
