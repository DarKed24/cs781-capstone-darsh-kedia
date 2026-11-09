# Phase 6: Summarisation

**Task.** Summarise a judgment (median 3.1k words) into a headnote-style summary (median 638 words).
Metric: BERTScore F1 (roberta-large, no rescaling; ROUGE secondary). 7,030 train pairs, 100 test docs, no dev
split: a fixed 150-document held-out slice of train (seed 781) is used for every local estimate.

**Best submission: Codabench 0.8522**, `centroid400` (TF-IDF centroid extractive, 400 words). Best generator
(fine-tuned Phase 1 gpt, extract-then-generate): 0.8330.

## Files
- `p6_extract.py`: lead, header-skipping lead, TextRank, centroid, supervised sentence scorer (`Scorer`), zip writer.
- `make_ladder.py`, `make_fusion.py`: build the extractive and rank-fusion submissions (CPU, ~10 min). Set `CS781_ROOT` to the folder holding `data/`.
- `eval_holdout.py`, `eval_fusion.py`: BERTScore/ROUGE on the held-out slice.
- Generators: `p6_summ.py`, notebook `CS781-CP-Phase6-Darsh-Kedia-250311.ipynb`.

## Key design facts
- The scorer truncates at 512 roberta tokens, so only the first ~400 words of a summary affect BERTScore: all budgets are ~400 words.
- Generators: decoder-only window `[source 576 tokens] <sep> [summary 447 tokens]`, loss on summary tokens only, greedy decoding, no-repeat 4-gram, repetition penalty 1.15. `src=ext` feeds the 470 words chosen by the supervised sentence scorer (extract-then-generate) instead of the lead (`src=head`).

## Every Codabench submission (18 scored; held-out = 60-100 train documents never trained on)
| zip | method | held-out BERTScore | Codabench |
|---|---|---|---|
| centroid400 | sentences most similar to the TF-IDF document centroid, 400 words, document order | 0.8503 | **0.8522** |
| fuse_trcen_400 | mean rank of TextRank and centroid | 0.8505 | 0.8520 |
| textrank400 | TextRank over TF-IDF sentence cosine, 400 words | 0.8512 | 0.8519 |
| fuse3_400 / fuse3_450 | mean rank of TextRank, centroid, supervised scorer; 400 / 450 words | 0.8518 / 0.8508 | 0.8513 / 0.8499 |
| supsent400 / supsent600 | supervised sentence scorer (oracle-overlap targets), 400 / 600 words | 0.8451 / 0.8450 | 0.8450 / 0.8443 |
| supsent400_rank | as supsent400, sentences in score order | 0.8443 | 0.8437 |
| gpt_full_ext | gpt full FT, source = scorer-selected sentences (extract-then-generate) | 0.8318 | 0.8330 |
| unilm_full_ext | unilm full FT, ext source | 0.8313 | 0.8312 |
| gpt_full_head | gpt full FT, source = header-skipped lead | 0.8329 | 0.8304 |
| unilm_full_head | unilm full FT, lead source | 0.8342 | 0.8301 |
| unilm_lora_ext / unilm_lorma_ext | unilm LoRA / LoRMA+ r16, ext source | 0.8303 / 0.8307 | 0.8297 / 0.8294 |
| leadskip400 / lead400 / leadskip250 | first 400 words (after/without header skip), 250 words | 0.8336 / 0.8335 / 0.8273 | 0.8267 / 0.8264 / 0.8214 |
| retnet_full_ext | retnet full FT, ext source | 0.8266 | 0.8261 |

Extractive upper bound (oracle sentence choice, 400 words): 0.8774 held-out.
