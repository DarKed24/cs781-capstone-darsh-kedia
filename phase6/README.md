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
- Generators: `../finetune/p6_summ.py`, notebook `../finetune/kaggle_p6/CS781-CP-Phase6-Darsh-Kedia-250311.ipynb`.
- `NOTES.md`: full analysis.

## Key design facts
- The scorer truncates at 512 roberta tokens, so only the first ~400 words of a summary affect BERTScore: all budgets are ~400 words.
- Generators: decoder-only window `[source 576 tokens] <sep> [summary 447 tokens]`, loss on summary tokens only, greedy decoding, no-repeat 4-gram, repetition penalty 1.15. `src=ext` feeds the 470 words chosen by the supervised sentence scorer (extract-then-generate) instead of the lead (`src=head`).

## Codabench ladder (18 scored)
centroid400 **0.8522**, fuse_trcen_400 0.8520, textrank400 0.8519, fuse3_400 0.8513, fuse3_450 0.8499, supsent400 0.8450,
supsent600 0.8443, supsent400_rank 0.8437, gpt_full_ext 0.8330, unilm_full_ext 0.8312, gpt_full_head 0.8304,
unilm_full_head 0.8301, unilm_lora_ext 0.8297, unilm_lorma_ext 0.8294, leadskip400 0.8267, lead400 0.8264,
retnet_full_ext 0.8261, leadskip250 0.8214. Held-out oracle extractive upper bound: 0.8774.
