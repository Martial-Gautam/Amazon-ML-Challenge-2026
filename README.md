# Amazon ML Challenge 2026 · Business Entity Resolution · Team Amizon

Code and methodology of the final submission (public leaderboard macro F0.5 **0.986691**). The output files
(`matching_results.tsv`, `candidate_pairs.tsv`) are only in the submitted zip, not in this repository.

```
code/business_entity_resolution/
    src/                        all source code (pipeline + experiments/)
    README.md                   end-to-end reproduction steps
    requirements.txt            pinned CPU dependencies (requirements-gpu.txt for the GPU steps)
Documentation_template.md       methodology write-up
```

Pipeline in one line: per-country blocking + dense retrieval (~25 candidates per S1, 99.2% recall) → LightGBM pairwise
classifier → e5-base cross-encoder re-ranking and stacking on uncertain pairs (France: two domain-adapted cross-encoders)
→ per-country decision layer (one-to-one assignment, ambiguity rejection, look-alike rules, noise-variant recovery).
Models: LightGBM and `intfloat/multilingual-e5-small/-base` (MIT). No external data or APIs. The dataset is not included.
