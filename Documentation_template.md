# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Amizon  
**Team Members:** Ranveer Gautam, Satyam Kumar Pandey, Shreepal  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A **blocking → learned matching → second stage** pipeline. Records are normalised by a stdlib-only Indic-script
transliterator and a native-word dictionary mined from the training pairs. Five label-free candidate sources keep
**~25 candidates per Source 1 record at ~99.2% training recall**. A tuned LightGBM matcher (3 seeds) combines string,
rarity and context features with neural signals from MIT-licensed `multilingual-e5` models; a second stage re-scores
the uncertain pairs with an e5-base cross-encoder, adapted to France (test-only) with synthetic pairs.
Final: US/India rows from v10 (CV 0.98921 with the conflict rule), French rows from v9 with the France-adapted cross-encoders P5 + P5b (France threshold 0.91), plus French copies that rule P1 had removed although both models were confident ("Services", "Associés", "France", initials) re-accepted; public leaderboard **0.986691**. No external data, APIs or lookups.

## 2. Methodology
### 2.1 Problem Analysis
2.2M Source 1 / 10.3M Source 2+3 training records; ~3.7 true matches per matched S1; 5.6% singletons; 26% of
S2/S3 records are decoys. Matches never cross the country label, so blocking is per country (open set: France, test-only,
runs through the same code). Noise: typos, leet digits (`Freed0m`), extra/duplicated words, website names
(`rajsoftware.com`), **native-script translations from a fixed vocabulary** (2,000 words = 98.6%: `குளோபல்` = *global*),
empty addresses, generic names shared by many businesses.

### 2.2 Solution Strategy
**Approach:** blocking with several complementary candidate sources → gradient-boosted matcher with neural
pair features and a one-owner rule → cross-encoder second stage, with France handled by label-free rules and synthetic
pairs. **Normalisation** (`prep.py`): transliteration from Unicode character names; the mined native-word dictionary;
legal-form, abbreviation, French street-type, elision and leet-digit rules; consonant *skeletons*; house-number keys.

## 3. Candidate Generation (Blocking)
| Source | What it adds |
|---|---|
| Composite keys (country-scoped: name skeletons, token pairs, prefixes, name/street + house number, address word pairs) | IDF-scored pool of 1,500 per S1, fuzzy re-rank, **top 20** |
| Bidirectional rescue | the pair is also kept if the S1 is the candidate's best S1 (+0.5% pairs, +0.0010 CV) |
| e5-small neighbours | top-5 same-country records by cosine of "name \| address" |
| **Fine-tuned retriever** | e5-small bi-encoder trained (contrastive, hard negatives) only on 99,948 held-out S1; top-10 finds 65.4% of the remaining misses |
| Rare-name rule | empty-address candidates whose exact name is unique among the country's S1 (97.6% precision) |

Training recall: 92.6% (v7) → 96.6% (v8) → 97.8% (v9) → **~99.2%** (v10); test: 43.9M pairs (25.3 per S1).
Every source was adopted only for its measured recall per added pair; the remaining misses are mostly empty-address
records with common names, which the data cannot resolve.

## 4. Matching Model
**Features:** name / core / skeleton rapidfuzz ratios, Jaro-Winkler, char-3-gram TF-IDF cosines; address ratios,
postcode, house-number overlap; blocking context and neighbour rank / cosine; **name rarity** (number of S1 and of
candidates sharing the exact core name, empty-address flag and interactions: +0.0038 CV, our largest feature gain);
e5-small name / address cosines; a **cross-encoder** logit (e5-small fine-tuned on 2.94M pairs of 99,948 training S1
that are excluded from the matcher's training rows, so the feature is never optimistic).
**Matcher:** LightGBM (MIT), Optuna-tuned, 2.1M training S1, mean of 3 seeds. **Decision:** accept if probability ≥ *t*
and this S1 is the candidate's most probable owner; *t* maximises the exact metric on 5-fold GroupKFold (by S1)
out-of-fold predictions. US/India: drop a winner whose second-best S1 is within 0.2 and ≥ 0.5 (+0.00018 CV).
**Second stage:** for the uncertain pairs (probability 0.02–0.98) a `multilingual-e5-base` cross-encoder (ce2, two
out-of-fold halves) feeds a small LightGBM on [logit(probability), ce2] (+0.00074 CV on v10).
**France (P5):** ce2 fine-tuned on 885,573 synthetic French pairs built from the French test records: confident
one-owner matches as positives, negatives from the generator's own look-alike recipes (sibling-word swaps, decoy words,
near house numbers); its score replaces ce2 for French pairs (LB +0.0017). Two label-free France rules reject
"one common word swapped" sibling look-alikes at the same address (LB +0.0011) and at a different address
(checked by training on one country and applying the rule to the other).

## 5. Results & Error Analysis
| Version | Main change | Recall | CV F0.5 | Public LB |
|---|---|---|---|---|
| v3 | composite keys, fuzzy re-rank | 93% | 0.9487 | 0.936 |
| v7 | TF-IDF, embeddings, tuned LightGBM, cross-encoder | 92.6% | 0.9603 | 0.953 |
| v8 | larger pool + address keys, all 2.1M training S1 | 96.6% | 0.9804 | 0.974 |
| v9 + ce2 + P5 + France rules | dictionary, rescue, neighbours, rarity features, second stage | 97.8% | 0.98706 | 0.983893 |
| v11 | v10 (retriever, normalisation fixes, 3 seeds) + second stage + P5 + rules | ~99.2% | 0.98921 | 0.98533 |
| 0.91 file | US/India from v11; France from v9 + P5/P5b ensemble at threshold 0.91 (v10's extra French candidates hurt France) | ~99.2% | 0.98921 | 0.986371 |
| + noise copies | + French noise copies ("Services", "Associés") that rule P1 had removed | ~99.2% | 0.98921 | 0.986525 |
| + P1 reversal | + P1-removed same-house copies the models score >= 0.91 ("France", initials) | ~99.2% | 0.98921 | 0.986673 |
| **final** | + all remaining P1/P2'-removed winners the models score >= 0.91 | ~99.2% | **0.98921** | **0.986691** |

Final assembly: 5,811,466 matches (US/India threshold 0.58 + conflict rule; France threshold 0.91 + rules P1/P2'). A diagnostic upload with the French rows emptied measured US/India at 0.9897 on the LB (above CV) and France at 0.9673: France recall is the remaining gap.
Also tried and not kept (measured on CV or LB): learned re-ranker, context stacking, anchor features, CatBoost blends,
more trees, re-tuning, importance weighting, word TF-IDF (France −0.0017 on the LB), an e5-large second-stage model,
a learned unseen-country corrector, pseudo-labelling of the matcher.

## 6. Conclusion
Candidate recall set the ceiling, so each candidate source was bought with measured recall per added pair; rarity
features taught the matcher when "same name" is evidence; cross-encoders trained on held-out records gave the largest
neural gains; and France, absent from training, was handled with label-free rules and synthetic pairs checked on the
leaderboard.

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/` with the full run order in its `README.md`: `src/` (`prep.py`, `native_dict.py`,
`blocking.py`, `features.py`, `embed.py`, `ce_data.py`, `ce.py`, `tune.py`, `run.py`) and `src/experiments/` (neighbour
search, retriever, rarity columns, retraining, second stage, P5 data, France rules, one-command final assembly).
Models: LightGBM, `multilingual-e5-small` (118M) and `-base` (278M), all MIT. GPU steps need `requirements-gpu.txt`
(transformers pinned to 4.57.6: 5.x changes cross-encoder logits on borderline pairs).

### B. Additional Results: data use and fair play
Only the provided data is used. Everything learned from **labels** (matcher, cross-encoders, dictionary, thresholds) is
fitted on training data only. Label-free statistics use each split's own text: char TF-IDF is fitted on train + test
text; rarity counts and neighbour searches run within each split. The France-only decisions were chosen with public-LB
probes and training-data simulations. **P5** fine-tunes the cross-encoder on pairs built from the French test records
(text only; confident matches as positives and synthetic negatives), a self-training use the organisers confirmed as
allowed. No test labels exist or were used.
