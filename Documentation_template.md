# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Amizon  
**Team Members:** Ranveer Gautam, Satyam Kumar Pandey, Shreepal  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We find each business's copies in three steps: gather a small set of likely candidates, score every candidate pair
with a LightGBM model, and have a multilingual transformer cross-encoder take a second look at the pairs the model is
unsure about. France never appears in training, so we taught the cross-encoder French by fine-tuning it on pairs built
from the unlabelled French test records. The final submission scores **0.986691** on the public leaderboard (macro
F0.5), with a cross-validated **0.98921** on US and India, using no external data.

---

## 2. Methodology

### 2.1 Problem Analysis

The training data has 2.2 million Source 1 businesses and 10.3 million Source 2/3 records from the US and India. The
test set adds France (15 % of the 1.7 million test businesses), with no French examples to learn from.

**Exploratory data analysis (EDA)** showed:

- **The data follows the same recipe in every country.** US and India both have 3.46 true copies per business, 5.6 %
  of businesses with no copy at all, and 74 % of Source 2/3 records matched. So what the model learns on US and India
  should carry over to France.
- **Copies never cross countries,** so candidates are searched within one country at a time.
- **Names are noisy in predictable ways:** typos, digits for letters (`Freed0m`), website-style names
  (`rajsoftware.com`), legal forms (Pvt Ltd, LLC, SARL), and Indian names written in native scripts. The native-script
  words come from a small vocabulary (2,000 words cover 98.6 %), so a dictionary learned from training pairs translates
  them.
- **Some copies change one word on purpose.** A copy often swaps its last word for *center*, *services* or *partners*;
  decoy records add a word such as *groupe* or *holding* and sit at a nearby house number.
- **France is harder:** it has 10–40 times more businesses with near-identical names at the same address, so a wrong
  match is easier to make there.
- **Addresses are often partial or empty,** and many small businesses share generic names.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier + cross-encoder re-ranking (hybrid).  
**Core Innovation:** A transformer cross-encoder re-reads the uncertain pairs. For France, a country the model never
saw, the same cross-encoder is adapted by self-training on the unlabelled French test records, with synthetic
look-alikes that mimic how the data was generated.

The models doing the heavy lifting:

| Model | Size | What it does |
|---|---|---|
| `intfloat/multilingual-e5-base` cross-encoder | 278 M parameters | reads both records of an uncertain pair together and judges whether they are the same business; two French-adapted copies handle France |
| `intfloat/multilingual-e5-small` dense retriever | 118 M parameters | fine-tuned to find a business's copies by meaning, for candidates that share no keyword |
| `intfloat/multilingual-e5-small` cross-encoder and embeddings | 118 M parameters | quick neural similarity signals for every candidate pair |
| LightGBM | gradient-boosted trees | the main matcher over all features, and the model that combines its score with the cross-encoder's |

All models are MIT-licensed and far below the 8 billion parameter limit.

**Data preprocessing:** before anything else, every record is cleaned and normalised: native scripts are transliterated
to Latin letters, abbreviations expanded, legal forms removed, and house numbers extracted.

---

## 3. Candidate Generation (Blocking)

Comparing every Source 1 business with every Source 2/3 record would mean trillions of pairs, so we first pick about 25
likely candidates per business, always within the same country.

- **Blocking keys used:** words of the normalised name, consonant "skeletons" of names (robust to typos), name + street
  + house number combinations, and address word pairs. Keys are weighted by rarity (IDF), the best 1,500 records are
  re-ranked by fuzzy string similarity, and the top 20 are kept. Extra sources add what keys miss:
  - the 5 nearest records by `multilingual-e5-small` embedding similarity;
  - the 10 nearest records by our dense retriever, fine-tuned with contrastive learning and hard negatives;
  - a rescue rule that keeps a pair when the Source 1 business is that record's best match, and a rule for records
    with an empty address but a name unique in their country.
- **Candidate pairs generated:** 43.9 million test pairs, about 25 per business.
- **How you ensured true matches were not lost:** each source was kept only if it found enough extra true matches for
  the pairs it added, measured on training data. Recall of true matches grew from 92.6 % to **99.2 %**. The remaining
  misses are mostly records with an empty address and a common name, which the data itself cannot resolve.

---

## 4. Matching Model

**Features used (feature engineering):**
- Name features: fuzzy string ratios (RapidFuzz) on the full name, the name without legal words, and its consonant
  skeleton; Jaro-Winkler; character 3-gram TF-IDF cosine; **name rarity**, meaning how many businesses and records share
  the exact name (our largest single feature gain, +0.0038 CV).
- Address features: fuzzy address ratios, postcode agreement, house-number overlap, empty-address flags.
- Other: how the candidate was found (key score, rank, neighbour similarity); `multilingual-e5-small` similarity of
  names and of addresses; the score of a small `multilingual-e5-small` cross-encoder trained on 99,948 businesses that
  the matcher never trains on, so the feature is never over-optimistic.

**Model type:** gradient-boosted trees and fine-tuned transformers, combined by stacking.
1. **LightGBM matcher (gradient boosting):** hyperparameters tuned with Optuna, trained on 2.1 million businesses with
   5-fold GroupKFold cross-validation, final model an ensemble of 3 seeds. It scores all 43.9 million pairs: CV 0.98802.
2. **Cross-encoder re-ranker and stacking:** the 1.5 million pairs with a probability between 0.02 and 0.98 go to a
   `multilingual-e5-base` cross-encoder, fine-tuned (transfer learning) to read both records at once. It is trained twice, on each half of the
   businesses, so no training pair is scored by a model that saw it. A LightGBM meta-model
   stacks the matcher's and the cross-encoder's scores: CV 0.98876.
3. **Domain adaptation for France (self-training):** we built 885,573 French training pairs from the test records alone.
   Pseudo-labelled positives are 300,000 French pairs the model is at least 99.5 % sure of; synthetic hard negatives
   (data augmentation) are look-alikes made the way the data
   was generated (one generic word swapped, decoy words at a nearby house number). Both cross-encoders were fine-tuned
   on them for one epoch, and their ensemble average replaces the re-ranker for French pairs (about +0.011 on France on the
   leaderboard).

**Threshold selection method (post-processing):** each Source 2/3 record goes to at most one business, its most probable
one (one-to-one assignment). The US/India threshold (0.58) maximises macro F0.5 on out-of-fold predictions; a match is also dropped when a second
business scores almost as high (at least 0.5 and within 0.2), which brings CV to **0.98921**. France has no labels, so
its threshold (0.91) was chosen with a probability-calibration model fitted to leaderboard results. For France we also reject
a match whose name differs from the business's name by one common word (a typical look-alike), unless that word is a
known noise word such as *Services* or *Associés* and both French models are confident.

France uses the model version without the dense retriever: the retriever's extra French candidates were mostly
look-alikes and cost France 0.004.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.98921** cross-validated on US and India; **0.986691** on the public leaderboard.

| Version | Main change | Recall | CV F0.5 | Public LB |
|---|---|:---:|:---:|:---:|
| v3 | composite keys, fuzzy re-rank | 93.0 % | 0.9487 | 0.936 |
| v7 | TF-IDF, embeddings, tuned LightGBM, small cross-encoder | 92.6 % | 0.9603 | 0.953 |
| v8 | larger candidate pool, address keys, all training businesses | 96.6 % | 0.9804 | 0.9739 |
| v9 | dictionary, neighbours, rarity features, e5-base re-ranker, French cross-encoder and rules | 97.8 % | 0.98693 | 0.9839 |
| v10 | dense retriever, 3 seeds, tuned US/India threshold | 99.2 % | 0.98921 | 0.9853 |
| final | France from v9 with two French cross-encoders, threshold 0.91, noise-word recovery | 99.2 % | 0.98921 | **0.986691** |

A diagnostic upload with the French rows left empty measured the two parts separately: **US/India 0.9897** on the
leaderboard (slightly above CV) and **France 0.9673**.

- **Common false positives (wrong merges):** look-alike businesses at the same address whose names differ by one
  generic word (for example *Club* vs *Comité*), especially in France; and records with an empty address and a generic
  name that several businesses share.
- **Common false negatives (missed matches):** in France, copies whose noise the model never saw in training (French
  noise words, accents, reordered words), roughly 10 % of French copies; elsewhere, empty-address copies of businesses
  with common names, where two candidates are equally likely.

Tried and not kept, each measured: CatBoost blends, a LambdaRank re-ranker, an e5-large cross-encoder (−0.0004 on the
leaderboard), word TF-IDF features (−0.0017 on France), a second round of French self-training (−0.0002), and a learned
corrector for unseen countries.

---

## 6. Conclusion

Recall set the ceiling, so we added candidate sources one at a time and kept only those that paid for themselves. Name
rarity taught the matcher when "same name" really means "same business", and the multilingual-e5 cross-encoders gave
the biggest neural gains. France was the hardest part: with no labels, self-training the cross-encoder on French test
records brought the largest improvement, and better French recall is where the remaining gap lies.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds the runnable pipeline. One command rebuilds both output files:

```bash
pip install -r code/business_entity_resolution/requirements.txt -r code/business_entity_resolution/requirements-gpu.txt
python code/business_entity_resolution/reproduce.py --data /path/to/student_resource
```

`src/` has normalisation, blocking, features, the cross-encoders and the core driver; `src/pipeline/` has one module
per later step (candidates, extra features, classifier, re-ranking, France, decisions). The `README.md` there lists the
requirements and steps. GPU steps need a CUDA GPU; `transformers` is pinned to 4.57.6 because later versions change the
cross-encoder scores.

### B. Additional Results: data use and fair play

Only the provided data is used. Everything learned from labels (matcher, cross-encoders, dictionary, thresholds) is
fitted on training data only. Label-free statistics use each split's own text: the character TF-IDF is fitted on
training and test names, and rarity counts and neighbour searches run within each split. The French cross-encoders are
fine-tuned on pairs built from the French test records, using their text only, never labels; the organisers confirmed
this self-training is allowed. The French decisions were checked with public-leaderboard uploads and with simulations on
training data.
