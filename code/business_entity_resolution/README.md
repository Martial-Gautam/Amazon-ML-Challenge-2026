# Business Entity Resolution — Team Amizon

No external data, APIs or lookups. Licences: LightGBM (MIT), `intfloat/multilingual-e5-small` (MIT, 118M params),
polars / rapidfuzz / scikit-learn / optuna (MIT/BSD). CPU steps: Linux, 32 cores / 64 GB RAM for the full build
(blocking peaks ~36 GB); the neural steps (`embed.py`, `ce.py`, e5 neighbour search, retriever) need a CUDA GPU (Kaggle T4).

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
.venv/bin/pip install -r code/business_entity_resolution/requirements-gpu.txt   # GPU steps only, + a CUDA build of torch
#   transformers is pinned to 4.57.6 (the version that produced all our ce / ce2 / embedding scores; 5.x differs)
S=code/business_entity_resolution/src
X=$S/experiments                                 # helper scripts (in the git repo: X=experiments)
FLAGS="--work work9 --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys --rescue \
       --knn work/n4 --knn-top 5 --rare-empty --exclude-s1 work/ce/ce_train.parquet"
```

## Run order (exactly as v9 was produced; from the repo root, with `dataset/` present)
```bash
# 0. native-script dictionary (native word -> English word), mined from TRAINING match pairs only.
#    Committed as native_dict.tsv (5,484 tokens) and read by prep.py; re-mine with:
.venv/bin/python $S/native_dict.py --work work9

# 1. e5 neighbour candidates (GPU, Kaggle): top-20 same-country S2/S3 records per S1 by cosine of off-the-shelf
#    e5-small "name | address" embeddings (raw text, label-free) -> work/n4/{train,test}_knn.parquet
#    $X/n4_search.py (set SPLIT; inputs: raw text + country per record)

# 2. normalise + candidates (blocking C, rescue, kNN top-5, rare-empty rule); writes work9/emb_in/* for the GPU step
.venv/bin/python $S/run.py pairs $FLAGS

# 3. extra label-free feature columns per split
.venv/bin/python $X/tfw_cols.py --work work9 --tag k30_p1500_ap_ak_use20_rs_knn5_re      # word TF-IDF cosines
.venv/bin/python $X/rarity_cols.py --work work9 --tag k30_p1500_ap_ak_use20_rs_knn5_re   # name-rarity counts

# 4. string features without a model (runs in parallel with step 5)
.venv/bin/python $S/run.py train $FLAGS --features-only
.venv/bin/python $S/run.py predict $FLAGS --features-only

# 5. GPU pair features for every pair in work9/emb_in (raw text per record first)
mkdir -p work9/text work9/gpu && .venv/bin/python -c "import polars as pl
for s in ('train','test'):
    for w in ('s1','cand'):
        pl.scan_parquet(f'work9/{s}_{w}/*.parquet').select('idx','business_name','business_address').sink_parquet(f'work9/text/{s}_{w}.parquet')"
.venv/bin/python $S/embed.py --text work9/text --pairs work9/emb_in --out work9/gpu           # e5 cosines
#    cross-encoder: trained once on the pairs of 99,948 training S1 (ce_train.parquet), then score-only
.venv/bin/python $S/ce_data.py --n-train 600000 --out work/ce
.venv/bin/python $S/ce.py --text work/text --pairs work/ce --out work/gpu                     # fine-tune
.venv/bin/python $S/ce.py --text work9/text --pairs work9/emb_in --out work9/gpu --model-dir work/gpu/ce_model

# 6. join GPU + extra columns, 5-fold CV + threshold, final fit, predict (tuned LightGBM)
.venv/bin/python $X/retrain_extra.py --work work9 --params code/business_entity_resolution/best_params.json \
    --extra 'work9/gpu/{split}_emb.parquet' --extra 'work9/gpu/{split}_ce.parquet' \
    --extra 'work9/tfw_{split}.parquet' --extra 'work9/rarity_{split}.parquet' --out output_v9 --tag v9

# 7. ce2 second stage on the uncertain band (final = v9 + ce2)
#    ce2 = multilingual-e5-base cross-encoder, two out-of-fold halves: ce_data.py --half h builds the training
#    pairs of 200k half-h S1; ce.py fine-tunes e5-base on them (Kaggle 2xT4; $X/kaggle/ce2_train_h0.py, h1 alike):
.venv/bin/python $S/ce_data.py --half 0 --n-ce 200000 --k-use 20 --out work/ce2      # and --half 1
.venv/bin/python $S/ce.py --text work/text --pairs work/ce2 --train-file ce_train_h0.parquet --out work/ce2_h0 \
    --model intfloat/multilingual-e5-base --lr 3e-5 --epochs 1 --batch 64                # and h1
#    band = v9 pairs with probability in [0.02, 0.98]; model h0 scores train half 1, h1 half 0, both the test band
.venv/bin/python $X/band_export.py --oof work9/oof_v9.parquet --test-prob work9/test_prob_v9.parquet --out work9/band
for h in 0 1; do                     # score-only ($X/kaggle/band_score_h0.py, h1 alike)
  .venv/bin/python $S/ce.py --text work9/text --pairs work9/band/h$h --out work9/ce2band/h$h --model-dir work/ce2_h$h/ce_model
done
#    stage 2: LightGBM on [logit(v9 prob), ce2] over the band pairs (5-fold by S1, kept only if CV beats the
#    control [logit(v9 prob)]); writes the final submission (candidate_pairs.tsv unchanged)
.venv/bin/python $X/ce2_band.py --work work9 --band work9/ce2band --oof work9/oof_v9.parquet \
    --test-prob work9/test_prob_v9.parquet --meta work9/tune_meta.parquet --report work9/report_v9.json \
    --src output_v9 --out output_v9ce2

# 8. France rules (label-free, chosen by public-LB probes; France is test-only)
#    (all of step 8 + validation in one command, from the git repo:
#     experiments/final_assemble.sh work9 work9/test_prob_v9ce2.parquet output_v9ce2/ce2_band.json output)
#    a) France: the stage's CV threshold (from ce2_band.json) + 0.04 (pre-P5; 0 once P5 is used, see the final section);
#       US/India: CV threshold + near-tie conflict rule
#       (drop a record's winner when its 2nd-best S1 has probability >= 0.5 within 0.2; +0.00018 CV)
.venv/bin/python $X/country_threshold.py --work work9 --prob work9/test_prob_v9ce2.parquet \
    --json output_v9ce2/ce2_band.json --country France --offset 0.04 \
    --conflict-delta 0.2 --conflict-countries US,India --out output_v9ce2_fr
#    b) P1: reject French "one common word swapped at the SAME address" pairs (sibling look-alikes; LB +0.001118)
.venv/bin/python $X/france_rule.py --work work9 --pairs work9/test_prob_v9ce2.parquet --out work9/g_test_common.parquet
.venv/bin/python $X/france_probe.py --base output_v9ce2_fr --groups work9/g_test_common.parquet --mode reject --out output_p1
#    c) P2': the same symmetric sibling-word swap at a DIFFERENT address (checked in the unseen-country simulation)
.venv/bin/python $X/france_rule.py --work work9 --pairs work9/test_prob_v9ce2.parquet --address different \
    --out work9/g2_test_diffaddr_sym.parquet
.venv/bin/python $X/france_probe.py --base output_p1 --groups work9/g2_test_diffaddr_sym.parquet --mode reject --out output

# 9. validate and package
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
./make_zip.sh output <tag>          # -> Amizon_submission.zip (code + docs + helper scripts from the git tag)
```

## Final submission (v11 = v10 + ce2 + P5 + France/US-India rules)
v10 = the v9 run order on `work10` with v10's code (retriever candidates, prep/feature fixes D1/D3/D4/D6, char TF-IDF
on train+test, 3 seeds, NO word TF-IDF: it cost France −0.0017 on the LB). Exactly as run:
```bash
FLAGS10="--work work10 --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys --rescue --knn work/n4 --knn-top 5 \
         --knn2 work/ret --knn2-top 10 --rare-empty --exclude-s1 work/ce/ce_train.parquet --tfidf-all"
#   work/ret/{split}_knn.parquet = fine-tuned retriever top-20 ($X/ret_train.py on the CE's 99,948 held-out S1, $X/ret_search.py)
.venv/bin/python $S/run.py pairs $FLAGS10
.venv/bin/python $X/rarity_cols.py --work work10 --tag k30_p1500_ap_ak_use20_rs_knn5_rk10_re
.venv/bin/python $S/run.py train $FLAGS10 --features-only && .venv/bin/python $S/run.py predict $FLAGS10 --features-only
#   GPU: e5 embeddings + ce scores for work10/emb_in into work10/gpu (step 5)
.venv/bin/python $X/retrain_extra.py --work work10 --params code/business_entity_resolution/best_params.json \
    --extra 'work10/gpu/{split}_emb.parquet' --extra 'work10/gpu/{split}_ce.parquet' --extra 'work10/rarity_{split}.parquet' \
    --seeds 3 --out output_v10 --tag v10                              # CV 0.98802 @ 0.72
```
Then:
```bash
# F1. P5, France-adapted cross-encoder (organiser-approved self-training + synthetic pairs from the provided records):
#     synthetic French pairs from the TEST records (no labels), ce2 h0 fine-tuned on them (Kaggle, $X/kaggle/frce_train.py),
#     scores for every French v10 pair with p >= 0.02; pairs outside the Kaggle run are scored on CPU with the same model
.venv/bin/python $X/fr_synth.py --out work9/frce --score work/test_prob_v81ce2.parquet --score work9/test_prob_v9ce2.parquet
.venv/bin/python $S/ce.py --text work9/frce/text --pairs work9/frce/pairs --out work9/frce_out \
    --model work/ce2_h0/ce_model --lr 2e-5 --epochs 1 --batch 64      # 885,573 training pairs, 1,060,401 French pairs scored
.venv/bin/python $X/cpu_score.py work9/frce_out/ce_model work/text <v10 French pairs not scored above> <out>   # 210,562 pairs
#     (frce_out + cpu scores -> work10/frce_all/test_ce.parquet)
# F2. ce2 stage 2 on v10's uncertain band, P5 in ce2's slot for French pairs (p >= 0.02); band scoring as in step 7
.venv/bin/python $X/ce2_band.py --work work10 --band work10/band --oof work10/oof_v10.parquet \
    --test-prob work10/test_prob_v10.parquet --meta work10/tune_meta.parquet --src output_v10 --out output_v10ce2 \
    --frce work10/frce_all --frce-min-cov 0.999
# F3. France rule lists P1 / P2' with work9's prep (the prep the LB probe used) over v10's pairs
.venv/bin/python $X/france_rule.py --work work9 --pairs work10/test_prob_v10.parquet --out work10/g_test_common_w9prep.parquet
.venv/bin/python $X/france_rule.py --work work9 --pairs work10/test_prob_v10.parquet --address different \
    --out work10/g2_test_diffaddr_sym_w9prep.parquet
# F4. thresholds (France offset 0 with P5; US/India conflict rule δ 0.2) + P1 + P2' + validator, in one command
G1=work10/g_test_common_w9prep.parquet G2=work10/g2_test_diffaddr_sym_w9prep.parquet \
    experiments/final_assemble.sh work10 work10/test_prob_v10ce2.parquet output_v10ce2/ce2_band.json output_final_v10 0
./make_zip.sh output_final_v10 v10-final
```
Research simulations shipped for reference: `$X/rule_sim.py`, `$X/probe1_sim.py`, `$X/corrector_sim.py`
(with `proxy_cache.py` / `proxy_mine.py`); `$X/frce_probe.py` built the France-only P5 LB probe.

**Leakage rule.** The cross-encoder is trained on the candidate pairs of 99,948 training S1 (`ce_train.parquet`);
`--exclude-s1` keeps exactly these S1 out of the matcher's training rows, so the `ce` feature is never optimistic.
`run.py` stops with an error if any pair lacks a GPU score. Everything is cached in the work folder; delete a
cached parquet to force a rebuild. `candidate_pairs.tsv` is exactly the set of pairs the matcher scores.

## Source
| file | role |
|---|---|
| `src/prep.py` | Indic → Latin transliteration (stdlib `unicodedata`), native-word dictionary (`translate`), abbreviations, legal-word removal, French normalisation, consonant skeletons, house-number and blocking keys. Self-test: `python src/prep.py` |
| `src/native_dict.py` | mines `native_dict.tsv` from training match pairs |
| `src/blocking.py` | streaming normalisation to parquet; country-scoped key join, IDF pool, fuzzy re-rank, top-k |
| `src/features.py` | rapidfuzz / Jaccard / postcode / house-number / TF-IDF / blocking-context features, plus GPU and extra columns |
| `src/embed.py` | GPU: e5-small cosine(name), cosine(address) per pair |
| `src/ce_data.py` | builds the cross-encoder training set (S1 outside the matcher's sample) |
| `src/ce.py` | GPU: fine-tunes and applies the e5-small cross-encoder |
| `src/tune.py` | Optuna LightGBM search → `best_params.json` |
| `src/run.py` | stages `pairs`, `train`, `predict`, `all`; candidate options `--rescue`, `--knn`, `--rare-empty` |

## Final submission (public LB 0.986691): exact steps after v11
The final file merges two runs by country: US/India rows from v10 (above), French rows from v9 with two France-adapted
cross-encoders. Run from the repo root after the v9 and v10 sections (`X=experiments` there = `src/experiments` in the zip).
```bash
# L1. US/India threshold 0.58 (CV-optimal with the conflict rule: experiments/country_thr_cv.py on the stage-2 OOF,
#     CV 0.98903 -> 0.98921); France kept at 0.68 via offset 0.10 -> output_final_v11L1 (LB 0.98533)
T=0.58 G1=work10/g_test_common_w9prep.parquet G2=work10/g2_test_diffaddr_sym_w9prep.parquet \
  bash $X/final_assemble.sh work10 work10/test_prob_v10ce2.parquet output_v10ce2/ce2_band.json output_final_v11L1 0.10
# F5. second France-adapted cross-encoder (P5b): the F1 recipe with base = cross-encoder fold h1 (ce2 h1), same
#     885,573 French pairs (Kaggle kernel = kaggle/frce_train.py with the h1 model attached); it scores the France pairs
#     of work10/frce_all -> work10/frce_b/test_ce.parquet
# F6. French rows: v9 recipe (ce2 stage 2 fitted on half-1 S1, as LB upload A) with P5 + P5b averaged, France t 0.91
python $X/ce2_band.py --work work9 --band work9/ce2band --oof work9/oof_v9.parquet --test-prob work9/test_prob_v9.parquet \
  --meta work9/tune_meta.parquet --report work9/report_v9.json --src output_v9 --out output_v9ce3ab_control_p5b \
  --save-test-prob work9/test_prob_v9_p5b.parquet --ce3 work9/ce3ens --frce work9/frce_out work10/frce_b \
  --frce-min-cov 0.95 --write-arm control --margin -1
bash $X/final_assemble.sh work9 work9/test_prob_v9_p5b.parquet output_v9ce3ab_control_p5b/ce2_band.json output_final_v9_p5b_fr0.19 0.19
# F7. merge by country: US/India rows from v10 (L1), French rows from v9 (F6) -> LB 0.986371
python $X/country_merge.py --base output_final_v11L1 --take output_final_v9_p5b_fr0.19 --countries France \
  --out output_probe_hyb2_p5b_frt0.19
# F8. French noise-variant recovery + high-confidence rule reversal -> output/ (LB 0.986691; validator included)
python $X/france_noise_accept.py --base output_probe_hyb2_p5b_frt0.19 --reverse-rules --out output
```
`france_noise_accept.py --reverse-rules` reproduces the uploaded file exactly (identical match and candidate sets per
row, 5,817,951 matches). The France threshold 0.91 comes from `france_thr_model.py` (calibration fitted to LB probes).
