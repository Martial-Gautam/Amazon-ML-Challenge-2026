#!/usr/bin/env bash
# Rebuilds output/matching_results.tsv and output/candidate_pairs.tsv (public LB 0.986691) from the competition data.
#
#   bash code/business_entity_resolution/reproduce.sh            # every stage, in order
#   bash code/business_entity_resolution/reproduce.sh v10 final  # only the named stages
#
# Run from the package root, with dataset/ and utils/ (the organisers' validator) next to code/.
# Stages marked GPU need CUDA. A step whose output already exists is skipped, so an interrupted run resumes.
set -euo pipefail
PY=${PY:-python3}
S=code/business_entity_resolution/src
P=$S/pipeline
PARAMS=code/business_entity_resolution/best_params.json
export PY

step() {  # step OUTPUT command...  -> runs the command unless OUTPUT exists
  local out=$1; shift
  if [ -e "$out" ]; then echo "· skip   $out"; else echo "▶ build  $out"; "$@"; fi
}
text() {  # raw name/address per record for the GPU scripts: <work>/text/{split}_{s1,cand}.parquet
  mkdir -p "$1/text"
  $PY -c "import polars as pl
for s in ('train','test'):
    for w in ('s1','cand'):
        pl.scan_parquet(f'$1/{s}_{w}/*.parquet').select('idx','business_name','business_address').sink_parquet(f'$1/text/{s}_{w}.parquet')"
}
gpu_features() {  # GPU: e5 cosines + small cross-encoder score for every candidate pair of a build
  step "$1/text/test_cand.parquet" text "$1"
  step "$1/gpu/test_emb.parquet" $PY $S/embed.py --text "$1/text" --pairs "$1/emb_in" --out "$1/gpu"
  step "$1/gpu/test_ce.parquet"  $PY $S/ce.py --text "$1/text" --pairs "$1/emb_in" --out "$1/gpu" --model-dir work/gpu/ce_model
}
rerank_band() {  # GPU: score a build's uncertainty band with both re-ranker halves
  local w=$1
  for h in 0 1; do
    step "$w/ce2band/h$h/test_ce.parquet" $PY $S/ce.py --text "$w/text" --pairs "$w/band/h$h" --out "$w/ce2band/h$h" \
      --model-dir work/ce2_h$h/ce_model
  done
}

FLAGS9="--work work9 --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys --rescue \
  --knn work/n4 --knn-top 5 --rare-empty --exclude-s1 work/ce/ce_train.parquet"
FLAGS10="--work work10 --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys --rescue --knn work/n4 \
  --knn-top 5 --knn2 work/ret --knn2-top 10 --rare-empty --exclude-s1 work/ce/ce_train.parquet --tfidf-all"
TAG9=k30_p1500_ap_ak_use20_rs_knn5_re
TAG10=k30_p1500_ap_ak_use20_rs_knn5_rk10_re

stage_base() {  # normalised records + raw text + small cross-encoder (GPU)
  step work/train_pairs_k30_p1500_ap_ak_use20.parquet \
    $PY $S/run.py pairs --work work --n-train 2206821 --k-use 20 --pool 1500 --addr-partial --addr-keys
  step work/text/test_cand.parquet text work
  step work/ce/ce_train.parquet $PY $S/ce_data.py --n-train 600000 --out work/ce
  step work/gpu/ce_model $PY $S/ce.py --text work/text --pairs work/ce --out work/gpu
}
stage_neighbours() {  # GPU: two nearest-neighbour candidate sources
  step work/retriever $PY $P/ret_train.py --work work --ce work/ce/ce_train.parquet --out work/retriever
  for s in train test; do
    step work/n4/${s}_knn.parquet  $PY $P/knn_search.py --work work --split $s --model intfloat/multilingual-e5-small --out work/n4
    step work/ret/${s}_knn.parquet $PY $P/knn_search.py --work work --split $s --model work/retriever --out work/ret
  done
}
stage_v9() {  # build v9 (source of the French rows): candidates, features, classifier
  step work9/test_cand $PY $S/run.py pairs $FLAGS9
  step work9/tfw_test.parquet    $PY $P/tfw_cols.py    --work work9 --tag $TAG9
  step work9/rarity_test.parquet $PY $P/rarity_cols.py --work work9 --tag $TAG9
  step work9/train_features.parquet $PY $S/run.py train $FLAGS9 --features-only
  step work9/test_features         $PY $S/run.py predict $FLAGS9 --features-only
  gpu_features work9
  step work9/test_prob_v9.parquet $PY $P/retrain_extra.py --work work9 --params $PARAMS \
    --extra 'work9/gpu/{split}_emb.parquet' --extra 'work9/gpu/{split}_ce.parquet' \
    --extra 'work9/tfw_{split}.parquet' --extra 'work9/rarity_{split}.parquet' --out output_v9 --tag v9
}
stage_rerank() {  # GPU: e5-base re-ranker (one model per S1 half) + stacking on v9 + French rule lists
  for h in 0 1; do
    step work/ce2/ce_train_h$h.parquet $PY $S/ce_data.py --half $h --n-ce 200000 --k-use 20 --out work/ce2
    step work/ce2_h$h/ce_model $PY $S/ce.py --text work/text --pairs work/ce2 --train-file ce_train_h$h.parquet \
      --out work/ce2_h$h --model intfloat/multilingual-e5-base --lr 3e-5 --epochs 1 --batch 64
  done
  step work9/band/h1 $PY $P/band_export.py --oof work9/oof_v9.parquet --test-prob work9/test_prob_v9.parquet --out work9/band
  rerank_band work9
  step work9/test_prob_v9ce2.parquet $PY $P/ce2_band.py --work work9 --band work9/ce2band --oof work9/oof_v9.parquet \
    --test-prob work9/test_prob_v9.parquet --meta work9/tune_meta.parquet --report work9/report_v9.json \
    --src output_v9 --out output_v9ce2
  step work9/g_test_common.parquet $PY $P/france_rule.py --work work9 --pairs work9/test_prob_v9.parquet \
    --out work9/g_test_common.parquet
  step work9/g2_test_diffaddr_sym.parquet $PY $P/france_rule.py --work work9 --pairs work9/test_prob_v9.parquet \
    --address different --out work9/g2_test_diffaddr_sym.parquet
}
stage_france() {  # GPU: two domain-adapted French cross-encoders (self-training on the unlabelled French test records)
  step work9/frce/pairs/ce_train.parquet $PY $P/fr_synth.py --out work9/frce --score work9/test_prob_v9ce2.parquet
  step work9/frce_A/test_ce.parquet $PY $S/ce.py --text work9/frce/text --pairs work9/frce/pairs --out work9/frce_A \
    --model work/ce2_h0/ce_model --lr 2e-5 --epochs 1 --batch 64
  step work9/frce_B/test_ce.parquet $PY $S/ce.py --text work9/frce/text --pairs work9/frce/pairs --out work9/frce_B \
    --model work/ce2_h1/ce_model --lr 2e-5 --epochs 1 --batch 64
}
stage_v10() {  # build v10 (source of the US/India rows): retriever candidates, features, 3-seed classifier, re-ranker
  step work10/test_cand $PY $S/run.py pairs $FLAGS10
  step work10/rarity_test.parquet $PY $P/rarity_cols.py --work work10 --tag $TAG10
  step work10/train_features.parquet $PY $S/run.py train $FLAGS10 --features-only
  step work10/test_features         $PY $S/run.py predict $FLAGS10 --features-only
  gpu_features work10
  step work10/test_prob_v10.parquet $PY $P/retrain_extra.py --work work10 --params $PARAMS \
    --extra 'work10/gpu/{split}_emb.parquet' --extra 'work10/gpu/{split}_ce.parquet' --extra 'work10/rarity_{split}.parquet' \
    --seeds 3 --out output_v10 --tag v10
  step work10/band/h1 $PY $P/band_export.py --oof work10/oof_v10.parquet --test-prob work10/test_prob_v10.parquet --out work10/band
  rerank_band work10
  step work10/test_prob_v10ce2.parquet $PY $P/ce2_band.py --work work10 --band work10/ce2band --oof work10/oof_v10.parquet \
    --test-prob work10/test_prob_v10.parquet --meta work10/tune_meta.parquet --src output_v10 --out output_v10ce2
}
stage_final() {  # decisions per country, merge, French noise-variant recovery, validation -> output/
  # US/India rows: threshold 0.58 + ambiguity rejection (the French rows of this file are replaced below)
  step output_final_usin env T=0.58 G1=work9/g_test_common.parquet G2=work9/g2_test_diffaddr_sym.parquet \
    bash $P/final_assemble.sh work10 work10/test_prob_v10ce2.parquet output_v10ce2/ce2_band.json output_final_usin 0.10
  # French rows: v9 stacking fitted on S1 half 1 with French models A + B averaged; France threshold 0.72 + 0.19 = 0.91
  step work9/test_prob_final_fr.parquet $PY $P/ce2_band.py --work work9 --band work9/ce2band --oof work9/oof_v9.parquet \
    --test-prob work9/test_prob_v9.parquet --meta work9/tune_meta.parquet --src output_v9 --out output_v9_fr \
    --save-test-prob work9/test_prob_final_fr.parquet --half 1 --margin -1 \
    --frce ${FR_A:-work9/frce_A} ${FR_B:-work9/frce_B} --frce-min-cov 0.95
  step output_final_fr bash $P/final_assemble.sh work9 work9/test_prob_final_fr.parquet output_v9_fr/ce2_band.json \
    output_final_fr 0.19
  step output_merged $PY $P/country_merge.py --base output_final_usin --take output_final_fr --countries France --out output_merged
  $PY $P/france_noise_accept.py --base output_merged --reverse-rules --out output
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(base neighbours v9 rerank france v10 final)
for s in "${STAGES[@]}"; do echo "══ stage $s"; "stage_$s"; done
echo "done: output/matching_results.tsv, output/candidate_pairs.tsv"
