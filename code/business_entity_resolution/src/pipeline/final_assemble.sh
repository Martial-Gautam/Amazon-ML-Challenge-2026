#!/usr/bin/env bash
# Decision layer for one build, then validation: per-country thresholds (US/India + ambiguity rejection),
# French look-alike rules P1 / P2', validator. Run from the package root.
#
#   bash pipeline/final_assemble.sh WORK PROB JSON OUT [FR_OFFSET=0]
#     WORK  work folder of the build (test_s1 / test_cand parquet)
#     PROB  i, j, prob for every scored test pair (ce2_band.py --save-test-prob)
#     JSON  the stage's threshold / one-owner choice (ce2_band.json)
#     FR_OFFSET  France threshold = US/India threshold + offset
#   env T=0.58  US/India threshold (overrides the json); G1 / G2: P1 / P2' pair lists (built into WORK when missing);
#   env PY=python3  interpreter
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-python3}
W=$1 PROB=$2 JSON=$3 OUT=$4 FR_OFFSET=${5:-0}
G1=${G1:-$W/g_test_common.parquet} G2=${G2:-$W/g2_test_diffaddr_sym.parquet}
$PY "$D/country_threshold.py" --work "$W" --prob "$PROB" --json "$JSON" ${T:+--t "$T"} --country France --offset "$FR_OFFSET" \
    --conflict-delta 0.2 --conflict-countries US,India --out "${OUT}_thr"
[ -f "$G1" ] || $PY "$D/france_rule.py" --work "$W" --pairs "$PROB" --out "$G1"
[ -f "$G2" ] || $PY "$D/france_rule.py" --work "$W" --pairs "$PROB" --address different --out "$G2"
$PY "$D/france_probe.py" --base "${OUT}_thr" --groups "$G1" --mode reject --out "${OUT}_p1"
$PY "$D/france_probe.py" --base "${OUT}_p1" --groups "$G2" --mode reject --out "$OUT"
$PY utils/validate_submission.py -m "$OUT/matching_results.tsv" -c "$OUT/candidate_pairs.tsv" -t dataset/test --check-ids
