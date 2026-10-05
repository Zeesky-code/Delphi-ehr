#!/usr/bin/env bash
# Train the FiLM variants, one output folder per variant and seed:
#   $RUNS_DIR/<variant>/seed<seed>/  (ckpt.pt, train.log, summary.json)
#
# Usage:
#   scripts/run_film_variants.sh                         # every variant
#   scripts/run_film_variants.sh baseline qkv attn_pre   # just these
#   scripts/run_film_variants.sh qkv -- --device=cuda --max_iters=2000   # extra train.py args after --
#   scripts/run_film_variants.sh -- --device=cuda --wandb_log=True       # log to wandb
#
# With wandb on, each run is named <variant>-seed<seed> and grouped by variant, so seeds average together.
#
# Environment:
#   DATA=mimic4        dataset: mimic4 (default; build it with data/mimic/build_mimic4.py) or synthetic
#   SEEDS="42 43 44"   seeds to run (default 42); use several before trusting small differences
#   ANCHOR=qkv         film_location for the mode/context/layer/width sweeps (default both)
#   RUNS_DIR=...       output root (default runs/film-<DATA>)
#   PYTHON=python3     interpreter to use
#
# Compare finished runs with: python3 scripts/compare_runs.py runs/film-mimic4
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-mimic4}
RUNS_DIR=${RUNS_DIR:-runs/film-$DATA}
SEEDS=${SEEDS:-42}
ANCHOR=${ANCHOR:-both}
PYTHON=${PYTHON:-python3}
case "$DATA" in
    mimic4)    FILM=config/train_delphi_mimic4_film.py;  BASELINE=config/train_delphi_mimic4_baseline.py ;;
    synthetic) FILM=config/train_delphi_film.py;         BASELINE=config/train_delphi_labs_baseline.py ;;
    *) echo "DATA must be mimic4 or synthetic" >&2; exit 1 ;;
esac

# name  config  overrides
VARIANTS="
baseline        $BASELINE
attn_pre        $FILM --film_location=attn_pre
qkv             $FILM --film_location=qkv
attn            $FILM --film_location=attn
mlp             $FILM --film_location=mlp
both            $FILM --film_location=both
${ANCHOR}_gamma     $FILM --film_location=$ANCHOR --film_mode=gamma
${ANCHOR}_beta      $FILM --film_location=$ANCHOR --film_mode=beta
${ANCHOR}_numeric   $FILM --film_location=$ANCHOR --film_context=numeric
${ANCHOR}_layer0    $FILM --film_location=$ANCHOR --film_layers=0
${ANCHOR}_layers01  $FILM --film_location=$ANCHOR --film_layers=0,1
${ANCHOR}_h32       $FILM --film_location=$ANCHOR --film_hidden_dim=32
qkv_shuffled            $FILM --film_location=qkv --film_shuffle_values=True
${ANCHOR}_numeric_shuffled  $FILM --film_location=$ANCHOR --film_context=numeric --film_shuffle_values=True
"

selected=()
extra=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
    selected+=("$1"); shift
done

is_selected() {
    [ ${#selected[@]} -eq 0 ] && return 0
    for s in "${selected[@]}"; do [ "$s" = "$1" ] && return 0; done
    return 1
}

while read -r name config overrides; do
    [ -z "$name" ] && continue
    is_selected "$name" || continue
    for seed in $SEEDS; do
        out="$RUNS_DIR/$name/seed$seed"
        mkdir -p "$out"
        echo "=== $name (seed $seed) -> $out"
        # shellcheck disable=SC2086  # overrides is a word list on purpose
        "$PYTHON" train.py "$config" $overrides --seed="$seed" --out_dir="$out" \
            --wandb_run_name="$name-seed$seed" --wandb_group="$name" \
            ${extra[@]+"${extra[@]}"} 2>&1 | tee "$out/train.log"
    done
done <<< "$VARIANTS"
