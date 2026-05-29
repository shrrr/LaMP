#!/usr/bin/env bash
# Generate the synthetic forward fields for the Fig3 (TwinCircle 8x12) and
# Fig6 (Star 16x32) cases. Outputs to data/forward/<case>_ds_solution.npy.
# Fresnel (FoamTwinDiel) inverts measured .exp data directly, no forward needed.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

gen() {
    local CASE=$1; local N_INC=$2; local N_REC=$3; local TAG=$4
    local TMP="./data/forward/_tmp_${TAG}/"
    mkdir -p "$TMP"
    "$PYTHON_BIN" -u -m lamp.main \
        --expname "fwd" --basedir "$TMP" \
        --freq "3," --L_doi 0.3 --R_t 2 --R_r 2.2 --N_rec "$N_REC" --N_inc "$N_INC" \
        --params_path "./data/testcases/$CASE" \
        --method fwd --inc_wave cir --max_params 2 \
        --J_network single-mlp --netdepth 8 --netwidth 256 2>&1 | tail -1
    cp "$TMP"fwd_*/ds_solution.npy "./data/forward/${TAG}_ds_solution.npy"
    rm -rf "$TMP"
    echo "  -> data/forward/${TAG}_ds_solution.npy"
}

echo "=== TwinCircle 8x12 ==="; gen "epsilon_twinCircle.npy" 8 12 "twinCircle_8x12"
echo "=== Star 16x32 ===";      gen "epsilon_Star.npy" 16 32 "Star_16x32"
echo "=== DONE ==="
