#!/usr/bin/env bash
# TwinCircle 8x12 synthetic case. Runs DeepCSI, CasVAE and LaMP.
# Needs data/forward/twinCircle_8x12_ds_solution.npy (see scripts/gen_forwards.sh).
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
VAE_CKPT="./lamp/checkpoints/sd_vae_ft.pth"
VAE_CFG="./configs/kl_f8.yaml"
OUT="./results/fig3_8x12/"; mkdir -p "$OUT/_noise"
N="${OUT}_noise/n.npy"
FWD="./data/forward/twinCircle_8x12_ds_solution.npy"

run() {  # $1=label $2=epsilon_network $3=extra
    "$PYTHON_BIN" -u -m lamp.main \
        --expname "$1" --basedir "$OUT" \
        --freq "3," --L_doi 0.3 --R_t 2 --R_r 2.2 --N_rec 12 --N_inc 8 \
        --max_iter 3000 --lrate 5e-2 --lrate_decay 4 --params_lrate_decay 2 \
        --params_path "./data/testcases/epsilon_twinCircle.npy" --recdata_path "$FWD" \
        --method fdtot-isp --epsilon_network "$2" \
        --noise_ratio 0.05 --noise_type uniform_try \
        --vae_model sd_vae_ft --vae_latent_dim 256 --vae_ckpt_path "$VAE_CKPT" --vae_config_path "$VAE_CFG" \
        --result_file "${OUT}results.csv" --max_params 2 \
        --i_projection 200 --i_projection_start 300 \
        --proj_inner_max 50 --proj_inner_min 50 --proj_inner_tol 1e-2 \
        --proj_z_init safe_encode --proj_z_init_ood_tau 0.15 --proj_ood_metric mean_sigma \
        --proj_tv_weight 0.01 \
        --regularizer tv_l1 --regularizer_weight 0.1 --regularizer_decay 0.5 \
        --save_metric --seed 100 $3 2>&1 | tail -1
}

echo "=== Fig3 TwinCircle 8x12 ==="
run fig3_dc     proj_param  "--disable_projection --save_noisy_data $N"
run fig3_casvae paramzation "--load_noisy_data $N"
run fig3_lamp   proj_param  "--load_noisy_data $N"
echo "done -> $OUT"
