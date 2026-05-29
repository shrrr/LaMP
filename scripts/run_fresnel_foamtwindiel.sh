#!/usr/bin/env bash
# FoamTwinDiel Fresnel case, transceiver configurations 8x241 / 4x8 / 4x4.
# Runs DeepCSI, CasVAE and LaMP per configuration.
# Needs the measured FoamTwinDielTM data under data/Fresnel/ (see data/Fresnel/README.md).
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
VAE_CKPT="./lamp/checkpoints/sd_vae_ft.pth"
VAE_CFG="./configs/kl_f8.yaml"
OUT="./results/fresnel_foamtwindiel/"; mkdir -p "$OUT"

run() {  # $1=label $2=N_inc $3=N_rec $4=epsilon_network $5=extra
    "$PYTHON_BIN" -u -m lamp.main \
        --expname "$1" --basedir "$OUT" \
        --freq "3," --L_doi 0.17 --R_t 1.67 --R_r 1.67 \
        --N_rec 360 --N_inc "$2" --N_rec_data 241 --N_rec_use "$3" --N_inc_data 18 --N_inc_use "$2" \
        --max_iter 3000 --lrate 5e-2 --lrate_decay 4 --params_lrate_decay 2 \
        --params_path "./data/Fresnel/FoamTwinDielTM.npy" --recdata_path "./data/Fresnel/FoamTwinDielTM.exp" \
        --method fdtot-isp --epsilon_network "$4" \
        --noise_ratio 0.05 --noise_type gaussion \
        --vae_model sd_vae_ft --vae_latent_dim 256 --vae_ckpt_path "$VAE_CKPT" --vae_config_path "$VAE_CFG" \
        --result_file "${OUT}results.csv" --max_params 3 \
        --i_projection 200 --i_projection_start 300 \
        --proj_inner_max 200 --proj_inner_min 50 --proj_inner_tol 1e-2 \
        --proj_z_init safe_encode --proj_z_init_ood_tau 0.15 --proj_ood_metric mean_sigma \
        --proj_tv_weight 0.01 \
        --regularizer tv_l1 --regularizer_weight 0.1 --regularizer_decay 0.5 \
        --save_metric --seed 9 $5 2>&1 | tail -1
}

for CFG in "8 241" "4 8" "4 4"; do
    read NI NR <<< "$CFG"
    echo "=== Fresnel FoamTwinDiel ${NI}x${NR} ==="
    run "ft_${NI}x${NR}_dc"     "$NI" "$NR" proj_param  "--disable_projection"
    run "ft_${NI}x${NR}_casvae" "$NI" "$NR" paramzation ""
    run "ft_${NI}x${NR}_lamp"   "$NI" "$NR" proj_param  ""
done
echo "done -> $OUT"
