# LaMP

Code for *"Latent Manifold Projection for Plug-and-Play Generative Priors in
Electromagnetic Inverse Scattering"* (IEEE Trans. Antennas Propag., 2026).

LaMP solves 2D electromagnetic inverse scattering by alternating a physics-based
contrast-source update with periodic projection onto the latent manifold of a
finetuned Stable Diffusion VAE.

## Contents

```
lamp/      core package (solver, network, VAE wrapper)
ldm/       vendored Latent Diffusion VAE components
configs/   VAE configuration
data/      test targets and synthetic forward fields
scripts/   reproduction scripts
```

## Setup

```bash
pip install -r requirements.txt
wget -O lamp/checkpoints/sd_vae_ft.pth \
  https://github.com/shrrr/LaMP/releases/download/v0.1.0/sd_vae_ft.pth
```

## Usage

```bash
bash scripts/gen_forwards.sh               # synthetic forward fields (run once)
bash scripts/run_fig3_twincircle.sh        # TwinCircle 8x12 (synthetic)
bash scripts/run_fig6_star.sh              # Star 16x32 (synthetic)
bash scripts/run_fresnel_foamtwindiel.sh   # FoamTwinDiel (Fresnel measured data*)
python scripts/plot_results.py             # reconstruction panel
```

Each script runs DeepCSI, CasVAE and LaMP on one target. For the full CLI, see
`python -m lamp.main --help`.

\*The Fresnel measured data is not redistributed here; see `data/Fresnel/README.md`.

## Results

Each run writes contrast snapshots and a `results.csv` (rMSE, SSIM) to
`results/<case>/`. `plot_results.py` assembles the panel into
`figures/reconstructions.png`.

## Citing

```bibtex
@article{sun2026lamp,
  title   = {Latent Manifold Projection for Plug-and-Play Generative Priors in
             Electromagnetic Inverse Scattering},
  author  = {Sun, Haoran and Zhou, Hongyu and Li, Maokun and Xu, Shenheng and Yang, Fan},
  journal = {IEEE Transactions on Antennas and Propagation},
  year    = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)). `ldm/` is vendored from the
[Latent Diffusion Model project](https://github.com/CompVis/latent-diffusion)
under its own license.
