# LaMP VAE checkpoint

All reported experiments use one finetuned SD-VAE prior, `sd_vae_ft.pth`
(~320 MB). It is too large for a plain Git object and is distributed as a GitHub
release asset. Place it directly under `lamp/checkpoints/sd_vae_ft.pth`:

```bash
wget -O lamp/checkpoints/sd_vae_ft.pth \
  https://github.com/shrrr/LaMP/releases/download/v0.1.0/sd_vae_ft.pth
```

The inversion scripts load it through `--vae_ckpt_path lamp/checkpoints/sd_vae_ft.pth`.
