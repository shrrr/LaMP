# Test cases

This directory contains lightweight permittivity-map test cases used by the
shell scripts in `scripts/`. Each file is a `.npy` array of shape
`(grid_num, grid_num)` with values in `[1, max_params]`.

Common examples:

- `epsilon_twinCircle.npy`: two dielectric cylinders (the synthetic TwinCircle
  target in Section IV-B).
- `epsilon_australia.npy`: the Austria target used for the noise study in
  Section IV-B.
- `epsilon_Star.npy`: a star-shaped scatterer used for the iteration trajectory
  illustration in Section IV-B.
- `epsilon_LetterT.npy`, `epsilon_LetterH.npy`, `epsilon_LetterU.npy`: ablation
  geometries used in the supplemental analyses.

Out-of-distribution (MNIST and Fashion-MNIST) test cases are generated on the
fly by `lamp/mnist_fashion_case_utils.py`.

Fresnel experimental data is not included in this directory; see
`../Fresnel/README.md`.
