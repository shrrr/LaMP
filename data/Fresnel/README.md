# Fresnel database

The Fresnel experimental scattering database is not redistributed with this
repository. To run the compact Fresnel example in `scripts/`, please obtain the
FoamTwinDielTM measurement data from the official Fresnel release:

> J.-M. Geffrin, P. Sabouroux, and C. Eyraud, "Free space experimental
> scattering database continuation: experimental set-up and measurement
> precision," Inverse Problems, vol. 21, no. 6, pp. S117--S130, 2005.

Place the files used by the compact 4 x 4 sparse example in this directory:

```
data/Fresnel/
├── FoamTwinDielTM.exp
└── FoamTwinDielTM.npy
```

The `.npy` file is the ground-truth permittivity reference used only for metric
evaluation, not for inversion. Additional Fresnel cases can be placed in the
same directory using the same naming convention if broader experiments are
needed.
