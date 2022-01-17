# Shape Noise Generation

All the files within the directory can be used to propagate shape noise to the weak lensing convergence. All parameters in the noise map settings follow that of Massive Neutrino Simulations (MassiveNuS) and LSST-like surveys. One needs to install LensTools which is a python package containing many useful functions in doing weak lensing analysis beforehand in order to run these files.

## create kappa noise from KS

In this file, we first generate i.i.d Gaussian shear noise. Then based on Kaiser-Squires inversion we propagate it to convergence. Only noise maps are generated and one has to add them to convergence maps later on.

## create kappa noise from Cl

Ideally, the convergence noise should be indepedent among pixels. However, correlation will be introduced during the transformation from shear to convergence and we would like to simulate this correlation.

In this file, we implement the conclusion that the power spectrum of convergence noise should be the same as that of the shearand it is a constant. We input this white noise power spectrum and use the function fromConvPower to generate correlated Gaussian noise maps for convergence. However the function itself is not well-understood which makes this method less convincing than the first method.
