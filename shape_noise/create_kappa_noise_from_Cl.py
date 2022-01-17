import time
start_program = time.time()

import lenstools
from astropy import units as u
from lenstools import ConvergenceMap 
from astropy.io import fits
import numpy as np
import multiprocessing as mp

#######input parameters#########

i_min = int(input("Enter value of smallest map index:"))

i_max = int(input("Enter value of largest map index:"))


#construct a noise generator
noise_gen = lenstools.image.noise.GaussianNoiseGenerator(shape=(512,512), side_angle=3.5*u.degree)

#The power spectrum of shot noise coming from intrinsic ellipticities, with galaxy surface density at DES(~5)/LSST(~30) level
arcmin2rad = 1/60/180 * np.pi 
n_g = 30/(arcmin2rad)**2

l = np.logspace(2,4,30)
N_l = np.ones(len(l))*0.3**2/n_g

Cl_noise = np.zeros((2, len(l)))
Cl_noise[0, :] = l
Cl_noise[1, :] = N_l

#construct an array of seeds
seed = np.arange(i_min, i_max)

#add noise maps to the noiseless convergence maps from MassiveNuS
def add_noise(i_ini, i_fin, mv, om, As, maps, z):
    for i in np.arange(i_ini, i_fin):
        filename =  "/e/ocean1/users/lgong/convergence_gal_mnv%s_om%s_As%s/Maps%s/WLconv_z%s_%sr.fits"%('{:7.5f}'.format(mv), '{:7.5f}'.format(om), '{:6.4f}'.format(As), '{0:02d}'.format(maps), '{:3.2f}'.format(z), '{:04d}'.format(i))
        noiseless_map = ConvergenceMap.load(filename)
        noiseonly_map = noise_gen.fromConvPower(Cl_noise, seed=seed[i-1], fill_value="extrapolate")
        noisy_map = noiseless_map + noiseonly_map
        noisy_map.save("MassiveNuS_noisymaps_LSST/convergence_gal_mnv%s_om%s_As%s/Maps%s/WLconv_z%s_%sr_wnoise.fits"%('{:7.5f}'.format(mv), '{:7.5f}'.format(om), '{:6.4f}'.format(As), '{0:02d}'.format(maps), '{:3.2f}'.format(z), '{:04d}'.format(i)))
        print("computation of realization %s is done"%(i))
        
#parallelization
def add_noise_parallelisation(p):
    return add_noise(p[0], p[1], p[2], p[3], p[4], p[5], p[6])

####Other function arguments#######
mv=0.1

om=0.3

As=2.1

maps=[15]

z=[1.5]

pool = mp.Pool(processes=30)
pool.map(add_noise_parallelisation, [[i_min, i_max, mv, om, As, maps[i], z[i]] for i in range(1)])

end_program = time.time()
print('\nTime taken for execution (seconds): ', end_program - start_program)
