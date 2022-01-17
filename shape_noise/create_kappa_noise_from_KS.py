import time
start_program = time.time()

import lenstools
from astropy import units as u
from lenstools import ConvergenceMap 
from astropy.io import fits
import numpy as np
import multiprocessing as mp


import numpy as np
from astropy.io import fits
from lenstools import ShearMap
from lenstools import ConvergenceMap 
from lenstools.utils.defaults import load_fits_default_convergence
from lenstools.utils.defaults import load_fits_default_shear

#Choose how many noise maps to create (number of maps = i_max - i_min)
i_min = int(input("Enter value of smallest map index:"))
i_max = int(input("Enter value of largest map index:"))

#The size of the noise maps. It is adaptable. Here is the size of the convergence map from MassiveNuS.
xpixels = 512
ypixels = 512

#sigma is the shape noise parameter and N_g is the number of galaxies per pixel. Both of them are of LSST-survey level.
#30 is the number of observed galaxies per arcmin. 3.5 degree is the angular size of the convergence map from MassiveNuS.
sigma = 0.28
N_g = 30 * (3.5/512)**2 * 3600

#header information of shear shape noise file
hdr = fits.Header.fromstring("""\
SIMPLE  =                    T / conforms to FITS standard                      
BITPIX  =                  -32 / array data type                                
NAXIS   =                    3 / number of array dimensions                     
NAXIS1  =                   512                                                  
NAXIS2  =                   512
NAXIS3  =                    2 
EXTEND  =                    T  
H0      =                 70.0 / Hubble constant in km/s*Mpc                    
H       =                  0.7 / Dimensionless Hubble constant                  
OMEGA_M =              0.29997 / Dark Matter density                            
OMEGA_L =              0.69995 / Dark Energy density                            
W0      =                 -1.0 / Dark Energy equation of state                  
WA      =                  0.0 / Dark Energy running equation of state          
Z       =                  1.0 / Redshift of the background sources             
ANGLE   =                  3.5 / Side angle in degrees                          
ITYPE   = 'ShearMap'     / Image type 
""", sep='\n')

def generate_shear_noise(i_ini, i_fin, xpixels1, ypixels1, sigma1, N_g1):
    for i in np.arange(i_ini, i_fin):
        shape_noise = np.zeros((2, xpixels1, ypixels1))
        gamma1 = np.random.normal(loc=0.0, scale=sigma1/np.sqrt(N_g1), size=(xpixels1, ypixels1))
        gamma2 = np.random.normal(loc=0.0, scale=sigma1/np.sqrt(N_g1), size=(xpixels1, ypixels1))
        for j in range(xpixels1):
            for k in range(ypixels1):
                shape_noise[0,j,k] = gamma1[j,k]
                shape_noise[1,j,k] = gamma2[j,k]
        hdu = fits.PrimaryHDU(shape_noise, header=hdr)
        #Save the generated shape noise maps to a specific storage disk.
        hdu.writeto('/e/ocean1/users/lgong/shear_shape_noise_LSST/shear_shape_noise_LSST%s.fits'%(i), overwrite=True)
        print("computation of shear shape noise #%s is done"%(i))
        
#We use a function from LensTools which propagates shape noise to convergence based on the Kaiser-Squires inversion.
def generate_conv_noise(i_ini, i_fin):
    for i in np.arange(i_ini, i_fin):
        shear_noise = ShearMap.load('/e/ocean1/users/lgong/shear_shape_noise_LSST/shear_shape_noise_LSST%s.fits'%(i), format=load_fits_default_shear)
        conv_noise = shear_noise.convergence()
        conv_noise.save('/e/ocean1/users/lgong/conv_shape_noise_LSST/conv_shape_noise_LSST%s.fits'%(i))
        print("computation of conv shape noise #%s is done"%(i))
        
#parallel computation
def generate_shear_noise_para(p):
    return generate_shear_noise(p[0], p[1], p[2], p[3], p[4], p[5]) 

def generate_conv_noise_para(p):
    return generate_conv_noise(p[0], p[1]) 


pool = mp.Pool(processes=30)
pool.map(generate_shear_noise_para, [[i_min, i_max, xpixels, ypixels, sigma, N_g]])

pool = mp.Pool(processes=30)
pool.map(generate_conv_noise_para, [[i_min, i_max]])

end_program = time.time()
print('\nTime taken for execution (seconds): ', end_program - start_program)