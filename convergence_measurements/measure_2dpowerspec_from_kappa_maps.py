from lenstools import ConvergenceMap
from lenstools.statistics.ensemble import Ensemble
from lenstools.utils.decorators import Parallelize
from astropy.io import fits
from astropy import units as u
import logging
import numpy as np
import pandas as pd


def measure_power_spectrum(filename,l_edges):
    #l_edges is the bin edge of the multipole number
    #this function loads the convergence maps and compute its power spectrum
    conv_map = ConvergenceMap.load(filename)
    l,Pl = conv_map.powerSpectrum(l_edges)
    return Pl

logging.basicConfig(level=logging.DEBUG)

@Parallelize.masterworker
def main(pool, l_edges, map_list):
    #use the center of the bin as the multipole 
    l = 0.5*(l_edges[:-1] + l_edges[1:])
    #fast computation in a parallel way
    conv_ensemble = Ensemble.compute(map_list, callback_loader=measure_power_spectrum, pool=pool, l_edges=l_edges)
    mean = conv_ensemble.mean(0)
    error = np.sqrt(conv_ensemble.covariance().values.diagonal())
    return l, mean, error

l_edges = np.asarray(np.logspace(2, 4, 200), dtype=int)
l_edges = np.unique(l_edges)

map_list=[]

#load in simulated convergence maps with different cosmologies and source redshifts

'''
#this section is for computing the power spectrum of shape noise and verify that it is a white noise
for i in np.arange(1,5000):
    map_list.append("/e/ocean1/users/lgong/conv_shape_noise_LSST/conv_shape_noise_LSST%s.fits"%(i))
    
l = main(None, l_edges, map_list)[0]
Pk_noise = main(None, l_edges, map_list)[1]

Pk2D_conv_noise = np.zeros((len(l), 2))
Pk2D_conv_noise[:,0] = l
Pk2D_conv_noise[:,1] = Pk_noise
Pk2D_conv_noise = pd.DataFrame(Pk2D_conv_noise)
Pk2D_conv_noise.to_csv('Pk_2D_conv_noise.csv', sep=' ', index=False)
'''

map_list05=[]
map_list10=[]
map_list15=[]
map_list20=[]
map_list25=[]
map_mv01_list05=[]
map_mv01_list10=[]
map_mv01_list15=[]
map_mv01_list20=[]
map_mv01_list25=[]


for i in np.arange(1,10000):
    map_list05.append("/e/ocean1/users/lgong/convergence_gal_mnv0.00000_om0.30000_As2.1000/Maps05/WLconv_z0.50_%sr.fits"%('{:04d}'.format(i)))
    map_list10.append("/e/ocean1/users/lgong/convergence_gal_mnv0.00000_om0.30000_As2.1000/Maps10/WLconv_z1.00_%sr.fits"%('{:04d}'.format(i)))
    map_list15.append("/e/ocean1/users/lgong/convergence_gal_mnv0.00000_om0.30000_As2.1000/Maps15/WLconv_z1.50_%sr.fits"%('{:04d}'.format(i)))
    map_list20.append("/e/ocean1/users/lgong/convergence_gal_mnv0.00000_om0.30000_As2.1000/Maps20/WLconv_z2.00_%sr.fits"%('{:04d}'.format(i)))
    map_list25.append("/e/ocean1/users/lgong/convergence_gal_mnv0.00000_om0.30000_As2.1000/Maps25/WLconv_z2.50_%sr.fits"%('{:04d}'.format(i)))
    map_mv01_list05.append("/e/ocean1/users/lgong/convergence_gal_mnv0.10000_om0.30000_As2.1000/Maps05/WLconv_z0.50_%sr.fits"%('{:04d}'.format(i)))
    map_mv01_list10.append("/e/ocean1/users/lgong/convergence_gal_mnv0.10000_om0.30000_As2.1000/Maps10/WLconv_z1.00_%sr.fits"%('{:04d}'.format(i)))
    map_mv01_list15.append("/e/ocean1/users/lgong/convergence_gal_mnv0.10000_om0.30000_As2.1000/Maps15/WLconv_z1.50_%sr.fits"%('{:04d}'.format(i)))
    map_mv01_list20.append("/e/ocean1/users/lgong/convergence_gal_mnv0.10000_om0.30000_As2.1000/Maps20/WLconv_z2.00_%sr.fits"%('{:04d}'.format(i)))
    map_mv01_list25.append("/e/ocean1/users/lgong/convergence_gal_mnv0.10000_om0.30000_As2.1000/Maps25/WLconv_z2.50_%sr.fits"%('{:04d}'.format(i)))
    

l, Pk_05, Pk_05_err = main(None, l_edges, map_list05)
l, Pk_10, Pk_10_err = main(None, l_edges, map_list10)
l, Pk_15, Pk_15_err = main(None, l_edges, map_list15)
l, Pk_20, Pk_20_err = main(None, l_edges, map_list20)
l, Pk_25, Pk_25_err = main(None, l_edges, map_list25)
l, Pk_mv01_05, Pk_mv01_05_err = main(None, l_edges, map_mv01_list05)
l, Pk_mv01_10, Pk_mv01_10_err = main(None, l_edges, map_mv01_list10)
l, Pk_mv01_15, Pk_mv01_15_err = main(None, l_edges, map_mv01_list15)
l, Pk_mv01_20, Pk_mv01_20_err = main(None, l_edges, map_mv01_list20)
l, Pk_mv01_25, Pk_mv01_25_err = main(None, l_edges, map_mv01_list25)

Pk2D_sim_mv0 = np.zeros((len(l), 6))
Pk2D_sim_mv0[:,0] = l
Pk2D_sim_mv0[:,1] = Pk_05 
Pk2D_sim_mv0[:,2] = Pk_10
Pk2D_sim_mv0[:,3] = Pk_15
Pk2D_sim_mv0[:,4] = Pk_20
Pk2D_sim_mv0[:,5] = Pk_25
Pk2D_sim_mv0 = pd.DataFrame(Pk2D_sim_mv0)
Pk2D_sim_mv0.to_csv('powerspec/convergence_powerspec_simulationdata/Pk_2D_kappa_sim_mv0.csv', sep=' ', index=False)

Pk2D_sim_mv0_err = np.zeros((len(l), 6))
Pk2D_sim_mv0_err[:,0] = l
Pk2D_sim_mv0_err[:,1] = Pk_05_err 
Pk2D_sim_mv0_err[:,2] = Pk_10_err 
Pk2D_sim_mv0_err[:,3] = Pk_15_err
Pk2D_sim_mv0_err[:,4] = Pk_20_err
Pk2D_sim_mv0_err[:,5] = Pk_25_err
Pk2D_sim_mv0_err = pd.DataFrame(Pk2D_sim_mv0_err)
Pk2D_sim_mv0_err.to_csv('powerspec/convergence_powerspec_simulationdata/Pk_2D_kappa_sim_mv0_err.csv', sep=' ', index=False)

Pk2D_sim_mv01 = np.zeros((len(l), 6))
Pk2D_sim_mv01[:,0] = l
Pk2D_sim_mv01[:,1] = Pk_mv01_05
Pk2D_sim_mv01[:,2] = Pk_mv01_10
Pk2D_sim_mv01[:,3] = Pk_mv01_15
Pk2D_sim_mv01[:,4] = Pk_mv01_20
Pk2D_sim_mv01[:,5] = Pk_mv01_25
Pk2D_sim_mv01 = pd.DataFrame(Pk2D_sim_mv01)
Pk2D_sim_mv01.to_csv('powerspec/convergence_powerspec_simulationdata/Pk_2D_kappa_sim_mv01.csv', sep=' ', index=False)

Pk2D_sim_mv01_err = np.zeros((len(l), 6))
Pk2D_sim_mv01_err[:,0] = l
Pk2D_sim_mv01_err[:,1] = Pk_mv01_05_err
Pk2D_sim_mv01_err[:,2] = Pk_mv01_10_err
Pk2D_sim_mv01_err[:,3] = Pk_mv01_15_err
Pk2D_sim_mv01_err[:,4] = Pk_mv01_20_err
Pk2D_sim_mv01_err[:,5] = Pk_mv01_25_err
Pk2D_sim_mv01_err = pd.DataFrame(Pk2D_sim_mv01_err)
Pk2D_sim_mv01_err.to_csv('powerspec/convergence_powerspec_simulationdata/Pk_2D_kappa_sim_mv01_err.csv', sep=' ', index=False)
'''

