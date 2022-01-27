from lenstools import ConvergenceMap
from lenstools.statistics.ensemble import Ensemble
from lenstools.utils.decorators import Parallelize
from astropy.io import fits
from astropy import units as u
import logging
import numpy as np
import pandas as pd



def measure_power_spectrum(filename,l_edges):
    conv_map = ConvergenceMap.load(filename)
    l,Pl = conv_map.powerSpectrum(l_edges)
    return Pl

logging.basicConfig(level=logging.DEBUG)

@Parallelize.masterworker
def main(pool, l_edges, map_list):
    l = 0.5*(l_edges[:-1] + l_edges[1:])
    conv_ensemble = Ensemble.compute(map_list,callback_loader=measure_power_spectrum,pool=pool,l_edges=l_edges)
    mean = conv_ensemble.mean(0)
    errors = np.sqrt(conv_ensemble.covariance().values.diagonal())
    return l, mean, errors

#if __name__=="__main__":
#    main(None)

        
#l_edges = np.loadtxt('powerspec/multipoles_for_convergencepowerspec_lmax1e4.txt')
#l_edges = np.logspace(2, 4, 200)
l_edges = np.asarray(np.logspace(2, 4, 200), dtype=int)
l_edges = np.unique(l_edges)

map_list=[]
for i in np.arange(1,5000):
    map_list.append("/e/ocean1/users/lgong/conv_shape_noise_LSST/conv_shape_noise_LSST%s.fits"%(i))
    
l = main(None, l_edges, map_list)[0]
PK_conv_noise = main(None, l_edges, map_list)[1]

Pk2D_conv_noise = np.zeros((len(l), 2))
Pk2D_conv_noise[:,0] = l
Pk2D_conv_noise[:,1] = PK_conv_noise
Pk2D_conv_noise = pd.DataFrame(Pk2D_conv_noise)
Pk2D_conv_noise.to_csv('PK_2D_conv_noise.csv', sep=' ', index=False)

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
    



l, PK_sim1, PK_sim1_err = main(None, l_edges, map_list05)
l, PK_sim2, PK_sim2_err = main(None, l_edges, map_list10)
l, PK_sim3, PK_sim3_err = main(None, l_edges, map_list15)
l, PK_sim4, PK_sim4_err = main(None, l_edges, map_list20)
l, PK_sim5, PK_sim5_err = main(None, l_edges, map_list25)
l, PK_mv01_sim1, PK_mv01_sim1_err = main(None, l_edges, map_mv01_list05)
l, PK_mv01_sim2, PK_mv01_sim2_err = main(None, l_edges, map_mv01_list10)
l, PK_mv01_sim3, PK_mv01_sim3_err = main(None, l_edges, map_mv01_list15)
l, PK_mv01_sim4, PK_mv01_sim4_err = main(None, l_edges, map_mv01_list20)
l, PK_mv01_sim5, PK_mv01_sim5_err = main(None, l_edges, map_mv01_list25)

Pk2D_sim_mv0 = np.zeros((len(l), 6))
Pk2D_sim_mv0[:,0] = l
Pk2D_sim_mv0[:,1] = PK_sim1 
Pk2D_sim_mv0[:,2] = PK_sim2
Pk2D_sim_mv0[:,3] = PK_sim3
Pk2D_sim_mv0[:,4] = PK_sim4
Pk2D_sim_mv0[:,5] = PK_sim5
Pk2D_sim_mv0 = pd.DataFrame(Pk2D_sim_mv0)
Pk2D_sim_mv0.to_csv('powerspec/convergence_powerspec_simulationdata/PK_2D_Kappa_sim_mv0_1e4realizations_1.csv', sep=' ', index=False)

Pk2D_sim_mv0_err = np.zeros((len(l), 6))
Pk2D_sim_mv0_err[:,0] = l
Pk2D_sim_mv0_err[:,1] = PK_sim1_err 
Pk2D_sim_mv0_err[:,2] = PK_sim2_err 
Pk2D_sim_mv0_err[:,3] = PK_sim3_err
Pk2D_sim_mv0_err[:,4] = PK_sim4_err
Pk2D_sim_mv0_err[:,5] = PK_sim5_err
Pk2D_sim_mv0_err = pd.DataFrame(Pk2D_sim_mv0_err)
Pk2D_sim_mv0_err.to_csv('powerspec/convergence_powerspec_simulationdata/PK_2D_Kappa_sim_mv0_err_1e4realizations_1.csv', sep=' ', index=False)

Pk2D_sim_mv01 = np.zeros((len(l), 6))
Pk2D_sim_mv01[:,0] = l
Pk2D_sim_mv01[:,1] = PK_mv01_sim1
Pk2D_sim_mv01[:,2] = PK_mv01_sim2
Pk2D_sim_mv01[:,3] = PK_mv01_sim3
Pk2D_sim_mv01[:,4] = PK_mv01_sim4
Pk2D_sim_mv01[:,5] = PK_mv01_sim5
Pk2D_sim_mv01 = pd.DataFrame(Pk2D_sim_mv01)
Pk2D_sim_mv01.to_csv('powerspec/convergence_powerspec_simulationdata/PK_2D_Kappa_sim_mv01_1e4realizations_1.csv', sep=' ', index=False)

Pk2D_sim_mv01_err = np.zeros((len(l), 6))
Pk2D_sim_mv01_err[:,0] = l
Pk2D_sim_mv01_err[:,1] = PK_mv01_sim1_err
Pk2D_sim_mv01_err[:,2] = PK_mv01_sim2_err
Pk2D_sim_mv01_err[:,3] = PK_mv01_sim3_err
Pk2D_sim_mv01_err[:,4] = PK_mv01_sim4_err
Pk2D_sim_mv01_err[:,5] = PK_mv01_sim5_err
Pk2D_sim_mv01_err = pd.DataFrame(Pk2D_sim_mv01_err)
Pk2D_sim_mv01_err.to_csv('powerspec/convergence_powerspec_simulationdata/PK_2D_Kappa_sim_mv01_err_1e4realizations_1.csv', sep=' ', index=False)
'''

