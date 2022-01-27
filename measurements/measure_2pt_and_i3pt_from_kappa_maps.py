import time
start_program = time.time()

import treecorr
import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.io import fits

#############The function to compute correlation function within a circular patch###########

def measure_correlation_function(mv, om, As, maps, z, i_min, i_max, nbins):
    #i_min and i_max are map indices which select convergence maps to calculate correlation functions
    #mv, om and As are cosmological parameters. maps and z are necessary pointers to identify simulated maps
    x = np.array([])
    y = np.array([])
    varxi = np.zeros(nbins)
    varxi_i3pt = np.zeros(nbins)
    
    #Recording the resulting correlation function values for different realizations
    convergence_ave = np.zeros(i_max-i_min)
    xi_2pt = np.zeros((i_max-i_min, nbins))
    xi_i3pt = np.zeros((i_max-i_min, nbins))
    
    #Positions for x,y grids of a map. The positions are at the center of pixels
    #The side scale of a map is 3.5 degree and on each side there are 512 pixels in total
    x_ini = np.linspace(3.5/1024,3.5*(1-1/1024),512)
    y_ini = np.linspace(3.5/1024,3.5*(1-1/1024),512)
    
    #Specify (x,y) values for each of the 512**2 pixels
    for i in range(512):
        for j in range(512):
            x = np.append(x, x_ini[j])
            y = np.append(y, y_ini[i])
            
    ###This section is for measuring correlation functions within a circular patch of a map###
    #set up the arrays which will record selected pixels and a mask respectively
    #set up the center of the selected region
    mask = np.zeros(len(x), dtype=bool)
    xk = np.array([])
    yk = np.array([])
    xc = 3.5/1024 + 255*3.5/512
    yc = 3.5/1024 + 255*3.5/512
    
    #create the selected region and the mask array
    for m in range(len(x)):
        #75.0 is the radius of the circular patch and is in the unit of arcmin
        if np.sqrt((x[m]-xc)**2+(y[m]-yc)**2) <= 75.0/60.0:
            xk = np.append(xk, x[m])
            yk = np.append(yk, y[m])
            mask[m] = True
        else:
            mask[m] = False
     
    
    for i in np.arange(i_min,i_max):
        # read in the simulation maps from MassiveNuS
        filename =  "/e/ocean1/users/lgong/MassiveNuS_noisymaps_LSST/convergence_gal_mnv%s_om%s_As%s/Maps%s/WLconv_z%s_%sr_wnoise.fits"%('{:7.5f}'.format(mv), '{:7.5f}'.format(om), '{:6.4f}'.format(As), '{0:02d}'.format(maps), '{:3.2f}'.format(z), '{:04d}'.format(i))
        
        hdul = fits.open(filename)
        data_ini = hdul[0].data
        data = np.array([])
        #reshape the convergence 2d array into a 1d data vector
        data = np.reshape(data_ini, 262144)
    
        #mask out pixels outside the circular patch
        kappa = data[mask]
        kappa_ave = np.mean(kappa)
        
        #set up the treecorr catalog
        cat = treecorr.Catalog(x=xk,y=yk,k=kappa,x_units='deg',y_units='deg')
        kk = treecorr.KKCorrelation(min_sep=5.0, max_sep=140.0, nbins=nbins, sep_units='arcmin')
        
        #The resolution of the map is about 0.41 arcmin
        kk.process(cat)
        
        r = kk.rnom
        convergence_ave[i-i_min] = kappa_ave
        xi_2pt[i-i_min,:] = kk.xi
        xi_i3pt[i-i_min,:] = kk.xi*kappa_ave
        
        print("Computation for realization %s is done"%(i))

    corr_2pt = np.mean(xi_2pt, axis=0)
    scatter_2pt = np.std(xi_2pt, axis=0)
    summary_2pt = np.zeros((len(r), 3))
    summary_2pt[:,0] = r
    summary_2pt[:,1] = corr_2pt
    summary_2pt[:,2] = scatter_2pt
    summary_2pt = pd.DataFrame(summary_2pt)
    summary_2pt.to_csv("correlationfunc/2ptfunc_circpatch_wn/2pt_mnv%s_om%s_As%s_z%s_circular_LSST_wnoise.csv"%(mv,om,As,z), sep=' ', index=False)
    
    
    corr_i3pt = np.mean(xi_i3pt, axis=0)
    scatter_i3pt = np.std(xi_i3pt, axis=0)
    summary_i3pt = np.zeros((len(r), 3))
    summary_i3pt[:,0] = r
    summary_i3pt[:,1] = corr_i3pt
    summary_i3pt[:,2] = scatter_i3pt
    summary_i3pt = pd.DataFrame(summary_i3pt)
    summary_3pt.to_csv("correlationfunc/i3ptfunc_circpatch_wn/i3pt_mnv%s_om%s_As%s_z%s_circular_LSST_wnoise.csv"%(mv,om,As,z), sep=' ', index=False)
    
    
    convergence_ave = pd.DataFrame(convergence_ave)
    xi_2pt = pd.DataFrame(xi_2pt)
    xi_i3pt = pd.DataFrame(xi_i3pt)
    convergence_ave.to_csv('correlationfunc/2ptfunc_circpatch_wn/average_convergence_mnv%s_om%s_As%s_z%s_relizations%s_circular_LSST_wnoise.csv'%(mv, om, As, z, (i_max+1-i_min)), sep=' ', index=False)
    xi_2pt.to_csv('correlationfunc/2ptfunc_circpatch_wn/2pt_mnv%s_om%s_As%s_z%s_relizations%s_circular_LSST_wnoise.csv'%(mv, om, As, z, (i_max+1-i_min)), sep=' ', index=False)
    xi_i3pt.to_csv('correlationfunc/i3ptfunc_circpatch_wn/i3pt_mnv%s_om%s_As%s_z%s_relizations%s_circular_LSST_wnoise.csv'%(mv, om, As, z, (i_max+1-i_min)), sep=' ', index=False)
    


def measure_correlation_function_parallelisation(p):
    return measure_correlation_function(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])



#############Input parameters##############

mv=0.1

om=0.3

As=2.1

maps=[10, 15]

z=[1.0, 1.5]

i_min=1

i_max=10000

nbins=20

pool = mp.Pool(processes=30)
pool.map(measure_correlation_function_parallelisation, [[mv, om, As, maps[i], z[i], i_min, i_max, nbins] for i in range(len(z))])

end_program = time.time()
print('\nTime taken for execution (seconds): ', end_program - start_program)