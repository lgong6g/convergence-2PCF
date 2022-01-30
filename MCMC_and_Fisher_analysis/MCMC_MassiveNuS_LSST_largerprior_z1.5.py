import time
import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import astropy.units as u
import multiprocessing as mp
from multiprocessing import Pool
from scipy.optimize import fsolve
from scipy import integrate
from scipy import interpolate
from classy import Class
from astropy.cosmology import FlatLambdaCDM

####the function to get individual neutrino masses####
def get_masses(delta_m_squared_atm, delta_m_squared_sol, sum_masses, hierarchy):
    # massless neutrino case should be considered separately
    if sum_masses == 0.0:
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        return m1,m2,m3
    # any string containing letter 'n' will be considered as refering to normal hierarchy
    if 'n' in hierarchy.lower():
        # Normal hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_NH and deletes M_tot_NH
        #delta_m_squared_atm=2.45e-3 this is m_3^2 - (m_1^2 + m_2^2)/2
        #delta_m_squared_sol=7.50e-5 this is m_2^2 - m_1^2
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1,opt_output,success,output_message = fsolve(m1_func,sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1,m2,m3
    else:
        # Inverted hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_IH and deletes M_tot_IH
        #delta_m_squared_atm=-2.45e-3
        #delta_m_squared_sol=7.50e-5
        delta_m_squared_atm = -delta_m_squared_atm
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1,opt_output,success,output_message = fsolve(m1_func,sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1,m2,m3
    

####the function to compute density parameter of massive neutrinos####
def Mnu2omeganu(sum_masses, matter_density):
    m1, m2, m3 = get_masses(2.5e-3,7.37e-5, sum_masses, 'NH') 
    mnu_arr = np.array([m1, m2, m3])* u.eV
    cosmo = FlatLambdaCDM(H0=h*100, Om0=matter_density, Neff=3.046, m_nu=mnu_arr, Ob0=omega_b/(h**2), Tcmb0=2.7255)
    return cosmo.Onu0*(h**2)

#### Cosmologies besides those parameters probed by MCMC #####

kmax = 30.0

nonlinear_model = "halofit"

omega_b = 0.02254

n_s = 0.97

h = 0.7

#####Hartlap correction factor#########

hartlap_factor = (5000-20-2)/(5000-1)

##shell-correction fitting parameters##

a1 = 1.3606
        
a2 = 1.2674
        
a3 = 0.7028
        
c1 = 2.2041e-3
       
c2 = 1.0861e-2

################

#ell = np.loadtxt("powerspec/multipoles_for_convergencepowerspec_150l.txt", usecols=(0))
ell = np.logspace(0, np.log10(50001), 500)
ell = np.asarray(ell, dtype=int)
ell = np.unique(ell)

angular_bins = np.loadtxt('correlationfunc/angularsep_nominalcenter_corrfunc.txt', usecols=(0))
angular_bins_edge = np.loadtxt('correlationfunc/angularbin_edges.txt', usecols=(0))

steps = 6000

walkers = 30

z_source = 1.5

z_list_integration = np.linspace(1e-4, z_source, 50)

l1 = np.linspace(0, 49999, 50000)
    
l2 = np.linspace(1.5, 49999.5, 49999)

#l1 = np.linspace(0, 19999, 20000)

#l2 = np.linspace(1.5, 19999.5, 19999)

class class_obj:
    def __init__(self, theta):
        self.Mv, self.Omega_m, self.As = theta
        
    def class_computation(self):
        if self.Mv == 0.0:
            commonsettings  = {
                      'N_ur':3.046,
                      'N_ncdm':0,
                      'output':'mPk',
                      'P_k_max_1/Mpc':kmax,
                      #'background_verbose':10,
                      # omega_b here actually means Omega_b * h**2
                      'omega_b':omega_b,
                      'h':h,
                      'n_s':n_s,
                      'A_s':self.As * 1e-9, 
                      #'sigma8': sigma8,
                      'omega_cdm': self.Omega_m*(h**2)-omega_b,  
                      'Omega_k':0.0,
                      'Omega_fld': 0.0,
                      'Omega_scf': 0.0,
                      'YHe':0.24,
                      'z_max_pk':5.0,
                      'non linear':nonlinear_model,
                      'write warnings':'yes'
                     }
        else:
            m1, m2, m3 = get_masses(2.5e-3,7.37e-5, self.Mv, 'NH')
            omega_nu = Mnu2omeganu(self.Mv, self.Omega_m)
            commonsettings  = {
                      'N_ur':0.00641,
                      'N_ncdm':3,
                      'output':'mPk',
                      'P_k_max_1/Mpc':kmax,
                      #'background_verbose':10,
                      # omega_b here actually means Omega_b * h**2
                      'omega_b':omega_b,
                      'omega_cdm':self.Omega_m*(h**2)-(omega_b+omega_nu), 
                      'm_ncdm':str(m1)+','+str(m2)+','+str(m3),
                      'h':h,
                      'n_s':n_s,
                      'A_s':self.As * 1e-9, 
                      'Omega_k':0.0,
                      'Omega_fld': 0.0,
                      'Omega_scf': 0.0,          
                      'YHe':0.24, 
                      'z_max_pk':5.0,
                      #'nonlinear_verbose':10,
                      'non linear':nonlinear_model,
                      'write warnings':'yes'
                     }

        Nu_Cosmo = Class()
        Nu_Cosmo.set(commonsettings)
        Nu_Cosmo.compute()
        return Nu_Cosmo


def rz(obj_1, z):
    return obj_1.angular_distance(z)*(1+z)


def P_NL(obj_2, k, z):
    if k > kmax: 
        return 0
    else:
        return obj_2.pk(k, z)
    

def shell_fitting(obj_3, k, z):
    return  (1 + c1*(k/h)**(-a1))**(a1)/(1 + c2*(k/h)**(-a2))**(a3) * P_NL(obj_3, k, z)


#######this class is to turn CLASS object into normal python objects############
class my_class:
    def __init__(self, class_obj, z_list, l):
        self.class_obj = class_obj
        self.z_list = z_list
        self.l = l
        
    def angular_diameter_dist(self):
        DA = np.zeros(len(self.z_list))
        for i in range(len(self.z_list)):
            DA[i] = rz(self.class_obj, self.z_list[i])
        return DA
    
    def angular_diameter_dist_zs(self):
        return rz(self.class_obj, self.z_list[len(self.z_list)-1])
    
    def Hubble(self):
        H = np.zeros(len(self.z_list))
        for i in range(len(self.z_list)):
            H[i] = self.class_obj.Hubble(self.z_list[i])
        return H
    
    def Hubble_0(self):
        return self.class_obj.Hubble(0.0)
    
    def Omega_m0(self):
        return self.class_obj.Omega0_m()
    
    def Pk_value(self):
        Pk = np.zeros((len(self.l), len(self.z_list)))
        for i in range(len(self.l)):
            for j in range(len(self.z_list)):
                Pk[i,j] = shell_fitting(self.class_obj, self.l[i]/rz(self.class_obj, self.z_list[j]), self.z_list[j])
        return Pk


###############
def correlation_2pt(class_dict, l, z_list):
    #z_list = np.linspace(1e-4, z_s, 50)
    #p = my_class(cosmo_obj, z_list, l)
    P_k2D = np.zeros(len(l))
    for i in range(len(l)):
        dP_k2D = np.zeros(len(z_list))
        for j in range(len(z_list)):
            dP_k2D[j] = (1/class_dict['H'][j]) * (9./4.) * (class_dict['H0'])**4 * class_dict['Om0']**2 * ((class_dict['DA_s']-class_dict['DA'][j])/class_dict['DA_s'])**2 *(1+z_list[j])**2 * class_dict['Pk'][i,j]
            P_k2D[i] = np.trapz(dP_k2D, z_list)
    
    pk2D_interp = np.zeros(len(l1))
    '''
    #####logrithmic interpolation##########
    logl = np.log10(ell)
    logpk2D = np.log10(P_k2D)
    f = sp.interpolate.interp1d(logl, logpk2D, kind='linear', fill_value="extrapolate")
    pk2D_interp[0] = 0
    pk2D_interp[1:] = np.power(10.0*np.ones(len(l2)), f(np.log10(l2)))
    '''
    #####linear interpolation#########
    f = interpolate.interp1d(ell, P_k2D)
    pk2D_interp[0] = 0
    pk2D_interp[1:] = f(l2)
    
    #kitching correction
    for j in np.asarray(l1[1:], dtype=int):
        pk2D_interp[j] = (j+2)*(j+1)*j*(j-1)*pk2D_interp[j]/(j+0.5)**4
    
    '''
    ##the transformation from harmonic space to real space without angular bin average
    r = angular_bins 
    corr_2pt = np.zeros(len(r))
    
    x = np.cos(np.radians(r * 1/60))
    coeff = (2 * l1 + 1) * pk2D_interp/4/np.pi
    corr_2pt = np.polynomial.legendre.legval(x, coeff)
    
    '''
    ##the transformation from harmonic space to real space with angular bin average
    r = angular_bins
    r_edge = angular_bins_edge
    
    corr_2pt = np.zeros(len(r))
    coeff1 = np.zeros(len(l1)+1)
    coeff2 = np.zeros(len(l1)-1)
    for i in range(len(r)):
        x_max = np.cos(np.radians(r_edge[i+1] * 1/60))
        x_min = np.cos(np.radians(r_edge[i] * 1/60))
        coeff1[1:] = 1/(4*np.pi*(x_max - x_min)) * pk2D_interp
        coeff2 = 1/(4*np.pi*(x_max - x_min)) * pk2D_interp[1:]
        corr_2pt[i] = np.polynomial.legendre.legval(x_max, coeff1) - np.polynomial.legendre.legval(x_max, coeff2) - np.polynomial.legendre.legval(x_min, coeff1) + np.polynomial.legendre.legval(x_min, coeff2)
    
    return corr_2pt

'''
def P_k2D_integral(H, H0, Om0, DA_s, DA, Pk, l, z_list):
    #z_list = np.linspace(1e-4, z_s, 50)
    #p = my_class(cosmo_obj, z_list, l)
    P_k2D = np.zeros(len(l))
    for i in range(len(l)):
        dP_k2D = np.zeros(len(z_list))
        for j in range(len(z_list)):
            dP_k2D[j] = (1/H[j]) * (9./4.) * (H0)**4 * Om0**2 * ((DA_s-DA[j])/DA_s)**2 *(1+z_list[j])**2 * Pk[i,j]
            P_k2D[i] = np.trapz(dP_k2D, z_list)
    return P_k2D


def P_k2D_integral(p, l, z_list):
    #z_list = np.linspace(1e-4, z_s, 50)
    #p = my_class(cosmo_obj, z_list, l)
    P_k2D = np.zeros(len(l))
    for i in range(len(l)):
        dP_k2D = np.zeros(len(z_list))
        for j in range(len(z_list)):
            dP_k2D[j] = (1/p.Hubble()[j]) * (9./4.) * (p.Hubble_0())**4 * p.Omega_m0()**2 * ((p.angular_diameter_dist_zs()-p.angular_diameter_dist()[j])/p.angular_diameter_dist_zs())**2 *(1+z_list[j])**2 * p.Pk_value()[i,j]
            P_k2D[i] = np.trapz(dP_k2D, z_list)
    return P_k2D
'''
#def P_k2D_integral_parallelization(p):
#    return P_k2D_integral(p[0], p[1], p[2])


######the functional form of prior######
def log_prior(theta):
    Mv, Omega_m, As = theta
    #with the constraints on neutrino masses and the assumption of normal hierarchy, the total sum has a least value of 0.06eV
    #the upper limit of Mv is 0.17eV from Planck and 0.62eV is the one from Coulton's paper
    if 0.06 <= Mv <= 0.62 and 0.18 <= Omega_m <= 0.42 and 1.29 < As < 2.91:
        return 0.0
    return -np.inf


#####the functional form of likelihood######
def log_likelihood(theta, data, cov):
    
    Nu_Cosmo = class_obj(theta).class_computation()
    p = my_class(Nu_Cosmo, z_list_integration, ell)
    cosmo_dict = {'H': p.Hubble(),
                  'H0': p.Hubble_0(),
                  'Om0': p.Omega_m0(),
                  'DA_s': p.angular_diameter_dist_zs(),
                  'DA': p.angular_diameter_dist(),
                  'Pk': p.Pk_value()}
    
    #multiprocess in calculating 2pt functions
    #pool = mp.Pool(processes=16)
    #result = pool.apply_async(correlation_2pt, (cosmo_dict, ell, z_list_integration))
    #model = np.array(result.get())
    
    
    #serial method in calculating 2pt functions
    model = correlation_2pt(cosmo_dict, ell, z_list_integration)
    #pk2D = np.zeros(len(ell))
    #for i in range(len(ell)):
    #    pk2D[i] = calculate_pk(theta, zs, ell[i])
    
    '''
    l1 = np.linspace(0, 19999, 20000)
    l2 = np.linspace(1.5, 19999.5, 19999)
    
    pk2D_interp = np.zeros(len(l1))
    #####logrithmic interpolation##########
    logl = np.log10(ell)
    logpk2D = np.log10(pk2D)
    f = sp.interpolate.interp1d(logl, logpk2D, kind='linear')
    pk2D_interp[0] = 0
    pk2D_interp[1:] = np.power(10.0*np.ones(len(l2)), f(np.log10(l2)))
    #kitching correction
    for j in np.asarray(l1[1:], dtype=int):
        pk2D_interp[j] = (j+2)*(j+1)*j*(j-1)*pk2D_interp[j]/(j+0.5)**4

    
    #######this is normal interpolation#######
    pk2D_interp = np.zeros(len(l1)) 
    f = interpolate.interp1d(ell, pk2D)
    pk2D_interp[0] = 0
    pk2D_interp[1:] = f(l2)
    for j in np.asarray(l1[1:], dtype=int):
        pk2D_interp[j] = (j+2)*(j+1)*j*(j-1)*pk2D_interp[j]/(j+0.5)**4
    
    ########this is Legendre transformation#######
    r = angular_bins 
    ell_1 = np.linspace(0,19999,20000)
    
    corr_2pt = np.zeros(len(r))
    
    x = np.cos(np.radians(r * 1/60))
    coeff = (2 * ell_1 + 1) * pk2D_interp/4/np.pi
    corr_2pt = np.polynomial.legendre.legval(x, coeff)
    '''
    #model = corr_2pt
    diff = data - model
    
    return -0.5 * np.dot(diff, hartlap_factor*np.linalg.solve(cov, diff))


def log_probability(theta, data, cov):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data, cov)

param = [0.1, 0.3, 2.1]

#mean = pd.read_csv('correlationfunc/2ptfunc_circpatch/2pt_mnv0.1_om0.3_As2.1_z1.5_circular.csv', sep=' ')
data_vec = pd.read_csv('correlationfunc/2ptfunc_circpatch/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations9999_circular.csv', sep=' ')
#data_vec = pd.read_csv('correlationfunc/2ptfunc_circpatch_wn/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations10000_circular_LSST.csv', sep=' ')
'''
######shuffle the realization index#####
index = pd.read_csv('shuffled_indexlist_MassiveNuS_realizations.csv', sep=' ')
index = index.to_numpy()
index = np.squeeze(index)
'''
######use only 3000 realizations to compute the data vector###########
#Discard the first 4 bins 
data_vec = data_vec.iloc[:4999,:]
data_vec = data_vec.to_numpy()
data_vec = np.mean(data_vec, axis=0)

corr2pt = pd.read_csv('correlationfunc/2ptfunc_circpatch_wn/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations10000_circular_LSST_whitenoise.csv', sep=' ')
#corr2pt = corr2pt.iloc[index[5000:9999],:]
corr2pt = corr2pt.iloc[4999:,:]
covariance = corr2pt.cov()
covariance = covariance.to_numpy()
covariance = covariance * (5/18000)


######initial conditions of uniform distribution#######
np.random.seed(42)
theta1_0 = np.random.uniform(0.06, 0.62, size=30)
theta2_0 = np.random.uniform(0.18, 0.42, size=30)
theta3_0 = np.random.uniform(1.29, 2.91, size=30)
#p0 = param + 1e-3 * np.random.randn(walkers,3)
p0 = np.zeros((walkers, 3))
p0[:,0] = theta1_0
p0[:,1] = theta2_0
p0[:,2] = theta3_0
nwalkers, ndim = p0.shape 


#filename = "MCMC_test1_%s_steps_trapz_30walkers.h5"%(steps)
filename = "MCMC_30walkers_largerprior_z1.5_LSST_5e3cov_lmax5e4_w_binave_white.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(mean, covariance), backend=backend)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data_vec, covariance), backend=backend, pool=pool)
    
    max_n = 100000
    index = 0
    autocorr = np.empty(max_n)

    old_tau = np.inf

    for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        #print(tau)
        autocorr[index] = np.mean(tau)
        index += 1

    # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    
    '''
    start = time.time()
    sampler.run_mcmc(None, steps, progress=True)
    #sampler.run_mcmc(p0, 6000, progress=True)
    #tau = sampler.get_autocorr_time()
    #print(tau)
    end = time.time()
    multi_time = end - start
    print("MCMC took {0:.1f} seconds".format(multi_time))
    '''