import time
start = time.time()

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

############### function to get individual neutrino masses ########

def get_masses(delta_m_squared_atm, delta_m_squared_sol, sum_masses, hierarchy):
    # delta_m_squared_atm, delta_m_squared_sol are constraints of square differences of neutrino masses from oscillation experiments
    # massless neutrino case should be considered separately
    if sum_masses == 0.0:
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        return m1, m2, m3
    
    # any string containing letter 'n' will be considered as refering to normal hierarchy
    if 'n' in hierarchy.lower():
        # delta_m_squared_atm = m_3^2 - (m_1^2 + m_2^2)/2
        # delta_m_squared_sol = m_2^2 - m_1^2
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1, opt_output, success, output_message = fsolve(m1_func, sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol), full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1, m2, m3
    else:
        # Inverted hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_IH and deletes M_tot_IH
        #delta_m_squared_atm = -m_3^2 + (m_1^2 + m_2^2)/2
        #delta_m_squared_sol = m_2^2 - m_1^2
        delta_m_squared_atm = -delta_m_squared_atm
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1, opt_output, success, output_message = fsolve(m1_func, sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1, m2, m3
    

############### function to compute density parameter of massive neutrinos ######
def Mnu2omeganu(sum_masses, matter_density):
    # the neutrino mass constraints come from the MassiveNuS paper by Liu et al.
    m1, m2, m3 = get_masses(2.5e-3, 7.37e-5, sum_masses, 'NH') 
    mnu_arr = np.array([m1, m2, m3])* u.eV
    cosmo = FlatLambdaCDM(H0=h*100, Om0=matter_density, Neff=3.046, m_nu=mnu_arr, Ob0=omega_b/(h**2), Tcmb0=2.7255)
    return cosmo.Onu0*(h**2)


#### Cosmologies besides those parameters probed by MCMC #####

#### MassiveNuS cosmology ####
omega_b = 0.02254
n_s = 0.97
h = 0.7

kmax = 30.0
nonlinear_model = "halofit"


##### Hartlap correction factor for the precision matrix #########
# Number of simulation realizations used to estimate data covariance matrix
N_s = 5000
# Number of bins in the data vector
N_d = 20
hartlap_factor = (N_s - N_d - 2)/(N_s - 1)

## Shell-correction fitting parameters for MassiveNuS ##
a1 = 1.3606
a2 = 1.2674
a3 = 0.7028
c1 = 2.2041e-3
c2 = 1.0861e-2

#### Read in multipole numbers prepared for power spectrum computation and angular separations for 2PCF ######
ell = np.loadtxt("../2PCF_modelling/ell_array_120.txt", usecols=(0))
r = np.loadtxt("../2PCF_modelling/angularsep_arcmins_20bins.txt", usecols=(0))
r_min = np.loadtxt("../2PCF_modelling/angularsep_arcmins_min_20bins.txt", usecols=(0))
r_max = np.loadtxt("../2PCF_modelling/angularsep_arcmins_max_20bins.txt", usecols=(0))
l1 = np.linspace(0, 19999, 20000)
l2 = np.linspace(1.5, 19999.5, 19999)


# Number of Markov Chain steps 
steps = 6000

# Number of Markov Chains initiated simultaneously
walkers = 30

# Source redshift for the convergence 2PCF in the theoretical vector
z_source = 1.5
# z_source2 = 1.0 # The second source redshift in case a tomographic analysis is required 

# Number of cosmological parameters to be probed by MCMC
n_param = 3

# A list of redshifts between the observer and the source redshift used in the trapezoidal integration along the line of sight
z_list_integration = np.linspace(1e-4, z_source, 50)
#z_list_integration2 = np.linspace(1e-4, z_source2, 50)

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
                  # The next line should be uncommented fgor higher precision (but significantly slower running)
                  #'ncdm_fluid_approximation':3,
                  # You may uncomment this line to get more info on the ncdm sector from Class:
                  #'background_verbose':10,
                  'background_verbose':0,
                  # omega_b here actually means Omega_b * h**2
                  'omega_b':omega_b,
                  'h':h,
                  'n_s':n_s,
                  'A_s':self.As, 
                  #'sigma8': sigma8,
                  'omega_cdm':self.Omega_m*(h**2)-omega_b, #0.12470 Liu's input for massless neutrino and Omega_m=0.3 case 
                  #'omega_cdm': omega_cdm,
                  'Omega_k':0.0,
                  'Omega_fld':0.0,
                  'Omega_scf':0.0,
                  'YHe':0.24, #'BBN', 0.24 Liu's input
                  'z_max_pk':3.5,
                  'non linear':nonlinear_model,
                  'write warnings':'yes'
                 }
        else:
            m1, m2, m3 = get_masses(2.5e-3, 7.37e-5, self.Mv, 'NH')
            omega_nu = Mnu2omeganu(self.Mv, self.Omega_m)
            commonsettings  = {
                  'N_ur':0.00641,
                  'N_ncdm':3,
                  'output':'mPk',
                  'P_k_max_1/Mpc':kmax,
                  # The next line should be uncommented fgor higher precision (but significantly slower running)
                  #'ncdm_fluid_approximation':3,
                  # You may uncomment this line to get more info on the ncdm sector from Class:
                  #'background_verbose':10,
                  'background_verbose':0,
                  'omega_b':omega_b,
                  'omega_cdm':self.Omega_m*(h**2)-(omega_b+omega_nu), #0.12362 Liu's input for massive neutrinos and Omega_m=0.3
                  #'omega_cdm': omega_cdm,
                  'm_ncdm':str(m1)+','+str(m2)+','+str(m3),
                  'h':h,
                  'n_s':n_s,
                  'A_s':self.As, 
                  #'sigma8': sigma8,
                  'Omega_k':0.0,
                  'Omega_fld': 0.0,
                  'Omega_scf': 0.0,          
                  'YHe':0.24, #'BBN', 0.24 Liu's input
                  'z_max_pk':3.5,
                  'nonlinear_verbose':10,
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


#### This class is to turn CLASS object computed from class_obj into normal python objects ############
#### where we prepare a 2d table for matter power spectrum along the dimension of wave-number and redshift and other necessary quantities 
#### that will be applied in the later line-of-sight integration to compute convergence power spectrum
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

#### The function to calculate auto convergence 2PCF
def correlation_2pt(class_dict, l, z_list):
    Pk_2D = np.zeros(len(l))
    for i in range(len(l)):
        dPk_2D = np.zeros(len(z_list))
        for j in range(len(z_list)):
            # where we implement the trapezoidal integration instead of quad
            dPk_2D[j] = (1/class_dict['H'][j]) * (9./4.) * (class_dict['H0'])**4 * class_dict['Om0']**2 * ((class_dict['DA_s']-class_dict['DA'][j])/class_dict['DA_s'])**2 *(1+z_list[j])**2 * class_dict['Pk'][i,j]
            Pk_2D[i] = np.trapz(dPk_2D, z_list)
    
    pk2D_interp = np.zeros(len(l1))
    
    '''
    #####logrithmic interpolation##########
    logl = np.log10(ell)
    logpk2D = np.log10(Pk_2D)
    f = sp.interpolate.interp1d(logl, logpk2D, kind='linear', fill_value="extrapolate")
    pk2D_interp[0] = 0
    pk2D_interp[1:] = np.power(10.0*np.ones(len(l2)), f(np.log10(l2)))
    '''
    
    #####linear interpolation#########
    f = interpolate.interp1d(ell, Pk_2D)
    pk2D_interp[0] = 0
    pk2D_interp[1:] = f(l2)
    
    #kitching correction
    for j in np.asarray(l1[1:], dtype=int):
        pk2D_interp[j] = (j+2)*(j+1)*j*(j-1)*pk2D_interp[j]/(j+0.5)**4
    
    '''
    ##the transformation from harmonic space to real space without angular bin average
    corr_2pt = np.zeros(len(r))
    
    x = np.cos(np.radians(r * 1/60))
    coeff = (2 * l1 + 1) * pk2D_interp/4/np.pi
    corr_2pt = np.polynomial.legendre.legval(x, coeff)
    '''
    
    ##the transformation from harmonic space to real space with angular bin average
    corr_2pt = np.zeros(len(r))
    coeff1 = np.zeros(len(l1)+1)
    coeff2 = np.zeros(len(l1)-1)
    for i in range(len(r)):
        x_max = np.cos(np.radians(r_max[i] * 1/60))
        x_min = np.cos(np.radians(r_min[i] * 1/60))
        coeff1[1:] = 1/(4*np.pi*(x_max - x_min)) * pk2D_interp
        coeff2 = 1/(4*np.pi*(x_max - x_min)) * pk2D_interp[1:]
        corr_2pt[i] = np.polynomial.legendre.legval(x_max, coeff1) - np.polynomial.legendre.legval(x_max, coeff2) - np.polynomial.legendre.legval(x_min, coeff1) + np.polynomial.legendre.legval(x_min, coeff2)
    
    return corr_2pt

'''
##### The function computing convergence cross 2PCF#######
def correlation_2pt_cross(class_dict, l, z_list):
    Pk_2D_cross = np.zeros(len(l))
    for i in range(len(l)):
        dPk_2D_cross = np.zeros(len(z_list))
        for j in range(len(z_list)):
            dPk_2D_cross[j] = (1/class_dict['H'][j]) * (9./4.) * (class_dict['H0'])**4 * class_dict['Om0']**2 * ((class_dict['DA_s1']-class_dict['DA'][j])/class_dict['DA_s1'])* ((class_dict['DA_s2']-class_dict['DA'][j])/class_dict['DA_s2']) *(1+z_list[j])**2 * class_dict['Pk'][i,j]
            Pk_2D_cross[i] = np.trapz(dPk_2D_cross, z_list)
    
    pk2D_interp_cross = np.zeros(len(l1))
    
    
    #####logrithmic interpolation##########
    logl = np.log10(ell)
    logpk2D = np.log10(Pk_2D_cross)
    f = sp.interpolate.interp1d(logl, logpk2D, kind='linear', fill_value="extrapolate")
    pk2D_interp[0] = 0
    pk2D_interp[1:] = np.power(10.0*np.ones(len(l2)), f(np.log10(l2)))
    
    
    #####linear interpolation#########
    f = interpolate.interp1d(ell, Pk_2D_cross)
    pk2D_interp_cross[0] = 0
    pk2D_interp_cross[1:] = f(l2)
    #kitching correction
    for j in np.asarray(l1[1:], dtype=int):
        pk2D_interp_cross[j] = (j+2)*(j+1)*j*(j-1)*pk2D_interp_cross[j]/(j+0.5)**4
    
    
    ##the transformation from harmonic space to real space without angular bin average
    corr_2pt_cross = np.zeros(len(r))
    x = np.cos(np.radians(r * 1/60))
    coeff = (2 * l1 + 1) * pk2D_interp_cross/4/np.pi
    corr_2pt_cross = np.polynomial.legendre.legval(x, coeff)
    
    
    ##the transformation from harmonic space to real space with angular bin average
    corr_2pt_cross = np.zeros(len(r))
    coeff1 = np.zeros(len(l1)+1)
    coeff2 = np.zeros(len(l1)-1)
    for i in range(len(r)):
        x_max = np.cos(np.radians(r_max[i] * 1/60))
        x_min = np.cos(np.radians(r_min[i] * 1/60))
        coeff1[1:] = 1/(4*np.pi*(x_max - x_min)) * pk2D_interp_cross
        coeff2 = 1/(4*np.pi*(x_max - x_min)) * pk2D_interp_cross[1:]
        corr_2pt_cross[i] = np.polynomial.legendre.legval(x_max, coeff1) - np.polynomial.legendre.legval(x_max, coeff2) - np.polynomial.legendre.legval(x_min, coeff1) + np.polynomial.legendre.legval(x_min, coeff2)
    
    return corr_2pt_cross
'''

#### The functional form of prior #####
#### Here the prior is for the three varying parameters in MassiveNuS but it can change according to different tasks
def log_prior(theta):
    Mv, Omega_m, As = theta
    #with the constraints on neutrino masses and the assumption of normal hierarchy, the total sum has a least value of 0.06eV
    #the upper limit of Mv is 0.17eV from Planck and 0.62eV is the one from Coulton's paper
    if 0.06 <= Mv <= 0.62 and 0.18 <= Omega_m <= 0.42 and 1.29 < As < 2.91:
        return 0.0
    return -np.inf


#### The functional form of likelihood ######
def log_likelihood(theta, data, cov):
    Nu_Cosmo = class_obj(theta).class_computation()
    p = my_class(Nu_Cosmo, z_list_integration, ell)
    #p2 = my_class(Nu_Cosmo, z_list_integration2, ell) # for the second source redshift
    cosmo_dict = {'H': p.Hubble(),
                  'H0': p.Hubble_0(),
                  'Om0': p.Omega_m0(),
                  'DA_s': p.angular_diameter_dist_zs(),
                  'DA': p.angular_diameter_dist(),
                  'Pk': p.Pk_value()}
    '''
    # Another two dictionaries for one auto and cross 2PCF respectively
    cosmo_dict2 = {'H': p2.Hubble(),
                  'H0': p2.Hubble_0(),
                  'Om0': p2.Omega_m0(),
                  'DA_s': p2.angular_diameter_dist_zs(),
                  'DA': p2.angular_diameter_dist(),
                  'Pk': p2.Pk_value()}
    
    cosmo_dict12 = {'H': p.Hubble(),
                  'H0': p.Hubble_0(),
                  'Om0': p.Omega_m0(),
                  'DA_s1': p.angular_diameter_dist_zs(),
                  'DA_s2': p2.angular_diameter_dist_zs(),
                  'DA': p.angular_diameter_dist(),
                  'Pk': p.Pk_value()}
    
    '''
    # Compute the model vector in MCMC
    model = correlation_2pt(cosmo_dict, ell, z_list_integration)
    '''
    # Add two data vectors of auto 2PCF and one cross 2PCF together to form a tomographic data vector
    model_z2 = correlation_2pt(cosmo_dict2, ell, z_list_integration2)
    model_z12 = correlation_2pt_cross(cosmo_dict12, ell, z_list_integration)
    model = np.concatenate((model, model_z2, model_z12))
    '''
    diff = data - model
    # Here we use Gaussian likelihood and emcee only takes in log-likelihood
    gaussian_likelihood = -0.5 * np.dot(diff, hartlap_factor*np.linalg.solve(cov, diff))
    return gaussian_likelihood

#### The functional form of posterior probability
def log_probability(theta, data, cov):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data, cov)

# Read in and compute the data vector from noiseless simulation maps
data_vec = pd.read_csv('correlationfunc/2ptfunc_circpatch/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations9999_circular.csv', sep=' ')
'''
data_vec2 = pd.read_csv('correlationfunc/2ptfunc_circpatch/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations9999_circular.csv', sep=' ')
data_vec3 = pd.read_csv('correlationfunc/2ptfunc_circpatch/2pt_cross_mnv0.1_om0.3_As2.1_z11.0_z21.5_relizations10000_circular.csv', sep=' ')
data_vec = pd.concat([data_vec, data_vec2, data_vec3], axis=1)
'''
data_vec = data_vec.iloc[:4999,:]
data_vec = data_vec.to_numpy()
data_vec = np.mean(data_vec, axis=0)

# Read in and compute the data covariance matrix from noisy simulation maps
# We choose a separate set of simulation realizations to calculate the covariance matrix in order to avoid the dependence between the data vector and the covariance matrix  
corr2pt = pd.read_csv('correlationfunc/2ptfunc_circpatch_wn/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations10000_circular_LSST_whitenoise.csv', sep=' ')
'''
corr2pt2 = pd.read_csv('correlationfunc/2ptfunc_circpatch_wn/2pt_mnv0.1_om0.3_As2.1_z1.5_relizations10000_circular_LSST.csv', sep=' ')
corr2pt3 = pd.read_csv('correlationfunc/2ptfunc_circpatch_wn/2pt_cross_mnv0.1_om0.3_As2.1_z11.0_z21.5_relizations10000_circular_LSST.csv', sep=' ') 
corr2pt = pd.concat([corr2pt, corr2pt2, corr2pt3], axis=1)
'''
corr2pt = corr2pt.iloc[4999:,:]
covariance = corr2pt.cov()
covariance = covariance.to_numpy()
# Scale the covariance matrix with respect to the LSST survey area
covariance = covariance * (5/18000)


######initial conditions of uniform distribution#######
np.random.seed(42)
theta1_0 = np.random.uniform(0.06, 0.62, size=walkers)
theta2_0 = np.random.uniform(0.18, 0.42, size=walkers)
theta3_0 = np.random.uniform(1.29, 2.91, size=walkers)
p0 = np.zeros((walkers, n_param))
p0[:,0] = theta1_0
p0[:,1] = theta2_0
p0[:,2] = theta3_0
nwalkers, ndim = p0.shape 


filename = "MCMC_%swalkers_z%s_LSST_binave.h5"%(nwalkers, z_source)
#filename = "MCMC_%swalkers_z%s_LSST_binave_tomography.h5"%(nwalkers, z_source)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    # Multiprocessing in emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data_vec, covariance), backend=backend, pool=pool)
    # Maximum number of steps that can be carried out in MCMC
    max_n = 100000
    index = 0
    autocorr = np.empty(max_n)

    old_tau = np.inf

    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    
    '''
    # If one wants to run "steps" more after the first convergence of Markov Chains, one can read in the previously resulting MCMC, comment out the 
    # backend.reset function and the above lines below the sampler definition and run the below command
    sampler.run_mcmc(None, steps, progress=True)
    '''

    end = time.time()
    multi_time = end - start
    print("MCMC took {0:.1f} seconds".format(multi_time))