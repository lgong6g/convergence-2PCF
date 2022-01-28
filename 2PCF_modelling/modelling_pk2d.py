import time
start_program = time.time()

import numpy as np
import pandas as pd
from classy import Class
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import multiprocessing as mp
from scipy.optimize import fsolve
from scipy import integrate


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
    
###############above is a function to get individual neutrino masses########

def Mnu2omeganu(sum_masses, matter_density):
    # the neutrino mass constraints come from the MassiveNuS paper by Liu et al.
    m1, m2, m3 = get_masses(2.5e-3, 7.37e-5, sum_masses, 'NH') 
    mnu_arr = np.array([m1, m2, m3])* u.eV
    cosmo = FlatLambdaCDM(H0=h*100, Om0=matter_density, Neff=3.046, m_nu=mnu_arr, Ob0=omega_b/(h**2), Tcmb0=2.7255)
    return cosmo.Onu0*(h**2)

###############above is a function to compute density parameter of massive neutrinos######

print("Please provide cosmological parameters:")

Omega_m = float(input("Enter total matter density parameter(CDM+baryon+neutrino):"))

A_s = float(input("Enter primordial power spectrum amplitude:"))

Mv = float(input("Enter sum of massive neutrino masses:"))

n = int(input("Enter number of redshift bins:"))

zs = []
for i in range(n):
    bins = float(input("Enter redshift values:"))
    zs.append(bins)
    
kmax = float(input("Enter the maximum wave number (1/Mpc):"))
    
halofit_version = input("Enter the halofit version in CLASS:")

#multipole number range from 1 to 20000 with 150 bins in logspace
l = np.loadtxt("multipoles_for_convergencepowerspec_150l.txt", usecols=(0))

omega_b = 0.02254  #Liu's input, we have 0.02254 and 0.02254 is also the value in Takahashi simulation

n_s = 0.97 #n_s value for MassiveNuS

#sigma8 = 0.8

h = 0.7

Lbox = 512/h #simulaion box size in Mpc

##shell-correction fitting parameters##

a1 = 1.3606
        
a2 = 1.2674
        
a3 = 0.7028
        
c1 = 2.2041e-3
       
c2 = 1.0861e-2

########Input parameters#######

if Mv == 0.0:
    commonsettings  = {
                  'N_ur':3.046,
                  'N_ncdm':0,
                  'output':'mPk',
                  'P_k_max_1/Mpc':kmax,
                  # The next line should be uncommented fgor higher precision (but significantly slower running)
                  #'ncdm_fluid_approximation':3,
                  # You may uncomment this line to get more info on the ncdm sector from Class:
                  'background_verbose':10,
                  # omega_b here actually means Omega_b * h**2
                  'omega_b':omega_b,
                  'h':h,
                  'n_s':n_s,
                  'A_s':A_s, 
                  #'sigma8': sigma8,
                  'omega_cdm':Omega_m*(h**2)-omega_b, #0.12470 Liu's input for massless neutrino and Omega_m=0.3 case 
                  #value of cdm in takahashi simulation 0.11417,
                  'Omega_k':0.0,
                  'Omega_fld':0.0,
                  'Omega_scf':0.0,
                  'YHe':0.24, #'BBN', 0.24 Liu's input
                  'z_max_pk':5.0,
                  'non linear':halofit_version,
                  'write warnings':'yes'
                 }
else:
    m1, m2, m3 = get_masses(2.5e-3, 7.37e-5, Mv, 'NH')
    omega_nu = Mnu2omeganu(Mv, Omega_m)
    commonsettings  = {
                  'N_ur':0.00641,
                  'N_ncdm':3,
                  'output':'mPk',
                  'P_k_max_1/Mpc':kmax,
                  # The next line should be uncommented fgor higher precision (but significantly slower running)
                  #'ncdm_fluid_approximation':3,
                  # You may uncomment this line to get more info on the ncdm sector from Class:
                  'background_verbose':10,
                  'omega_b':omega_b,
                  'omega_cdm':Omega_m*(h**2)-(omega_b+omega_nu), #0.12362 Liu's input for massive neutrinos and Omega_m=0.3
                  'm_ncdm':str(m1)+','+str(m2)+','+str(m3),
                  'h':h,
                  'n_s':n_s,
                  'A_s':A_s, 
                  #'sigma8': sigma8,
                  'Omega_k':0.0,
                  'Omega_fld': 0.0,
                  'Omega_scf': 0.0,          
                  'YHe':0.24, #'BBN', 0.24 Liu's input
                  'z_max_pk':5.0,
                  'nonlinear_verbose':10,
                  'non linear':halofit_version,
                  'write warnings':'yes'
                 }


Nu_Cosmo = Class()
Nu_Cosmo.set(commonsettings)
Nu_Cosmo.compute()


def rz(z):
    return Nu_Cosmo.angular_distance(z)*(1+z)


def P_NL(k, z):
    # we do not compute matter power spectrum at scales smaller than the limit set by kmax
    if k > kmax: 
        return 0
    else:
        return Nu_Cosmo.pk(k, z) 

def PW_NL(k, z):
    return  (1 + c1*(k/h)**(-a1))**(a1)/(1 + c2*(k/h)**(-a2))**(a3) * P_NL(k, z)

def Pk_2D_integral(l, z_s):
    dPk_2D = lambda z : (1/Nu_Cosmo.Hubble(z)) * (9./4.) * (Nu_Cosmo.Hubble(0.0))**4 * Nu_Cosmo.Omega0_m()**2 * ((rz(z_s)-rz(z))/rz(z_s))**2 *(1+z)**2 * PW_NL(l/rz(z), z)  
    Pk_2D = integrate.quad(dPk_2D, 0., z_s)[0]
    return Pk_2D

def Pk_2D_integral_parallelisation(p):
    return Pk_2D_integral(p[0], p[1])

'''
# rather than using the fitting formula for the line-of-sight discretization correction, we explicitly compute the correction with integration in this part of the code
# in the following functions, delta_r is the shell-thickness in the ray-tracing process

def P_NL(k_perp, k_para, z):
    k = np.sqrt(k_para**2+k_perp**2)
    if k > kmax: 
        return 0
    else:
        return Nu_Cosmo.pk(k, z)

def dPW(k_para, delta_r, k_perp, z):
    return (delta_r/np.pi) * np.sinc(k_para*delta_r/2/np.pi)**2 * P_NL(k_perp, k_para, z)

def PW_NL(l, z, delta_r):
    k1 = 0.0 
    k2 = 600.0   
    #integrate the wave-vector component parallel to the line-of-sight
    shell_int = integrate.quad(dPW, k1, k2, args=(delta_r, l/rz(z), z), limit=5000)[0]
    return shell_int


def Pk_2D_integral(l, z_s, delta_r):
    dPk_2D = lambda z : (1/Nu_Cosmo.Hubble(z)) * (9./4.) * (Nu_Cosmo.Hubble(0.0))**4 * Nu_Cosmo.Omega0_m()**2 * ((rz(z_s)-rz(z))/rz(z_s))**2 *(1+z)**2 * PW_NL(l, z, delta_r)
    Pk_2D = integrate.quad(dPk_2D, 0., z_s)[0]
    return Pk_2D


def Pk_2D_integral_parallelisation(p):
    return Pk_2D_integral(p[0], p[1], p[2])
'''    

'''
#a piece of code for boxsize correction
def boxsize_correct(l, z):
    if l/rz(z) < 2*np.pi/Lbox:
        return 0
    else:
        return shell_fitting(l/rz(z), z)
'''
###############################

Pk2D = np.zeros((len(l), n+1))
Pk2D[:,0] = l

for i in range(1, len(zs)+1):
    print(zs[i-1])
    pool = mp.Pool(processes=30)
    result = pool.map(Pk_2D_integral_parallelisation, [[l[j], zs[i-1]] for j in range(len(l))])
    Pk2D[:,i] = np.array(result)

Pk2D_data = pd.DataFrame(Pk2D)
Pk2D_data.to_csv('angular_powerspec_forinterp/PK_2D_Kappa_Omega_m%s_M_v%s_A_s%s_zbins%s_lmax5e4_kmax%s_halofit_CLASS_shellfitting.csv'%(matter_density, sum_masses, A_s, n, kmax), sep=' ', index=False)

end_program = time.time()
print('\nTime taken for execution (seconds): ', end_program - start_program)