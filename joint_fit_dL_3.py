# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:40:04 2022

@author: Ana
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.optimize as opt
from scipy import integrate
from matplotlib import rc
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
from tqdm import tqdm
import os
import errno


def sigmoid_2(dL, dLmid, gamma, delta, alpha, emax=0.967):
    denom = 1. + (dL/dLmid) ** alpha * \
        np.exp(gamma * ((dL / dLmid) - 1.) + delta * ((dL**2 / dLmid**2) - 1.))
    return emax / denom


def integrand_2(dL_int, dLmid, gamma, delta, alpha, emax=0.967):
    return dLpdf_interp(dL_int) * sigmoid_2(dL_int, dLmid, gamma, delta, alpha, emax)


def lam_2(dL, pdL, dLmid, gamma, delta, alpha, emax=0.967):
    return pdL * sigmoid_2(dL, dLmid, gamma, delta, alpha, emax)


# in_param is a generic minimization variable, here ln(zmid)
def logL_quad_2(in_param, dL, pdL, Total_expected, gamma, delta, alpha):
    dLmid = np.exp(in_param[0])
    quad_fun = lambda dL_int: Total_expected * integrand_2(dL_int, dLmid, gamma, delta, alpha)
    # it's hard to check here what is the value of new_try_z
    Lambda_2 = integrate.quad(quad_fun, min(new_try_dL), max(new_try_dL))[0]
    lnL = -Lambda_2 + np.sum(np.log(Total_expected * lam_2(dL, pdL, dLmid, gamma, delta, alpha)))
    return lnL


def logL_quad_2_global(in_param, nbin1, nbin2, dLmid_inter):
    # in_param is a generic minimization variable, here it is a list [gamma, ln(delta), ln(gamma)]
    gamma, delta, alpha = in_param[0], np.exp(in_param[1]), np.exp(in_param[2])
    
    lnL_global = np.zeros([nbin1, nbin2])
    
    for i in range(0, nbin1):
        for j in range(0, nbin2):
            if j > i:
                continue
        
            m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
            m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
            mbin = m1inbin & m2inbin & found_any

            data = dL_origin[mbin]
            data_pdf = dL_pdf_origin[mbin]
            
            if len(data) < 1:
                continue
         
            index_sorted = np.argsort(data)
            dL = data[index_sorted]
            pdL = data_pdf[index_sorted]
            
            if dLmid_inter[i,j] < 0.1 * min(dL):
                dLmid_inter[i,j] = min(dL)

            Total_expected = NTOT * mean_mass_pdf[i,j]
            quad_fun = lambda dL_int: Total_expected * integrand_2(dL_int, dLmid_inter[i,j], gamma, delta, alpha)
            Lambda_2 = integrate.quad(quad_fun, min(new_try_dL), max(new_try_dL))[0]
            #print(Lambda_2)
            lnL = -Lambda_2 + np.sum(np.log(Total_expected * lam_2(dL, pdL, dLmid_inter[i,j], gamma, delta, alpha)))
            #sigmoid_2(dL, dLmid_inter[i,j], gamma, delta)
            if lnL == -np.inf:
                print("epsilon gives a zero value in ", i, j, " bin because zmid is zero or almost zero")
                #print(sigmoid_2(z, zmid_inter[i,j], gamma, delta))
                continue
            
            lnL_global[i,j] = lnL
    #print(lnL_global)
    print('\n', lnL_global.sum())            
    return lnL_global.sum()


# the nelder-mead algorithm has these default tolerances: xatol=1e-4, fatol=1e-4  

def MLE_2(dL, pdL, dLmid_guess, Total_expected, gamma_new, delta_new, alpha_new):
    res = opt.minimize(fun=lambda in_param, dL, pdL: -logL_quad_2(in_param, dL, pdL, Total_expected, gamma_new, delta_new, alpha_new), 
                       x0=np.array([np.log(dLmid_guess)]), 
                       args=(dL, pdL,), 
                       method='Nelder-Mead')
    
    dLmid_res = np.exp(res.x) 
    min_likelihood = res.fun                
    return dLmid_res, -min_likelihood


def MLE_2_global(nbin1, nbin2, dLmid_inter, gamma_guess, delta_guess, alpha_guess):
    res = opt.minimize(fun=lambda in_param: -logL_quad_2_global(in_param, nbin1, nbin2, dLmid_inter), 
                       x0=np.array([gamma_guess, np.log(delta_guess), np.log(alpha_guess)]), 
                       args=(), 
                       method='Nelder-Mead')
    
    gamma, logdelta, logalpha = res.x 
    delta = np.exp(logdelta)
    alpha = np.exp(logalpha)
    min_likelihood = res.fun                
    return gamma, delta, alpha, -min_likelihood


try:
    os.mkdir('dL_joint_fit_results_3')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('dL_joint_fit_results_3/dLmid')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('dL_joint_fit_results_3/maxL')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('dL_joint_fit_results_3/final_plots')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Is this needed ?
plt.close('all')

rc('text', usetex=True)
np.random.seed(42)

f = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')

NTOT = f.attrs['total_generated']
z_origin = f["injections/redshift"][:]
z_pdf_origin = f["injections/redshift_sampling_pdf"][:]

dL_origin = f["injections/distance"][:]

m1 = f["injections/mass1_source"][:]
m2 = f["injections/mass2_source"][:]
far_pbbh = f["injections/far_pycbc_bbh"][:]
far_gstlal = f["injections/far_gstlal"][:]
far_mbta = f["injections/far_mbta"][:]
far_pfull = f["injections/far_pycbc_hyperbank"][:]

mean_mass_pdf = np.loadtxt('mean_mpdf.dat')

###################################### for the dL_pdf interpolation 

H0 = 67.9 #km/sMpc
c = 3e5 #km/s
omega_m = 0.3065
A = np.sqrt(omega_m*(1+z_origin)**3+1-omega_m)

dL_dif_origin = (c*(1+z_origin)/H0)*(1/A)

dL_pdf_origin = z_pdf_origin/dL_dif_origin

index_all = np.argsort(dL_origin)
all_dL = dL_origin[index_all]
dL_pdf = dL_pdf_origin[index_all]

index = np.random.choice(np.arange(len(all_dL)), 200, replace=False)

try_dL = all_dL[index]
try_dLpdf = dL_pdf[index]

index_try = np.argsort(try_dL)
try_dL_ordered = try_dL[index_try]
try_dLpdf_ordered = try_dLpdf[index_try]

new_try_dL = np.insert(try_dL_ordered, 0, 0, axis=0)
new_try_dLpdf = np.insert(try_dLpdf_ordered, 0, 0, axis=0)

dLpdf_interp = interpolate.interp1d(new_try_dL, new_try_dLpdf)

#####################################

# FAR (false alarm rate) threshold for finding an injection
thr = 1.

nbin1 = 14
nbin2 = 14

mmin = 2.
mmax = 100.
m1_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin1+1), 1)
m2_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin2+1), 1)

found_pbbh = far_pbbh <= thr
found_gstlal = far_gstlal <= thr
found_mbta = far_mbta <= thr
found_pfull = far_pfull <= thr
found_any = found_pbbh | found_gstlal | found_mbta | found_pfull


## descoment for a new optimization


zmid_inter = np.loadtxt('maximization_results/zmid_2.dat')
#zmid_old is the zmid value from the old fit to the FC data
# -> This was a zmid value for one specific mass bin? 
zmid_old = 0.327

H0 = 67.9 #km/sMpc
omega_m = 0.3065
c = 3e5 #km/s

#A_inter = np.sqrt(omega_m*(1+zmid_inter)**3+1-omega_m)

def fun_A(t):
    return np.sqrt(omega_m*(1+t)**3+1-omega_m)

quad_fun_A = lambda t: 1/fun_A(t)

dLmid_inter = np.array([[(c*(1+zmid_inter[i,j])/H0)*integrate.quad(quad_fun_A, 0, zmid_inter[i,j])[0] for j in range(nbin2)] for i in range(nbin1)])

# I'm not seeing why the reason to use this one value of zmid to scale some initial
# values of gamma / delta ... it should be better to pick one mass bin where the
# previous fit with gamma, delta works well and then scale the fitted gamma / delta
# values with that bin's zmid.  However if this works anyway maybe it doesn't matter

dLmid_old = (c*(1+zmid_old)/H0)*integrate.quad(quad_fun_A, 0, zmid_old)[0]
delta_new = 0.1146989
gamma_new = -0.18395
alpha_new = 2.05


total_lnL = np.zeros([1])
all_delta = np.array([delta_new])
all_gamma = np.array([gamma_new])
all_alpha = np.array([alpha_new])

for k in range(0,10000):
    print('\n\n')
    print(k)
    
    gamma_new, delta_new, alpha_new, maxL_global = MLE_2_global(nbin1, nbin2, dLmid_inter, gamma_new, delta_new, alpha_new)
    
    all_delta = np.append(all_delta, delta_new) 
    all_gamma = np.append(all_gamma, gamma_new)
    all_alpha = np.append(all_alpha, alpha_new)

    maxL_inter = np.zeros([nbin1, nbin2])

    for i in range(0, nbin1):
        for j in range(0, nbin2):
            
            if j > i:
                continue
                
            print('\n\n')
            print(i, j)
            
            m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
            m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
            mbin = m1inbin & m2inbin & found_any
            
            data = dL_origin[mbin]
            data_pdf = dL_pdf_origin[mbin]
            
            if len(data) < 1:
                continue
            
            index3 = np.argsort(data)
            dL = data[index3]
            pdL = data_pdf[index3]
            
            if dLmid_inter[i,j] < 0.1 * min(dL):
                dLmid_inter[i,j] = min(dL)
                
            Total_expected = NTOT * mean_mass_pdf[i,j]
            dLmid_new, maxL = MLE_2(dL, pdL, dLmid_inter[i,j], Total_expected, gamma_new, delta_new, alpha_new)
            
            if maxL == -np.inf:
                #just in case, but in principle this does not happen
                print("epsilon gives a zero value in ", i, j, " bin")
                maxL = 0
            
            dLmid_inter[i, j] = dLmid_new
            maxL_inter[i, j] = maxL
    
    name = f"dL_joint_fit_results_3/dLmid/dLmid_{k}.dat"
    np.savetxt(name, dLmid_inter, fmt='%10.3f')
    
    name = f"dL_joint_fit_results_3/maxL/maxL_{k}.dat"
    np.savetxt(name, maxL_inter, fmt='%10.3f')
    
    total_lnL = np.append(total_lnL, maxL_inter.sum())
    
    print(maxL_inter.sum())
    print(total_lnL[k + 1] - total_lnL[k])
    
    if np.abs( total_lnL[k + 1] - total_lnL[k] ) <= 1e-2:
        break
print(k)

np.savetxt('dL_joint_fit_results_3/all_delta.dat', np.delete(all_delta, 0), fmt='%10.5f')
np.savetxt('dL_joint_fit_results_3/all_gamma.dat', np.delete(all_gamma, 0), fmt='%10.5f')
np.savetxt('dL_joint_fit_results_3/all_alpha.dat', np.delete(all_alpha, 0), fmt='%10.5f')
np.savetxt('dL_joint_fit_results_3/total_lnL.dat', np.delete(total_lnL, 0), fmt='%10.3f')


#compare_1 plots

#k = 3  #number of the last iteration

dLmid_plot = np.loadtxt(f'dL_joint_fit_results_3/dLmid/dLmid_{k}.dat')
gamma_plot = np.loadtxt('dL_joint_fit_results_3/all_gamma.dat')[-1]
delta_plot = np.loadtxt('dL_joint_fit_results_3/all_delta.dat')[-1]
alpha_plot = np.loadtxt('dL_joint_fit_results_3/all_alpha.dat')[-1]

for i in range(0,nbin1):
    for j in range(0,nbin2):
        
        try:
            data_binned = np.loadtxt(f'dL_binned/{i}{j}_data.dat')
        except OSError:
            continue
    
        mid_dL=data_binned[:,0]
        dL_com_1=np.linspace(0,max(mid_dL), 200)
        pdL_binned=data_binned[:,1]
        dLm_detections=data_binned[:,2]
        nonzero = dLm_detections > 0
        
        plt.figure()
        plt.plot(mid_dL, pdL_binned, '.', label='bins over dL')
        plt.errorbar(mid_dL[nonzero], pdL_binned[nonzero], yerr=pdL_binned[nonzero]/np.sqrt(dLm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
        plt.plot(dL_com_1, sigmoid_2(dL_com_1, dLmid_plot[i,j], gamma_plot, delta_plot, alpha_plot), '-', label=r'$\varepsilon_2$')
        plt.xlabel(r'$dL$', fontsize=14)
        plt.ylabel(r'$P_{det}(dL)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.legend(fontsize=14)
        name=f"dL_joint_fit_results_3/final_plots/{i}{j}.png"
        plt.savefig(name, format='png')
        
        plt.close()
