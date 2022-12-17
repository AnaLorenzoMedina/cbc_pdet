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


def sigmoid_2(z, zmid, gamma, delta, alpha=2.05, emax=0.967):
    denom = 1. + (z/zmid) ** alpha * \
        np.exp(gamma * ((z / zmid) - 1.) + delta * ((z**2 / zmid**2) - 1.))
    return emax / denom


def integrand_2(z_int, zmid, gamma, delta, alpha=2.05, emax=0.967):
    return zpdf_interp(z_int) * sigmoid_2(z_int, zmid, gamma, delta, alpha, emax)


def lam_2(z, pz, zmid, gamma, delta, alpha=2.05, emax=0.967):
    return pz * sigmoid_2(z, zmid, gamma, delta, alpha, emax)


# in_param is a generic minimization variable, here ln(zmid)
def logL_quad_2(in_param, z, pz, Total_expected, gamma, delta):
    zmid = np.exp(in_param[0])

    quad_fun = lambda z_int: Total_expected * integrand_2(z_int, zmid, gamma, delta)
    # it's hard to check here what is the value of new_try_z
    Lambda_2 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
    lnL = -Lambda_2 + np.sum(np.log(Total_expected * lam_2(z, pz, zmid, gamma, delta)))
    return lnL


def logL_quad_2_global(in_param, nbin1, nbin2, zmid_inter):
    # in_param is a generic minimization variable, here it is a list [gamma, ln(delta)]
    gamma, delta = in_param[0], np.exp(in_param[1])
    
    lnL_global = np.zeros([nbin1, nbin2])
    
    for i in range(0, nbin1):
        for j in range(0, nbin2):
            if j > i:
                continue
        
            m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
            m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
            mbin = m1inbin & m2inbin & found_any

            data = z_origin[mbin]
            data_pdf = z_pdf_origin[mbin]
            
            if len(data) < 1:
                continue
         
            index_sorted = np.argsort(data)
            z = data[index_sorted]
            pz = data_pdf[index_sorted]
            
            if zmid_inter[i,j] < 0.1 * min(z):
                zmid_inter[i,j] = min(z)

            Total_expected = NTOT * mean_mass_pdf[i,j]
            quad_fun = lambda z_int: Total_expected * integrand_2(z_int, zmid_inter[i,j], gamma, delta)
            Lambda_2 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
            lnL = -Lambda_2 + np.sum(np.log(Total_expected * lam_2(z, pz, zmid_inter[i,j], gamma, delta)))
            
            if lnL == -np.inf:
                print("epsilon gives a zero value in ", i, j, " bin because zmid is zero or almost zero")
                #print(sigmoid_2(z, zmid_inter[i,j], gamma, delta))
                continue
            
            lnL_global[i,j] = lnL

    print('\n', lnL_global.sum())            
    return lnL_global.sum()


# the nelder-mead algorithm has these default tolerances: xatol=1e-4, fatol=1e-4  

def MLE_2(z, pz, zmid_guess, Total_expected, gamma_new, delta_new):
    res = opt.minimize(fun=lambda in_param, z, pz: -logL_quad_2(in_param, z, pz, Total_expected, gamma_new, delta_new), 
                       x0=np.array([np.log(zmid_guess)]), 
                       args=(z, pz,), 
                       method='Nelder-Mead')
    
    zmid_res = np.exp(res.x) 
    min_likelihood = res.fun                
    return zmid_res, -min_likelihood


def MLE_2_global(nbin1, nbin2, zmid_inter, gamma_guess, delta_guess):
    res = opt.minimize(fun=lambda in_param: -logL_quad_2_global(in_param, nbin1, nbin2, zmid_inter), 
                       x0=np.array([gamma_guess, np.log(delta_guess)]), 
                       args=(), 
                       method='Nelder-Mead')
    
    gamma, logdelta = res.x 
    delta = np.exp(logdelta)
    min_likelihood = res.fun                
    return gamma, delta, -min_likelihood


try:
    os.mkdir('joint_fit_results')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('joint_fit_results/zmid')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('joint_fit_results/maxL')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('joint_fit_results/final_plots')
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

m1 = f["injections/mass1_source"][:]
m2 = f["injections/mass2_source"][:]
far_pbbh = f["injections/far_pycbc_bbh"][:]
far_gstlal = f["injections/far_gstlal"][:]
far_mbta = f["injections/far_mbta"][:]
far_pfull = f["injections/far_pycbc_hyperbank"][:]

mean_mass_pdf = np.loadtxt('mean_mpdf.dat')

###################################### for the z_pdf interpolation 

index_all = np.argsort(z_origin)
all_z = z_origin[index_all]
z_pdf = z_pdf_origin[index_all]

index = np.random.choice(np.arange(len(all_z)), 200, replace=False)

try_z = all_z[index]
try_zpdf = z_pdf[index]

index_try = np.argsort(try_z)
try_z_ordered = try_z[index_try]
try_zpdf_ordered = try_zpdf[index_try]

new_try_z = np.insert(try_z_ordered, 0, 0, axis=0)
new_try_zpdf = np.insert(try_zpdf_ordered, 0, 0, axis=0)

zpdf_interp = interpolate.interp1d(new_try_z, new_try_zpdf)

#####################################

# FAR threshold for finding an injection
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

# I'm not seeing why the reason to use this one value of zmid to scale some initial
# values of gamma / delta ... it should be better to pick one mass bin where the
# previous fit with gamma, delta works well and then scale the fitted gamma / delta
# values with that bin's zmid.  However if this works anyway maybe it doesn't matter
delta_new = 4*zmid_old**2
gamma_new = -0.6*zmid_old


total_lnL = np.zeros([1])
all_delta = np.array([delta_new])
all_gamma = np.array([gamma_new])

for k in range(0,10000):
    print('\n\n')
    print(k)
    
    gamma_new, delta_new, maxL_global = MLE_2_global(nbin1, nbin2, zmid_inter, gamma_new, delta_new)
    
    all_delta = np.append(all_delta, delta_new) 
    all_gamma = np.append(all_gamma, gamma_new)

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
            
            data = z_origin[mbin]
            data_pdf = z_pdf_origin[mbin]
            
            if len(data) < 1:
                continue
            
            index3 = np.argsort(data)
            z = data[index3]
            pz = data_pdf[index3]
            
            if zmid_inter[i,j] < 0.1 * min(z):
                zmid_inter[i,j] = min(z)
                
            Total_expected = NTOT * mean_mass_pdf[i,j]
            zmid_new, maxL = MLE_2(z, pz, zmid_inter[i,j], Total_expected, gamma_new, delta_new)
            
            if maxL == -np.inf:
                #just in case, but in principle this does not happen
                print("epsilon gives a zero value in ", i, j, " bin")
                maxL = 0
            
            zmid_inter[i, j] = zmid_new
            maxL_inter[i, j] = maxL
    
    name = f"joint_fit_results/zmid/zmid_{k}.dat"
    np.savetxt(name, zmid_inter, fmt='%10.3f')
    
    name = f"joint_fit_results/maxL/maxL_{k}.dat"
    np.savetxt(name, maxL_inter, fmt='%10.3f')
    
    total_lnL = np.append(total_lnL, maxL_inter.sum())
    
    print(maxL_inter.sum())
    print(total_lnL[k + 1] - total_lnL[k])
    
    if np.abs( total_lnL[k + 1] - total_lnL[k] ) <= 1e-2:
        break
print(k)

np.savetxt('joint_fit_results/all_delta.dat', np.delete(all_delta, 0), fmt='%e')
np.savetxt('joint_fit_results/all_gamma.dat', np.delete(all_gamma, 0), fmt='%10.5f')
np.savetxt('joint_fit_results/total_lnL.dat', np.delete(total_lnL, 0), fmt='%10.3f')


#compare_1 plots

#k = 3  #number of the last iteration

zmid_plot = np.loadtxt(f'joint_fit_results/zmid/zmid_{k}.dat')
gamma_plot = np.loadtxt('joint_fit_results/all_gamma.dat')[-1]
delta_plot = np.loadtxt('joint_fit_results/all_delta.dat')[-1]

for i in range(0,nbin1):
    for j in range(0,nbin2):
        
        try:
            data_binned = np.loadtxt(f'z_binned/{i}{j}_data.dat')
        except OSError:
            continue
    
        mid_z=data_binned[:,0]
        z_com_1=np.linspace(0,max(mid_z), 200)
        pz_binned=data_binned[:,1]
        zm_detections=data_binned[:,2]
        nonzero = zm_detections > 0
        
        plt.figure()
        plt.plot(mid_z, pz_binned, '.', label='bins over z')
        plt.errorbar(mid_z[nonzero], pz_binned[nonzero], yerr=pz_binned[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
        plt.plot(z_com_1, sigmoid_2(z_com_1, zmid_plot[i,j], gamma_plot, delta_plot), '-', label=r'$\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.ylabel(r'$P_{det}(z)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.legend(fontsize=14)
        name=f"joint_fit_results/final_plots/{i}{j}.png"
        plt.savefig(name, format='png')
        
        plt.close()
