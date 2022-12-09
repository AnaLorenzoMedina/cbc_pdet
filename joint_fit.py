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

def sigmoid_2(z, zmid,delta, gamma, alpha=2.05, emax=0.967):
    return emax/(1+(z/zmid)**alpha*np.exp(delta*(z**2-zmid**2)+gamma*(z-zmid)))

def integrand_2(z_int, zmid, delta, gamma, alpha=2.05, emax=0.967):
    return zpdf_interp(z_int)*sigmoid_2(z_int, zmid, delta, gamma)

def lam_2(z, pz, zmid, delta, gamma, alpha=2.05, emax=0.967):
    return pz*sigmoid_2(z, zmid, delta, gamma, alpha, emax)


def logL_quad_2(in_param, z, pz):
    
    delta = delta_new; gamma = gamma_new
    
    zmid = np.exp(in_param[0])
    
    quad_fun = lambda z_int: Total_expected*integrand_2(z_int, zmid, delta, gamma)
    
    Landa_2 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
    
    lnL = -Landa_2 + np.sum(np.log(Total_expected*lam_2(z, pz, zmid, delta, gamma)) )
    return lnL

def logL_quad_2_global(in_param):
    
    lnL_global = np.zeros([nbin1,nbin2])
    
    for i in range(0,nbin1):
        for j in range(0,nbin2):
            if j>i:
                continue
        
            m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
            m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
            mbin = m1inbin & m2inbin & found_any
            
            data = z_origin[mbin]
            data_pdf = z_pdf_origin[mbin]
            
            if len(data)<1:
                continue
            
            index3 = np.argsort(data)
            z = data[index3]
            pz = data_pdf[index3]
            
            Total_expected = NTOT*mean_mass_pdf[i,j]
    
            delta, gamma = np.exp(in_param[0]),  in_param[1]
            
            quad_fun = lambda z_int: Total_expected*integrand_2(z_int, zmid_inter[i,j], delta, gamma)
            
            Landa_2 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
            
            lnL = -Landa_2 + np.sum(np.log(Total_expected*lam_2(z, pz, zmid_inter[i,j], delta, gamma)) )
            
            if lnL==-np.inf:
                #print(i,j)
                #print(sigmoid_2(z, zmid_inter[i,j], delta, gamma))
                continue
            
            lnL_global[i,j] = lnL
            
   
    print(lnL_global.sum())        
            
    return lnL_global.sum()


# the nelder-mead algorithm has these default tolerances: xatol=1e-4, fatol=1e-4  

def MLE_2(z, pz, zmid_guess):
    res = opt.minimize( fun = lambda in_param, z, pz: -logL_quad_2(in_param, z, pz), 
                        x0 = np.array([np.log(zmid_guess)]), 
                        args = (z, pz,), 
                        method='Nelder-Mead')
    
    zmid_res = np.exp(res.x) 
    min_likelihood = res.fun                
    return zmid_res, -min_likelihood

def MLE_2_global():
    res = opt.minimize( fun = lambda in_param: -logL_quad_2_global(in_param), 
                        x0 = np.array([np.log(delta_new), gamma_new]), 
                        args = (), 
                        method='Nelder-Mead')
    
    delta, gamma = np.exp(res.x) 
    # we don't exponentiate gamma though
    gamma = np.log(gamma)
    min_likelihood = res.fun                
    return delta, gamma, -min_likelihood


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
        

plt.close('all')

rc('text', usetex=True)

np.random.seed(42)

f = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')

NTOT = f.attrs['total_generated']

z_origin = f["injections/redshift"][:]
z_pdf_origin = f["injections/redshift_sampling_pdf"][:]

m1 = f["injections/mass1_source"][:]
m2 = f["injections/mass2_source"][:]
far_pbbh = f["injections/far_pycbc_bbh"][:]  # rename this far_pbbh
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

new_try_z = np.insert(try_z_ordered , 0, 0, axis=0)
new_try_zpdf = np.insert(try_zpdf_ordered , 0, 0, axis=0)

zpdf_interp = interpolate.interp1d(new_try_z, new_try_zpdf)

#####################################

thr = 1

nbin1 = 14
nbin2 = 14

mmin = 2 ; mmax = 100

m1_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin1+1) , 1)
m2_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin2+1) , 1)

found_pbbh = far_pbbh <= thr
found_gstlal = far_gstlal <= thr
found_mbta = far_mbta <= thr
found_pfull = far_pfull <= thr
found_any = found_pbbh | found_gstlal | found_mbta | found_pfull

## descoment for a new optimization


'''
zmid_inter = np.loadtxt('maximization_results/zmid_2.dat')
delta_new = 4
gamma_new = -0.6

total_lnL = np.zeros([1])
all_delta = np.array([delta_new])
all_gamma = np.array([gamma_new])


for k in range(0,10000):
    
    print('\n\n')
    print(k)
    
    delta_new, gamma_new, maxL_global = MLE_2_global()
    
    all_delta = np.append(all_delta, delta_new) 
    all_gamma = np.append(all_gamma, gamma_new)
    

    maxL_inter = np.zeros([nbin1,nbin2])

    for i in range(0, nbin1):
        for j in range(0, nbin2):
            
            print('\n\n')
            print(i,j)
            
            m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
            m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
            mbin = m1inbin & m2inbin & found_any
            
            data = z_origin[mbin]
            data_pdf = z_pdf_origin[mbin]
            
            if len(data)<1:
                continue
            
            index3 = np.argsort(data)
            z = data[index3]
            pz = data_pdf[index3]
            
            Total_expected = NTOT*mean_mass_pdf[i,j]
            
            zmid_new, maxL = MLE_2(z, pz, zmid_inter[i,j])
            
            zmid_inter[i,j] = zmid_new
            maxL_inter[i,j] = maxL
    
    name = f"joint_fit_results/zmid/zmid_{k}.dat"
    np.savetxt(name, zmid_inter, fmt='%10.3f')
    
    name = f"joint_fit_results/maxL/maxL_{k}.dat"
    np.savetxt(name, maxL_inter, fmt='%10.3f')
    
    total_lnL = np.append(total_lnL, maxL_inter.sum() )
    
    print( maxL_inter.sum())
    print(total_lnL[k+1] - total_lnL[k])
    
    if np.abs( total_lnL[k+1] - total_lnL[k] ) <= 1e-2:
        break
print(k)

np.savetxt('joint_fit_results/all_delta.dat', np.delete(all_delta, 0), fmt='%e')
np.savetxt('joint_fit_results/all_gamma.dat', np.delete(all_gamma,0), fmt='%10.5f')
np.savetxt('joint_fit_results/total_lnL.dat', np.delete(total_lnL,0), fmt='%10.3f')

'''
#compare_1 plots

k = 10  #number of the last iteration

zmid_plot = np.loadtxt(f'joint_fit_results/zmid/zmid_{k}.dat')
delta_plot = np.loadtxt('joint_fit_results/all_delta.dat')[-1]
gamma_plot = np.loadtxt('joint_fit_results/all_gamma.dat')[-1]


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
        plt.plot(z_com_1, sigmoid_2(z_com_1, zmid_plot[i,j], delta_plot, gamma_plot), '-', label=r'$\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.ylabel(r'$P_{det}(z)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.legend(fontsize=14)
        name=f"joint_fit_results/final_plots/{i}{j}.png"
        plt.savefig(name, format='png')
        
        plt.close('all')
