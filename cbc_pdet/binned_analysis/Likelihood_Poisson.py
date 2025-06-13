# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:50:59 2022

@author: Ana
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.optimize as opt
from scipy import integrate
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
#from tqdm import tqdm
import os
import errno


def epsilon(z, zmid, a, zeta, emax=0.967):
    return emax/(1+(z/zmid)**(a+zeta*np.tanh(z/zmid)))

def sigmoid_2(z, zmid,delta, gamma, alpha=2.05, emax=0.967):
    return emax/(1+(z/zmid)**alpha*np.exp(delta*(z**2-zmid**2)+gamma*(z-zmid)))

def integrand_1(z_int,zmid, a, zeta, emax=0.967):
    return Total_expected*zpdf_interp(z_int)*epsilon(z_int, zmid, a, zeta)

def integrand_2(z_int,zmid, delta, gamma, alpha=2.05, emax=0.967):
    return Total_expected*zpdf_interp(z_int)*sigmoid_2(z_int, zmid, delta, gamma)

def lam_1(z, pz, zmid, a, zeta, emax=0.967):
    return Total_expected*pz*epsilon(z, zmid, a, zeta, emax)

def lam_2(z, pz, zmid, delta, gamma, alpha=2.05, emax=0.967):
    return Total_expected*pz*sigmoid_2(z, zmid, delta, gamma, alpha, emax)

def logL_quad_1(in_param, z, pz):
    
    zmid, a, zeta = np.exp(in_param[0]), np.exp(in_param[1]), np.exp(in_param[2])
    
    quad_fun = lambda z_int: integrand_1(z_int, zmid, a, zeta)
    
    Landa_1 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
    
    lnL = -Landa_1 + np.sum(np.log(lam_1(z, pz, zmid, a, zeta)) )
    return lnL

def logL_quad_2(in_param, z, pz):
    
    zmid, delta, gamma = np.exp(in_param[0]), np.exp(in_param[1]), in_param[2]
    
    quad_fun = lambda z_int: integrand_2(z_int, zmid, delta, gamma)
    
    Landa_2 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
    
    lnL = -Landa_2 + np.sum(np.log(lam_2(z, pz, zmid, delta, gamma)) )
    print(lnL)
    return lnL

# the nelder-mead algorithm has these default tolerances: xatol=1e-4, fatol=1e-4  

def MLE_1(z, pz):
    res = opt.minimize( fun = lambda in_param, z, pz: -logL_quad_1(in_param, z, pz), 
                        x0 = np.array([np.log(np.average(z)), 0, 0]), 
                        args = (z, pz,), 
                        method='Nelder-Mead')
    
    zmid, a, zeta = np.exp(res.x)   
    min_likelihood = res.fun                
    return zmid, a, zeta, -min_likelihood 

def MLE_2(z, pz):
    res = opt.minimize( fun = lambda in_param, z, pz: -logL_quad_2(in_param, z, pz), 
                        x0 = np.array([np.log(np.average(z)), np.log(4), -0.6]), 
                        args = (z, pz,), 
                        method='Nelder-Mead')
    
    zmid, delta, gamma = np.exp(res.x) 
    # we don't exponentiate gamma though
    gamma = np.log(gamma)
    min_likelihood = res.fun                
    return zmid, delta, gamma, -min_likelihood

try:
    os.mkdir('maximization_results')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('maximization_results/fit_normal')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('maximization_results/fit_logscale')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('maximization_results/poisson_compare_1')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('maximization_results/poisson_compare_2')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('maximization_results/poisson_expected_nf')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        

plt.close('all')

rc('text', usetex=True)

np.random.seed(42)

#from matrices import *

N=26 #number of bins we want in each mass bin for plotting

file = h5py.File('../samples-rpo4a_v2_20250220153231UTC-1366933504-23846400.hdf', 'r')

NTOT = file.attrs['total_generated']

m1 = file['events'][:]['mass1_source']
m2 = file['events'][:]['mass2_source']

m1_pdf = np.exp(file["events"][:]["lnpdraw_mass1_source"])
m2_pdf = np.exp(file["events"][:]["lnpdraw_mass2_source_GIVEN_mass1_source"])
m_pdf = m1_pdf * m2_pdf

z_origin = file['events'][:]['z']
dL_origin = file['events'][:]['luminosity_distance']

z_pdf_origin = np.exp(file["events"][:]["lnpdraw_z"])

far_pbbh = file["events"][:]["pycbc_far"]
far_gstlal = file["events"][:]["gstlal_far"]
far_mbta = file["events"][:]["mbta_far"]
far_cwb = file["events"][:]["cwb-bbh_far"]

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
zpdf_interp_all = interpolate.interp1d(np.insert(all_z , 0, 0, axis=0), np.insert(z_pdf, 0, 0, axis=0))

#####################################

thr = 1

nbin1 = 15
nbin2 = 15

mmin = 1.
mmax = 1000.
m1_bin = np.round(np.logspace(np.log10(1), np.log10(1000), nbin1+1), 1)
m2_bin = np.round(np.logspace(np.log10(1), np.log10(1000), nbin2+1), 1)

found_pbbh = far_pbbh <= thr
found_gstlal = far_gstlal <= thr
found_mbta = far_mbta <= thr
found_cwb = far_cwb <= thr
found_any = found_pbbh | found_gstlal | found_mbta | found_cwb

## descoment for a new optimization

zmid_1 = np.zeros([nbin1,nbin2])
zmid_2 = np.zeros([nbin1,nbin2])
maxL_1 = np.zeros([nbin1,nbin2])
maxL_2 = np.zeros([nbin1,nbin2])
a_1 = np.zeros([nbin1,nbin2])
zeta_1 = np.zeros([nbin1,nbin2])
delta_2 = np.zeros([nbin1,nbin2])
gamma_2 = np.zeros([nbin1,nbin2])
n_points = np.zeros([nbin1,nbin2])
index_n = np.zeros([nbin1,nbin2])


## comment for a new optimization

# zmid_1 = np.loadtxt('maximization_results/zmid_1.dat')

# zmid_2 = np.loadtxt('maximization_results/zmid_2.dat')

# maxL_1 = np.loadtxt('maximization_results/maxL_1.dat')

# maxL_2 = np.loadtxt('maximization_results/maxL_2.dat')

# a_1 = np.loadtxt('maximization_results/a_1.dat')

# zeta_1 = np.loadtxt('maximization_results/zeta_1.dat')

# delta_2 = np.loadtxt('maximization_results/delta_2.dat')

# gamma_2 = np.loadtxt('maximization_results/gamma_2.dat')

# n_points = np.loadtxt('maximization_results/n_points.dat')

# index_n = np.array([[f'{i}{j}' for j in range(0,nbin2)] for i in range(0,nbin1)])

## OPTIMIZATION AND PLOTTING

for i in range(0,nbin1):
    for j in range(0,nbin2):
        
        index_n[i,j]=f'{i}{j}'
        
        if j>i:
            continue

        plt.close('all')

        np.random.seed(42)
        
        print('\n\n\n')
        print(i,j)
        
        m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
        m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
        mbin = m1inbin & m2inbin & found_any
        
        data = z_origin[mbin]
        data_pdf = z_pdf_origin[mbin]
        
        if len(data)<1:
            n_points[i,j]=0
            continue
        
        index3 = np.argsort(data)
        z = data[index3]
        pz = data_pdf[index3]
        
        n_points[i,j] = len(z)
        
        index_n[i,j]=f'{i}{j}'
        
        Total_expected = NTOT*mean_mass_pdf[i,j]
        
        ### ALREADY OPTIMIZED (comment for new opt values)
        
        #zmid, a, zeta, lnL = zmid_1[i,j], a_1[i,j], zeta_1[i,j], maxL_1[i,j]
        #zmid_new, delta, gamma, lnL_new = zmid_2[i,j], delta_2[i,j], gamma_2[i,j], maxL_2[i,j]

        ### OPTIMIZATION (descomment for a new optimization)

        zmid, a, zeta, lnL = MLE_1(z, pz)
        zmid_new, delta, gamma, lnL_new = MLE_2(z, pz)
        
        zmid_1[i,j] = zmid      ;    maxL_1[i,j] = lnL
        a_1[i,j] = a            ;    zeta_1[i,j] = zeta
    
        zmid_2[i,j]= zmid_new   ;    maxL_2[i,j]= lnL_new
        delta_2[i,j]= delta     ;    gamma_2[i,j]= gamma
        
        
        ############   PLOTTING    ########
        
        zplot=np.linspace(0, max(z), 200)
        zpdf_plot=zpdf_interp_all(zplot)
        
        #compare both fits (normal scale)
        plt.figure()
        plt.plot(zplot, epsilon(zplot, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
        plt.plot(zplot, sigmoid_2(zplot, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.ylabel(r'$P_{det}(z)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.ylim(-0.05,1)
        plt.legend()
        name=f"maximization_results/fit_normal/{i}{j}.png"
        plt.savefig(name, format='png')
        
        
        #compare both fits (log scale)
        plt.figure()
        plt.plot(zplot, epsilon(zplot, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
        plt.plot(zplot, sigmoid_2(zplot, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.ylabel(r'$P_{det}(z)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.yscale('log')
        plt.ylim(0,1.5)
        plt.legend()
        name=f"maximization_results/fit_logscale/{i}{j}.png"
        plt.savefig(name, format='png')
        '''
        #compare_1 plot 
        
        data_binned = np.loadtxt(f'dL_binned/{i}{j}_data.dat')
        mid_dL=data_binned[:,0]
        dL_com_1=np.linspace(0,max(mid_dL), 200)
        pdL_binned=data_binned[:,1]
        dLm_detections=data_binned[:,2]
        nonzero = dLm_detections > 0

        plt.figure()
        plt.plot(mid_dL, pdL_binned, '.', label='bins over dL')
        plt.errorbar(mid_dL[nonzero], pdL_binned[nonzero], yerr=pdL_binned[nonzero]/np.sqrt(dLm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
        plt.plot(dL_com_1, epsilon(dL_com_1, zmid, a, zeta), '-', label=r'$\varepsilon_1$')
        plt.plot(dL_com_1, sigmoid_2(dL_com_1, zmid_new, delta, gamma), '-', label=r'$\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.ylabel(r'$P_{det}(z)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.legend(fontsize=14)
        name=f"maximization_results/poisson_compare_1/{i}{j}.png"
        plt.savefig(name, format='png')
        
        
        #compare_2 plot

        quad_fun_1 = lambda z_int: zpdf_interp_all(z_int)*epsilon(z_int, zmid, a, zeta)
        quad_fun_2 = lambda z_int: zpdf_interp_all(z_int)*sigmoid_2(z_int, zmid_new, delta, gamma)

        C1=integrate.quad(quad_fun_1, 0, max(all_z))[0]
        C2=integrate.quad(quad_fun_2, 0, max(all_z))[0]

        edges = np.linspace(0,max(z),30)
        
        plt.figure()
        plt.hist(z, edges ,alpha=0.5, density=True, label='actual detected events')
        plt.plot(zplot, zpdf_plot*epsilon(zplot, zmid, a, zeta)/C1, '-', label=r'$p(z)\cdot\varepsilon_1$')
        plt.plot(zplot, zpdf_plot*sigmoid_2(zplot, zmid_new, delta, gamma)/C2, '-', label=r'$p(z)\cdot\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.legend(fontsize=11)
        name=f"maximization_results/poisson_compare_2/{i}{j}.png"
        plt.savefig(name, format='png')
        
        
        #expected nf plot
        
        z_edges = np.linspace(0,max(z),N)
        delta_z = z_edges[2]-z_edges[1]
        mid = (z_edges[:-1]+z_edges[1:])/2
    
        exp_tot_1=np.array([Total_expected*integrate.quad(quad_fun_1, z_edges[u], z_edges[u+1])[0] for u in range(len(z_edges)-1)])
        exp_tot_2=np.array([Total_expected*integrate.quad(quad_fun_2, z_edges[u], z_edges[u+1])[0] for u in range(len(z_edges)-1)])

        nf_1 = exp_tot_1
        nf_2 = exp_tot_2

        plt.figure()
        n,_,_ = plt.hist(z, z_edges, alpha=0.5, label=r'Actual $n_f$')
        plt.errorbar(np.array(mid)[np.where(n>0)[0]], n[n>0], yerr=np.sqrt(n[n>0]), fmt="none", color="k", capsize=1, elinewidth=0.05)
        plt.plot(mid, nf_1, '.', label=r'Expected $n_f$ with $\varepsilon_1$')
        plt.plot(mid, nf_2, '.', label=r'Expected $n_f$ with $\varepsilon_2$')
        plt.xlabel(r'$z$', fontsize=14)
        plt.ylabel(r'$n_f (z)$', fontsize=14)
        plt.title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
        plt.legend(fontsize=11)
        name=f"maximization_results/poisson_expected_nf/{i}{j}.png"
        plt.savefig(name, format='png')
        '''
        plt.close('all')
#%%

### SAVE DATA (descoment to save  new opt values)

np.savetxt('maximization_results/zmid_1.dat', zmid_1, fmt='%10.3f')
np.savetxt('maximization_results/zmid_2_rounded.dat', zmid_2, fmt='%10.3f')
np.savetxt('maximization_results/zmid_2.dat', zmid_2, fmt='%e')
np.savetxt('maximization_results/a_1.dat', a_1, fmt='%10.3f')
np.savetxt('maximization_results/maxL_1.dat', maxL_1, fmt='%10.3f')
np.savetxt('maximization_results/zeta_1.dat', zeta_1, fmt='%10.3f')
np.savetxt('maximization_results/maxL_2.dat', maxL_2, fmt='%10.3f')
np.savetxt('maximization_results/delta_2.dat', delta_2, fmt='%10.3f')
np.savetxt('maximization_results/gamma_2.dat', gamma_2, fmt='%10.3f') 
np.savetxt('maximization_results/n_points.dat', n_points, fmt='%10.3f')


name = 'maximization_results/all_together.dat'
data = np.column_stack((np.hstack(index_n), np.hstack(n_points),np.hstack(zmid_1), np.hstack(zmid_2), np.hstack(maxL_1), np.hstack(maxL_2), np.hstack(a_1), np.hstack(zeta_1), np.hstack(delta_2), np.hstack(gamma_2)))
header = "mass_bin, # detections, zmid_1, zmid_2, maxL_1, maxL_2, a_1, zeta_1, delta_2, gamma_2"
np.savetxt(name, data, header=header, fmt='%10.3f')


plt.close('all')

#%%

# making a figure with 4 different mass bins

'''
m_bin = np.arange(2,100+7,7)

i,j=13,10

N=30

m1inbin = (m1 >= m1_bin[i]) & (m1 < m1_bin[i+1])
m2inbin = (m2 >= m2_bin[j]) & (m2 < m2_bin[j+1])
mbin = m1inbin & m2inbin & found_any

data = z_origin[mbin]
data_pdf = z_pdf_origin[mbin]

index3 = np.argsort(data)
z = data[index3]
pz = data_pdf[index3]

Total_expected = NTOT*mean_mass_pdf[i,j]
 
zzz_fits = np.linspace(0,max(z), 200)

zagh=np.linspace(0,max(z), 200)
zzzpdf=zpdf_interp_all(zagh)

zmid, a, zeta, lnL = zmid_1[i,j], a_1[i,j], zeta_1[i,j], maxL_1[i,j]
zmid_new, delta, gamma, lnL_new = zmid_2[i,j], delta_2[i,j], gamma_2[i,j], maxL_2[i,j]


data_binned = np.loadtxt(f'z_binned/{i}{j}_data.dat')
#index4=np.argsort(data_binned[:,0])
mid_z=data_binned[:,0]
pz_binned=data_binned[:,1]
zm_detections=data_binned[:,2]
nonzero = zm_detections > 0

z_com_1=np.linspace(0,max(mid_z), 200)


if i==0 and j==0:
    plt.figure(figsize=(11,9))
    plt.subplots_adjust(hspace=0.325, wspace=0.250)

    
    zax1=plt.subplot(221)   
    zax1.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    zax1.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    zax1.set_xlabel(r'$z$', fontsize=14)
    zax1.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax1.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax1.set_ylim(-0.05,1)
    
    a1 = plt.axes([0,0,1,1])
    ip = InsetPosition(zax1, [0.4,0.35,0.58,0.6])
    a1.set_axes_locator(ip)

    a1.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    a1.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    a1.set_yscale('log')
    a1.set_ylim(0,1.5)
    a1.legend(fontsize=11)

    
if i==6 and j==5:
    zax2=plt.subplot(222)   
    zax2.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    zax2.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    zax2.set_xlabel(r'$z$', fontsize=14)
    zax2.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax2.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax2.set_ylim(-0.05,1)
    
    a2 = plt.axes([0,0,1,1])
    ip = InsetPosition(zax2, [0.4,0.35,0.58,0.6])
    a2.set_axes_locator(ip)

    a2.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    a2.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    a2.set_yscale('log')
    a2.set_ylim(0,1.5)
    a2.legend(fontsize=11)
    

    
if i==9 and j==9:
    zax3=plt.subplot(223)   
    zax3.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    zax3.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    zax3.set_xlabel(r'$z$', fontsize=14)
    zax3.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax3.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax3.set_ylim(-0.05,1)
    
    a3 = plt.axes([0,0,1,1])
    ip = InsetPosition(zax3, [0.45,0.35,0.58,0.6])
    a3.set_axes_locator(ip)

    a3.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    a3.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    a3.set_yscale('log')
    a3.set_ylim(0,1.5)
    a3.legend(fontsize=11)
    


if i==13 and j==10:
    zax4=plt.subplot(224)   
    zax4.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    zax4.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    zax4.set_xlabel(r'$z$', fontsize=14)
    zax4.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax4.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax4.set_ylim(-0.05,1)
    
    a4 = plt.axes([0,0,1,1])
    ip = InsetPosition(zax4, [0.65,0.35,0.58,0.6])
    a4.set_axes_locator(ip)

    a4.plot(zzz_fits, epsilon(zzz_fits, zmid, a, zeta), '-', label=r'$\varepsilon_1$') 
    a4.plot(zzz_fits, sigmoid_2(zzz_fits, zmid_new, delta, gamma), '-',  label=r'$\varepsilon_2$')
    a4.set_yscale('log')
    a4.set_ylim(0,1.5)
    a4.legend(fontsize=11)
    
    name="general_plots/fits.png"
    plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
    name="general_plots/fits.pdf"
    plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")



if i==0 and j==0:
    plt.figure(figsize=(11,9))
    plt.subplots_adjust(hspace=0.325, wspace=0.250)

    
    zax1=plt.subplot(221)   
    zax1.plot(mid_z, pz_binned, '.', label='bins over z')
    zax1.errorbar(mid_z[nonzero], pz_binned[nonzero], yerr=pz_binned[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    zax1.plot(z_com_1, epsilon(z_com_1, zmid, a, zeta), '-', label=r'$\varepsilon_1$')
    zax1.plot(z_com_1, sigmoid_2(z_com_1, zmid_new, delta, gamma), '-', label=r'$\varepsilon_2$')
    zax1.set_xlabel(r'$z$', fontsize=14)
    zax1.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax1.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax1.legend(fontsize=14)

    
if i==6 and j==5:
    zax2=plt.subplot(222)   
    zax2.plot(mid_z, pz_binned, '.', label='bins over z')
    zax2.errorbar(mid_z[nonzero], pz_binned[nonzero], yerr=pz_binned[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    zax2.plot(z_com_1, epsilon(z_com_1, zmid, a, zeta), '-', label=r'$\varepsilon_1$')
    zax2.plot(z_com_1, sigmoid_2(z_com_1, zmid_new, delta, gamma), '-', label=r'$\varepsilon_2$')
    zax2.set_xlabel(r'$z$', fontsize=14)
    zax2.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax2.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax2.legend(fontsize=14)
    

    
if i==9 and j==9:
    zax3=plt.subplot(223)   
    zax3.plot(mid_z, pz_binned, '.', label='bins over z')
    zax3.errorbar(mid_z[nonzero], pz_binned[nonzero], yerr=pz_binned[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    
    zax3.plot(z_com_1, epsilon(z_com_1, zmid, a, zeta), '-', label=r'$\varepsilon_1$')
    zax3.plot(z_com_1, sigmoid_2(z_com_1, zmid_new, delta, gamma), '-', label=r'$\varepsilon_2$')
    zax3.set_xlabel(r'$z$', fontsize=14)
    zax3.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax3.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax3.legend(fontsize=14)
    


if i==13 and j==10:
    zax4=plt.subplot(224)   
    zax4.plot(mid_z, pz_binned, '.', label='bins over z')
    zax4.errorbar(mid_z[nonzero], pz_binned[nonzero], yerr=pz_binned[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    
    zax4.plot(z_com_1, epsilon(z_com_1, zmid, a, zeta), '-', label=r'$\varepsilon_1$')
    zax4.plot(z_com_1, sigmoid_2(z_com_1, zmid_new, delta, gamma), '-', label=r'$\varepsilon_2$')
    zax4.set_xlabel(r'$z$', fontsize=14)
    zax4.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax4.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax4.legend(fontsize=14)
    
    name="general_plots/compare1.png"
    plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
    name="general_plots/compare1.pdf"
    plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")



quad_fun_1 = lambda z_int: zpdf_interp_all(z_int)*epsilon(z_int, zmid, a, zeta)
quad_fun_2 = lambda z_int: zpdf_interp_all(z_int)*sigmoid_2(z_int, zmid_new, delta, gamma)


C1=integrate.quad(quad_fun_1, 0, max(all_z))[0]
C2=integrate.quad(quad_fun_2, 0, max(all_z))[0]

edges = np.linspace(0,max(z),30)

if i==0 and j==0:
    plt.figure(figsize=(11,9))
    plt.subplots_adjust(hspace=0.325, wspace=0.250)

    zax1=plt.subplot(221)  
    
    zax1.hist(z, edges ,alpha=0.5, density=True, label='actual detected events')
    zax1.plot(zagh, zzzpdf*epsilon(zagh, zmid, a, zeta)/C1, '-', label=r'$p(z)\cdot\varepsilon_1$')
    zax1.plot(zagh, zzzpdf*sigmoid_2(zagh, zmid_new, delta, gamma)/C2, '-', label=r'$p(z)\cdot\varepsilon_2$')
    zax1.set_xlabel(r'$z$', fontsize=14)
    zax1.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax1.legend(fontsize=11)

    
if i==6 and j==5:
    zax2=plt.subplot(222)  
    
    zax2.hist(z, edges ,alpha=0.5, density=True, label='actual detected events')
    zax2.plot(zagh, zzzpdf*epsilon(zagh, zmid, a, zeta)/C1, '-', label=r'$p(z)\cdot\varepsilon_1$')
    zax2.plot(zagh, zzzpdf*sigmoid_2(zagh, zmid_new, delta, gamma)/C2, '-', label=r'$p(z)\cdot\varepsilon_2$')
    zax2.set_xlabel(r'$z$', fontsize=14)
    #zax1.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax2.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax2.legend(fontsize=11)
    

if i==9 and j==9:
    zax3=plt.subplot(223)
    
    zax3.hist(z, edges ,alpha=0.5, density=True, label='actual detected events')
    zax3.plot(zagh, zzzpdf*epsilon(zagh, zmid, a, zeta)/C1, '-', label=r'$p(z)\cdot\varepsilon_1$')
    zax3.plot(zagh, zzzpdf*sigmoid_2(zagh, zmid_new, delta, gamma)/C2, '-', label=r'$p(z)\cdot\varepsilon_2$')
    zax3.set_xlabel(r'$z$', fontsize=14)
    #zax1.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax3.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax3.legend(fontsize=11)
    

if i==13 and j==10:
    zax4=plt.subplot(224)   
    
    zax4.hist(z, edges ,alpha=0.5, density=True, label='actual detected events')
    zax4.plot(zagh, zzzpdf*epsilon(zagh, zmid, a, zeta)/C1, '-', label=r'$p(z)\cdot\varepsilon_1$')
    zax4.plot(zagh, zzzpdf*sigmoid_2(zagh, zmid_new, delta, gamma)/C2, '-', label=r'$p(z)\cdot\varepsilon_2$')
    zax4.set_xlabel(r'$z$', fontsize=14)
    #zax1.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax4.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax4.legend(fontsize=11)
    
    name="general_plots/compare2.png"
    plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
    name="general_plots/compare2.pdf"
    plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")



z_edges = np.linspace(0,max(z),N)
#z_edges_new=np.insert(z_edges, 0, min(all_z), axis=0)
delta_z = z_edges[2]-z_edges[1]

mid = (z_edges[:-1]+z_edges[1:])/2
center_index = np.array([find_nearest(z, mid[i]) for i in range(len(mid))])

#new_z = np.insert(mid, 0, min(all_z), axis=0)
new_z=mid
new_pdf = zpdf_interp_all(new_z)


exp_tot_1=np.array([Total_expected*integrate.quad(quad_fun_1, z_edges[u], z_edges[u+1])[0] for u in range(len(z_edges)-1)])
exp_tot_2=np.array([Total_expected*integrate.quad(quad_fun_2, z_edges[u], z_edges[u+1])[0] for u in range(len(z_edges)-1)])


landa_1 = Total_expected**new_pdf*epsilon(new_z, zmid, a, zeta)
landa_2 = Total_expected*new_pdf*sigmoid_2(new_z, zmid_new, delta, gamma)

nf_1 = exp_tot_1
nf_2 = exp_tot_2

if i==0 and j==0:
    plt.figure(figsize=(11,9))
    plt.subplots_adjust(hspace=0.325, wspace=0.250)

    zax1=plt.subplot(221)  
    
    n,_,_ = zax1.hist(z, z_edges, alpha=0.5, label=r'Actual $n_f$')
    zax1.errorbar(np.array(mid)[np.where(n>0)[0]], n[n>0], yerr=np.sqrt(n[n>0]), fmt="none", color="k", capsize=1, elinewidth=0.05)
    zax1.plot(new_z, nf_1, '.', label=r'Expected $n_f$ with $\varepsilon_1$')
    zax1.plot(new_z, nf_2, '.', label=r'Expected $n_f$ with $\varepsilon_2$')
    zax1.set_xlabel(r'$z$', fontsize=14)
    zax1.set_ylabel(r'$n_f (z)$', fontsize=14)
    zax1.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax1.legend(fontsize=11)
    #zax1.set_ylim(bottom=0)

    
if i==6 and j==5:
    zax2=plt.subplot(222)  
    
    n,_,_ = zax2.hist(z, z_edges, alpha=0.5, label=r'Actual $n_f$')
    zax2.errorbar(np.array(mid)[np.where(n>0)[0]], n[n>0], yerr=np.sqrt(n[n>0]), fmt="none", color="k", capsize=1, elinewidth=0.05)
    zax2.plot(new_z, nf_1, '.', label=r'Expected $n_f$ with $\varepsilon_1$')
    zax2.plot(new_z, nf_2, '.', label=r'Expected $n_f$ with $\varepsilon_2$')
    zax2.set_xlabel(r'$z$', fontsize=14)
    zax2.set_ylabel(r'$n_f (z)$', fontsize=14)
    zax2.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax2.legend(fontsize=11)
    

if i==9 and j==9:
    zax3=plt.subplot(223)
    
    n,_,_ = zax3.hist(z, z_edges, alpha=0.5, label=r'Actual $n_f$')
    zax3.errorbar(np.array(mid)[np.where(n>0)[0]], n[n>0], yerr=np.sqrt(n[n>0]), fmt="none", color="k", capsize=1, elinewidth=0.05)
    zax3.plot(new_z, nf_1, '.', label=r'Expected $n_f$ with $\varepsilon_1$')
    zax3.plot(new_z, nf_2, '.', label=r'Expected $n_f$ with $\varepsilon_2$')
    zax3.set_xlabel(r'$z$', fontsize=14)
    zax3.set_ylabel(r'$n_f (z)$', fontsize=14)
    zax3.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax3.legend(fontsize=11)
    

if i==13 and j==10:
    zax4=plt.subplot(224)   
    
    n,_,_ = zax4.hist(z, z_edges, alpha=0.5, label=r'Actual $n_f$')
    zax4.errorbar(np.array(mid)[np.where(n>0)[0]], n[n>0], yerr=np.sqrt(n[n>0]), fmt="none", color="k", capsize=1, elinewidth=0.05)
    zax4.plot(new_z, nf_1, '.', label=r'Expected $n_f$ with $\varepsilon_1$')
    zax4.plot(new_z, nf_2, '.', label=r'Expected $n_f$ with $\varepsilon_2$')
    zax4.set_xlabel(r'$z$', fontsize=14)
    zax4.set_ylabel(r'$n_f (z)$', fontsize=14)
    zax4.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m_bin[i], m_bin[i+1], m_bin[j], m_bin[j+1]) )
    zax4.legend(fontsize=11)
    
    name="general_plots/ndetected.png"
    plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
    name="general_plots/ndetected.pdf"
    plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")
'''