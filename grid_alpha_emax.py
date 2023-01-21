# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:21:23 2023

@author: Ana
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy import integrate
from matplotlib import rc
import os
import errno

def sigmoid_2(z, zmid, alpha, emax, gamma=-0.32640, delta=2.751057e-01):
    denom = 1. + (z/zmid) ** alpha * \
        np.exp(gamma * ((z / zmid) - 1.) + delta * ((z**2 / zmid**2) - 1.))
    return emax / denom


def integrand_2(z_int, zmid, alpha, emax, gamma=-0.32640, delta=2.751057e-01):
    return zpdf_interp(z_int) * sigmoid_2(z_int, zmid, alpha, emax, gamma, delta)


def lam_2(z, pz, zmid, alpha, emax, gamma=-0.32640, delta=2.751057e-01):
    return pz * sigmoid_2(z, zmid, alpha, emax, gamma, delta)


# in_param is a generic minimization variable, here ln(zmid)
def logL_quad_2(z, pz, Total_expected, zmid, alpha, emax, gamma=-0.32640, delta=2.751057e-01):
    
    quad_fun = lambda z_int: Total_expected * integrand_2(z_int, zmid, alpha, emax, gamma, delta)
    # it's hard to check here what is the value of new_try_z
    Lambda_2 = integrate.quad(quad_fun, min(new_try_z), max(new_try_z))[0]
    lnL = -Lambda_2 + np.sum(np.log(Total_expected * lam_2(z, pz, zmid, alpha, emax, gamma, delta)))
    return lnL

try:
    os.mkdir('emax_apha_logL')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.mkdir('emax_apha_logL/plots_2D_hist')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('emax_apha_logL/tot_logL')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

        
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

k=3 #last iteration of the joint fit, to get the optimized values of zmid
zmid = np.loadtxt(f'joint_fit_results/zmid/zmid_{k}.dat')

emax_old = 0.967
alpha_old = 2.05

emax = np.linspace(0.5,1.5,11)*emax_old
alpha = np.linspace(0.5,1.5,11)*alpha_old

max_index_lnL = np.zeros([nbin1, nbin2])
max_lnL = np.zeros([nbin1, nbin2])

indexes_1 = np.zeros([nbin1, nbin2], dtype='int')
indexes_2 = np.zeros([nbin1, nbin2], dtype='int')

best_emax = np.zeros([nbin1, nbin2])
best_alpha = np.zeros([nbin1, nbin2])

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
     
        # index_sorted = np.argsort(data)
        # z = data[index_sorted]
        # pz = data_pdf[index_sorted]
        # Total_expected = NTOT * mean_mass_pdf[i,j]
        
        # tot_lnL = np.array([[logL_quad_2(z, pz, Total_expected, zmid[i,j], alpha[u], emax[w]).sum() for w in range(len(emax))] for u in range(len(alpha))])
        
        # color = 'viridis'

        # plt.figure()
        # plt.imshow(tot_lnL, cmap=color, origin='lower', norm=LogNorm())
        # plt.colorbar()
        # plt.xlabel(r'$\alpha$', fontsize=15)
        # plt.ylabel(r'$\epsilon_{max}$', fontsize=15)
        # plt.title(r'tot_lnL $m_1$ %.1f-%.1f M$_{\odot}$ \& $m_2$ %.1f-%.1f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]), fontsize=15)

        # name=f"emax_apha_logL/plots_2D_hist/{i}{j}.png"
        # plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
        
        # plt.close()
        
        # np.savetxt(f'emax_apha_logL/tot_logL/{i}{j}.dat', tot_lnL, fmt='%10.5f')
        
        tot_lnL = np.loadtxt(f'emax_apha_logL/tot_logL/{i}{j}.dat')
        
        ind = np.unravel_index(np.argmax(tot_lnL), tot_lnL.shape)
        s = '%i.%i' %(ind[0], ind[1])
        max_index_lnL[i,j] = s
        max_lnL[i,j] = np.max(tot_lnL)
        
        indexes_1[i,j] = ind[0]
        indexes_2[i,j] = ind[1]
        
        best_emax[i,j] = emax[ind[0]]
        best_alpha[i,j] = alpha[ind[1]]
        
#total log likelihood over the mass bins

l = []

for i in range(0,nbin1):
    for j in range(0,nbin2):
        try:
            l.append(np.loadtxt(f'emax_apha_logL/tot_logL/{i}{j}.dat'))
        except OSError:
            continue
        
tot_L_allbin = sum(l)
np.savetxt('emax_apha_logL/max_tot_lnL.dat', tot_L_allbin, fmt='%-19s')

max_tot_index = np.unravel_index(np.argmax(tot_L_allbin), tot_L_allbin.shape)
best_tot_emax = emax[max_tot_index[0]]
best_tot_alpha = alpha[max_tot_index[1]]

print('Best overall values are alpha=%f and emax=%f' %(best_tot_alpha, best_tot_emax))
np.savetxt('emax_apha_logL/best_values.dat', np.array([best_tot_alpha, best_tot_emax]), header='alpha, emax', fmt='%s')

# np.savetxt('emax_apha_logL/max_index_lnL.dat', max_index_lnL, fmt='%-5s')
# np.savetxt('emax_apha_logL/max_lnL.dat', max_lnL, fmt='%-14s')

# np.savetxt('emax_apha_logL/best_emax.dat', np.round(best_emax, 4), fmt='%-7s')
# np.savetxt('emax_apha_logL/best_alpha.dat', np.round(best_alpha, 4), fmt='%-7s')

# most_rep_1 = np.argmax(np.bincount(indexes_1.flatten())[1:]) + 1  #the zero values dont count
# most_rep_2 = np.argmax(np.bincount(indexes_2.flatten())[1:]) + 1
# print('Best overall values are alpha=%f and emax=%f' %(alpha[most_rep_2], emax[most_rep_1]))
# np.savetxt('emax_apha_logL/best_values.dat', np.array([alpha[most_rep_2], emax[most_rep_1]]), header='alpha, emax', fmt='%s')
