#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:59:47 2025

@author: ana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:11:32 2023

@author: ana
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from matplotlib import rc
import sys
import os
import matplotlib.ticker as ticker


# # Save the current working directory
#original_working_directory = os.getcwd()

# # Change the current working directory to the parent directory
#os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#sys.path.append('../')

# Import the class from the module
from cbc_pdet.gwtc_found_inj import Found_injections




plt.close('all')

run_fit = 'o3'
run_dataset = 'o3'

# function for dmid and emax we wanna use
#dmid_fun = 'Dmid_mchirp_fdmid_fspin_cubic'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin_4'
#dmid_fun = 'Dmid_mchirp_fdmid'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin'
#dmid_fun = 'Dmid_mchirp_expansion_noa30'
#dmid_fun = 'Dmid_mchirp_expansion_exp'
#dmid_fun = 'Dmid_mchirp_expansion_a11'
#dmid_fun = 'Dmid_mchirp_expansion_asqrt'
#dmid_fun = 'Dmid_mchirp_expansion'
#dmid_fun = 'Dmid_mchirp'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin_c21'
#emax_fun = 'emax_exp'
emax_fun = 'emax_gaussian'
#emax_fun = 'emax_sigmoid'
#dmid_fun = 'Dmid_mchirp_mixture'
#dmid_fun = 'Dmid_mchirp_mixture_logspin'
dmid_fun = 'Dmid_mchirp_mixture_logspin_corr'

alpha_vary = None

rc('text', usetex=True)

        
#ini_files = [[94], [-0.7435398875859958, 0.2511010365079828, 2.05, -5.632796685161368, 0.025575280643270047, -3.1030795279503675e-05]]
#ini_files = [[82.86080855190852, -3.843054082930893e-06, -0.4726476591881541, 4.304672026161589e-06, 0.0004746871925697556, -0.0014672425957000098], [-0.7435398875859958, 0.2511010365079828, -5.632796685161368, 0.025575280643270047, -3.1030795279503675e-05]]
#ini_files = [[80.19, -6.089e-06, -0.4567, 1.113e-05, 0.0008987, -0.00316, 0.7, 0.001], [-0.737, 0.2493,-5.458, 0.024, -2.903994e-05]]
#ini_files = [[82.018, -7.915e-06, -0.529, -1.673e-06, 0.0008857, -0.00157, 0.167, 0.001997, 1e-6], [-0.96959, 0.3238, -5.8534, 0.02368, -2.79575e-05]]
#ini_files = [[82.018934, -7.91546e-06, -0.529322, -1.673239e-06, 0.000885705916, -0.001573180, 0,0], [-0.9695, 0.32388, -5.8534, 0.02368, -2.7957e-05]]
#ini_files_cubic = [[ 97.2546, -2.653e-05, -0.30749, 4.0609e-05, 0.004749, -0.016725, 9.67741e-09, -1.5317e-08, 0.1336, 0.00442], 
#                       [ 0.4148, 1.3314e-07, -3.1692, 0.51967, -1.18356e-06]]

#ini_files_4 = [[ 1.3e+02, -7.27382274e-06, -4.15308800e-01, -3.80434134e-07,
#                        1.04326772e-03, -1.04753609e-02, -8.95047613e-09,  4.83547088e-08, 0.,
#                        1.25960150e-01,  4.41568954e-03],  [5.42613833e-01,  2.71685598e-79, -1.49074349e+00,  1.62358016e-03,
#                                1.47645182e-05] ]
                                                            

#ini_files = [[1.438e2,  4.292e-03 , 1.541 , 1.433e2,  7.32e-1,  -5.86e-1,  -3.699e-3,  3.74e-6], [0.804, 4.9203e-14, 0.3, 0.5, 150, 1]]
#ini_files_spin = [[98.832, 0.0023, 0.7082, 200.722, 1.058, -0.5713, -0.00336, 2.9764e-07, 0.1548, 0.00195, 0 ], [-1.37, 0.381, 0.20327, 0.9999, 2418.1926, 1.393]]
#ini_files_spin_log = [[99.70, 0.00237, 0.696, 202.94, 1.052,-0.567, -0.00348, 8.332e-07, -0.1235, 0.00108, 0.086, 0 ], [-1.378, 0.3825, 0.2025, 0.999, 2491.30475, 1.412 ]]

ini_files_4_sources = [[ 83.016, 0.00314, 0.74181, 218.231, 0.87566, -0.5712, -0.003573, 1.3588e-06, -0.0922916, 0.00121, 0.0766, -0.09137],  [-1.3639, 0.378019,0.18802, 0.99999, 3239.7326, 1.604237] ]


data = Found_injections(dmid_fun, emax_fun, alpha_vary, ini_files_4_sources)
path = f'{run_dataset}/' + data.path

data.make_folders(run_dataset)

data.load_inj_set(run_dataset) 
#data.get_opt_params(run_fit)
#81.7746046336622 -8.091776490302833e-06 -0.4444955581608646 9.016393698238174e-06 0.0009175736469746053 -0.003975600149591415 0.16487147248641948 0.0020706799396616876 -2.994036429373754e-07 -340801.1469014259
#-0.9695926056697535 0.3238884595756078 -5.8534140256510625 0.023685941008609972 -2.7957506633390077e-05 -340802.9348490275ye

#%%

data.joint_MLE(run_dataset)
 
#%%

#data.load_inj_set(run_dataset)
data.get_opt_params(run_fit)
data.set_shape_params()

#%%

dL = data.dL
Mc = data.Mc
Mtot = data.Mtot
eta = data.eta
found = data.found_any
z = data.z
Mc_det = data.Mc_det
Mtot_det = data.Mtot_det

plt.figure()
plt.hist(dL, bins=np.logspace(np.log10(min(dL)),np.log10(max(dL)), 100), log=True, alpha = 0.5)
plt.hist(dL[found], bins=np.logspace(np.log10(min(dL)),np.log10(max(dL)), 100), log=True, alpha = 0.5)
plt.semilogx()
plt.xlabel('dL')
plt.savefig(f'{run_dataset}/diagnostics/dL.png')

plt.figure()
plt.hist(Mc_det, alpha = 0.5, bins=np.logspace(np.log10(min(Mc_det)),np.log10(max(Mc_det)), 100), log=True)
plt.hist(Mc_det[found], alpha = 0.5, bins=np.logspace(np.log10(min(Mc_det)),np.log10(max(Mc_det)), 100), log=True)
plt.semilogx()
plt.xlabel('Mc_det')
plt.savefig(f'{run_dataset}/diagnostics/Mc_det.png')

plt.figure()
plt.hist(eta, bins = 50, alpha=0.5, log=True)
plt.hist(eta[found], bins = 50, alpha=0.5, log=True)
plt.xlabel('eta')
plt.savefig(f'{run_dataset}/diagnostics/eta.png')

plt.figure()
plt.scatter(dL, Mtot_det, s=1)
plt.scatter(dL[found], Mtot_det[found], s=1)
plt.loglog()
plt.xlabel('dL')
plt.ylabel('Mtot_det')
plt.savefig(f'{run_dataset}/diagnostics/Mtotdet_vs_dL.png')

m1 = data.m1
m2 = data.m2

plt.figure()
plt.scatter(m1, m2, s=1)
plt.scatter(m1[found], m2[found], s=1)
plt.loglog()
plt.xlabel('m1')
plt.ylabel('m2')
plt.savefig(f'{run_dataset}/diagnostics/m1_vs_m2.png')

np.logspace(1,1000, 0.2)

#%%

npoints = 100
index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
m1 = data.m1[index]
m2=data.m2[index]

vsensitive = np.array([data.sensitive_volume(run_fit, m1[i], m2[i]) for i in range(len(m1))])

plt.figure()
plt.scatter(m1, m2, s=1, c=vsensitive/1e9, norm=LogNorm())
plt.xlabel('m1')
plt.ylabel('m2')
plt.colorbar(label=r'Sensitive volume [Gpc$^3$]')
plt.savefig( path + f'/Vsensitive_{npoints}.png')

#%%

m1_det = data.m1*(1+data.z)
m2_det = data.m2*(1+data.z)
mtot_det = data.Mtot_det

dmid_values = data.dmid(m1_det, m2_det, data.dmid_params)
data.set_shape_params()
'''

pdet = data.run_pdet(data.dL, m1_det, m2_det, 'o3')

plt.figure()
plt.scatter(data.dL/data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params), pdet, s=1)
#plt.scatter(data.dL/dmid_values, pdet, s=1)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 15)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 15)
plt.savefig( path + '/pdet_o3.png')
name = path + '/pdet_o3.pdf'
plt.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")


#plt.figure()
#plt.plot(data.dL, data.chi_eff, '.')
#plt.plot(data.dL, data.chieff_d, '.', alpha=0.1)
#plt.plot(data.dL, data.chi_eff - data.chieff_d, '.')
'''

#%%
data.cumulative_dist(run_dataset, run_fit,'dL', ks = False)
data.cumulative_dist(run_dataset, run_fit,'Mtot', ks = False)
data.cumulative_dist(run_dataset, run_fit,'Mc', ks = False)
data.cumulative_dist(run_dataset, run_fit,'Mtot_det', ks = False)
data.cumulative_dist(run_dataset, run_fit,'Mc_det', ks = False)
data.cumulative_dist(run_dataset, run_fit,'eta', ks = False)
data.cumulative_dist(run_dataset, run_fit,'chi_eff', ks = False)
#%%
nbins = 10

data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'dL')
data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mtot')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mc')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mtot_det')
data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mc_det')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'eta')



#%%


npoints = 100
index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
m1 = data.m1[index]
m2=data.m2[index]

tot_vsensitive = np.array([data.total_sensitive_volume(m1[i], m2[i]) for i in range(len(m1))])

plt.figure()
plt.scatter(m1, m2, s=1, c=tot_vsensitive/1e9, norm=LogNorm())
plt.xlabel('m1')
plt.ylabel('m2')
plt.colorbar(label=r'Total sensitive volume [Gpc$^3$]')
plt.savefig( path + f'/total_Vsensitive_{npoints}.png')
name = path + f'/total_Vsensitive_{npoints}.pdf'
plt.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")

#%%

# cmds_bins = ['dL' , 'Mtot', 'Mc', 'Mtot_det', 'Mc_det', 'eta']
# chi_eff_opt_params = {}
# mid_values = {}

# for i in cmds_bins:
#     chi_eff_opt_params[f'{i}_bins'] = np.loadtxt(f'o3/Dmid_mchirp_expansion_noa30/emax_exp_cmds/{i}_bins/chi_eff_cmd/chi_eff_opt_param')
#     mid_values[f'{i}_bins'] = np.loadtxt(f'o3/Dmid_mchirp_expansion_noa30/emax_exp_cmds/{i}_bins/chi_eff_cmd/mid_values')
            
# for i in cmds_bins:
#     plt.figure()
#     plt.plot(mid_values[f'{i}_bins'], chi_eff_opt_params[f'{i}_bins'], 'o')
#     plt.xlabel(f'{i} (middle bin points)')
#     plt.ylabel('c1')
#     plt.title(r'$\chi_{eff}$ correction $ = exp(\chi_{eff} \, c_1)$')
#     plt.savefig(f'o3/Dmid_mchirp_expansion_noa30/emax_exp_cmds/chi_eff_corr_{i}_bins.png')

'''
m1_det = data.m1*(1+data.z)
m2_det = data.m2*(1+data.z)

order=np.argsort(data.dL)
dL=data.dL[order]
m1=data.m1[order]
m2=data.m2[order]
z=data.z[order]
chieff = data.chi_eff[order]

plt.figure(figsize=(7,6))
plt.scatter(m1*(1+z), m2*(1+z), s=1, c=data.dmid(m1*(1+z), m2*(1+z), chieff, data.dmid_params))
#plt.scatter(m1*(1+z), m2*(1+z), s=1, c=data.dmid(m1*(1+z), m2*(1+z), data.dmid_params))
plt.xlabel('m1_det')
plt.ylabel('m2_det')
plt.colorbar(label='dL_mid')
plt.show()
plt.savefig( path + '/m1m2det_dmid.png')
name = path + '/m1m2det_dmid.pdf'
plt.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")

'''
#%%
#### CHIEFF 
data.get_opt_params('o3')

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)

mtot_det = data.Mtot_det

dmid_values = data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params)

data.set_shape_params()

#%%

plt.figure(figsize=(7,6))
plt.scatter(data.dL/dmid_values, data.sigmoid(data.dL,dmid_values, data.emax(m1_det, m2_det, data.emax_params), data.gamma, data.delta), s=1, rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 18)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 18)
plt.savefig( path + '/pdet_o3.png')
name = path + '/pdet_o3.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.dL/dmid_values, data.sigmoid(data.dL,dmid_values, data.emax(m1_det, m2_det, data.emax_params), data.gamma, data.delta), s=1, c=mtot_det, rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 24)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$M_z$', fontsize=24)
plt.show()
plt.savefig( path + '/pdet_o3_Mtot.png')
name = path + '/pdet_o3_Mtot.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.dL/dmid_values, data.sigmoid(data.dL,dmid_values, data.emax(m1_det, m2_det, data.emax_params), data.gamma, data.delta), s=1, c=data.emax(m1_det, m2_det, data.emax_params), rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 24)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$\varepsilon_\mathrm{max}$', fontsize=24)
#plt.xlim(-0.1, 6)
plt.savefig( path + '/pdet_o3_emax.png')
name = path + '/t_pdet_o3_emax.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight", transparent = True)

#%%
order = np.argsort(dmid_values)
dmid_values_ordered = dmid_values[order]
m1_det_ordered = m1_det[order]
m2_det_ordered = m2_det[order]

#%%
plt.figure(figsize=(7,6))
im = plt.scatter(m1_det_ordered, m2_det_ordered, s=1, c=dmid_values_ordered, norm=LogNorm(), rasterized=True)
plt.loglog()
plt.xlabel(r'$m_{1z} [M_{\odot}]$', fontsize=24)
plt.ylabel('$m_{2z} [M_{\odot}]$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$d_\mathrm{mid}$', fontsize=24)
plt.savefig( path + '/logm1m2det_dmid.png')
name = path + '/t_m1m2det_dmid.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight", transparent = True)

#%%
#WHITE
plt.figure(figsize=(7,6))
im = plt.scatter(m1_det_ordered, m2_det_ordered, s=1, c=dmid_values_ordered, norm=LogNorm(), rasterized=True)
plt.loglog()
plt.xlabel(r'$m_{1z} [M_{\odot}]$', fontsize=24, color = 'white')
plt.ylabel('$m_{2z} [M_{\odot}]$', fontsize=24, color = 'white')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15, colors='white')
cbar.set_label(r'$d_\mathrm{mid}$', fontsize=24, color = 'white')
plt.tick_params(axis='x', colors='white')  # x-axis ticks and labels
plt.tick_params(axis='y', colors='white')
plt.savefig( path + '/logm1m2det_dmid.png')
name = path + '/twhite_m1m2det_dmid.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight", transparent = True)

#%%
data.set_shape_params()

m1det = np.linspace(min(m1_det), max(m1_det), 200)
m2det = np.linspace(min(m2_det), max(m2_det), 200)
mtot = m1det + m2det
emax = data.emax(m1det, m2det, data.emax_params)
#y = (1 - np.exp(params_emax[0])) / (1 + np.exp(params_emax[1]*(x-params_emax[2])))
plt.figure(figsize=(7,4.8))
plt.plot(mtot, emax, '-')
plt.semilogx()
#plt.ylim(0, 2)
#plt.semilogx()
#plt.grid()
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize=24)
plt.ylabel(r'$\varepsilon_{max}$', fontsize=24)
plt.savefig( path + '/emax.png')
name = path + '/t_emax.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight", transparent = True)


#%%

data.bootstrap_resampling(100, 'o3', 'o3')


#os.chdir(original_working_directory)







