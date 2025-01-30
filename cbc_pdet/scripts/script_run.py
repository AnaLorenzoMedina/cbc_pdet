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


# Save the current working directory
original_working_directory = os.getcwd()

# Change the current working directory to the parent directory
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append('../')

# Import the class from the module
from o123_class_found_inj_general import Found_injections




plt.close('all')

run_fit = 'o3'
run_dataset = 'o3'
sources = 'bbh, bns, nsbh, imbh'

# function for dmid and emax we wanna use
#dmid_fun = 'Dmid_mchirp_fdmid'
dmid_fun = 'Dmid_mchirp_fdmid_fspin'
#dmid_fun = 'Dmid_mchirp_expansion_noa30'
#dmid_fun = 'Dmid_mchirp_expansion_exp'
#dmid_fun = 'Dmid_mchirp_expansion_a11'
#dmid_fun = 'Dmid_mchirp_expansion_asqrt'
#dmid_fun = 'Dmid_mchirp_expansion'
#dmid_fun = 'Dmid_mchirp'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin_c21'
emax_fun = 'emax_exp'
#emax_fun = 'emax_sigmoid'

alpha_vary = None

rc('text', usetex=True)

        
#ini_files = [[94], [-0.7435398875859958, 0.2511010365079828, 2.05, -5.632796685161368, 0.025575280643270047, -3.1030795279503675e-05]]
#ini_files = [[82.86080855190852, -3.843054082930893e-06, -0.4726476591881541, 4.304672026161589e-06, 0.0004746871925697556, -0.0014672425957000098], [-0.7435398875859958, 0.2511010365079828, -5.632796685161368, 0.025575280643270047, -3.1030795279503675e-05]]
#ini_files = [[80.19, -6.089e-06, -0.4567, 1.113e-05, 0.0008987, -0.00316, 0.7, 0.001], [-0.737, 0.2493,-5.458, 0.024, -2.903994e-05]]
#ini_files = [[82.018, -7.915e-06, -0.529, -1.673e-06, 0.0008857, -0.00157, 0.167, 0.001997, 1e-6], [-0.96959, 0.3238, -5.8534, 0.02368, -2.79575e-05]]
#ini_files = [[82.018934, -7.91546e-06, -0.529322, -1.673239e-06, 0.000885705916, -0.001573180, 0,0], [-0.9695, 0.32388, -5.8534, 0.02368, -2.7957e-05]]
ini_files_bbh_bns = [[ 8.46541813e+01, -3.65634417e-06, -2.64650682e-01, -2.43506667e-06,
        2.68401892e-04, -5.37960156e-03,  9.28916039e-02,  1.72064631e-03], [-1.02623620e+00,  3.36482981e-01, -4.91342679e+00,  2.27222889e-02,
               -3.61757871e-05]]
                                                                             
ini_files_4_sources = [[8.71047422e+01, -7.24775437e-06, -3.94322637e-01,  4.73292056e-05,
        3.62306966e-04, -1.15754512e-02,  2.04977352e-01,  1.90820585e-03 ], [-8.18965437e-01,  2.77526065e-01, -4.51593709e+00,  1.28444148e-02,
               -8.55554484e-06]]

data = Found_injections(dmid_fun, emax_fun, alpha_vary, ini_files = ini_files_bbh_bns)

if isinstance(sources, str):
    each_source = [source.strip() for source in sources.split(',')] 
    
sources_folder = "_".join(sorted(each_source)) 

path = f'{run_dataset}/{sources_folder}/' + data.path

data.make_folders(run_fit, sources)

#data.load_inj_set(run_dataset, source)
#data.get_opt_params(run_fit)
#81.7746046336622 -8.091776490302833e-06 -0.4444955581608646 9.016393698238174e-06 0.0009175736469746053 -0.003975600149591415 0.16487147248641948 0.0020706799396616876 -2.994036429373754e-07 -340801.1469014259
#-0.9695926056697535 0.3238884595756078 -5.8534140256510625 0.023685941008609972 -2.7957506633390077e-05 -340802.9348490275ye

#%%

data.joint_MLE(run_dataset, run_fit, sources)

#%%
[data.load_inj_set(run_dataset, source) for source in each_source]

data.get_opt_params(run_fit, sources)
data.set_shape_params()

#%%

# npoints = 10000
# index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
# m1 = data.m1[index]
# m2=data.m2[index]

# vsensitive = np.array([data.sensitive_volume(run_dataset, run_fit, m1[i], m2[i]) for i in range(len(m1))])

# plt.figure()
# plt.scatter(m1, m2, s=1, c=vsensitive/1e9, norm=LogNorm())
# plt.xlabel('m1')
# plt.ylabel('m2')
# plt.colorbar(label=r'Sensitive volume [Gpc$^3$]')
# plt.savefig( path + f'/Vsensitive_{npoints}.png')

m1_det = data.m1*(1+data.z)
m2_det = data.m2*(1+data.z)
mtot_det = data.Mtot_det

dmid_values = data.dmid(m1_det, m2_det, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)
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
data.cumulative_dist(run_dataset, run_fit, sources, 'dL')
data.cumulative_dist(run_dataset, run_fit, sources, 'Mtot')
data.cumulative_dist(run_dataset, run_fit, sources, 'Mc')
data.cumulative_dist(run_dataset, run_fit, sources, 'Mtot_det')
data.cumulative_dist(run_dataset, run_fit, sources, 'Mc_det')
data.cumulative_dist(run_dataset, run_fit, sources, 'eta')
data.cumulative_dist(run_dataset, run_fit, sources, 'chi_eff')
#%%
nbins = 5

data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'dL')
data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mtot')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mc')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mtot_det')
data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mc_det')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'eta')



#%%

# npoints = 10000
# index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
# m1 = data.m1[index]
# m2=data.m2[index]

# tot_vsensitive = np.array([data.total_sensitive_volume(run_fit, m1[i], m2[i]) for i in range(len(m1))])

# plt.figure()
# plt.scatter(m1, m2, s=1, c=tot_vsensitive/1e9, norm=LogNorm())
# plt.xlabel('m1')
# plt.ylabel('m2')
# plt.colorbar(label=r'Total sensitive volume [Gpc$^3$]')
# plt.savefig( path + f'/total_Vsensitive_{npoints}.png')
# name = path + f'/total_Vsensitive_{npoints}.pdf'
# plt.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")

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

m1_det_bbh = data.sets['bbh']['m1'] * (1 + data.sets['bbh']['z'])
m2_det_bbh = data.sets['bbh']['m2'] * (1 + data.sets['bbh']['z'])

mtot_det_bbh = data.sets['bbh']['Mtot_det']

chi_eff_bbh = data.sets['bbh']['chi_eff']
dL_bbh = data.sets['bbh']['dL']

#%%

m1_det_nsbh = data.sets['nsbh']['m1'] * (1 + data.sets['nsbh']['z'])
m2_det_nsbh = data.sets['nsbh']['m2'] * (1 + data.sets['nsbh']['z'])

mtot_det_nsbh = data.sets['nsbh']['Mtot_det']

chi_eff_nsbh = data.sets['nsbh']['chi_eff']
dL_nsbh = data.sets['nsbh']['dL']

#%%

m1_det_bns = data.sets['bns']['m1'] * (1 + data.sets['bns']['z'])
m2_det_bns = data.sets['bns']['m2'] * (1 + data.sets['bns']['z'])

mtot_det_bns = data.sets['bns']['Mtot_det']

chi_eff_bns = data.sets['bns']['chi_eff']
dL_bns = data.sets['bns']['dL']

#%%

m1_det_all = np.concatenate([m1_det_bbh, m1_det_bns, m1_det_nsbh])
m2_det_all = np.concatenate([m2_det_bbh, m2_det_bns, m2_det_nsbh])

mtot_det_all = np.concatenate([mtot_det_bbh, mtot_det_bns, mtot_det_nsbh])

chi_eff_all = np.concatenate([chi_eff_bbh, chi_eff_bns, chi_eff_nsbh])
dL_all = np.concatenate([dL_bbh, dL_bns, dL_nsbh])

#%%

dmid_values_all = data.dmid(m1_det_all, m2_det_all, chi_eff_all, data.dmid_params)
#data.apply_dmid_mtotal_max(dmid_values_all, mtot_det_all)
data.set_shape_params()
#%%

dmid_values_bns = data.dmid(m1_det_bns, m2_det_bns, chi_eff_bns, data.dmid_params)
data.set_shape_params()

#%%

plt.figure(figsize=(7,6))
plt.scatter(dL/dmid_values, data.sigmoid(dL,dmid_values, data.emax(m1_det, m2_det, data.emax_params), data.gamma, data.delta), s=1, rasterized=True)
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
im = plt.scatter(dL_all/dmid_values_all, data.sigmoid(dL_all,dmid_values_all, data.emax(m1_det_all, m2_det_all, data.emax_params), data.gamma, data.delta), s=1, c=data.emax(m1_det_all, m2_det_all, data.emax_params), rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 24)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$\varepsilon_\mathrm{max}$', fontsize=24)
plt.show()
plt.savefig( path + '/pdet_o3_emax.png')
name = path + '/pdet_o3_emax.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
order = np.argsort(dmid_values_all)
dmid_values_ordered = dmid_values_all[order]
m1_det_ordered = m1_det_all[order]
m2_det_ordered = m2_det_all[order]


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
plt.savefig( path + '/m1m2det_dmid.png')
name = path + '/m1m2det_dmid.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
data.set_shape_params()

m1det = np.linspace(min(m1_det_all), max(m1_det_all), 200)
m2det = np.linspace(min(m2_det_all), max(m2_det_all), 200)
mtot = m1det + m2det
emax = data.emax(m1det, m2det, data.emax_params)

plt.figure(figsize=(7,4.8))
plt.plot(mtot, emax, '-')
plt.ylim(0, 1.2)
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize=24)
plt.ylabel(r'$\varepsilon_\mathrm{max}$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name = path + '/emax.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")


#%%

data.bootstrap_resampling(100, 'o3', 'o3')

#%%
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000

plt.figure()
plt.loglog(m1_det_all, m2_det_all, 'r*', alpha =0.1, rasterized = True)
plt.loglog(m1_det_bbh, m2_det_bbh, 'g.', alpha =0.1, rasterized = True)
plt.loglog(m1_det_bns, m2_det_bns, 'b.', alpha =0.1, rasterized = True)


#%%


os.chdir(original_working_directory)
