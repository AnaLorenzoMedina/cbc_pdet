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
#sources = 'imbh'
#sources = 'bbh'

# function for dmid and emax we wanna use
#dmid_fun = 'Dmid_mchirp_fdmid'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin_cubic'
dmid_fun = 'Dmid_mchirp_fdmid_fspin_4'
#dmid_fun = 'Dmid_mchirp_expansion_noa30'
#dmid_fun = 'Dmid_mchirp_expansion_exp'
#dmid_fun = 'Dmid_mchirp_expansion_a11'
#dmid_fun = 'Dmid_mchirp_expansion_asqrt'
#dmid_fun = 'Dmid_mchirp_expansion'
#dmid_fun = 'Dmid_mchirp'
#dmid_fun = 'Dmid_mchirp_fdmid_fspin_c21'
#emax_fun = 'emax_exp'
#emax_fun = 'emax_mix'
emax_fun = 'emax_sigmoid'
#emax_fun = 'emax_sigmoid_nolog'

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
                                                                             
ini_files_3_sources = [[8.71047422e+01, -7.24775437e-06, -3.94322637e-01,  4.73292056e-05,
                        3.62306966e-04, -1.15754512e-02,  2.04977352e-01,  1.90820585e-03 ], 
                       [-8.18965437e-01,  2.77526065e-01, -4.51593709e+00,  1.28444148e-02, -8.55554484e-06]]
                                                                              
ini_files_4_sources = [[ 1.15341023e+02, -1.11041171e-05, -1.45113888e+00,  1.11057545e-05,
                         3.28819168e-04, -4.27337660e-03, 0., 0.,  3.39429723e-02,  5.61212292e-03], 
                       [ 5.45981998e-01,  1.44198708e-21, -2.26423282e+00,  7.75663512e-03, -6.75911668e-06]]

ini_files_4_sources = [[ 1.08806961e+02, -7.27382274e-06, -4.15308800e-01, -3.80434134e-07,
                        1.04326772e-03, -1.04753609e-02, -8.95047613e-09,  4.83547088e-08, 0.,
                        1.25960150e-01,  4.41568954e-03],  [5.42613833e-01,  2.71685598e-79, -1.49074349e+00,  1.62358016e-03,
                                1.47645182e-05] ]

ini_files_imbh = [[8.71047422e+01, -7.24775437e-06, -3.94322637e-01,  4.73292056e-05,
                   3.62306966e-04, -1.15754512e-02,  2.04977352e-01,  1.90820585e-03 ], 
                  [-8.18965437e-01,  2.77526065e-01, -4.51593709e+00,  1, 2e-02]]


#ini_files = ini_files_4_sources
data = Found_injections(dmid_fun, emax_fun, alpha_vary, ini_files = ini_files_4_sources)

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
#%%
data.get_opt_params(run_fit, sources)
data.set_shape_params()

#%%

npoints = 10000
index = np.random.choice(np.arange(len(data.sets['bbh']['dL'])), npoints, replace=False)
m1 = data.sets['bbh']['m1'][index]
m2=data.sets['bbh']['m2'][index]

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
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'dL')
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'Mtot')
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'Mc')
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'Mtot_det')
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'Mc_det')
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'eta')
data.cumulative_dist(run_dataset, run_fit, 'imbh', 'chi_eff')
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

found = data.sets['bbh']['found_any']

m1_det_bbh = data.sets['bbh']['m1'] * (1 + data.sets['bbh']['z'])
m2_det_bbh = data.sets['bbh']['m2'] * (1 + data.sets['bbh']['z'])

mtot_det_bbh = data.sets['bbh']['Mtot_det']

chi_eff_bbh = data.sets['bbh']['chi_eff']
dL_bbh = data.sets['bbh']['dL']

#%%

found = data.sets['nsbh']['found_any']

m1_det_nsbh = data.sets['nsbh']['m1'] * (1 + data.sets['nsbh']['z'])
m2_det_nsbh = data.sets['nsbh']['m2'] * (1 + data.sets['nsbh']['z'])

mtot_det_nsbh = data.sets['nsbh']['Mtot_det']

chi_eff_nsbh = data.sets['nsbh']['chi_eff']
dL_nsbh = data.sets['nsbh']['dL']

#%%

found = data.sets['bns']['found_any']

m1_det_bns = data.sets['bns']['m1'] * (1 + data.sets['bns']['z'])
m2_det_bns = data.sets['bns']['m2']* (1 + data.sets['bns']['z'])

mtot_det_bns = data.sets['bns']['Mtot_det']

chi_eff_bns = data.sets['bns']['chi_eff']
dL_bns = data.sets['bns']['dL']

#%%

#%%

found = data.sets['imbh']['found_any']

m1_det_imbh = data.sets['imbh']['m1'] * (1 + data.sets['imbh']['z'])
m2_det_imbh = data.sets['imbh']['m2'] * (1 + data.sets['imbh']['z'])

mtot_det_imbh = data.sets['imbh']['Mtot_det']

chi_eff_imbh = data.sets['imbh']['chi_eff']
dL_imbh = data.sets['imbh']['dL']

#%%

m1_det_all = np.concatenate([m1_det_bbh, m1_det_bns, m1_det_nsbh, m1_det_imbh])
m2_det_all = np.concatenate([m2_det_bbh, m2_det_bns, m2_det_nsbh, m2_det_imbh])

mtot_det_all = np.concatenate([mtot_det_bbh, mtot_det_bns, mtot_det_nsbh, mtot_det_imbh])

chi_eff_all = np.concatenate([chi_eff_bbh, chi_eff_bns, chi_eff_nsbh, chi_eff_imbh])
dL_all = np.concatenate([dL_bbh, dL_bns, dL_nsbh, dL_imbh])

#%%

dmid_values_all = data.dmid(m1_det_all, m2_det_all, chi_eff_all, data.dmid_params)
#data.apply_dmid_mtotal_max(dmid_values_all, mtot_det_all)
data.set_shape_params()
#%%

dmid_values_bbh = data.dmid(m1_det_bbh, m2_det_bbh, chi_eff_bbh, data.dmid_params)
data.set_shape_params()

#%%
dmid_values_imbh = data.dmid(m1_det_imbh, m2_det_imbh, chi_eff_imbh, data.dmid_params)
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
plt.semilogx()
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 24)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$\varepsilon_\mathrm{max}$', fontsize=24)
plt.show()
plt.savefig( path + '/pdet_o3_emax_logscale.png')
name = path + '/pdet_o3_emax_logscale.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(dL_all/dmid_values_all, data.sigmoid(dL_all,dmid_values_all, data.emax(m1_det_all, m2_det_all, data.emax_params), data.gamma, data.delta), s=1, c=data.emax(m1_det_all, m2_det_all, data.emax_params), rasterized=True)
plt.xlim(-0.05, 6)
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
im = plt.scatter(m1_det_ordered, m2_det_ordered, s=1, c=dmid_values_ordered, norm = LogNorm(), rasterized=True)
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
#dmid_params = [45, -2e-06, -3.94e-01,  -1.6e-05,
#        -8.2e-05, -1.15754512e-02,  2.04977352e-01,  1.90820585e-03 ]

#shape_params = [0.4,  2.77526065e-04, -1,  0.8e-03,
#      -5.8e-06]

dmid_params = [ 43, -7.993179639e-06, -4e-01, -3.10825257e-06,
        8.89180835e-04, -1.57344205e-03,  1.67257226e-01,  2.00068413e-03]

shape_params =[-9.85e-01,  3.25e-01, -6.5e+00,  2.368e-02,
       -2.9e-05]

#%%

emax_params = data.emax_params
gamma = data.gamma
delta = data.delta

variables = ['dL', 'z', 'Mc', 'Mtot', 'eta', 'Mc_det', 'Mtot_det', 'chi_eff']
names_plotting = {'dL': '$d_L$', 'z': '$z$', 'Mc': '$\mathcal{M}$', 'Mtot': '$M$', 'eta': '$\eta$', 'Mc_det': '$\mathcal{M}_z$', 'Mtot_det': '$M_z$', 'chi_eff': '$\chi_\mathrm{eff}$'}
emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds'}


for j in variables:
    
    #var_bbh = self.sets['bbh'][f'{j}']
    #var_bns = self.sets['bns'][f'{j}']
    var = data.sets['imbh'][f'{j}']
    #var = np.concatenate([var_bbh, var_bns])
    indexo = np.argsort([var])
    varo = var[indexo]
    '''
    #bbh
    dL_bbh = self.sets['bbh']['dL']
    m1_bbh = self.sets['bbh']['m1']
    m2_bbh = self.sets['bbh']['m2']
    z_bbh = self.sets['bbh']['z']
    chieff_bbh = self.sets['bbh']['chi_eff']
    
    #bns
    dL_bns = self.sets['bns']['dL']
    m1_bns = self.sets['bns']['m1']
    m2_bns = self.sets['bns']['m2']
    z_bns = self.sets['bns']['z']
    chieff_bns = self.sets['bns']['chi_eff']
    '''
    #imbh
    dLo = data.sets['imbh']['dL'][indexo]
    m1o = data.sets['imbh']['m1'][indexo]
    m2o = data.sets['imbh']['m2'][indexo]
    zo = data.sets['imbh']['z'][indexo]
    chieffo = data.sets['imbh']['chi_eff'][indexo]
    '''
    #together
    dLo = np.concatenate([dL_bbh, dL_bns])[indexo]
    m1o = np.concatenate([m1_bbh, m1_bns])[indexo]
    m2o = np.concatenate([m2_bbh, m2_bns])[indexo]
    zo = np.concatenate([z_bbh, z_bns])[indexo]
    chieffo = np.concatenate([chieff_bbh, chieff_bns])[indexo]
    '''
    m1o_det = m1o * (1 + zo) 
    m2o_det = m2o * (1 + zo)
    mtoto_det = m1o_det + m2o_det
    
    dmid_values = data.dmid(m1o_det, m2o_det, chieffo, data.dmid_params)
    
    emax = data.emax(m1o_det, m2o_det, emax_params)
    
    cmd = np.cumsum(data.sigmoid(dLo, dmid_values, emax, gamma, delta))
    
    #found injections
    #var_found_bbh = var_bbh[self.sets['bbh']['found_any']]
    #var_found_bns = var_bns[self.sets['bns']['found_any']]
    #var_found = np.concatenate([var_found_bbh, var_found_bns])
    var_found = var[data.sets['imbh']['found_any']]
    indexo_found = np.argsort(var_found)
    var_foundo = var_found[indexo_found]
    real_found_inj = np.arange(len(var_foundo))+1

    plt.figure()
    plt.scatter(varo, cmd, s=1, label='model', rasterized=True)
    plt.scatter(var_foundo, real_found_inj, s=1, label='found injections', rasterized=True)
    plt.xlabel(names_plotting[j], fontsize = 24)
    plt.ylabel('Cumulative found injections', fontsize = 24)
    plt.legend(loc='best', fontsize = 20, markerscale=3.)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    name = f'o3/imbh/Dmid_mchirp_fdmid_fspin/{emax_dic[data.emax_fun]}/{j}_cumulative.pdf'
    plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")
    #plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")
    plt.close()

#%%
var = mtot_det_imbh
#var = np.concatenate([var_bbh, var_bns])
indexo = np.argsort([var])
varo = var[indexo]

dLo = dL_imbh[indexo]
m1o = m1_det_imbh[indexo]
m2o = m2_det_imbh[indexo]
chieffo = chi_eff_imbh[indexo]

dmid_values = data.dmid(m1o, m2o, chieffo, dmid_params)
emax_values = data.emax(m1o, m2o, emax_params)

cmd = np.cumsum(data.sigmoid(dLo, dmid_values, emax_values, gamma, delta))

var_found = var[data.sets['imbh']['found_any']]
indexo_found = np.argsort(var_found)
var_foundo = var_found[indexo_found]
real_found_inj = np.arange(len(var_foundo))+1

plt.figure()
plt.scatter(varo, cmd, s=1, label='model', rasterized=True)
plt.scatter(var_foundo, real_found_inj, s=1, label='found injections', rasterized=True)
plt.xlabel('mtot_det', fontsize = 24)
plt.ylabel('Cumulative found injections', fontsize = 24)
plt.legend(loc='best', fontsize = 20, markerscale=3.)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name ='o3/imbh/mtot_det.png'
plt.savefig(name, format='png', bbox_inches="tight")

#%%

m1_imbh = data.sets['imbh']['m1']
pdet = data.sigmoid(dL_imbh,dmid_values_imbh, data.emax(m1_det_imbh, m2_det_imbh, data.emax_params), data.gamma, data.delta)

plt.figure()
im = plt.scatter(m1_det_imbh, pdet, c = dL_imbh)
plt.xlabel('m1_det')
plt.ylabel('pdet')
cbar = plt.colorbar(im)
cbar.set_label('dL')
name ='o3/imbh/pdet_vs_m1det_dL.png'
plt.savefig(name, format='png', bbox_inches="tight")

#%%
m1det = np.linspace(min(m1_det_imbh), max(m1_det_imbh), 200)
m2det = np.linspace(min(m2_det_imbh), max(m2_det_imbh), 200)
mtot = m1det + m2det
emax = data.emax(m1det, m2det, data.emax_params)
emax = data.emax(m1_det_imbh, m2_det_imbh, data.emax_params)



plt.figure()
plt.plot(mtot_det_imbh, emax, '.')
plt.xlabel('mtot_det')
plt.ylabel('emax')
name ='o3/imbh/emax.png'
plt.savefig(name, format='png', bbox_inches="tight")

#%%
dmid_p = [ 1.10110113e2 ,-1.23841432e-5 ,-5.30370626e-1 ,-5.88862138e-7,
  9.28161233e-04 ,-7.13790302e-03 ,-4.91418470e-09 , 2.51905117e-08,
  9.91407583e-14,  1.27955312e-01,  4.30731861e-03]
emax_p = [-1.57677634,  1.15051936e-01, -6.66585524e-04]


gamma =  4.23858382e-01
delta = 0.

dmid = data.dmid(m1_det_all, m2_det_all, chi_eff_all, dmid_p)
emax = data.emax(m1_det_all, m2_det_all, emax_p)

pdet = data.sigmoid(dL_all, dmid, emax, gamma, delta)

plt.plot(dL_all/dmid, pdet, '.')
plt.xlim(0, 6)

#%%
os.chdir(original_working_directory)
