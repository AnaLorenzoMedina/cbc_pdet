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
from o123_class_found_inj_general import Found_injections

plt.close('all')

run_fit = 'o3'
run_dataset = 'o3'

# function for dmid and emax we wanna use
#dmid_fun = 'Dmid_mchirp_expansion_noa30'
#dmid_fun = 'Dmid_mchirp_expansion_exp'
#dmid_fun = 'Dmid_mchirp_expansion_a11'
#dmid_fun = 'Dmid_mchirp_expansion_asqrt'
#dmid_fun = 'Dmid_mchirp_expansion'
dmid_fun = 'Dmid_mchirp'
emax_fun = 'emax_exp'
#emax_fun = 'emax_sigmoid'

alpha_vary = True

# path = f'{run_dataset}/{dmid_fun}' if alpha_vary is None else f'{run_dataset}/alpha_vary/{dmid_fun}'
# if emax_fun is not None:
#     path = path + f'/{emax_fun}'

# try:
#     os.mkdir(f'{run_dataset}')
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise
        
# if alpha_vary is not None:
#     try:
#         os.mkdir(f'{run_dataset}/alpha_vary')
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
        
# try:
#     os.mkdir(path)
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise
        
# try:
#     os.mkdir(path + f'/{emax_fun}')
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise
        
ini_files = [[94], [-0.7435398875859958, 0.2511010365079828, 2.05, -5.632796685161368, 0.025575280643270047, -3.1030795279503675e-05]]
data = Found_injections(dmid_fun, emax_fun, alpha_vary, ini_files)

data.make_folders(run_fit)

data.load_inj_set(run_dataset)
#data.joint_MLE(run_dataset, run_fit)

#data.load_inj_set(run_dataset)
#data.get_opt_params(run_fit)

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

# m1_det = data.m1*(1+data.z)
# m2_det = data.m2*(1+data.z)

# total_pdet = data.total_pdet(data.dL, m1_det, m2_det)

# plt.figure()
# plt.scatter(data.dL/data.dmid(data, m1_det, m2_det, data.dmid_params), total_pdet, s=1)
# plt.xlabel('dL/dmid')
# plt.ylabel('total Pdet')
#plt.savefig( path + '/total_pdet.png')

#data.load_inj_set('o2')

#plt.figure()
#plt.plot(data.dL, data.chi_eff, '.')
#plt.plot(data.dL, data.chieff_d, '.', alpha=0.1)
#plt.plot(data.dL, data.chi_eff - data.chieff_d, '.')

nbins = 5

data.binned_cumulative_dist(run_dataset, run_fit, nbins,'Mtot_det', 'dL')
data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mtot')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mc')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mtot_det')
data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mc_det')
data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'eta')

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
