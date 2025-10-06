#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:28:40 2023

@author: ana
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import rc
#import corner
import sys
import os
import matplotlib.ticker as ticker

# # Save the current working directory
# original_working_directory = os.getcwd()

# # Change the current working directory to the parent directory
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# sys.path.append('../')

# Import the class from the module
from cbc_pdet.gwtc_found_inj import Found_injections

plt.close('all')

run_fit = 'o3'
run_dataset = 'o3'
sources = 'bbh, bns, nsbh'


#dmid_fun = 'Dmid_mchirp_fdmid'
dmid_fun = 'Dmid_mchirp_mixture_logspin_corr'
emax_fun = 'emax_gaussian'
alpha_vary = None

data = Found_injections(dmid_fun, emax_fun, alpha_vary)

if isinstance(sources, str):
    each_source = [source.strip() for source in sources.split(',')] 
    
sources_folder = "_".join(sorted(each_source)) 

path = f'{run_dataset}/{sources_folder}/' + data.path

data.make_folders(run_fit, sources)

data.load_all_inj_sets(run_dataset, sources)
data.get_opt_params(run_fit, sources)
data.set_shape_params()

rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

#%%
nbin1 = 14
nbin2 = 14

m1_bin = np.round(np.logspace(np.log10(data.mmin), np.log10(data.mmax), nbin1+1), 1)
m2_bin = np.round(np.logspace(np.log10(data.mmin), np.log10(data.mmax), nbin2+1), 1)

mid_1 = (m1_bin[:-1]+ m1_bin[1:])/2
mid_2 = (m2_bin[:-1]+ m2_bin[1:])/2

Mc = np.array ([[(mid_1[i] * mid_2[j])**(3/5) / (mid_1[i] + mid_2[j])**(1/5) for j in range(len(mid_2))] for i in range(len(mid_1))] )
Mtot = np.array ([[(mid_1[i] + mid_2[j]) for j in range(len(mid_2))] for i in range(len(mid_1))] )
q = np.array ([[(mid_2[j] / mid_1[i]) for j in range(len(mid_2))] for i in range(len(mid_1))] )
mu = np.array ([[(mid_1[i] * mid_2[j]) / (mid_1[i] + mid_2[j]) for j in range(len(mid_2))] for i in range(len(mid_1))] ) 


#%% 
#DMID PLOTS

k = 145
dmid = np.loadtxt(f'binned_analysis/dL_joint_fit_results_emax_vary/dLmid/dLmid_{k}.dat')
toplot = np.nonzero(dmid)


Mc_plot = Mc[toplot]
dmid_plot = dmid[toplot]
Mtot_plot = Mtot[toplot]
q_plot = q[toplot]
mu_plot = mu[toplot]
eta_plot = mu_plot / Mtot_plot

rc('text', usetex=True)

plt.close('all')

plt.figure()
plt.scatter(Mtot_plot, dmid_plot/(Mc_plot)**(5/6), c=eta_plot)
plt.colorbar(label=r'$\eta$')
plt.xlabel(r'$M_z$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, \mathcal{M}^{5/6}$')
plt.grid(True, which='both')
name="binned_analysis/dL_joint_fit_results_emax_vary/Mtot.png"
plt.savefig(name, format='png', dpi=300)
name="binned_analysis/dL_joint_fit_results_emax_vary/Mtot.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")


#plt.plot(Mc_plot.flatten(), 70*(Mc_plot.flatten())**(5/6), 'r-', label='cte = 70')
#plt.legend()


plt.figure()
plt.loglog(q_plot, dmid_plot/(Mc_plot)**(5/6), '.')
plt.xlabel(r'$q = m2_ / m_1$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="binned_analysis/dL_joint_fit_results_emax_vary/q.png"
plt.savefig(name, format='png', dpi=300)
name="binned_analysis/dL_joint_fit_results_emax_vary/q.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

plt.figure()
plt.scatter(eta_plot, dmid_plot/(Mc_plot)**(5/6), c=Mtot_plot)
plt.colorbar(label=r'$M_z$')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="binned_analysis/dL_joint_fit_results_emax_vary/eta.png"
plt.savefig(name, format='png', dpi=300)
name="binned_analysis/dL_joint_fit_results_emax_vary/eta.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), gridspec_kw=dict(top=1, left=-0.3, right=1.2, bottom=0))
ax.scatter(Mtot_plot, eta_plot, dmid_plot/(Mc_plot)**(5/6), c=dmid_plot/(Mc_plot)**(5/6), cmap='viridis', alpha=0.8)
ax.set_xlabel(r'$M_z$', fontsize = 15)
ax.set_ylabel(r'$\eta$', fontsize = 15)
ax.set_zlabel(r'$d_\mathrm{mid} \, / \, \mathcal{M}^{5/6}$', fontsize = 15)
fig.subplots_adjust(right=1.6, top=0.9, bottom=0.1, left=-0.6)
name="binned_analysis/dL_joint_fit_results_emax_vary/3D_plot.png"
plt.savefig(name, format='png', dpi=300)
name="binned_analysis/dL_joint_fit_results_emax_vary/3D_plot.pdf"
plt.savefig(name, format='pdf', dpi=300)
#%%
fig = plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_trisurf(Mtot_plot, eta_plot, dmid_plot/(Mc_plot)**(5/6), cmap='viridis')
ax.set_xlabel('Mtotal ')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
name="binned_analysis/dL_joint_fit_results_emax_vary/3D_surface_plot.png"
plt.savefig(name, format='png', dpi=300)

#%%
#EMAX PLOTS 

k = 145
emax = np.loadtxt(f'binned_analysis/dL_joint_fit_results_emax_vary/emax/emax_{k}.dat')
morethan10 = np.zeros([nbin1,nbin2])

for i in range(nbin1):
    for j in range(nbin2):
        
        if j > i:
            continue
            
        m1inbin = (data.m1 >= m1_bin[i]) & (data.m1 < m1_bin[i+1])
        m2inbin = (data.m2 >= m2_bin[j]) & (data.m2 < m2_bin[j+1])
        mbin = m1inbin & m2inbin & data.found_any
        
        if mbin.sum() < 10:
            continue
        
        morethan10[i,j] = mbin.sum()

toplot = np.nonzero(morethan10)

emax_plot = emax[toplot]
Mc_plot = Mc[toplot]
Mtot_plot = Mtot[toplot]
q_plot = q[toplot]
mu_plot = mu[toplot]
eta_plot = mu_plot / Mtot_plot
n_plot = morethan10[toplot]

#%%
plt.close('all')

plt.figure()
im = plt.scatter(Mtot_plot, emax_plot, c=n_plot)
cbar = plt.colorbar(im)
cbar.set_label('N events', fontsize=15)
plt.xlabel(r'$M_z$', fontsize = 15)
plt.ylabel(r'$\varepsilon_{max}$', fontsize = 15)
plt.grid(True, which='both')
name="dL_joint_fit_results_emax_vary/Mtot.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax_vary/Mtot.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
#%%
plt.figure()
im = plt.scatter(Mtot_plot, emax_plot, c=n_plot, norm=LogNorm())
plt.xscale('log')
cbar = plt.colorbar(im)
cbar.set_label(r'N events', fontsize=20)
cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
cbar.ax.tick_params(axis='y', which='both', labelsize=15)
plt.xlabel(r'$M_z$', fontsize = 24)
plt.ylabel(r'$\varepsilon_\mathrm{max}$', fontsize = 24)
plt.grid(True, which='both')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name="binned_analysis/dL_joint_fit_results_emax_vary/Mtot_log.png"
plt.savefig(name, format='png', dpi=300)
name="binned_analysis/dL_joint_fit_results_emax_vary/Mtot_log.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
plt.figure()
im = plt.scatter(Mc_plot, emax_plot, c=n_plot)
cbar = plt.colorbar(im)
cbar.set_label('N events', fontsize=15)
plt.xlabel(r'$M_c$', fontsize = 15)
plt.ylabel(r'$\varepsilon_{max}$', fontsize = 15)
plt.grid(True, which='both')
name="dL_joint_fit_results_emax_vary/Mc.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax_vary/Mc.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
plt.figure()
im = plt.scatter(Mc_plot, emax_plot, c=n_plot)
plt.xscale('log')
cbar = plt.colorbar(im)
cbar.set_label('N events', fontsize=15)
plt.xlabel(r'$\log M_c$', fontsize = 15)
plt.ylabel(r'$\varepsilon_{max}$', fontsize = 15)
plt.grid(True, which='both')
name="dL_joint_fit_results_emax_vary/Mc_log.png"
plt.savefig(name, format='png', dpi=1000)
name="dL_joint_fit_results_emax_vary/Mc_log.pdf"
plt.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")
#%%

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), gridspec_kw=dict(top=1, left=-0.3, right=1.2, bottom=0))
ax.scatter(Mtot_plot, Mc_plot, emax_plot, c=n_plot, cmap='viridis', alpha=0.8)
ax.set_xlabel(r'$M_z$', fontsize = 15)
ax.set_ylabel(r'$M_c$', fontsize = 15)
ax.set_zlabel(r'$\varepsilon_{max}$', fontsize = 15)
name="dL_joint_fit_results_emax_vary/3D_plot.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax_vary/3D_plot.pdf"
plt.savefig(name, format='pdf', dpi=300)

fig = plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_trisurf(Mtot_plot, Mc_plot, emax_plot, cmap='viridis')
ax.set_xlabel(r'$M_z$', fontsize = 15)
ax.set_ylabel(r'$M_c$', fontsize = 15)
ax.set_zlabel(r'$\varepsilon_{max}$', fontsize = 15)
name="dL_joint_fit_results_emax_vary/3D_surface_plot.png"
plt.savefig(name, format='png', dpi=300)


#%% 
#pdet ideal/real plots

d2 = np.linspace(0, 3, 1000)
fun1 = 1 / (1+(d2/0.7)**(3+35*np.tanh(d2/0.7)))
fun2 = 1 / (1+(d2/0.7)**(2+2*np.tanh(d2/0.7)))

fig, ax = plt.subplots()
ax.plot(d2, fun1, 'r-')
ax.hlines(0.5, 0, 0.7, color='tab:blue', linestyle='dashed')
ax.vlines(0.7, 0, 0.5, color='tab:blue', linestyle='dashed')
ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 2.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=25)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.set_xlabel(r'$d_L$', fontsize=20)
ax.set_ylabel(r'$P_\mathrm{det}$', fontsize=22)
ax.spines[["left", "bottom"]].set_position(('data', 0))
ax.spines[["top", "right"]].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
#ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
plt.savefig('plots/pdet_ideal.pdf', format='pdf', dpi=300, bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(d2, fun2, 'r-')
ax.hlines(0.5, 0, 0.7, color='tab:blue', linestyle='dashed')
ax.vlines(0.7, 0, 0.5, color='tab:blue', linestyle='dashed')
ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 2.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=25)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.set_xlabel(r'$d_L$', fontsize=20)
ax.set_ylabel(r'$P_\mathrm{det}$', fontsize=22)
ax.spines[["left", "bottom"]].set_position(('data', 0))
ax.spines[["top", "right"]].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
#ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
plt.savefig('plots/pdet_real.pdf', format='pdf', dpi=300, bbox_inches="tight")

#%%
run_dataset = 'o3'   
data.load_inj_set(run_dataset)

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)
mtot_det = m1_det + m2_det 

#dmid_values = data.dmid(m1_det, m2_det, data.dmid_params)
dmid_values = data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)
#total_pdet = data.total_pdet(data.dL, m1_det, m2_det)
data.set_shape_params()

#%%
plt.figure(figsize=(7,6))
im = plt.scatter(m1_det, m2_det, s=1, c=dmid_values, rasterized=True)
plt.xlabel(r'$m_{1z} [M_{\odot}]$', fontsize=24)
plt.ylabel('$m_{2z} [M_{\odot}]$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$d_\mathrm{mid}$', fontsize=24)
plt.show()
plt.savefig( path + '/m1m2det_dmid.png')
name = path + '/m1m2det_dmid.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
plt.figure()
#plt.scatter(data.dL/data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params), total_pdet, s=1)
plt.scatter(data.dL/data.dmid(m1_det, m2_det, data.dmid_params), total_pdet, s=1, rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 15)
plt.ylabel(r'total $P_\mathrm{det}$', fontsize = 15)
plt.savefig( path + '/total_pdet.png')
name = path + '/total_pdet.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%

#PDET / OPT_EPSILON_PLOT FOR ONLY 1 RUN
#IMPORTANT TO RERUN THE LOAD INJ SET IN CASE YOU RUNNED OTHER METHODS BEFORE THAT CHANGED DE SET LOADED SUCH AS run_pdet for o1 or o2
dmid_fun = 'Dmid_mchirp_fdmid'
run_dataset = 'o3'   
data = Found_injections(dmid_fun, emax_fun, alpha_vary)
path = f'{run_dataset}/' + data.path
data.load_inj_set(run_dataset)
data.get_opt_params(run_fit)

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)
mtot_det = m1_det + m2_det 

dmid_values = data.dmid(m1_det, m2_det, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)

data.set_shape_params()

rc('text', usetex=True)

#%%

plt.figure()
plt.scatter(data.dL/dmid_values, data.run_pdet(data.dL, m1_det, m2_det, 'o3'), s=1, rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 24)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name = path + '/pdet_o3.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%

plt.figure()
im = plt.scatter(data.dL/dmid_values, data.run_pdet(data.dL, m1_det, m2_det, 'o3'), s=1, c=mtot_det, rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 24)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'$M_z$', fontsize=24)
plt.show()
name = path + '/pdet_o3_Mtot.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.dL, data.run_pdet(data.dL, m1_det, m2_det, 'o3'), s=1, c = mtot_det)
cbar = plt.colorbar(im)
cbar.set_label(r'$M_z [M_{\odot}]$', fontsize=15)
plt.xlabel(r'$d_L$ [Mpc]', fontsize = 15)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 15)
name = path + '/pdet_o3_cmap.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(m1_det, m2_det,  c = data.run_pdet(data.dL, m1_det, m2_det, 'o3'), s=1)
cbar = plt.colorbar(im)
cbar.set_label(r'$P_\mathrm{det}$', fontsize=15)
plt.xlabel(r'$m_{1z} [M_{\odot}]$', fontsize = 15)
plt.ylabel(r'$m_{2z} [M_{\odot}]$', fontsize = 15)
name = path + '/m12_pdet.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

data.set_shape_params()

m1det = np.linspace(min(m1_det), max(m1_det), 200)
m2det = np.linspace(min(m2_det), max(m2_det), 200)
mtot = m1det + m2det
emax = data.emax(m1det, m2det, data.emax_params)
#y = (1 - np.exp(params_emax[0])) / (1 + np.exp(params_emax[1]*(x-params_emax[2])))
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

data.set_shape_params()

index = np.random.choice(np.arange(len(data.m1)), 100000, replace=False)
m1det = m1_det[index]
m2det = m2_det[index]
mtot = m1det + m2det
emax = data.emax(m1det, m2det, data.emax_params)
#y = (1 - np.exp(params_emax[0])) / (1 + np.exp(params_emax[1]*(x-params_emax[2])))
plt.figure()
plt.scatter(mtot, emax, s=1)
plt.ylim(0, 2)
#plt.semilogx()
plt.grid()
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize=15)
plt.ylabel(r'$\varepsilon_{max}$', fontsize=15)
#name = path + '/emax.pdf'
#plt.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")

#%%

plt.figure()
im = plt.scatter(m1_det, m2_det,  c = data.emax(m1_det, m2_det, data.emax_params), s=1)
cbar = plt.colorbar(im)
cbar.set_label(r'$\varepsilon_{max}$', fontsize=15)
plt.xlabel(r'$m_{1z} [M_{\odot}]$', fontsize = 15)
plt.ylabel(r'$m_{2z} [M_{\odot}]$', fontsize = 15)
name = path + '/m12_emax.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

plt.figure()
plt.scatter(m1_det[data.found_any], m2_det[data.found_any], s=1)
plt.xlabel('m1')
plt.ylabel('m2')

#%%
plt.close('all')

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.Mc, dmid_values, c=data.dL, s=1)
cbar = plt.colorbar(im)
cbar.set_label(r'$d_L$ [Mpc]', fontsize=15)
plt.xlabel(r'$M_c [M_{\odot}]$', fontsize=15)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize=15)
name = path + '/dmid_vs_mchirp.png'
#plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
plt.savefig(name, format='png', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.Mc_det, dmid_values, c=data.dL, s=1)
cbar = plt.colorbar(im)
cbar.set_label(r'$d_L$ [Mpc]', fontsize=15)
plt.xlabel(r'$M_c det[M_{\odot}]$', fontsize=15)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize=15)
name = path + '/dmid_vs_mchirp_det.png'
#plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
plt.savefig(name, format='png', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(7,6))
im = plt.scatter(data.Mc_det, dmid_values, c=data.eta, s=1, rasterized=True)
cbar = plt.colorbar(im)
cbar.set_label(r'$\eta$', fontsize=24)
cbar.ax.tick_params(labelsize=15)
plt.xlabel(r'$\mathcal{M}_z [M_{\odot}]$', fontsize=24)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name = path + '/dmid_vs_mchirp_eta.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")
name = path + '/dmid_vs_mchirp_eta.png'
plt.savefig(name, format='png', dpi=150, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.dL, data.Mc, c=dmid_values, s=1)
cbar = plt.colorbar(im )
cbar.set_label(r'$d_\mathrm{mid}$', fontsize=15)
plt.ylabel(r'$M_c [M_{\odot}]$', fontsize=15)
plt.xlabel(r'$d_L$ [Mpc]', fontsize=15)
name = path + '/dL_vs_mchirp.png'
#plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
plt.savefig(name, format='png', dpi=300, bbox_inches="tight")

#%%

# class MidPointLogNorm(LogNorm):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
#         self.midpoint=midpoint
#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(np.log(value), x, y))
    
#chieff_corr = np.exp( (data.dmid_params[6] + data.dmid_params[7] * data.Mtot_det) * data.chi_eff )
chieff_corr = np.exp( (data.dmid_params[6] + data.dmid_params[7] * data.Mtot_det) * data.chi_eff )

plt.close('all')
plt.figure(figsize=(8,4.8))
#im = plt.scatter(data.chi_eff, data.Mc_det, s=1, c=chieff_corr, norm=MidPointLogNorm(vmin=chieff_corr.min(), vmax=chieff_corr.max(), midpoint=1), cmap = 'Spectral')
im = plt.scatter(data.chi_eff, data.Mtot_det, s=1, c=chieff_corr, norm=LogNorm(vmin=0.4, vmax=2.6), cmap = 'Spectral', rasterized=True)
cbar = plt.colorbar(im)
#cbar.ax.tick_params(axis='both', which='major', labelsize=15)
cbar.set_label(r'$\exp (f_{AS})$', fontsize=24)
plt.xlabel(r'$\chi_\mathrm{eff}$', fontsize=24)
plt.ylabel(r'$M_{z}$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
cbar.ax.tick_params(axis='y', which='both', labelsize=15)
#cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
#cbar.ax.tick_params(labelsize=15)
name = path + '/chieff_corr_mtot.png'
plt.savefig(name, format='png', dpi=150, bbox_inches="tight")
name = path + '/chieff_corr_mtot.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

# plt.figure()
# plt.hist(chieff_corr, bins=50)
# plt.xlabel(r'$\exp (f_{AS})$', fontsize=15)
# name = path + '/chieff_corr_hist.png'
# #plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
# plt.savefig(name, format='png', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(7,6))
im = plt.scatter(data.Mc_det, dmid_values, c=data.chi_eff, s=1, rasterized=True)
cbar = plt.colorbar(im)
cbar.set_label(r'$\chi_\mathrm{eff}$', fontsize=24)
cbar.ax.tick_params(labelsize=15)
plt.xlabel(r'$\mathcal{M}_z [M_{\odot}]$', fontsize=24)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name = path + '/dmid_vs_mchirp_chieff.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")
name = path + '/dmid_vs_mchirp_chieff.png'
plt.savefig(name, format='png', dpi=150, bbox_inches="tight")

#%%

dL_real = data.dL[data.found_any]
pdet = data.run_pdet(data.dL, m1_det, m2_det, 'o3')

plt.figure()
plt.hist(dL_real, bins = 100, histtype=u'step', label = 'real detected inj' )
plt.hist(data.dL, bins = 100, weights = pdet, histtype=u'step', label = 'model')
plt.xlabel(r'$d_L$')
plt.legend()
name = path + '/dL_detected_hist.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
name = path + '/dL_detected_hist.png'
plt.savefig(name, format='png', dpi=300, bbox_inches="tight")

#%%
nboots = 100
params_data = np.loadtxt(f'{path}/100_boots_opt_params.dat')[1:]
#%%
#figure = corner.corner(params_data, labels = [r'$\gamma$', r'$\delta$', r'$b_0$', r'$b_1$', r'$b_2$', r'$cte$', r'$a_{20}$', r'$a_{01}$', r'$a_{21}$', r'$a_{10}$', r'$a_{11}$'])
#figure.savefig(f'{path}/50_boots_corner_plot.png')
figure = corner.corner(params_data[:, :5], labels = [r'$\gamma$', r'$\delta$', r'$b_0$', r'$b_1$', r'$b_2$'])
figure.savefig(f'{path}/{nboots}_boots_corner_plot_shape.png')
#%%
figure = corner.corner(params_data[:, 5:], labels = [r'$D_0$', r'$a_{20}$', r'$a_{01}$', r'$a_{21}$', r'$a_{10}$', r'$a_{11}$', r'$c_{1}$', r'$c_{11}$'])
figure.savefig(f'{path}/{nboots}_boots_corner_plot_dmid.png')
#%%
emax_params = params_data[:, 2:6]
dmid_params = params_data[:, 6:]
gamma = params_data[:,:1]
delta = params_data[:,1:2]

data.load_all_inj_sets('o3', sources)
m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)
mtot_det = m1_det + m2_det 

dmid_values = np.array([data.dmid(m1_det, m2_det, data.chi_eff, dmid_params[i]) for i in range(len(dmid_params[:,0]))])
[data.apply_dmid_mtotal_max(dmid_values[i], mtot_det) for i in range(len(dmid_params[:,0]))]

data.set_shape_params()

#%%

m1det = np.linspace(min(m1_det), max(m1_det), 200)
m2det = np.linspace(min(m2_det), max(m2_det), 200)
mtot = m1det + m2det
emax = np.array([data.emax(m1det, m2det, emax_params[i]) for i in range(len(emax_params[:,0]))])
#y = (1 - np.exp(params_emax[0])) / (1 + np.exp(params_emax[1]*(x-params_emax[2])))
plt.figure()
for i in range(len(emax)):
    
    plt.plot(mtot, emax[i], '-', alpha = 0.5)
#plt.ylim(0, 1.2)
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize=24)
plt.ylabel(r'$\varepsilon_\mathrm{max}$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
name = path + f'/{nboots}_boots_emax.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%

m1 = 50.
m2 = 45.
chieff = 0.75
dL = np.linspace(min(data.dL), max(data.dL), 500)
dmid_values = np.array([data.dmid(m1, m2, chieff, dmid_params[i]) for i in range(len(dmid_params[:,0]))])
emax = np.array([data.emax(m1, m2, emax_params[i]) for i in range (len(emax_params[:,0]))])
#[data.apply_dmid_mtotal_max(dmid_values[i], np.array([m1+m2])) for i in range(len(dmid_params[:,0]))]
pdet = np.array([data.sigmoid(dL, dmid_values[i], emax[i] , gamma[i] , delta[i]) for i in range(len(emax_params[:,0]))])
plt.figure()
for i in range(len(dmid_values)):
    plt.plot(dL/dmid_values[i], pdet[i], alpha = 0.2, rasterized=True)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 15)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 15)
plt.title(r'$m_{1z} = 50 M_{\odot}$, $m_{2z} = 45 M_{\odot}$, $\chi_\mathrm{eff} = 0.75$')
name = path + f'/{nboots}_boots_pdet_o3.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
#no chieff

cte = dmid_params[:,0]
a_20 = dmid_params[:,1]
a_01 = dmid_params[:,2]
a_21 = dmid_params[:,3]
a_10 = dmid_params[:,4]
a_11 = dmid_params[:,5]


eta = np.linspace(min(data.eta), max(data.eta), 500)

M = 50
Mc = eta**(3/5) * M
pol = np.array([cte[i] *(1+ a_20[i] * M**2  + a_01[i] * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) + a_10[i] * M + a_11[i] * M * (1 - 4*eta)) for i in range(len(cte)) ])
dmid = pol * Mc**(5/6)

plt.figure()
for i in range(len(cte)):
    plt.semilogy(eta, dmid[i],color = 'r', alpha = 0.2, rasterized=True, label= r'$M_z = 50 [M_{\odot}]$')
    

M = 150
Mc = eta**(3/5) * M
pol = np.array([cte[i] *(1+ a_20[i] * M**2  + a_01[i] * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) + a_10[i] * M + a_11[i] * M * (1 - 4*eta)) for i in range(len(cte)) ])
dmid = pol * Mc**(5/6)

for i in range(len(cte)):
    plt.semilogy(eta, dmid[i],color = 'g', alpha = 0.2, rasterized=True, label= r'$M_z = 150 [M_{\odot}]$')
    
M = 250
Mc = eta**(3/5) * M
pol = np.array([cte[i] *(1+ a_20[i] * M**2  + a_01[i] * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) + a_10[i] * M + a_11[i] * M * (1 - 4*eta)) for i in range(len(cte)) ])
dmid = pol * Mc**(5/6)

for i in range(len(cte)):
    plt.semilogy(eta, dmid[i], color = 'b', alpha = 0.2, rasterized=True, label= r'$M_z = 250 [M_{\odot}]$')

    
M = 450
Mc = eta**(3/5) * M
pol = np.array([cte[i] *(1+ a_20[i] * M**2  + a_01[i] * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) + a_10[i] * M + a_11[i] * M * (1 - 4*eta)) for i in range(len(cte)) ])
dmid = pol * Mc**(5/6)

for i in range(len(cte)):
    plt.semilogy(eta, dmid[i], color = 'm', alpha = 0.2, rasterized=True, label= r'$M_z = 450 [M_{\odot}]$')
    
#plt.legend()
plt.xlabel(r'$\eta$', fontsize = 15)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize = 15)
name = path + '/{nboots}_boots_dmid_variousMz.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
#chieff
#dmid vs eta
# cte = dmid_params[:,0]
# a_20 = dmid_params[:,1]
# a_01 = dmid_params[:,2]
# a_21 = dmid_params[:,3]
# a_10 = dmid_params[:,4]
# a_11 = dmid_params[:,5]
# c_1 = dmid_params[:,6]
# c_11 = dmid_params[:,7]

D0 = dmid_params[:,0] 
B = dmid_params[:,1] 
C = dmid_params[:,2]  
mu = dmid_params[:,3] 
sigma = dmid_params[:,4] 
a_01 = dmid_params[:,5] 
a_11 = dmid_params[:,6] 
a_21 = dmid_params[:,7] 
c_01 = dmid_params[:,8] 
c_11 = dmid_params[:,9] 
d_11 = dmid_params[:,10] 
L = dmid_params[:,11] 


eta = np.linspace(min(data.eta), max(data.eta), 500)
M_values = [20, 50, 150, 250, 450]
colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#e78ac3']
labels = [r'$M_z = 20 [M_{\odot}]$', r'$M_z = 50 [M_{\odot}]$', r'$M_z = 150 [M_{\odot}]$', r'$M_z = 250 [M_{\odot}]$', r'$M_z = 450 [M_{\odot}]$']

plt.figure()

for M, color, label in zip(M_values, colors, labels):
    
    Mc = eta**(3/5) * M
    chieff = 0.75
    pol = np.array([cte[i] * np.exp((c_1[i] + c_11[i] * M) * chieff) * np.exp(a_20[i] * M**2  + a_01[i] * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) + a_10[i] * M + a_11[i] * M * (1 - 4*eta)) for i in range(len(cte)) ])
    dmid = pol * Mc**(5/6)
    
    for i in range(len(cte)):
        if i == 0:
            plt.semilogy(eta, dmid[i], color=color, alpha=0.2, rasterized=True, label=label)
        else:
            plt.semilogy(eta, dmid[i], color=color, alpha=0.2, rasterized=True)
   
leg = plt.legend(fontsize = 18)
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.xlabel(r'$\eta$', fontsize = 24)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('chieff = 0.75')
name = path + f'/{nboots}_boots_dmid_variousMz.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
#chieff 
#dmid vs M
M = np.linspace(min(data.Mtot_det), max(data.Mtot_det), 500)
eta_values = [0.05, 0.1, 0.15, 0.2, 0.25]
colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#e78ac3']
labels = [r'$\eta = 0.05$', r'$\eta = 0.1$', r'$\eta = 0.15$', r'$\eta = 0.2$', r'$\eta = 0.25$']

plt.figure()

for eta, color, label in zip(eta_values, colors, labels):
    
    Mc = eta**(3/5) * M
    chi_eff = 0.75
    fexp = np.array([np.exp(-B[i] * M - L[i] * np.log(M)) for i in range(len(L))])
    fgauss = np.array([np.exp(-(np.log(M)-np.log(mu[i]))**2 / (2*sigma[i]**2)) for i in range(len(L)) ])
    
    f_M = np.array([D0[i] * (fexp[i] + C[i] * fgauss[i]) for i in range(len(L)) ])
    
    f_eta = np.array([ a_01[i] * (1 - 4*eta)  + a_11[i] * M * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) for i in range(len(L)) ])
    f_as = np.array([(c_01[i] + c_11[i] * M + d_11[i] * np.log(M)) * chi_eff for i in range(len(L)) ])
    
    dmid = Mc**(5/6) * f_M * np.exp(f_eta) * np.exp(f_as)
    
    for i in range(len(L)):
        if i == 0:
            plt.loglog(M, dmid[i], color=color, alpha=0.3, rasterized=True, label=label)
        else:
            plt.loglog(M, dmid[i], color=color, alpha=0.3, rasterized=True)
   
leg = plt.legend(fontsize = 18)
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize = 24)
plt.ylabel(r'$d_\mathrm{mid} \, / \, \mathcal{M}^{5/6}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title(r'$\chi_\mathrm{eff} = 0.75$', fontsize = 20)
name = path + f'/{nboots}_boots_dmid_variouseta.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
#chieff 
#dmid vs chieff various Mz
chieff = np.linspace(min(data.chi_eff), max(data.chi_eff), 500)
M_values = [20, 50, 150, 250, 450]
colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#e78ac3']
labels = [r'$M_z = 20 \, M_{\odot}$', r'$M_z = 50  \, M_{\odot}$', r'$M_z = 150 \, M_{\odot}$', r'$M_z = 250 \, M_{\odot}$', r'$M_z = 450 \, M_{\odot}$']

plt.figure()

for M, color, label in zip(M_values, colors, labels):
    
    eta = 0.175
    M = np.ones(len(chieff)) * M
    Mc = eta**(3/5) * M
    fexp = np.array([np.exp(-B[i] * M - L[i] * np.log(M)) for i in range(len(L))])
    fgauss = np.array([np.exp(-(np.log(M)-np.log(mu[i]))**2 / (2*sigma[i]**2)) for i in range(len(L)) ])
    
    f_M = np.array([D0[i] * (fexp[i] + C[i] * fgauss[i]) for i in range(len(L)) ])
    
    f_eta = np.array([ a_01[i] * (1 - 4*eta)  + a_11[i] * M * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) for i in range(len(L)) ])
    f_as = np.array([(c_01[i] + c_11[i] * M + d_11[i] * np.log(M)) * chieff for i in range(len(L)) ])
    
    dmid = Mc**(5/6) * f_M * np.exp(f_eta) * np.exp(f_as)
    
    for i in range(len(L)):
        if i == 0:
            plt.semilogy(chieff, dmid[i], color=color, alpha=0.2, rasterized=True, label=label)
        else:
            plt.semilogy(chieff, dmid[i], color=color, alpha=0.2, rasterized=True)
   
leg = plt.legend(fontsize = 14)
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.xlabel(r'$\chi_\mathrm{eff}$', fontsize = 24)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize = 24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title(r'$\eta =  0.175$', fontsize = 20)

name = path + f'/{nboots}_boots_dmid_vs_chieff_variousMz.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
#chieff 
#dmid vs M various eta
chieff = np.linspace(min(data.chi_eff), max(data.chi_eff), 500)
eta_values = [0.05, 0.1, 0.15, 0.2, 0.25]
colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#e78ac3']
labels = [r'$\eta = 0.05$', r'$\eta = 0.1$', r'$\eta = 0.15$', r'$\eta = 0.2$', r'$\eta = 0.25$']

plt.figure()

for eta, color, label in zip(eta_values, colors, labels):
    
    M = 300
    Mc = eta**(3/5) * M
    pol = np.array([cte[i] * np.exp((c_1[i] + c_11[i] * M) * chieff) * np.exp(a_20[i] * M**2  + a_01[i] * (1 - 4*eta) + a_21[i] * M**2 * (1 - 4*eta) + a_10[i] * M + a_11[i] * M * (1 - 4*eta)) for i in range(len(cte)) ])
    dmid = pol * Mc**(5/6)
    
    for i in range(len(cte)):
        if i == 0:
            plt.semilogy(chieff, dmid[i], color=color, alpha=0.2, rasterized=True, label=label)
        else:
            plt.semilogy(chieff, dmid[i], color=color, alpha=0.2, rasterized=True)
   
plt.legend()
plt.xlabel(r'$\chi_\mathrm{eff}$', fontsize = 15)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize = 15)
plt.title('$M_z = 300 [M_{\odot}]$')
name = path + f'/{nboots}_boots_dmid_vs_chieff_variouseta.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
xf = data.dL[data.found_any]/(data.Mc_det[data.found_any]**(5/6))
yf = data.eta[data.found_any]

xn = data.dL[~data.found_any]/(data.Mc_det[~data.found_any]**(5/6))
yn = data.eta[~data.found_any]

plt.figure()
plt.scatter(xn, yn, alpha = 0.1, label = 'missed inj')
plt.scatter(xf, yf, alpha = 0.1, label = 'found inj')
plt.xlabel('dL/Mc_det**(5/6)')
plt.ylabel('eta')
plt.legend()
name = path + 'eta_vs_dL'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")


#%%

#emax plt for both chieff and not chieff comparison
dmid_fun = 'Dmid_mchirp_fdmid_fspin'

data = Found_injections(dmid_fun, emax_fun, alpha_vary)
data.load_inj_set(run_dataset)
data.get_opt_params(run_fit)
data.set_shape_params()

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)

m1det = np.linspace(min(m1_det), max(m1_det), 200)
m2det = np.linspace(min(m2_det), max(m2_det), 200)
mtot = m1det + m2det
emax = data.emax(m1det, m2det, data.emax_params)

plt.figure(figsize=(7,4.8))
plt.plot(mtot, emax, '-', label = r'fit including $\chi_\mathrm{eff}$ ')


#no chieff
dmid_fun = 'Dmid_mchirp_fdmid'

data = Found_injections(dmid_fun, emax_fun, alpha_vary)
data.load_inj_set(run_dataset)
data.get_opt_params(run_fit)
data.set_shape_params()

emax_nochieff = data.emax(m1det, m2det, data.emax_params)
plt.plot(mtot, emax_nochieff, linestyle = 'dashed', label = r'fit without $\chi_\mathrm{eff}$ ')


plt.ylim(0, 1.2)
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize=24)
plt.ylabel(r'$\varepsilon_\mathrm{max}$', fontsize=24)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.legend(fontsize=18)
name = path + '/emax_comparison.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
data.get_opt_params('o3')

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)

mtot_det = data.Mtot_det

dmid_values = data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)

data.set_shape_params()

pdet = data.sigmoid(data.dL,dmid_values, data.emax(m1_det, m2_det, data.emax_params), data.gamma, data.delta)

import h5py
file = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')
snr = file["injections/optimal_snr_net"][:]

plt.figure()
plt.scatter(snr, pdet, s = 0.2, c = mtot_det)
plt.loglog()
plt.colorbar()

#%%

far_pful = data.far_pbbh[data.found_any]
snr = data.snr[data.found_any]
mtot = mtot_det[data.found_any]
plt.figure()
plt.scatter(snr, far_pful, s = 0.2, c = mtot)
plt.loglog()
plt.colorbar()
plt.ylim(1e-5, 100)

#%%

snr = data.snr

#cumulative distribution over snr
indexo = np.argsort(snr)
varo = snr[indexo]
dLo = data.dL[indexo]
m1o = data.m1[indexo]
m2o = data.m2[indexo]
zo = data.z[indexo]
chieffo = data.chi_eff[indexo]
m1o_det = m1o * (1 + zo) 
m2o_det = m2o * (1 + zo)
mtoto_det = m1o_det + m2o_det

data.get_opt_params('o3')
dmid_values = data.dmid(m1o_det, m2o_det, chieffo, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)
data.set_shape_params()
emax = data.emax(m1o_det, m2o_det, data.emax_params)

cmd = np.cumsum(data.sigmoid(dLo, dmid_values, emax, data.gamma, data.delta))

#found injections
var_found = snr[data.found_any]
indexo_found = np.argsort(var_found)
var_foundo = var_found[indexo_found]
real_found_inj = np.arange(len(var_foundo))+1

plt.figure()
plt.scatter(varo, cmd, s=1, label='model', rasterized=True)
plt.scatter(var_foundo, real_found_inj, s=1, label='found injections', rasterized=True)
plt.semilogx()
plt.xlabel('optimal SNR net', fontsize = 20)
plt.ylabel('Cumulative found injections', fontsize = 20)
plt.legend(loc='best', fontsize = 20)
name = path + '/SNR_cumulative.png'
plt.savefig(name, format='png', bbox_inches="tight")
name = path + '/SNR_cumulative.pdf'
plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")

#%%
#KS test
from scipy.stats import kstest

m1_det = data.m1 * (1 + data.z) 
m2_det = data.m2 * (1 + data.z)
mtot_det = m1_det + m2_det

dmid_values = data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)
data.set_shape_params()
emax = data.emax(m1_det, m2_det, data.emax_params)

       
pdet = data.sigmoid(data.dL, dmid_values, emax, data.gamma, data.delta)

def cdf(x):
    values = [np.sum(pdet[snr<value])/np.sum(pdet) for value in x]
    return np.array(values)
    
stat, pvalue = kstest(var_foundo, lambda x: cdf(x) )
print('optimal SNR net KStest : statistic = %s , pvalue = %s' %(stat, pvalue))


#%%
from astropy.cosmology import FlatLambdaCDM
dL = data.dL
z = data.z
pdL = data.dL_pdf

A = np.sqrt(data.omega_m * (1 + z)**3 + 1 - data.omega_m)
dL_dif = (data.c * (1 +z) / data.H0) * (1/A)

dC = np.array(data.cosmo.comoving_distance(data.z))

plt.figure()
plt.plot(z, dL, '.', label='dL')
plt.plot(z, dL_dif, '.', label = 'd(dL)')
plt.plot(z, dL_dif + dL/(1+z), '.', label = 'd(dL) + dL/(1+z)')
plt.plot(z, dC+dL_dif, '.', label='new d(dL)')
plt.xlabel('z')
plt.legend()
plt.savefig('plots/dL_derivative.png', format='png', dpi=300, bbox_inches="tight")







#%%

os.chdir(original_working_directory)



#%%

os.chdir(original_working_directory)
