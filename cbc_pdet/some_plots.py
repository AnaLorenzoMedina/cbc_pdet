#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:28:40 2023

@author: ana
"""

import matplotlib.pyplot as plt
import numpy as np
from o123_class_found_inj_general import Found_injections
from matplotlib.colors import LogNorm
from matplotlib import rc

plt.close('all')

run_fit = 'o3'
run_dataset = 'o3'

dmid_fun = 'Dmid_mchirp_expansion_noa30'
emax_fun = 'emax_exp'
alpha_vary = None

data = Found_injections(dmid_fun, emax_fun, alpha_vary)
path = f'{run_dataset}/' + data.path
data.load_inj_set(run_dataset)
data.get_opt_params(run_fit)

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

k = 42
dmid = np.loadtxt(f'dL_joint_fit_results_emax/dLmid/dLmid_{k}.dat')
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
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="dL_joint_fit_results_emax/Mtot.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax/Mtot.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")


#plt.plot(Mc_plot.flatten(), 70*(Mc_plot.flatten())**(5/6), 'r-', label='cte = 70')
#plt.legend()


plt.figure()
plt.loglog(q_plot, dmid_plot/(Mc_plot)**(5/6), '.')
plt.xlabel(r'$q = m2_ / m_1$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="dL_joint_fit_results_emax/q.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax/q.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

plt.figure()
plt.scatter(eta_plot, dmid_plot/(Mc_plot)**(5/6), c=Mtot_plot)
plt.colorbar(label=r'$M_z$')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="dL_joint_fit_results_emax/eta.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax/eta.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), gridspec_kw=dict(top=1, left=-0.3, right=1.2, bottom=0))
ax.scatter(Mtot_plot, eta_plot, dmid_plot/(Mc_plot)**(5/6), c=dmid_plot/(Mc_plot)**(5/6), cmap='viridis', alpha=0.8)
ax.set_xlabel(r'$M_z$', fontsize = 15)
ax.set_ylabel(r'$\eta$', fontsize = 15)
ax.set_zlabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$', fontsize = 15)
fig.subplots_adjust(right=1.6, top=0.9, bottom=0.1, left=-0.6)
name="dL_joint_fit_results_emax/3D_plot.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax/3D_plot.pdf"
plt.savefig(name, format='pdf', dpi=300)
#%%
fig = plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_trisurf(Mtot_plot, eta_plot, dmid_plot/(Mc_plot)**(5/6), cmap='viridis')
ax.set_xlabel('Mtotal ')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
name="dL_joint_fit_results_emax/3D_surface_plot.png"
plt.savefig(name, format='png', dpi=300)

#%%
#EMAX PLOTS 

k = 10
emax = np.loadtxt(f'dL_joint_fit_results_emax_vary/emax/emax_{k}.dat')
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

rc('text', usetex=True)
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

plt.figure()
im = plt.scatter(Mtot_plot, emax_plot, c=n_plot, norm=LogNorm())
plt.xscale('log')
cbar = plt.colorbar(im)
cbar.set_label('N events', fontsize=15)
plt.xlabel(r'$\log M_z$', fontsize = 15)
plt.ylabel(r'$\varepsilon_{max}$', fontsize = 15)
plt.grid(True, which='both')
name="dL_joint_fit_results_emax_vary/Mtot_log.png"
plt.savefig(name, format='png', dpi=300)
name="dL_joint_fit_results_emax_vary/Mtot_log.pdf"
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

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

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)
mtot_det = m1_det + m2_det 

m1 = data.m1
m2 = data.m2
z = data.z
dmid_values = data.dmid(m1_det, m2_det, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)
#total_pdet = data.total_pdet(data.dL, m1_det, m2_det)

#%%
plt.figure(figsize=(7,6))
im = plt.scatter(m1*(1+z), m2*(1+z), s=1, c=dmid_values)
plt.xlabel(r'$m_{1z} [M_{\odot}]$', fontsize=15)
plt.ylabel('$m_{2z} [M_{\odot}]$', fontsize=15)
cbar = plt.colorbar(im)
cbar.set_label(r'$d_\mathrm{mid}$', fontsize=15)
plt.show()
plt.savefig( path + '/m1m2det_dmid.png')
name = path + '/m1m2det_dmid.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%
plt.figure()
#plt.scatter(data.dL/data.dmid(m1_det, m2_det, data.chi_eff, data.dmid_params), total_pdet, s=1)
plt.scatter(data.dL/data.dmid(m1_det, m2_det, data.dmid_params), total_pdet, s=1)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 15)
plt.ylabel(r'total $P_\mathrm{det}$', fontsize = 15)
plt.savefig( path + '/total_pdet.png')
name = path + '/total_pdet.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

#%%

#PDET / OPT_EPSILON_PLOT FOR ONLY 1 RUN
#IMPORTANT TO RERUN THE LOAD INJ SET IN CASE YOU RUNNED OTHER METHODS BEFORE THAT CHANGED DE SET LOADED SUCH AS run_pdet for o1 or o2

run_dataset = 'o3'   
data.load_inj_set(run_dataset)

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)
mtot_det = m1_det + m2_det 

dmid_values = data.dmid(m1_det, m2_det, data.dmid_params)
data.apply_dmid_mtotal_max(dmid_values, mtot_det)

rc('text', usetex=True)

#%%
plt.figure()
plt.scatter(data.dL/dmid_values, data.run_pdet(data.dL, m1_det, m2_det, 'o3'), s=1)
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize = 15)
plt.ylabel(r'$P_\mathrm{det}$', fontsize = 15)
name = path + '/pdet_o3.pdf'
plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")

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
plt.figure()
plt.plot(mtot, emax, '-')
plt.ylim(0, 2)
#plt.semilogx()
plt.grid()
plt.xlabel(r'$M_z [M_{\odot}]$', fontsize=15)
plt.ylabel(r'$\varepsilon_{max}$', fontsize=15)
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

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.Mc_det, dmid_values, c=data.eta, s=1)
cbar = plt.colorbar(im)
cbar.set_label(r'$\eta$', fontsize=15)
plt.xlabel(r'$M_c det [M_{\odot}]$', fontsize=15)
plt.ylabel(r'$d_\mathrm{mid}$', fontsize=15)
name = path + '/dmid_vs_mchirp_eta.png'
#plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
plt.savefig(name, format='png', dpi=300, bbox_inches="tight")

#%%

plt.figure(figsize=(8,4.8))
im = plt.scatter(data.dL, data.Mc, c=dmid_values, s=1)
cbar = plt.colorbar(im, )
cbar.set_label(r'$d_\mathrm{mid}$', fontsize=15)
plt.ylabel(r'$M_c [M_{\odot}]$', fontsize=15)
plt.xlabel(r'$d_L$ [Mpc]', fontsize=15)
name = path + '/dL_vs_mchirp.png'
#plt.savefig(name, format='pdf', dpi=300, bbox_inches="tight")
plt.savefig(name, format='png', dpi=300, bbox_inches="tight")