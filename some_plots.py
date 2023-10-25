#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:28:40 2023

@author: ana
"""

import matplotlib.pyplot as plt
import numpy as np
from class_found_inj_general import *
from matplotlib.colors import LogNorm
from matplotlib import rc

plt.close('all')

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

k = 42
dmid = np.loadtxt(f'dL_joint_fit_results_emax/dLmid/dLmid_{k}.dat')
toplot = np.nonzero(dmid)

# k = 10
# emax = np.loadtxt(f'dL_joint_fit_results_emax_vary/emax/emax_{k}.dat')
# morethan10 = np.zeros([nbin1,nbin2])

# for i in range(nbin1):
#     for j in range(nbin2):
        
#         if j > i:
#             continue
            
#         m1inbin = (data.m1 >= m1_bin[i]) & (data.m1 < m1_bin[i+1])
#         m2inbin = (data.m2 >= m2_bin[j]) & (data.m2 < m2_bin[j+1])
#         mbin = m1inbin & m2inbin & data.found_any
        
#         if mbin.sum() < 10:
#             continue
        
#         morethan10[i,j] = mbin.sum()

# toplot = np.nonzero(morethan10)

#emax_plot = emax[toplot]
Mc_plot = Mc[toplot]
dmid_plot = dmid[toplot]
Mtot_plot = Mtot[toplot]
q_plot = q[toplot]
mu_plot = mu[toplot]
eta_plot = mu_plot / Mtot_plot
#n_plot = morethan10[toplot]


plt.figure()
plt.scatter(Mtot_plot, dmid_plot/(Mc_plot)**(5/6), c=eta_plot)
plt.colorbar(label=r'$\eta$')
plt.xlabel(r'$M_z$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="dL_joint_fit_results_emax/Mtot.png"
plt.savefig(name, format='png', dpi=1000)


#plt.plot(Mc_plot.flatten(), 70*(Mc_plot.flatten())**(5/6), 'r-', label='cte = 70')
#plt.legend()


plt.figure()
plt.loglog(q_plot, dmid_plot/(Mc_plot)**(5/6), '.')
plt.xlabel('q = m2 / m1')
plt.ylabel('dL_mid / Mc^(5/6)')
plt.grid(True, which='both')
name="dL_joint_fit_results_emax/q.png"
plt.savefig(name, format='png', dpi=1000)

plt.figure()
plt.scatter(eta_plot, dmid_plot/(Mc_plot)**(5/6), c=Mtot_plot)
plt.colorbar(label=r'$M_z$')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
plt.grid(True, which='both')
name="dL_joint_fit_results_emax/eta.png"
plt.savefig(name, format='png', dpi=1000)

fig = plt.figure()
ax=fig.add_subplot(projection='3d')

ax.scatter(Mtot_plot, eta_plot, dmid_plot/(Mc_plot)**(5/6), c=dmid_plot/(Mc_plot)**(5/6), cmap='viridis', alpha=0.8)
ax.set_xlabel(r'$M_z$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$d_\mathrm{mid} \, / \, M_\mathrm{c} ^{5/6}$')
name="dL_joint_fit_results_emax/3D_plot.png"
plt.savefig(name, format='png', dpi=1000)

fig = plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot_trisurf(Mtot_plot, eta_plot, dmid_plot/(Mc_plot)**(5/6), cmap='viridis')
ax.set_xlabel('Mtotal ')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel('dL_mid / Mc^(5/6)')
name="dL_joint_fit_results_emax/3D_surface_plot.png"
plt.savefig(name, format='png', dpi=1000)


# plt.figure()
# plt.scatter(Mtot_plot, emax_plot, c=n_plot)
# plt.colorbar(label='N events')
# plt.xlabel('Mtot = m1 + m2')
# plt.ylabel('emax')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax_vary/Mtot.png"
# plt.savefig(name, format='png', dpi=1000)

# plt.figure()
# plt.scatter(Mtot_plot, emax_plot, c=n_plot, norm=LogNorm())
# plt.xscale('log')
# plt.colorbar(label='N events')
# plt.xlabel('log Mtot = m1 + m2')
# plt.ylabel('emax')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax_vary/Mtot_log.png"
# plt.savefig(name, format='png', dpi=1000)

# plt.figure()
# plt.scatter(Mc_plot, emax_plot, c=n_plot)
# plt.colorbar(label='N events')
# plt.xlabel('Mc')
# plt.ylabel('emax')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax_vary/Mc.png"
# plt.savefig(name, format='png', dpi=1000)

# plt.figure()
# plt.scatter(Mc_plot, emax_plot, c=n_plot)
# plt.xscale('log')
# plt.colorbar(label='N events')
# plt.xlabel('log Mc')
# plt.ylabel('emax')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax_vary/Mc_log.png"
# plt.savefig(name, format='png', dpi=1000)


# fig = plt.figure()
# ax=fig.add_subplot(projection='3d')

# ax.scatter(Mtot_plot, Mc_plot, emax_plot, c=n_plot, cmap='viridis', alpha=0.8)
# ax.set_xlabel('Mtotal ')
# ax.set_ylabel('Mc')
# ax.set_zlabel('emax')
# name="dL_joint_fit_results_emax_vary/3D_plot.png"
# plt.savefig(name, format='png', dpi=1000)

# fig = plt.figure()
# ax=fig.add_subplot(projection='3d')
# ax.plot_trisurf(Mtot_plot, Mc_plot, emax_plot, cmap='viridis')
# ax.set_xlabel('Mtotal ')
# ax.set_ylabel('Mc')
# ax.set_zlabel('emax')
# name="dL_joint_fit_results_emax_vary/3D_surface_plot.png"
# plt.savefig(name, format='png', dpi=1000)

rc('text', usetex=True)

d2 = np.linspace(0, 3, 1000)
fun1 = 1 / (1+(d2/0.7)**(3+35*np.tanh(d2/0.7)))
fun2 = 1 / (1+(d2/0.7)**(2+2*np.tanh(d2/0.7)))

plt.close('all')
fig, ax = plt.subplots()
ax.plot(d2, fun1, 'r-')
ax.hlines(0.5, 0, 0.7, color='tab:blue', linestyle='dashed')
ax.vlines(0.7, 0, 0.5, color='tab:blue', linestyle='dashed')
ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 2.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1], fontsize=25)
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
plt.savefig('plots/pdet_ideal.pdf', format='pdf', dpi=1000)

fig, ax = plt.subplots()
ax.plot(d2, fun2, 'r-')
ax.hlines(0.5, 0, 0.7, color='tab:blue', linestyle='dashed')
ax.vlines(0.7, 0, 0.5, color='tab:blue', linestyle='dashed')
ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 2.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1], fontsize=25)
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
plt.savefig('plots/pdet_real.pdf', format='pdf', dpi=1000)




