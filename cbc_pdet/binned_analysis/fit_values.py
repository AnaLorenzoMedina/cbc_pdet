# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:54:45 2022

@author: Ana
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')

rc('text', usetex=True)

zmid_1=np.loadtxt('maximization_results/zmid_1.dat')

zmid_2=np.loadtxt('maximization_results/zmid_2.dat')

maxL_1=np.loadtxt('maximization_results/maxL_1.dat')

maxL_2=np.loadtxt('maximization_results/maxL_2.dat')

a_1=np.loadtxt('maximization_results/a_1.dat')

zeta_1=np.loadtxt('maximization_results/zeta_1.dat')

delta_2=np.loadtxt('maximization_results/delta_2.dat')

gamma_2=np.loadtxt('maximization_results/gamma_2.dat')


n_points=np.loadtxt('maximization_results/n_points.dat')

#total log likelihoods

tot_1 = maxL_1.sum()
tot_2 = maxL_2.sum()

name = 'maximization_results/total_likelihoods.dat'
data = np.array([tot_1, tot_2])
header = "tot_log_like_1 , tot_log_like_2  "
np.savetxt(name, data, header=header, fmt='%10.3f')


maxL_1[maxL_1==0]=np.nan
maxL_2[maxL_2==0]=np.nan

a_1[a_1==0]=np.nan
zeta_1[zeta_1==0]=np.nan

zmid_1[zmid_1==0]=np.nan
zmid_2[zmid_2==0]=np.nan

gamma_2[gamma_2==0]=np.nan
delta_2[delta_2==0]=np.nan

n_points[n_points==0]=np.nan

x=delta_2.flatten()
y=gamma_2.flatten()

plt.figure()
plt.plot(x,y,'.')
plt.semilogx()
plt.xlim(1e-3, 1e4)


'''
plt.figure()
plt.imshow(maxL_1.T, cmap='viridis', origin='lower', norm=LogNorm())
plt.colorbar()

plt.figure()
plt.imshow(maxL_2.T, cmap='viridis', origin='lower', norm=LogNorm())
plt.colorbar()
'''

tick = np.array((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14))

mmin =2; mmax = 100
nbin1 = nbin2 = 14

m1_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin1+1) , 1)
m2_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin2+1) , 1)


plt.figure()
plt.imshow((maxL_2.T-maxL_1.T)/n_points.T, cmap='seismic', origin='lower', extent=[0, nbin1,0, nbin2])
plt.clim(-0.2,0.2) 
plt.colorbar()
plt.title(r'$(\max \ln \mathcal{L}_2 - \max \ln \mathcal{L}_1)/n_f$', fontsize=14)
plt.xlabel(r'$m_1$   $(M_{\odot})$', fontsize=14)
plt.ylabel(r'$m_2$   $(M_{\odot})$', fontsize=14)
plt.xticks(ticks=tick, labels=m1_bin, fontsize=8)
plt.yticks(ticks=tick, labels=m2_bin)

name="general_plots/plot_like.png"
plt.savefig(name, format='png', dpi=100, bbox_inches="tight")

name="general_plots/plot_like.pdf"
plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")

'''
plt.figure()
plt.imshow(a_1.T, cmap='viridis', origin='lower')
plt.colorbar()
plt.clim(0,5.3)

plt.figure()
plt.imshow(zeta_1.T, cmap='viridis', origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(gamma_2.T, cmap='viridis', origin='lower', norm=LogNorm())
plt.colorbar()

plt.figure()
plt.imshow(delta_2.T, cmap='viridis', origin='lower')
plt.colorbar()


plt.figure()
plt.imshow(zmid_1.T, cmap='viridis', origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(zmid_2.T, cmap='viridis', origin='lower')
plt.colorbar()
'''

'''

fig=plt.figure(figsize=(12,6))
#plt.subplots_adjust(hspace=0.325, wspace=0.250)

minmin=np.min([np.nanmin(zmid_1.T), np.nanmin(zmid_2.T)])
maxmax=np.max([np.nanmax(zmid_1.T), np.nanmax(zmid_2.T)])

ax1=plt.subplot(121)  
im=ax1.imshow(zmid_1.T, cmap='viridis', origin='lower', vmin=minmin, vmax=maxmax, extent=[0, nbin1,0, nbin2])
ax1.set_title(r'$z_{mid_1}$', fontsize=18)
ax1.set_xlabel(r'$m_1$   $(M_{\odot})$', fontsize=14)
ax1.set_ylabel(r'$m_2$   $(M_{\odot})$', fontsize=14)
ax1.set_xticks(ticks=tick, labels=m1_bin, fontsize=8)
ax1.set_yticks(ticks=tick, labels=m2_bin)

ax2=plt.subplot(122)  
img=ax2.imshow(zmid_2.T, cmap='viridis', origin='lower', vmin=minmin, vmax=maxmax, extent=[0, nbin1,0, nbin2])
ax2.set_title(r'$z_{mid_2}$', fontsize=18)
ax2.set_xlabel(r'$m_1$   $(M_{\odot})$', fontsize=14)
ax2.set_ylabel(r'$m_2$   $(M_{\odot})$', fontsize=14)
ax2.set_xticks(ticks=tick, labels=m1_bin, fontsize=8)
ax2.set_yticks(ticks=tick, labels=m2_bin)

axin=inset_axes(ax2, width="5%", height="100%", loc="lower left",
                bbox_to_anchor=(1.08, 0., 1, 1), bbox_transform=ax2.transAxes,
                borderpad=0)
fig.colorbar(img, cax=axin)

name="general_plots/plot_zmid.png"
plt.savefig(name, format='png', dpi=100, bbox_inches="tight")

name="general_plots/plot_zmid.pdf"
plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")
'''