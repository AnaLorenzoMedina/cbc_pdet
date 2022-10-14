# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:04:18 2022

@author: Ana
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
from scipy.integrate import quad, dblquad
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
from scipy import interpolate
import os
import errno

def fun_pdf(m1, m2):
    if m2 > m1:
        return 0
    mmin = 2 ; mmax = 100
    alpha = -2.35 ; beta = 1
    m1_norm = (1. + alpha) / (mmax ** (1. + alpha) - mmin ** (1. + alpha))
    m2_norm = (1. + beta) / (m1 ** (1. + beta) - mmin ** (1. + beta))
    return m1**alpha * m2**beta * m1_norm * m2_norm

def m1_integrand_diag(m1, alpha, beta, mmin, mmax, m1m):
    return m1**alpha * (m1**(1. + beta) - m1m**(1. + beta)) / (m1**(1. + beta) - mmin**(1. + beta))

#%%
plt.close('all')

rc('text', usetex=True)

#we create folders needed to save results

try:
    os.mkdir('z_data')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('z_binned')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('Ntot_bin')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('mean_mass_pdf')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('mean_z_pdf')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('general_plots')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('zm_plots_dic')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
        

f = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')


variables = f['injections'].keys()
attributes = f.attrs.keys()

m1 = f["injections/mass1_source"][:]
m2 = f["injections/mass2_source"][:]

m_pdf = f["injections/mass1_source_mass2_source_sampling_pdf"][:]

z = f["injections/redshift"][:]
z_pdf = f["injections/redshift_sampling_pdf"][:]

index = np.argsort(z)
try_z = z[index]
try_zpdf = z_pdf[index]

zpdf_interp = interpolate.interp1d(try_z, try_zpdf)

total = f.attrs['total_generated']
far_pbbh = f["injections/far_pycbc_bbh"][:]  # rename this far_pbbh
far_gstlal = f["injections/far_gstlal"][:]
far_mbta = f["injections/far_mbta"][:]
far_pfull = f["injections/far_pycbc_hyperbank"][:]

mmin = 2 ; mmax = 100
alpha = -2.35 ; beta = 1

m1_norm = (1. + alpha) / (mmax**(1. + alpha) - mmin**(1. + alpha))


b = 7          #bin length of masses
b2 = 7
bz = 0.1       #bin length of z
thr = 1      #threshold far

#nbin1 = int((max(m1)-0)/b)
#nbin2 = int((max(m2)-0)/b2)
nbinz = round((max(z)-0)/bz)

nbin1=14
nbin2=14

#m1_bin = np.arange(2,100+b,b)
#m2_bin = np.arange(2,100+b2,b2)

m1_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin1+1) , 1)
m2_bin = np.round(np.logspace(np.log10(mmin), np.log10(mmax), nbin2+1) , 1)

size=m1_bin[1:]-m1_bin[:-1]



mbin_data = np.array([[{'m1': [],
                      'm2': [],
                      'z': [],
                      'far_pbbh': [],
                      'far_gstlal': [],
                      'far_mbta': [],
                      'far_pfull': [],
                      'm_pdf': [],
                      'z_pdf': []
                      ,} for j in range(nbin2)] for i in range(nbin1)])

N = mbin_data.shape[0]
N2 = mbin_data.shape[1]

for i in range(len(m1)):
    for k1 in range(N):
        if m1[i]>=m1_bin[k1] and m1[i]<m1_bin[k1+1]:
            for k2 in range(N2):
                if m2[i]>=m2_bin[k2] and m2[i]<m2_bin[k2+1]:
                    mbin_data[k1,k2]['m1'].append(m1[i])
                    mbin_data[k1,k2]['m2'].append(m2[i])
                    mbin_data[k1,k2]['z'].append(z[i])
                    mbin_data[k1,k2]['far_pbbh'].append(far_pbbh[i])
                    mbin_data[k1,k2]['far_gstlal'].append(far_gstlal[i])
                    mbin_data[k1,k2]['far_mbta'].append(far_mbta[i])
                    mbin_data[k1,k2]['far_pfull'].append(far_pfull[i])
                    mbin_data[k1,k2]['m_pdf'].append(m_pdf[i])
                    mbin_data[k1,k2]['z_pdf'].append(z_pdf[i])


for i in range(N):
    for j in range(N2):
        m1_b=np.array(mbin_data[i,j]['m1'])
        m2_b=np.array(mbin_data[i,j]['m2'])
        
        mbin_data[i,j]['Ntot_bin']=len(m1_b)
        
        if len(m1_b)!=0:
        
            quad_fun = lambda y,x: fun_pdf(x, y)
            quad_fun_diag = lambda x: m1_integrand_diag(x, alpha, beta, mmin, mmax, m1_bin[i])
            
            if m2_bin[j] == m1_bin[i] and m2_bin[j+1] == m1_bin[i+1]:
                mbin_data[i,j]['mean_m_pdf'] = m1_norm * quad(quad_fun_diag, m1_bin[i], m1_bin[i+1])[0] 

            else:
                mbin_data[i,j]['mean_m_pdf']=dblquad(quad_fun, m1_bin[i],m1_bin[i+1], m2_bin[j],m2_bin[j+1], epsabs=1e-11)[0]
            
        else:
            mbin_data[i,j]['mean_m_pdf']=np.nan
        # precalculate detection condition
        found_pbbh = np.array(mbin_data[i,j]['far_pbbh']) <= thr
        found_gstlal = np.array(mbin_data[i,j]['far_gstlal']) <= thr
        found_mbta = np.array(mbin_data[i,j]['far_mbta']) <= thr
        found_pfull = np.array(mbin_data[i,j]['far_pfull']) <= thr
        found_any = found_pbbh | found_gstlal | found_mbta | found_pfull
        mbin_data[i,j]['n_det'] = found_any.sum()
        mbin_data[i,j]['z_det'] = np.array(mbin_data[i,j]['z'])[found_any]
        mbin_data[i,j]['z_pdf_det'] = np.array(mbin_data[i,j]['z_pdf'])[found_any]
        
#m detected with far>=1
detections=np.array([[mbin_data[i,j]['n_det'] for j in range(N2)] for i in range(N)]).T

#mean pdfs
mean_pdf = np.array([[mbin_data[i,j]['mean_m_pdf'] for j in range(N2)] for i in range(N)]).T

m_prob = detections/(mean_pdf * total)

#%%
#####  2D plot ####

tick= np.array((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14))
color = 'viridis'

plt.figure()
plt.imshow(m_prob, cmap=color, extent=[0, nbin1,0, nbin2], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlabel(r'$m_1$   $(M_{\odot})$', fontsize=15)
plt.ylabel(r'$m_2$   $(M_{\odot})$', fontsize=15)
plt.title(r'$P_{det} (m_1,m_2)$', fontsize=15)
plt.xticks(ticks=tick, labels=m1_bin, fontsize=8)
plt.yticks(ticks=tick, labels=m2_bin)

name="general_plots/2dhist_z.png"
plt.savefig(name, format='png', dpi=100, bbox_inches="tight")

#name="LATEX_IMAGES/2dhist_z.pdf"
#plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")


'''
#  z histogram

z_bin = np.arange(min(z), max(z)+bz, bz)

zbin_data = np.array([{ 'z': [],
                      'far_pbbh': [],
                      'far_gstlal': [],
                      'far_mbta': [],
                      'far_pfull': [],
                      'z_pdf': []
                      ,} for i in range(nbinz)])

Nz = zbin_data.shape[0]

for i in range(len(z)):
    for k in range(len(z_bin)-1):
        if z[i]>=z_bin[k] and z[i]<z_bin[k+1]:
            zbin_data[k]['z'].append(z[i])
            zbin_data[k]['far_pbbh'].append(far_pbbh[i])
            zbin_data[k]['far_gstlal'].append(far_gstlal[i])
            zbin_data[k]['far_mbta'].append(far_mbta[i])
            zbin_data[k]['far_pfull'].append(far_pfull[i])
            zbin_data[k]['z_pdf'].append(z_pdf[i])


for i in range(Nz):
    zbin_data[i]['mean_z_pdf'] = np.average(zbin_data[i]['z_pdf'])
    zbin_data[i]['z_det'] = len(np.array(zbin_data[i]['z'])[np.logical_or(np.logical_or(np.logical_or(np.array(zbin_data[i]['far_pbbh'])<=1, np.array(zbin_data[i]['far_gstlal'])<=1), np.array(zbin_data[i]['far_mbta'])<=1), np.array(zbin_data[i]['far_pfull'])<=1)])
                   
    
#z detected with far>=1
z_detections = np.array([zbin_data[i]['z_det'] for i in range(Nz)])

#mean pdfs
mean_z_pdf = np.array([zbin_data[i]['mean_z_pdf'] for i in range(Nz)])

z_prob = z_detections/(mean_z_pdf*bz*total)

#%%
color='#2a788e'

fig, ax1 = plt.subplots()
ax1.bar(z_bin[:-1], z_prob, align='edge', width=np.diff(z_bin), color=color)
ax1.set_xlabel(r'$z$', fontsize=14)
ax1.set_ylabel(r'$P_{det}$', fontsize=14)
ax1.set_title(r'$P_{det}(z)$', fontsize=14)

ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax1, [0.35,0.3,0.58,0.6])
ax2.set_axes_locator(ip)

ax2.bar(z_bin[:-1], z_prob, align='edge', width=np.diff(z_bin), color=color, alpha=0.5)
ax2.set_yscale('log')
#ax2.set_xlabel(r'$z$')
#ax2.set_ylabel(r'$P_{det}$')

name="general_plots/prob_z.png"
plt.savefig(name, format='png', dpi=100, bbox_inches="tight")

name="LATEX_IMAGES/prob_z.pdf"
plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")

#%%
'''
##################### z histograms for each mass bin ##############

for i in range(N):
    for j in range(N2):
        if mbin_data[i,j]['n_det']!=0:
            
            print(i,j)
            
            Ntot_bin = mbin_data[i,j]['Ntot_bin']
            
            # detected values
            zdeti = mbin_data[i,j]['z_det']
            zpdfdeti = mbin_data[i,j]['z_pdf_det']
            
            # all values detected or not
            z_i = mbin_data[i,j]['z']
            z_pdf_i = mbin_data[i,j]['z_pdf']
            mean_m_pdf_i = mbin_data[i,j]['mean_m_pdf']
            
            #zm_bin=np.arange(min(z_i),max(z_i)+bz,bz)
            zm_bin=np.linspace(min(z_i),max(z_i),20)
            delta_z = zm_bin[1]-zm_bin[0]

            zmbin_data=np.array([{'zdet': [],
                                  'zdet_pdf': [],
                                  'm_pdf': [],
                                  'z': [],
                                  'z_pdf': []
                                  ,} for i in range(len(zm_bin)-1)])

            Nzm = zmbin_data.shape[0]  # number of z bins in this mass bin
            
            
            for u in range(len(zdeti)):
                for k in range(len(zm_bin)-1):
                    if zdeti[u] >= zm_bin[k] and zdeti[u] < zm_bin[k+1]:
                        zmbin_data[k]['zdet'].append(zdeti[u])
                        zmbin_data[k]['zdet_pdf'].append(zpdfdeti[u])
                        
            for u in range(len(z_i)):
                for k in range(len(zm_bin)-1):
                    if z_i[u] >= zm_bin[k] and z_i[u] < zm_bin[k+1]:
                        zmbin_data[k]['z'].append(z_i[u])
                        zmbin_data[k]['z_pdf'].append(z_pdf_i[u])


            for u in range(Nzm):
                quad_fun = lambda x: zpdf_interp(x)
                zmbin_data[u]['mean_z_pdf'] = quad(quad_fun, zm_bin[u], zm_bin[u+1])[0] 
                #zmbin_data[u]['mean_z_pdf'] = np.average(zmbin_data[u]['z_pdf'])
                zmbin_data[u]['n_det'] = len(np.array(zmbin_data[u]['zdet']))

            #z detected with far>=1
            zm_detections = np.array([zmbin_data[u]['n_det'] for u in range(Nzm)])
            
            z_det_hist=np.array(zdeti)
            
            # mean pdfs
            mean_zm_pdf = np.array([zmbin_data[u]['mean_z_pdf'] for u in range(Nzm)])
            
            #mass bins on the diagonal have half of the number of injections that there should be
            #if i==j:
            #    zm_detections=zm_detections*2
            
            zm_prob = zm_detections/(mean_zm_pdf * mean_m_pdf_i * total )
            
            mid = (zm_bin[:-1]+zm_bin[1:])/2
            
            nonzero = zm_detections > 0
            
            plt.figure()
            #mid=np.array([(zm_bin[u]+zm_bin[u+1])/2 for u in range(Nzm)])
            plt.plot(mid, zm_prob, 'o')
            plt.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
            #plt.yscale('log')
            #plt.ylim(0.001,2)
            plt.xlabel(r'$z$', fontsize=14)
            plt.ylabel(r'$P_{det}(z)$', fontsize=14)
            plt.title(r'$P_{det}(z)$ $m_1$ %.1f-%.1f M$_{\odot}$ \& $m_2$ %.1f-%.1f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
            name=f"zm_plots_dic/{i}{j}.png"
            plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
            
            plt.close()
            
            name=f'z_data/{i}{j}_data.dat'
            data = np.column_stack((zdeti, zpdfdeti))
            header = "det_z, det_zpdf, Ntot_bin"
            np.savetxt(name, data, header=header)
            
            name=f'z_binned/{i}{j}_data.dat'
            data = np.column_stack((mid, zm_prob, zm_detections))
            header = "z_mid, z_binned_prob, zm_detections"
            np.savetxt(name, data, header=header)
            
            name=f'Ntot_bin/{i}{j}_data.dat'
            data = np.array([Ntot_bin])
            header = "Ntot_bin"
            np.savetxt(name, data, header=header)
            
            name=f'mean_mass_pdf/{i}{j}_data.dat'
            data = np.array([mean_m_pdf_i])
            header = "mean_zm_pdf"
            np.savetxt(name, data, header=header)
            
            name=f'mean_z_pdf/{i}{j}_data.dat'
            data = np.array([mean_zm_pdf])
            header = "mean_zm_pdf"
            np.savetxt(name, data, header=header)
            

''' 
              
#%%

######### SUBPLOTS 

#for plotting results

i,j=13,10

Ntot_bin = mbin_data[i,j]['Ntot_bin']

# detected values
zdeti = mbin_data[i,j]['z_det']
zpdfdeti = mbin_data[i,j]['z_pdf_det']

# all values detected or not
z_i = mbin_data[i,j]['z']
z_pdf_i = mbin_data[i,j]['z_pdf']
mean_m_pdf_i = mbin_data[i,j]['mean_m_pdf']

#zm_bin=np.arange(min(z_i),max(z_i)+bz,bz)
zm_bin=np.linspace(min(z_i),max(z_i),20)

zmbin_data=np.array([{'zdet': [],
                      'zdet_pdf': [],
                      'm_pdf': [],
                      'z': [],
                      'z_pdf': []
                      ,} for i in range(len(zm_bin)-1)])

Nzm = zmbin_data.shape[0]  # number of z bins in this mass bin


for u in range(len(zdeti)):
    for k in range(len(zm_bin)-1):
        if zdeti[u] >= zm_bin[k] and zdeti[u] < zm_bin[k+1]:
            zmbin_data[k]['zdet'].append(zdeti[u])
            zmbin_data[k]['zdet_pdf'].append(zpdfdeti[u])
            
for u in range(len(z_i)):
    for k in range(len(zm_bin)-1):
        if z_i[u] >= zm_bin[k] and z_i[u] < zm_bin[k+1]:
            zmbin_data[k]['z'].append(z_i[u])
            zmbin_data[k]['z_pdf'].append(z_pdf_i[u])


for u in range(Nzm):
    quad_fun = lambda x: zpdf_interp(x)
    zmbin_data[u]['mean_z_pdf'] = quad(quad_fun, zm_bin[u], zm_bin[u+1])[0] 
    #zmbin_data[u]['mean_z_pdf'] = np.average(zmbin_data[u]['z_pdf'])
    zmbin_data[u]['n_det'] = len(np.array(zmbin_data[u]['zdet']))

#z detected with far>=1
zm_detections = np.array([zmbin_data[u]['n_det'] for u in range(Nzm)])

z_det_hist=np.array(zdeti)

# mean pdfs
mean_zm_pdf = np.array([zmbin_data[u]['mean_z_pdf'] for u in range(Nzm)])

#mass bins on the diagonal have half of the number of injections that there should be
#if i==j:
#    zm_detections=zm_detections*2

zm_prob = zm_detections/(mean_zm_pdf * mean_m_pdf_i  * total )

mid = (zm_bin[:-1]+zm_bin[1:])/2

nonzero = zm_detections > 0




if i==0 and j==0:
    plt.figure(figsize=(11,9))
    plt.subplots_adjust(hspace=0.325, wspace=0.250)

    
    zax1=plt.subplot(221)   
    zax1.plot(mid, zm_prob, 'o')
    zax1.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    #plt.yscale('log')
    #plt.ylim(0.001,2)
    zax1.set_xlabel(r'$z$', fontsize=14)
    zax1.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax1.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
    
    a1 = plt.axes([0,0,1,1])
    ip = InsetPosition(zax1, [0.4,0.35,0.58,0.6])
    a1.set_axes_locator(ip)

    a1.plot(mid, zm_prob, 'o')
    a1.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    a1.set_yscale('log')
    

    
if i==6 and j==5:
    zax2=plt.subplot(222)   
    zax2.plot(mid, zm_prob, 'o')
    zax2.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    #plt.yscale('log')
    #plt.ylim(0.001,2)
    zax2.set_xlabel(r'$z$', fontsize=14)
    zax2.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    zax2.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
    
    a2 = plt.axes([0,0,1,1])
    ip = InsetPosition(zax2, [0.4,0.35,0.58,0.6])
    a2.set_axes_locator(ip)

    a2.plot(mid, zm_prob, 'o')
    a2.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    a2.set_yscale('log')
    

    
if i==9 and j==9:
    ax3=plt.subplot(223)   
    ax3.plot(mid, zm_prob, 'o')
    ax3.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    #plt.yscale('log')
    #plt.ylim(0.001,2)
    ax3.set_xlabel(r'$z$', fontsize=14)
    ax3.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    ax3.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
   
    a3 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax3, [0.4,0.35,0.58,0.6])
    a3.set_axes_locator(ip)

    a3.plot(mid, zm_prob, 'o')
    a3.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    a3.set_yscale('log')
    


if i==13 and j==10:
    ax4=plt.subplot(224)   
    ax4.plot(mid, zm_prob, 'o')
    ax4.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    #plt.yscale('log')
    #plt.ylim(0.001,2)
    ax4.set_xlabel(r'$z$', fontsize=14)
    ax4.set_ylabel(r'$P_{det}(z)$', fontsize=14)
    ax4.set_title(r'$m_1:$ %.0f-%.0f M$_{\odot}$ \& $m_2:$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
   
    a4 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax4, [0.4,0.35,0.58,0.6])
    a4.set_axes_locator(ip)

    a4.plot(mid, zm_prob, 'o')
    a4.errorbar(mid[nonzero], zm_prob[nonzero], yerr=zm_prob[nonzero]/np.sqrt(zm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    a4.set_yscale('log')
    
    name="general_plots/4zm.png"
    plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
    name="general_plots/4zm.pdf"
    plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")
    


if i==0 and j==0:
    plt.figure(figsize=(11,7))
    plt.subplots_adjust(hspace=0.325, wspace=0.250)

    ax1=plt.subplot(221)   
    ax1.hist(z_det_hist, zm_bin, density=True, alpha=0.5)
    ax1.errorbar(np.array(mid)[np.where(n>0)[0]], n[n!=0], yerr=n[n>0]/np.sqrt(zm_detections[n>0]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    ax1.plot(mid, mean_zm_pdf, 'r-', label=r'$p(z)$')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$z$', fontsize=14)
    ax1.set_ylabel(r'$\log{n_f}$', fontsize=14)
    ax1.set_title(r'$m_1$ %.0f-%.0f M$_{\odot}$ \& $m_2$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
    
if i==5 and j==3:
    ax2=plt.subplot(222)   
    ax2.hist(z_det_hist, zm_bin, density=True, alpha=0.5)
    ax2.errorbar(np.array(mid)[np.where(n>0)[0]], n[n!=0], yerr=n[n>0]/np.sqrt(zm_detections[n>0]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    ax2.plot(mid, mean_zm_pdf, 'r-', label=r'$p(z)$')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$z$', fontsize=14)
    ax2.set_ylabel(r'$\log{n_f}$', fontsize=14)
    ax2.set_title(r'$m_1$ %.0f-%.0f M$_{\odot}$ \& $m_2$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
    
if i==9 and j==9:
    ax3=plt.subplot(223)   
    ax3.hist(z_det_hist, zm_bin, density=True, alpha=0.5)
    ax3.errorbar(np.array(mid)[np.where(n>0)[0]], n[n!=0], yerr=n[n>0]/np.sqrt(zm_detections[n>0]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    ax3.plot(mid, mean_zm_pdf, 'r-', label=r'$p(z)$')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$z$', fontsize=14)
    ax3.set_ylabel(r'$\log{n_f}$', fontsize=14)
    ax3.set_title(r'$m_1$ %.0f-%.0f M$_{\odot}$ \& $m_2$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
    
if i==13 and j==11:
    ax4=plt.subplot(224)   
    ax4.hist(z_det_hist, zm_bin, density=True, alpha=0.5)
    ax4.errorbar(np.array(mid)[np.where(n>0)[0]], n[n!=0], yerr=n[n>0]/np.sqrt(zm_detections[n>0]), fmt="none", color="k", capsize=2, elinewidth=0.4)
    ax4.plot(mid, mean_zm_pdf, 'r-', label=r'$p(z)$')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.set_xlabel(r'$z$', fontsize=14)
    ax4.set_ylabel(r'$\log{n_f}$', fontsize=14)
    ax4.set_title(r'$m_1$ %.0f-%.0f M$_{\odot}$ \& $m_2$ %.0f-%.0f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
    
    name="general_plots/4zbinned.png"
    plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
    name="general_plots/4zbinned.pdf"
    plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")

'''
