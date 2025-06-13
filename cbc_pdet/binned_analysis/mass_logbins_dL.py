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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
from scipy import interpolate
import os
import errno

def fun_pdf(m1, m2):
    if m2 > m1:
        return 0
    mmin = 1 ; mmax = 1000
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
    os.mkdir('dL_binned')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('general_plots_dL')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir('dLm_plots_dic')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
        

file = h5py.File('../samples-rpo4a_v2_20250220153231UTC-1366933504-23846400.hdf', 'r')

attributes = file.attrs.keys()

m1 = file['events'][:]['mass1_source']
m2 = file['events'][:]['mass2_source']

m1_pdf = np.exp(file["events"][:]["lnpdraw_mass1_source"])
m2_pdf = np.exp(file["events"][:]["lnpdraw_mass2_source_GIVEN_mass1_source"])
m_pdf = m1_pdf * m2_pdf

z = file['events'][:]['z']
dL = file['events'][:]['luminosity_distance']

z_pdf = np.exp(file["events"][:]["lnpdraw_z"])

total = file.attrs['total_generated']
far_pbbh = file["events"][:]["pycbc_far"]
far_gstlal = file["events"][:]["gstlal_far"]
far_mbta = file["events"][:]["mbta_far"]
far_cwb = file["events"][:]["cwb-bbh_far"]

mmin = 1 ; mmax = 1000
alpha = -2.35 ; beta = 1

m1_norm = (1. + alpha) / (mmax**(1. + alpha) - mmin**(1. + alpha))

H0 = 67.9 #km/sMpc
c = 3e5 #km/s
omega_m = 0.3065
A = np.sqrt(omega_m*(1+z)**3+1-omega_m)

dC_dif = (c/ H0) / A
#second term is equal to comoving distance 
dL_dif = (1 + z) * dC_dif + dL / (1 + z)

dL_pdf = z_pdf/dL_dif

index = np.argsort(dL)
try_dL = dL[index]
try_dLpdf = dL_pdf[index]

dLpdf_interp = interpolate.interp1d(try_dL, try_dLpdf)

bdL = 1000       #bin length of z
thr = 1      #threshold far

#nbin1 = int((max(m1)-0)/b)
#nbin2 = int((max(m2)-0)/b2)
nbindL = int((np.max(dL)-np.min(dL))/bdL)

nbin1=15
nbin2=15

#m1_bin = np.arange(2,100+b,b)
#m2_bin = np.arange(2,100+b2,b2)

m1_bin = np.round(np.logspace(np.log10(1), np.log10(1000), nbin1+1), 1)
m2_bin = np.round(np.logspace(np.log10(1), np.log10(1000), nbin2+1), 1)

size=m1_bin[1:]-m1_bin[:-1]



mbin_data = np.array([[{'m1': [],
                      'm2': [],
                      'dL': [],
                      'far_pbbh': [],
                      'far_gstlal': [],
                      'far_mbta': [],
                      'far_cwb': [],
                      'm_pdf': [],
                      'dL_pdf': []
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
                    mbin_data[k1,k2]['dL'].append(dL[i])
                    mbin_data[k1,k2]['far_pbbh'].append(far_pbbh[i])
                    mbin_data[k1,k2]['far_gstlal'].append(far_gstlal[i])
                    mbin_data[k1,k2]['far_mbta'].append(far_mbta[i])
                    mbin_data[k1,k2]['far_cwb'].append(far_cwb[i])
                    mbin_data[k1,k2]['m_pdf'].append(m_pdf[i])
                    mbin_data[k1,k2]['dL_pdf'].append(z_pdf[i])

Ntot_inbin = np.zeros([nbin1,nbin2])
mean_mpdf_inbin = np.zeros([nbin1,nbin2])

for i in range(N):
    for j in range(N2):
        m1b_min, m1b_max = m1_bin[i], m1_bin[i+1]
        m2b_min, m2b_max = m2_bin[j], m2_bin[j+1]
        bin_area = (m1b_max - m1b_min) * (m2b_max - m2b_min)
        
        m1_b=np.array(mbin_data[i,j]['m1'])
        m2_b=np.array(mbin_data[i,j]['m2'])
        m_pdf_b = np.array(mbin_data[i,j]['m_pdf'])
        
        mbin_data[i,j]['Ntot_bin'] = len(m1_b)
        Ntot_inbin[i,j] = len(m1_b)
        
        if len(m1_b)!=0:
            '''
            quad_fun = lambda y,x: fun_pdf(x, y)
            quad_fun_diag = lambda x: m1_integrand_diag(x, alpha, beta, mmin, mmax, m1_bin[i])
            '''
 
            if m2_bin[j] == m1_bin[i] and m2_bin[j+1] == m1_bin[i+1]:
                bin_area = (m1b_max - m1b_min) * (m2b_max - m2b_min) / 2
                
            else:
                bin_area = (m1b_max - m1b_min) * (m2b_max - m2b_min)
                
            mbin_data[i,j]['mean_m_pdf'] = np.average(m_pdf_b) * bin_area
        
        else:
            mbin_data[i,j]['mean_m_pdf']=np.nan
            
        mean_mpdf_inbin[i,j] = mbin_data[i,j]['mean_m_pdf']
        
        # precalculate detection condition
        found_pbbh = np.array(mbin_data[i,j]['far_pbbh']) <= thr
        found_gstlal = np.array(mbin_data[i,j]['far_gstlal']) <= thr
        found_mbta = np.array(mbin_data[i,j]['far_mbta']) <= thr
        found_cwb = np.array(mbin_data[i,j]['far_cwb']) <= thr
        found_any = found_pbbh | found_gstlal | found_mbta | found_cwb
        mbin_data[i,j]['n_det'] = found_any.sum()
        mbin_data[i,j]['dL_det'] = np.array(mbin_data[i,j]['dL'])[found_any]
        mbin_data[i,j]['dL_pdf_det'] = np.array(mbin_data[i,j]['dL_pdf'])[found_any]
        
np.savetxt('Ntot.dat', Ntot_inbin)
np.savetxt('mean_mpdf.dat', mean_mpdf_inbin)


#m detected with far>=1
detections=np.array([[mbin_data[i,j]['n_det'] for j in range(N2)] for i in range(N)]).T

#mean pdfs
mean_pdf = np.array([[mbin_data[i,j]['mean_m_pdf'] for j in range(N2)] for i in range(N)]).T

m_prob = detections/(mean_pdf * total)


#%%
#####  2D plot ####

tick= np.array((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
color = 'viridis'

plt.figure()
plt.imshow(m_prob, cmap=color, extent=[0, nbin1,0, nbin2], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlabel(r'$m_1$   $(M_{\odot})$', fontsize=15)
plt.ylabel(r'$m_2$   $(M_{\odot})$', fontsize=15)
plt.title(r'$P_{det} (m_1,m_2)$', fontsize=15)
plt.xticks(ticks=tick, labels=m1_bin, fontsize=8)
plt.yticks(ticks=tick, labels=m2_bin)

name="general_plots_dL/2dhist_dL.png"
plt.savefig(name, format='png', dpi=100, bbox_inches="tight")

#name="LATEX_IMAGES/2dhist_dL.pdf"
#plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")


#  dL histogram

#dL_bin = np.arange(min(dL), max(dL)+bdL, bdL)
dL_bin = np.linspace(np.min(dL), np.min(dL) + bdL*nbindL, nbindL+1)

dLbin_data = np.array([{ 'dL': [],
                      'far_pbbh': [],
                      'far_gstlal': [],
                      'far_mbta': [],
                      'far_cwb': [],
                      'dL_pdf': []
                      ,} for i in range(nbindL)])

NdL = dLbin_data.shape[0]

for i in range(len(dL)):
    for k in range(len(dL_bin)-1):
        if dL[i]>=dL_bin[k] and dL[i]<dL_bin[k+1]:
            dLbin_data[k]['dL'].append(dL[i])
            dLbin_data[k]['far_pbbh'].append(far_pbbh[i])
            dLbin_data[k]['far_gstlal'].append(far_gstlal[i])
            dLbin_data[k]['far_mbta'].append(far_mbta[i])
            dLbin_data[k]['far_cwb'].append(far_cwb[i])
            dLbin_data[k]['dL_pdf'].append(dL_pdf[i])


for i in range(NdL):
    dLbin_data[i]['mean_dL_pdf'] = np.average(dLbin_data[i]['dL_pdf'])
    dLbin_data[i]['dL_det'] = len(np.array(dLbin_data[i]['dL'])[np.logical_or(np.logical_or(np.logical_or(np.array(dLbin_data[i]['far_pbbh'])<=1, np.array(dLbin_data[i]['far_gstlal'])<=1), np.array(dLbin_data[i]['far_mbta'])<=1), np.array(dLbin_data[i]['far_cwb'])<=1)])
                   
    
#z detected with far>=1
dL_detections = np.array([dLbin_data[i]['dL_det'] for i in range(NdL)])

#mean pdfs
mean_dL_pdf = np.array([dLbin_data[i]['mean_dL_pdf'] for i in range(NdL)])

dL_prob = dL_detections/(mean_dL_pdf*bdL*total)

#%%
color='#2a788e'

fig, ax1 = plt.subplots()
ax1.bar(dL_bin[:-1], dL_prob, align='edge', width=np.diff(dL_bin), color=color)
ax1.set_xlabel(r'$dL$', fontsize=14)
ax1.set_ylabel(r'$P_{det}$', fontsize=14)
ax1.set_title(r'$P_{det}(dL)$', fontsize=14)

ax2 = inset_axes(ax1, width="58%", height="60%", loc='lower left',
                 bbox_to_anchor=(0.35, 0.3, 1, 1),
                 bbox_transform=ax1.transAxes, borderpad=0)

ax2.bar(dL_bin[:-1], dL_prob, align='edge', width=np.diff(dL_bin), color=color, alpha=0.5)
ax2.set_yscale('log')
#ax2.set_xlabel(r'$z$')
#ax2.set_ylabel(r'$P_{det}$')

name="general_plots_dL/prob_dL.png"
plt.savefig(name, format='png', dpi=100, bbox_inches="tight")

#name="LATEX_IMAGES/prob_dL.pdf"
#plt.savefig(name, format='pdf', dpi=100, bbox_inches="tight")

#%%


##################### z histograms for each mass bin ##############

for i in range(N):
    for j in range(N2):
        if mbin_data[i,j]['n_det']!=0:
            
            print(i,j)
            
            Ntot_bin = mbin_data[i,j]['Ntot_bin']
            
            # detected values
            dLdeti = mbin_data[i,j]['dL_det']
            dLpdfdeti = mbin_data[i,j]['dL_pdf_det']
            
            # all values detected or not
            dL_i = mbin_data[i,j]['dL']
            dL_pdf_i = mbin_data[i,j]['dL_pdf']
            mean_m_pdf_i = mbin_data[i,j]['mean_m_pdf']
            
            #zm_bin=np.arange(min(z_i),max(z_i)+bz,bz)
            dLm_bin=np.linspace(min(dL_i),max(dL_i),20)
            delta_dL = dLm_bin[1]-dLm_bin[0]

            dLmbin_data=np.array([{'dLdet': [],
                                  'dLdet_pdf': [],
                                  'm_pdf': [],
                                  'dL': [],
                                  'dL_pdf': []
                                  ,} for i in range(len(dLm_bin)-1)])

            NdLm = dLmbin_data.shape[0]  # number of z bins in this mass bin
            
            
            for u in range(len(dLdeti)):
                for k in range(len(dLm_bin)-1):
                    if dLdeti[u] >= dLm_bin[k] and dLdeti[u] < dLm_bin[k+1]:
                        dLmbin_data[k]['dLdet'].append(dLdeti[u])
                        dLmbin_data[k]['dLdet_pdf'].append(dLpdfdeti[u])
                        
            for u in range(len(dL_i)):
                for k in range(len(dLm_bin)-1):
                    if dL_i[u] >= dLm_bin[k] and dL_i[u] < dLm_bin[k+1]:
                        dLmbin_data[k]['dL'].append(dL_i[u])
                        dLmbin_data[k]['dL_pdf'].append(dL_pdf_i[u])


            for u in range(NdLm):
                quad_fun = lambda x: dLpdf_interp(x)
                dLmbin_data[u]['mean_dL_pdf'] = quad(quad_fun, dLm_bin[u], dLm_bin[u+1])[0] 
                dLmbin_data[u]['n_det'] = len(np.array(dLmbin_data[u]['dLdet']))

            #z detected with far>=1
            dLm_detections = np.array([dLmbin_data[u]['n_det'] for u in range(NdLm)])
            
            dL_det_hist=np.array(dLdeti)
            
            # mean pdfs
            mean_dLm_pdf = np.array([dLmbin_data[u]['mean_dL_pdf'] for u in range(NdLm)])
           
            dLm_prob = dLm_detections/(mean_dLm_pdf * mean_m_pdf_i * total )
            
            mid = (dLm_bin[:-1]+dLm_bin[1:])/2
            
            nonzero = dLm_detections > 0
            
            plt.figure()
            plt.plot(mid, dLm_prob, 'o')
            plt.errorbar(mid[nonzero], dLm_prob[nonzero], yerr=dLm_prob[nonzero]/np.sqrt(dLm_detections[nonzero]), fmt="none", color="k", capsize=2, elinewidth=0.4)
            #plt.yscale('log')
            #plt.ylim(0.001,2)
            plt.xlabel(r'$dL$', fontsize=14)
            plt.ylabel(r'$P_{det}(dL)$', fontsize=14)
            plt.title(r'$P_{det}(dL)$ $m_1$ %.1f-%.1f M$_{\odot}$ \& $m_2$ %.1f-%.1f M$_{\odot}$' %(m1_bin[i], m1_bin[i+1], m2_bin[j], m2_bin[j+1]) )
            name=f"dLm_plots_dic/{i}{j}.png"
            plt.savefig(name, format='png', dpi=100, bbox_inches="tight")
            
            plt.close()
            
            name=f'dL_binned/{i}{j}_data.dat'
            data = np.column_stack((mid, dLm_prob, dLm_detections))
            header = "dL_mid, dL_binned_prob, dLm_detections"
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
