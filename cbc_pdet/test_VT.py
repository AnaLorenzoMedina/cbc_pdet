#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 17:10:55 2026

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
import astropy.cosmology
import astropy.constants as const
from scipy import interpolate
from scipy import integrate
from cbc_pdet.gwtc_found_inj import Found_injections
import time

plt.close('all')

run_fit = 'o4'
run_dataset = 'o4'
sources = 'all'


#dmid_fun = 'Dmid_mchirp_fdmid_fspin'
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

#%%

def sensitive_volume(run_fit, m1, m2, dl_test, chieff=0., sources='all', zmax=3.1, rescale_o3=True):
    '''
    Sensitive volume for a merger with given masses (m1 and m2), computed from the fit to whichever observed run we want.
    Integrated within the total range of redshift available in the injection's dataset.

    Parameters
    ----------
    run_fit : str. Observing run from which we want to use its fit. Must be 'o1', 'o2', 'o3' or 'o4'.
    m1 : float. Mass 1 (source)
    m2 : float. Mass 2 (source)
    chieff : float. Effective spin. The default is 0, If you use a fit that includes a dependence on chieff in the dmid function
            (it has to be on the list of spin functions), it will use chieff. if not, it won't be used for anything.
    zmax : float. Maximum redshift for cosmological calculation, if an injection set is not loaded:
           3.1 is just larger than any O4 injection
    sources : str or list with the types of sources you want. Must be 'bbh' for o1 and o2, \
             'nsbh' 'bns' 'imbh' or 'bbh' for o3 (or a combination of them) and 'all' for o4
    rescale_o3 : True or False, optional. The default is True. If True, we use the rescaled fit for o1 and o2. If False, the direct fit.

    Returns
    -------
    pdet * Vtot : float. Sensitive volume
    '''
    data.get_opt_params(run_fit, 'all', rescale_o3) if run_fit == 'o4' else data.get_opt_params(run_fit, sources, rescale_o3)
    # Reference inj set for interpolating cosmology quantities
    source_interp_dL_pdf = 'all' if run_fit == 'o4' else 'bbh'

    if 'interp_z' in data.sets[source_interp_dL_pdf]:  # Interpolator derived from injection set
        #m1_det = lambda dL_int : m1 * (1 + data.sets[source_interp_dL_pdf]['interp_z'](dL_int))
        #m2_det = lambda dL_int : m2 * (1 + data.sets[source_interp_dL_pdf]['interp_z'](dL_int))
        z = lambda dL_int : data.sets[source_interp_dL_pdf]['interp_z'](dL_int)
        
    else:  # We compute some values of dl for some z to make an interpolator
        if data.zinterp_VT is None:
            fun_A = lambda t : np.sqrt(data.cosmo.Om0 * (1 + t)**3 + 1 - data.cosmo.Om0)
            quad_fun_A = lambda t: 1/fun_A(t)

            z_inter = np.linspace(0.002, zmax, 100)
            z0 = np.insert(z_inter, 0, 0, axis=0)
            dL = np.array([(const.c.value*1e-3 / data.cosmo.H0.value) * (1 + i) * integrate.quad(quad_fun_A, 0, i)[0] for i in z0])
            data.zinterp_VT = interpolate.interp1d(dL, z0)
            data.zinterp_VT = np.interp(dL, z0)

        if data.sets[sources]['dLmax'] is None:
            data.sets[sources]['dLmax'] = dL.max()
            
        z = lambda dL_int : data.zinterp_VT(dL_int)

        #m1_det = lambda dL_int : m1 * (1 + data.zinterp_VT(dL_int))
        #m2_det = lambda dL_int : m2 * (1 + data.zinterp_VT(dL_int))
        
    m1_det = lambda dL_int : m1 * (1 + z(dL_int))
    m2_det = lambda dL_int : m2 * (1 + z(dL_int))

    if data.dmid_fun in data.spin_functions:
        dmid = lambda dL_int : data.dmid(m1_det(dL_int), m2_det(dL_int), chieff, data.dmid_params)
    else:
        dmid = lambda dL_int : data.dmid(m1_det(dL_int), m2_det(dL_int), data.dmid_params)

    emax_params, gamma, delta, alpha = data.get_shape_params()

    if data.emax_fun is not None:
        emax = lambda dL_int : data.emax(m1_det(dL_int), m2_det(dL_int), emax_params)
        quad_fun = lambda dL_int : data.sigmoid(dL_int, dmid(dL_int), emax(dL_int), gamma, delta, alpha) \
                   * data.sets[source_interp_dL_pdf]['interp_dL_pdf'](dL_int)
        #emax = 0.8    
        #quad_fun = lambda dL_int : data.sigmoid(dL_int, dmid(dL_int), emax, gamma, delta, alpha) \
        #           * data.sets[source_interp_dL_pdf]['interp_dL_pdf'](dL_int)
        
    else:
        emax = np.copy(emax_params)
        #emax = 0.8
        quad_fun = lambda dL_int : data.sigmoid(dL_int, dmid(dL_int), emax, gamma, delta, alpha) \
                   * data.sets[source_interp_dL_pdf]['interp_dL_pdf'](dL_int)

    pdet = integrate.quad(quad_fun, 0, data.sets[source_interp_dL_pdf]['dLmax'])[0]

    if data.Vtot is None:
        # NB the factor of 1/(1+z) for time dilation in the signal rate
        vquad = lambda z_int : 4 * np.pi * data.cosmo.differential_comoving_volume(z_int).value / (1 + z_int)
        data.Vtot = integrate.quad(vquad, 0, data.sets[source_interp_dL_pdf]['zmax'])[0]

    #return pdet * data.Vtot
    return quad_fun(dl_test)

def new_sensitive_volume(run, m1, m2, dl_test, chieff=0., sources='all', zmax=3.1, rescale_o3=True):
    '''
    Sensitive volume for a merger with given masses (m1 and m2), computed from the fit to whichever observed run we want.
    Integrated within the total range of redshift available in the injection's dataset.

    Parameters
    ----------
    run : str. Observing run from which we want to use its fit. Must be 'o1', 'o2', 'o3' or 'o4'.
    m1 : float. Mass 1 (source)
    m2 : float. Mass 2 (source)
    chieff : float. Effective spin. The default is 0, If you use a fit that includes a dependence on chieff in the dmid function
            (it has to be on the list of spin functions), it will use chieff. if not,t it won't be used for anything.
    zmax : float. Maximum redshift for cosmological calculation, if an injection set is not loaded:
           3.1 is just larger than any O4 injection
    sources : str or list with the types of sources you want. Must be 'bbh' for o1 and o2, \
             'nsbh' 'bns' 'imbh' or 'bbh' for o3 (or a combination of them) and 'all' for o4
    rescale_o3 : True or False, optional. The default is True. If True, we use the rescaled fit for o1 and o2. If False, the direct fit.

    Returns
    -------
    pdet * Vtot : float. Sensitive volume
    '''
    data.get_opt_params(run, 'all', rescale_o3) if run == 'o4' else data.get_opt_params(run, sources, rescale_o3)
    # Reference inj set for interpolating cosmology quantities
    source_interp_dL_pdf = 'all' if run == 'o4' else 'bbh'
    
    fun_A = lambda t : np.sqrt(data.cosmo.Om0 * (1 + t)**3 + 1 - data.cosmo.Om0)
    quad_fun_A = lambda t: 1/fun_A(t)

    z_inter = np.linspace(0.002, zmax, 100)
    z0 = np.insert(z_inter, 0, 0, axis=0)
    dL = np.array([(const.c.value*1e-3 / data.cosmo.H0.value) * (1 + i) * integrate.quad(quad_fun_A, 0, i)[0] for i in z0])
    
    emax_params, gamma, delta, alpha = data.get_shape_params()
    #emax = 0.8
    
    def integrand_VT(dL_int):

        if data.sets[sources]['dLmax'] is None:
            data.sets[sources]['dLmax'] = dL.max()

        z = np.interp(dL_int, dL, z0)
            
        m1_det = m1 * (1 + z)
        m2_det = m2 * (1 + z)
            
        if data.dmid_fun in data.spin_functions:
            dmid =  data.dmid(m1_det, m2_det, chieff, data.dmid_params)
        else:
            dmid = data.dmid(m1_det, m2_det, data.dmid_params)
        
        if data.emax_fun is not None:
            emax = data.emax(m1_det, m2_det, emax_params)
            
        else:
            emax = np.copy(emax_params)
            
        integrand = data.sigmoid(dL_int, dmid, emax, gamma, delta, alpha) \
                * np.interp(dL_int, data.inter_dL, data.inter_dLpdf)
                
        return integrand
    
    pdet_integrated = integrate.quad(integrand_VT, 0, data.sets[source_interp_dL_pdf]['dLmax'])[0]

    if data.Vtot is None:
        # NB the factor of 1/(1+z) for time dilation in the signal rate
        vquad = lambda z_int : 4 * np.pi * data.cosmo.differential_comoving_volume(z_int).value / (1 + z_int)
        data.Vtot = integrate.quad(vquad, 0, data.sets[source_interp_dL_pdf]['zmax'])[0]

    #return pdet_integrated * data.Vtot
    return integrand_VT(dl_test)

#%%

npoints = 1000
index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
m1 = data.m1[index]
m2 = data.m2[index]
#chieff = data.chi_eff[index]

#%%
m1 = 10
m2 = 8
chieff = 0.2

dl_test = np.linspace(100, max(data.dL), 100)

vsensitive = np.array([data.sensitive_volume(run_fit, m1, m2, dl_test[i]) for i in range(len(dl_test))])


#%%

vsensitive_new = np.array([data.new_sensitive_volume(run_fit, m1, m2, dl_test[i]) for i in range(len(dl_test))])
#%%
plt.figure()
plt.plot(dl_test, vsensitive/1e9, '.')
plt.xlabel('dl')
plt.ylabel('integrand')
plt.savefig( path + '/integrand_dl.png')
#%%
plt.figure()
plt.plot(dl_test, vsensitive_new/1e9, '.')
plt.xlabel('dl')
plt.ylabel('integrand new')
plt.savefig( path + '/integrand_new_dl.png')


#%%
plt.figure()
plt.plot(vsensitive/1e9, vsensitive_new/1e9, '.')
plt.loglog()
plt.xlabel('integrand old')
plt.ylabel('integrand new')
plt.savefig( path + f'/integrand_comparison_{npoints}.png')

#%%
m1_det = m1*(1+data.z[index])
m2_det = m2*(1+data.z[index])
chieff = np.zeros(len(m1))
dmid = data.dmid(m1_det, m2_det, chieff, data.dmid_params)

plt.figure()
plt.plot(m1_det + m2_det, vsensitive/vsensitive_new, '.')
plt.loglog()
plt.xlabel('Mtot det')
plt.ylabel('VT / VT new')
plt.savefig( path + f'/integrand_mtot_det__{npoints}.png')

#%%

fun_A = lambda t : np.sqrt(data.cosmo.Om0 * (1 + t)**3 + 1 - data.cosmo.Om0)
quad_fun_A = lambda t: 1/fun_A(t)

z_inter = np.linspace(0.002, 2.3, 100)
z0 = np.insert(z_inter, 0, 0, axis=0)
dL = np.array([(const.c.value*1e-3 / data.cosmo.H0.value) * (1 + i) * integrate.quad(quad_fun_A, 0, i)[0] for i in z0])

m1s = np.zeros(len(dl_test)) + 10
m2s = np.zeros(len(dl_test)) + 8
chieffs = np.zeros(len(dl_test)) + 0.2
z = np.interp(dl_test, dL, z0)
     
m1_det = m1s * (1 + z)
m2_det = m2s * (1 + z)

pdet = data.run_pdet(dl_test, m1_det, m2_det, chieffs, 'o4', 'all')
inter_dl_test = data.sets['all']['interp_dL_pdf'](dl_test)

plt.figure()
plt.plot(dl_test, pdet, '.')
plt.xlabel('dl')
plt.ylabel('pdet')
plt.savefig( path + '/pdet_dl.png')

plt.figure()
plt.plot(dl_test, inter_dl_test, '.')
plt.xlabel('dl')
plt.ylabel('interp dl pdf')
plt.savefig( path + '/pdf_dl.png')


plt.figure()
plt.plot(dl_test, pdet * inter_dl_test, '.')
plt.xlabel('dl')
plt.ylabel('pdet * interp dl pdf')
plt.savefig( path + '/pdetpdf_dl.png')

#%%
emax_params, gamma, delta, alpha = data.get_shape_params()
dmid =  data.dmid(m1_det, m2_det, chieffs, data.dmid_params)
emax = data.emax(m1_det, m2_det, emax_params)
sigmoid = data.sigmoid(dl_test, dmid, emax, gamma, delta, alpha)

plt.figure()
plt.plot(dl_test, sigmoid, '.')
plt.xlabel('dl')
plt.ylabel('sigmoid')
#plt.savefig( path + '/pdet_dl.png')

plt.figure()
plt.plot(dl_test, inter_dl_test, '.')
plt.xlabel('dl')
plt.ylabel('interp dl pdf')
#plt.savefig( path + '/pdf_dl.png')


plt.figure()
plt.plot(dl_test, sigmoid * inter_dl_test, '.')
plt.xlabel('dl')
plt.ylabel('sigmoid * interp dl pdf')
#plt.savefig( path + '/sigmoidpdf_dl.png')