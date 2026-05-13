#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:33:14 2025

@author: ana
"""

import numpy as np
from cbc_pdet.pdet_api import PdetEstimation as pdet

mdict = {'observing_run':'o4', 'sources': 'all', 'thr_far': 1, 'dmid_fun': 'Dmid_mchirp_mixture_logspin_corr', 'emax_fun': 'emax_gaussian'}
estimation = pdet(method_dict=mdict)

# you can give the param dict either luminosity distance (in Mpc) or redshift
# if you give d_lum and source masses, it's gonna compute the detector frame masses (so, effectively, z) with the default cosmology
# you can specify the cosmolgy with your_cosmo_parameters = {'name': 'FlatLambdaCDM', 'H0': 67.9, 'Om0': 0.3065} (or your fav cosmology)
# and then when you initialise the class, you set estimation = pdet(method_dict=mdict, cosmo_parameters=your_cosmo_parameters)
# or you can give it both d_lum and redshift, and in that case you need to also specify override_redshift = True:
# so estimation = pdet(method_dict=mdict, override_redshift = True)
# basically for pdet you need d_lum, detector frame masses and chieff
# take into account that the conversion from d_lum to z or the other way around is not super fast, so you may consider doing 
# an interpolation to you data points with a few hundred so you get both values before calling it

#with source masses
param_dict = {'d_lum' : np.array([1000, 3000]), 'mass1' : np.array([70, 80]), 'mass2' : np.array([50, 40]), 'chi_eff' : np.array([0.8, 0])}

#with redshifted masses
param_dict = {'d_lum' : np.array([1000, 3000]), 'mass1_det' : np.array([70, 80]), 'mass2_det' : np.array([50, 40]), 'chi_eff' : np.array([0.8, 0])}

Pdet = estimation.predict(param_dict) #array with the probabilities

