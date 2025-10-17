#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:33:14 2025

@author: ana
"""

import numpy as np
from cbc_pdet.pdet_api import PdetEstimation as pdet

mdict = {'observing_run':'o4', 'sources': 'all', 'dmid_fun': 'Dmid_mchirp_mixture_logspin_corr', 'emax_fun': 'emax_gaussian'}
estimation = pdet(method_dict=mdict)

#with source masses
param_dict = {'d_lum' : np.array([1000, 3000]), 'mass1' : np.array([70, 80]), 'mass2' : np.array([50, 40]), 'chi_eff' : np.array([0.8, 0])}

#with redshifted masses
param_dict = {'d_lum' : np.array([1000, 3000]), 'mass1_det' : np.array([70, 80]), 'mass2_det' : np.array([50, 40]), 'chi_eff' : np.array([0.8, 0])}

Pdet = estimation.predict(param_dict) #array with the probabilities

