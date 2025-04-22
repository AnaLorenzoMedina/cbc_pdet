#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:53:26 2025

@author: ana
"""

import numpy as np
import pandas
import astropy.units as u
from astropy.cosmology.funcs import z_at_value
from .gwtc_found_inj import Found_injections

class PdetEstimation():
    def __init__(self, method_dict=None, cosmo_parameters=None):
        
        if method_dict is None:  # Current defaults use a fit on O3 injections
            method_dict = {'observing_run': 'o3', 'dmid_fun': 'Dmid_mchirp_fdmid_fspin', 'emax_fun': 'emax_exp'}        
        self.run = method_dict.pop('observing_run')

        self.fit = Found_injections(**method_dict, cosmo_parameters=cosmo_parameters)
        
        self.fit.get_opt_params(self.run)
        self.fit.set_shape_params()

    def check_parameter_dict(self, pdict):
        # Check that all parameters given are ones that we can use
        
        self.allowed_distance_params = ['d_lum',
                                        'redshift',
                                        'comoving_distance']
        
        self.allowed_mass_params = ['mass1', 
                                    'mass2', 
                                    'mass1_det',
                                    'mass2_det']
        
        self.allowed_spin_params = ['spin1',
                                    'spin2',
                                    'spin1z',
                                    'spin2z',
                                    'cos_theta_1',
                                    'cos_theta_2',
                                    'chi_eff']
        # At the moment we don't have any allowed extrinsic params (yet!)
        
        self.allowed_params = self.allowed_distance_params + self.allowed_mass_params + self.allowed_spin_params
        if not all(p in self.allowed_params for p in pdict):
            invalid_params = [p for p in pdict if p not in self.allowed_params]
            raise ValueError(f"Invalid parameters: {invalid_params}")

    def check_distance(self, pdict):
        # Find which allowed distance parameters exist
        found_params = [p for p in self.allowed_distance_params if p in pdict]
        
        # We need exactly one distance parameter
        if len(found_params) == 0:
            raise RuntimeError(f"Missing distance parameter. Requires one of: {self.allowed_distance_params}")
        
        if len(found_params) > 1:
            raise RuntimeError(f"Too many distance parameters provided: {found_params}. Requires exactly one.")


    def check_masses(self, pdict):
        # Currently allow only the source or detector frame masses
        allowed_combinations = [{'mass1', 'mass2'}, 
                                {'mass1_det', 'mass2_det'},
                                #{'mass1', 'q'}, 
                                #{'mass2', 'q'},
                                #{'mass1', 'total_mass'},
                                #{'mass2', 'total_mass'},
                                #{'total_mass', 'q'},
                                #{'total_mass', 'eta'},
                                #{'chirp_mass', 'q'},
                                #{'chirp_mass', 'eta'},
                                #{'mass1_det', 'q'}, 
                                #{'mass2_det', 'q'},
                                #{'mass1_det', 'total_mass_det'},
                                #{'mass2_det', 'total_mass_det'},
                                #{'total_mass_det', 'q'},
                                #{'total_mass_det', 'eta'},
                                #{'chirp_mass_det', 'q'},
                                #{'chirp_mass_det', 'eta'},
                                ]

        given_mass_params = set(pdict.keys()) & set(self.allowed_mass_params)
        
        # Find all valid combinations given
        found_combos = [combo for combo in allowed_combinations if combo.issubset(given_mass_params)]

        # We need exactly one mass parameter combination
        if not found_combos:
            raise ValueError(f"Invalid mass parameter combination: {given_mass_params}. Requires one of: {allowed_combinations}")

        if len(found_combos) > 1:
            raise ValueError(f"Too many mass parameter combinations: {found_combos}. Provide only one valid combination of: {allowed_combinations}")
        
    def check_spins(self, pdict):
        # Currently the fit uses only chieff, thus allow any set of parameters from which it can be calculated
        allowed_combinations = [{'spin1z', 'spin2z'},
                                {'chi_eff'},
                                {'spin1', 'spin2', 'cos_theta_1', 'cos_theta_2'}]

        if all(param in pdict for param in ['spin1z', 'spin2z']):
            if np.any(np.abs(pdict['spin1z']) > 1) or np.any(np.abs(pdict['spin2z']) > 1):
                raise ValueError("Invalid spin parameters: spin1z and spin2z must be between -1 and 1")

        if all(p in pdict for p in ['spin1', 'spin2', 'cos_theta_1', 'cos_theta_2']):
            s1z = pdict['spin1'] * pdict['cos_theta_1']
            s2z = pdict['spin2'] * pdict['cos_theta_2']
            if np.any(np.abs(s1z) > 1) or np.any(np.abs(s2z) > 1):
                raise ValueError("Invalid spin parameters: |spin1|*cos(theta_1) and |spin2|*cos(theta_2) must be between -1 and 1")

        if 'chi_eff' in pdict:
            if  np.any(np.abs(pdict['chi_eff']) > 1):
                raise ValueError("Invalid spin parameters: effective spin must be between -1 and 1")

        # if 'chi_p' in parameter_dict:
        #     if not (0 <= pdict['chi_p'] <= 1):
        #         raise ValueError("Invalid spin parameters: precessing spin must be between 0 and 1")
        
        given_spin_params = set(pdict.keys()) & set(self.allowed_spin_params)
        
        # Find all valid combinations given
        found_combos = [combo for combo in allowed_combinations if combo.issubset(given_spin_params)]

        # We need exactly one spin parameter combination
        if not found_combos:
            raise ValueError(f"Invalid spin parameter combination: {given_spin_params}. Requires one of: {allowed_combinations}")

        if len(found_combos) > 1:
            raise ValueError(f"Too many spin parameter combinations: {found_combos}. Provide only one valid combination of: {allowed_combinations}")
        
    def check_extrinsics(self, pdict):
        # At present a placeholder as the fit does not use any extrinsic (angular) parameter
        # allowed_extrinsic_params = ['ra',
        #                        'dec', 
        #                        'sin_dec',
        #                        'inclination',
        #                        'cos_inclination',
        #                        'polarization',
        #                        ]

        #allowed_combinations = []

        # if 'ra' not in pdict:
        #     warnings.warn("Parameter 'ra' not present.")
        
        # if 'dec' not in pdict:
        #     if 'sin_dec' not in pdict:
        #         warnings.warn("Parameter 'dec' or 'sin_dec' not present.")

        # if 'dec' in pdict:
        #     if 'sin_dec' not in pdict:
        #         pdict['sin_dec'] = np.sin(pdict['dec'])

        # if 'inclination' not in pdict:
        #     if 'cos_inclination' not in pdict:
        #         warnings.warn("Parameter 'inclination' or 'cos_inclination' not present.")
            
        # if 'inclination' in pdict:
        #     if 'cos_inclination' not in pdict:
        #         pdict['cos_inclination'] = np.cos(pdict['inclination'])

        # if 'polarization' not in pdict:
        #     warnings.warn("Parameter 'polarization' not present.")
        
        return
        

    def check_input(self, parameter_dict):

        # Convert from pandas table to dictionary, if necessary
        if type(parameter_dict) == pandas.core.frame.DataFrame:
            parameter_dict = parameter_dict.to_dict(orient='list')

        # Check parameters
        self.check_parameter_dict(parameter_dict)
        self.check_distance(parameter_dict)
        self.check_masses(parameter_dict)
        self.check_spins(parameter_dict)
        self.check_extrinsics(parameter_dict)

        return

    def transform_parameters(self, p_dict):
        # Pre-process parameters to the ones that will be input to the fit

        if 'd_lum' not in p_dict:
            if 'redshift' not in p_dict:
                # Convert comoving distance to Mpc astropy units
                dC_Mpc = p_dict['comoving_distance'] * u.Mpc
                p_dict['redshift'] = z_at_value(self.fit.cosmo.luminosity_distance, dC_Mpc).value
            
            p_dict['d_lum'] = self.fit.cosmo.luminosity_distance(p_dict['redshift']).value

        if 'mass1_det' not in p_dict and 'mass2_det' not in p_dict:
            if 'redshift' not in p_dict:
                # Convert luminosity distance to Mpc astropy units
                dL_Mpc = p_dict['d_lum'] * u.Mpc
                p_dict['redshift'] = z_at_value(self.fit.cosmo.luminosity_distance, dL_Mpc).value

            p_dict['mass1_det'] = p_dict['mass1'] * (1 + p_dict['redshift'])
            p_dict['mass2_det'] = p_dict['mass2'] * (1 + p_dict['redshift'])

        if 'chi_eff' not in p_dict:
            m1_det = p_dict['mass1_det']
            m2_det = p_dict['mass2_det']
            if not('spin1z' in p_dict and 'spin2z' in p_dict):
                p_dict['spin1z'] = p_dict['spin1'] * p_dict['cos_theta_1']
                p_dict['spin2z'] = p_dict['spin2'] * p_dict['cos_theta_2']
    
            p_dict['chi_eff'] = (m1_det * p_dict['spin1z'] + m2_det * p_dict['spin2z']) / (m1_det + m2_det)

        return p_dict

    def predict(self, input_parameter_dict):
        # Copy so that we can safely modify dictionary in-place
        p_dict = input_parameter_dict.copy()

        # Check input
        self.check_input(p_dict)
        
        p_dict = self.transform_parameters(p_dict)
        
        m1_det = p_dict['mass1_det']
        m2_det = p_dict['mass2_det']
        chi_eff = p_dict['chi_eff']
        d_lum = p_dict['d_lum']
        
        pdet = self.fit.evaluate(d_lum, m1_det, m2_det, chi_eff, self.fit.dmid_params, self.fit.emax_params, self.fit.gamma, self.fit.delta)

        return pdet
    
