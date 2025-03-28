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
from .o123_class_found_inj_general import Found_injections

class PdetEstimation():

    def __init__(self, method_dict=None, cosmo_parameters=None):
        
        if method_dict is None:
            method_dict = {'observing_run': 'o3', 'dmid_fun': 'Dmid_mchirp_fdmid_fspin', 'emax_fun': 'emax_exp'}

        #method keys: observing run, name dmid and emax functions
        
        self.run = method_dict.pop('observing_run')
        
        if cosmo_parameters is None:
            cosmo_parameters = {'name': 'FlatLambdaCDM', 'H0': 67.9, 'Om0': 0.3065}

        
        self.fit = Found_injections(**method_dict, cosmo_parameters = cosmo_parameters)
        
        self.cosmo = self.fit.cosmo
    


    def check_parameter_dict(self, parameter_dict):
        
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
        
        #at the moment we don't have any allowed extrinsic params YET
        
        self.allowed_params = self.allowed_distance_params + self.allowed_mass_params + self.allowed_spin_params
        
        if not all(param in self.allowed_params for param in parameter_dict):
            invalid_params = [param for param in parameter_dict if param not in self.allowed_params]
            raise ValueError(f"Invalid parameters: {invalid_params}")


    def check_distance(self, parameter_dict):

        if not any(param in parameter_dict for param in self.allowed_distance_params):
            raise RuntimeError("Missing distance parameter. Requires one of:", self.allowed_distance_params)


    def check_masses(self, parameter_dict):
    
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

        given_mass_params = set(parameter_dict.keys()) & set(self.allowed_mass_params)

        if given_mass_params not in allowed_combinations:
            raise ValueError(f"Invalid mass parameter combination. Requires at least one combination of: {allowed_combinations}")
        

    def check_spins(self, parameter_dict):

        allowed_combinations = [{'spin1z', 'spin2z'},
                                {'chi_eff'},
                                {'spin1', 'spin2', 'cos_theta_1', 'cos_theta_2'}]

        if all(param in parameter_dict for param in ['spin1z', 'spin2z']):
            #a1_module = np.sqrt(parameter_dict['spin1x']**2 + parameter_dict['spin1y']**2 + parameter_dict['spin1z']**2)
            #a2_module = np.sqrt(parameter_dict['spin2x']**2 + parameter_dict['spin2y']**2 + parameter_dict['spin2z']**2)
            if not (0 <= np.all(parameter_dict['spin1z']) <= 1) and (0 <= np.all(parameter_dict['spin2z']) <= 1):
                raise ValueError("Invalid spin parameters: spin1z and spin2z must be between 0 and 1")

        if all(param in parameter_dict for param in ['spin1', 'spin2', 'cos_theta_1', 'cos_theta_2']):
            s1z = parameter_dict['spin1'] * np.cos(parameter_dict['cos_theta_1'])
            s2z = parameter_dict['spin2'] * np.cos(parameter_dict['cos_theta_2'])
            if not (-1 <= np.all(s1z) <= 1) and (-1 <= np.all(s2z) <= 1):
                raise ValueError("Invalid spin parameters: |spin1|*cos(theta_1) and |spin2|*cos(theta_2) must be between -1 and 1")

        if 'chi_eff' in parameter_dict:
            if not (0 <= np.all(parameter_dict['chi_eff']) <= 1):
                raise ValueError("Invalid spin parameters: effective spin must be between 0 and 1")

        # if 'chi_p' in parameter_dict:
        #     if not (0 <= parameter_dict['chi_p'] <= 1):
        #         raise ValueError("Invalid spin parameters: precessing spin must be between 0 and 1")
        
        given_spin_params = set(parameter_dict.keys()) & set(self.allowed_spin_params)

        if given_spin_params not in allowed_combinations:
            raise ValueError(f"Invalid mass parameter combination. Requires at least one combination of: {allowed_combinations}")

        
        

    def check_extrinsics(self, parameter_dict):
        # allowed_extrinsic_params = ['ra',
        #                        'dec', 
        #                        'sin_dec',
        #                        'inclination',
        #                        'cos_inclination',
        #                        'polarization',
        #                        ]

        #allowed_combinations = []

        # if 'ra' not in parameter_dict:
        #     warnings.warn("Parameter 'ra' not present.")
        
        # if 'dec' not in parameter_dict:
        #     if 'sin_dec' not in parameter_dict:
        #         warnings.warn("Parameter 'dec' or 'sin_dec' not present.")

        # if 'dec' in parameter_dict:
        #     if 'sin_dec' not in parameter_dict:
        #         parameter_dict['sin_dec'] = np.sin(parameter_dict['dec'])

        # if 'inclination' not in parameter_dict:
        #     if 'cos_inclination' not in parameter_dict:
        #         warnings.warn("Parameter 'inclination' or 'cos_inclination' not present.")
            
        # if 'inclination' in parameter_dict:
        #     if 'cos_inclination' not in parameter_dict:
        #         parameter_dict['cos_inclination'] = np.cos(parameter_dict['inclination'])

        # if 'polarization' not in parameter_dict:
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

        if 'd_lum' not in p_dict:
            if 'redshift' not in p_dict:
                #convert comoving distance to Mpc astropy units
                dC_Mpc = p_dict['comoving_distance'] * u.Mpc
                #get redshift
                p_dict['redshift'] = z_at_value(self.cosmo.luminosity_distance, dC_Mpc).value
            
            p_dict['d_lum'] = self.cosmo.luminosity_distance(p_dict['redshift']).value

        if 'mass1_det' not in p_dict and 'mass1_det' not in p_dict:
            if 'redshift' not in p_dict:
                #convert luminosity distance to Mpc astropy units
                dL_Mpc = p_dict['d_lum'] * u.Mpc
                p_dict['redshift'] = z_at_value(self.cosmo.luminosity_distance, dL_Mpc).value

            p_dict['mass1_det'] = p_dict['mass1'] * (1 + p_dict['redshift'])
            p_dict['mass2_det'] = p_dict['mass2'] * (1 + p_dict['redshift'])


        if 'chi_eff' not in p_dict:
            if 'spin1z' in p_dict and 'spin2z' in p_dict:
                m1_det = p_dict['mass1_det']
                m2_det = p_dict['mass2_det']
                s1z = p_dict['spin1z']
                s2z = p_dict['spin2z']
                
                p_dict['chi_eff'] =  (m1_det * s1z + m2_det * s2z) / (m1_det + m2_det)
                
            else:
                s1 = p_dict['spin1']
                s2 = p_dict['spin2']
                cos1 = p_dict['cos_theta_1']
                cos2 = p_dict['cos_theta_2']
                
                p_dict['chi_eff'] =  (m1_det * s1 * cos1 + m2_det * s2 * cos2) / (m1_det + m2_det)
                
                
        return p_dict


    def predict(self, input_parameter_dict):
        # Copy so that we can safely modify dictionary in-place
        parameter_dict = input_parameter_dict.copy()

        # Check input
        self.check_input(parameter_dict)
        
        parameter_dict = self.transform_parameters(parameter_dict)
        
        self.fit.get_opt_params(self.run)
        self.fit.set_shape_params()
        
        m1_det = parameter_dict['mass1_det']
        m2_det = parameter_dict['mass2_det']
        chi_eff = parameter_dict['chi_eff']
        d_lum = parameter_dict['d_lum']
        
        dmid_values = self.fit.dmid(m1_det, m2_det, chi_eff, self.fit.dmid_params)
        emax_values = self.fit.emax(m1_det, m2_det, self.fit.emax_params)
        pdet = self.fit.sigmoid(d_lum, dmid_values, emax_values, self.fit.gamma, self.fit.delta)

        return pdet
 
    