# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:14:27 2023

@author: Ana
"""

import numpy as np
import h5py
from scipy import interpolate
from scipy import integrate
from scipy.stats import kstest
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os
import errno
from scipy.optimize import fsolve

class Found_injections:
    """
    Class for an algorithm of GW detected found injections from signal templates 
    of binary black hole mergers
    
    Input: an h5py file with the parameters of every sample and the threshold for 
    the false alarm rate (FAR). The default value is thr = 1, which means we are 
    considering a signal as detected when FAR <= 1.

    """
    
    def __init__(self, dmid_fun, emax_fun = None, alpha_vary = None, thr_far = 1, thr_snr = 10):
        
        assert isinstance(thr_far, float) or isinstance(thr_far, int),\
        "Argument (thr_far) must be a float or an integrer."
        
        assert isinstance(thr_snr, float) or isinstance(thr_snr, int),\
        "Argument (thr_snr) must be a float or an integrer."
        
        self.thr_far = thr_far
        self.thr_snr = thr_snr
        
        self.dmid_fun = dmid_fun
        self.emax_fun = emax_fun
        self.dmid = getattr(Found_injections, dmid_fun)
        self.emax = getattr(Found_injections, emax_fun)
        self.alpha_vary = alpha_vary
        
        self.dmid_ini_values, self.shape_ini_values = self.get_ini_values()
        
        self.dmid_params = self.dmid_ini_values
        self.shape_params = self.shape_ini_values
        
        self.obs_time = { 'o1' : 0.1331507, 'o2' : 0.323288, 'o3' : 0.75435365296528 } #years
        self.total_obs_time = self.obs_time['o1'] + self.obs_time['o2'] + self.obs_time['o3']
        self.prop_obs_time = np.array([self.obs_time['o1']/self.total_obs_time, self.obs_time['o2']/self.total_obs_time, self.obs_time['o3']/self.total_obs_time ])
        self.obs_nevents = { 'o1' : 3, 'o2' : 7, 'o3' : 59 }
        self.det_rates = {'o1' : self.obs_nevents['o1'] / self.obs_time['o2'], 
                          'o2' : self.obs_nevents['o2'] / self.obs_time['o2'], 
                          'o3' : self.obs_nevents['o3'] / self.obs_time['o3'] }
        
        self.dmid_params_names = {'Dmid_mchirp': 'cte', 
                                  'Dmid_mchirp_expansion': ['cte', 'a20', 'a01', 'a21', 'a30', 'a10'], 
                                  'Dmid_mchirp_expansion_noa30': ['cte', 'a20', 'a01', 'a21', 'a10','a11'],
                                  'Dmid_mchirp_expansion_a11': ['cte', 'a20', 'a01', 'a21', 'a30', 'a10','a11'],
                                  'Dmid_mchirp_expansion_exp': ['cte', 'a20', 'a01', 'a21', 'a30', 'a10','a11', 'Mstar'],
                                  'Dmid_mchirp_expansion_asqrt': ['cte', 'a20', 'a01', 'a21', 'a30', 'asqrt'], 
                                  'Dmid_mchirp_power': ['cte', 'a20', 'a01', 'a21', 'a30', 'power_param']}
        
        self.emax_params_names = {'emax_exp' : ['gamma_opt, delta_opt, b_0, b_1, b_2'],
                                  'emax_sigmoid' : ['gamma_opt, delta_opt, b_0, k, M_0'],
                                  }
        
        
    def injection_set(self, run_dataset):
        
        assert run_dataset =='o1' or run_dataset == 'o2' or run_dataset == 'o3',\
        "Argument (run_dataset) must be 'o1' or 'o2' or 'o3'. "
        
        if run_dataset == 'o1' or run_dataset =='o2' :
            
            file = h5py.File(f'{run_dataset}-bbh-IMRPhenomXPHMpseudoFourPN.hdf5', 'r')
            
            atr = dict(file.attrs.items())
            
            #Total number of generated injections
            self.Ntotal = atr['total_generated'] 
            
            #Mass 1 and mass 2 values in the source frame in solar units
            self.m1 = file["events"][:]["mass1_source"]
            self.m2 = file["events"][:]["mass2_source"]

            #Redshift and luminosity distance [Mpc] values 
            self.z = file["events"][:]["z"]
            self.dL = file["events"][:]["distance"]
          
            #Joint mass sampling pdf (probability density function) values, p(m1,m2)
            self.m1_pdf = np.exp(file["events"][:]["logpdraw_mass1_source_GIVEN_z"])
            self.m2_pdf = np.exp(file["events"][:]["logpdraw_mass2_source_GIVEN_mass1_source"])
            self.m_pdf = self.m1_pdf * self.m2_pdf
            
            #Redshift sampling pdf values, p(z), corresponding to a redshift defined by a flat Lambda-Cold Dark Matter cosmology
            self.z_pdf = np.exp(file["events"][:]["logpdraw_z"])
            
            H0 = 67.9 #km/sMpc
            c = 3e5 #km/s
            omega_m = 0.3065
            A = np.sqrt(omega_m * (1 + self.z)**3 + 1 - omega_m)
            dL_dif = (c * (1 + self.z) / H0) * (1/A)
            
            #Luminosity distance sampling pdf values, p(dL), computed for a flat Lambda-Cold Dark Matter cosmology from the z_pdf values
            self.dL_pdf = self.z_pdf / dL_dif
            
            #mass chirp
            self.Mc = file["events"][:]["Mc_source"]
            self.Mc_det = file["events"][:]["Mc"]
            
            #total mass (m1+m2)
            self.Mtot = self.m1 + self.m2
            self.Mtot_det = self.m1 * (1+self.z) + self.m2 * (1+self.z)
            self.Mtot_max = 510.25378 #maximum Mtotal used to make the bbh fit in o3
            
            #eta aka symmetric mass ratio
            self.mu = (self.m1 * self.m2) / (self.m1 + self.m2)
            self.eta = self.mu / self.Mtot
            self.q = file["events"][:]["q"]
            
            self.s1x = file["events"][:]["spin1x"]
            self.s1y = file["events"][:]["spin1y"]
            self.s1z = file["events"][:]["spin1z"]
            
            self.s2x = file["events"][:]["spin2x"]
            self.s2y = file["events"][:]["spin2y"]
            self.s2z = file["events"][:]["spin2z"]
            
            self.a1 = np.sqrt(self.s1x**2 + self.s1y**2 + self.s1z**2)
            self.a2 = np.sqrt(self.s2x**2 + self.s2y**2 + self.s2z**2)
            
            #SNR
            self.snr = file["events"][:]["snr_net"]
            found_snr = self.snr >= self.thr_snr
            
            #indexes of the found injections
            self.found_any = found_snr
            print(f'Found inj in {run_dataset} set: ', self.found_any.sum())   
            
            self.max_index = np.argmax(self.dL)
            self.dLmax = self.dL[self.max_index]
            self.zmax = np.max(self.z)
            
            index = np.random.choice(np.arange(len(self.dL)), 200, replace=False)
            if self.max_index not in index:
                index = np.insert(index, -1, self.max_index)
                
            try_dL = self.dL[index]
            try_dLpdf = self.dL_pdf[index]
            
            self.max_dL_inter = np.max(try_dL)
        
            #we add 0 value
            inter_dL = np.insert(try_dL, 0, 0, axis=0)
            inter_dLpdf = np.insert(try_dLpdf, 0, 0, axis=0)
            self.interp_dL_pdf = interpolate.interp1d(inter_dL, inter_dLpdf)
            
            try_z = self.z[index]
            inter_z = np.insert(try_z, 0, 0, axis=0)
            
            #add a value for self.zmax
            new_dL = np.insert(inter_dL, -1, self.dLmax, axis=0)
            new_z = np.insert(inter_z, -1, self.zmax, axis=0)
            
            self.interp_z = interpolate.interp1d(new_dL, new_z)
        
        elif run_dataset == 'o3' : 
            
            file = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')
            
            #Total number of generated injections
            self.Ntotal = file.attrs['total_generated'] 
            
            #Mass 1 and mass 2 values in the source frame in solar units
            self.m1 = file["injections/mass1_source"][:]
            self.m2 = file["injections/mass2_source"][:]
            
            #Redshift and luminosity distance [Mpc] values 
            self.z = file["injections/redshift"][:]
            self.dL = file["injections/distance"][:]
          
            #Joint mass sampling pdf (probability density function) values, p(m1,m2)
            self.m_pdf = file["injections/mass1_source_mass2_source_sampling_pdf"][:]
            
            #Redshift sampling pdf values, p(z), corresponding to a redshift defined by a flat Lambda-Cold Dark Matter cosmology
            self.z_pdf = file["injections/redshift_sampling_pdf"][:]
            
            H0 = 67.9 #km/sMpc
            c = 3e5 #km/s
            omega_m = 0.3065
            A = np.sqrt(omega_m * (1 + self.z)**3 + 1 - omega_m)
            dL_dif = (c * (1 + self.z) / H0) * (1/A)
            
            #Luminosity distance sampling pdf values, p(dL), computed for a flat Lambda-Cold Dark Matter cosmology from the z_pdf values
            self.dL_pdf = self.z_pdf / dL_dif
            
            #mass chirp
            self.Mc = (self.m1 * self.m2)**(3/5) / (self.m1 + self.m2)**(1/5) 
            self.Mc_det = (self.m1 * self.m2 * (1+self.z)**2 )**(3/5) / (self.m1 * (1+self.z) + self.m2 * (1+self.z))**(1/5) 
            
            #total mass (m1+m2)
            self.Mtot = self.m1 + self.m2
            self.Mtot_det = self.m1 * (1+self.z) + self.m2 * (1+self.z)
            self.Mtot_max = np.max(self.Mtot_det)
            
            #eta aka symmetric mass ratio
            mu = (self.m1 * self.m2) / (self.m1 + self.m2)
            self.eta = mu / self.Mtot
            self.q = self.m2/self.m1
            
            #False alarm rate statistics from each pipeline
            self.far_pbbh = file["injections/far_pycbc_bbh"][:]
            self.far_gstlal = file["injections/far_gstlal"][:]
            self.far_mbta = file["injections/far_mbta"][:]
            self.far_pfull = file["injections/far_pycbc_hyperbank"][:]
            self.snr = file['injections/optimal_snr_net'][:]
            
            found_pbbh = self.far_pbbh <= self.thr_far
            found_gstlal = self.far_gstlal <= self.thr_far
            found_mbta = self.far_mbta <= self.thr_far
            found_pfull = self.far_pfull <= self.thr_far

            #indexes of the found injections
            self.found_any = found_pbbh | found_gstlal | found_mbta | found_pfull
            print(f'Found inj in {run_dataset} set: ', self.found_any.sum())      
            
            self.pow_m1 = file.attrs['pow_mass1']
            self.pow_m2 = file.attrs['pow_mass2']
            self.mmax = file.attrs['max_mass1']
            self.mmin = file.attrs['min_mass1']
            self.zmax = file.attrs['max_redshift']
            self.max_index = np.argmax(self.dL)
            self.dLmax = self.dL[self.max_index]
            
            index = np.random.choice(np.arange(len(self.dL)), 200, replace=False)
            if self.max_index not in index:
                index = np.insert(index, -1, self.max_index)
                
            try_dL = self.dL[index]
            try_dLpdf = self.dL_pdf[index]
            
            self.max_dL_inter = np.max(try_dL)
        
            #we add 0 value
            inter_dL = np.insert(try_dL, 0, 0, axis=0)
            inter_dLpdf = np.insert(try_dLpdf, 0, 0, axis=0)
            self.interp_dL_pdf = interpolate.interp1d(inter_dL, inter_dLpdf)
            
            try_z = self.z[index]
            inter_z = np.insert(try_z, 0, 0, axis=0)
            
            #add a value for self.zmax
            new_dL = np.insert(inter_dL, -1, self.dLmax, axis=0)
            new_z = np.insert(inter_z, -1, self.zmax, axis=0)
            
            self.interp_z = interpolate.interp1d(new_dL, new_z)
        
            
        return
    
    def get_opt_params(self, run_fit):
        
        assert run_fit =='o1' or run_fit == 'o2' or run_fit == 'o3',\
        "Argument (run_fit) must be 'o1' or 'o2' or 'o3'. "
        
        try:
            if self.alpha_vary is None:
                path = f'{run_fit}/{self.dmid_fun}' if self.emax_fun is None else f'{run_fit}/{self.dmid_fun}/{self.emax_fun}'
                    
            else: 
                path = f'{run_fit}/alpha_vary/{self.dmid_fun}' if self.emax_fun is None else f'{run_fit}/alpha_vary/{self.dmid_fun}/{self.emax_fun}'
        
            self.dmid_params = np.loadtxt( path + '/joint_fit_dmid.dat')[-1, :-1]
            self.shape_params = np.loadtxt( path + '/joint_fit_shape.dat')[-1, :-1]
        
        except:
            print('ERROR in self.get_opt_params: There are not such files because there is not a fit yet with these options.')
    
        return
    
    def get_ini_values(self):
        
        try:
            if self.alpha_vary is None:
                path = f'ini_values/{self.dmid_fun}' if self.emax_fun is None else f'ini_values/{self.dmid_fun}_{self.emax_fun}'
                    
            else: 
                path = f'ini_values/alpha_vary_{self.dmid_fun}' if emax_fun is None else f'ini_values/alpha_vary_{self.dmid_fun}_{self.emax_fun}'
            
            dmid_ini_values = np.loadtxt( path + '_dmid.dat')[-1]
            shape_ini_values = np.loadtxt( path + '_shape.dat')[-1]
        
        except:
            print('ERROR in self.get_ini_values: Files not found. Please create .dat files with the initial param guesses for this fit.')
    
        return dmid_ini_values, shape_ini_values
    
    def sigmoid(self, dL, dLmid, emax , gamma , delta , alpha = 2.05):
        """
        Sigmoid function used to estime the probability of detection of bbh events

        Parameters
        ----------
        dL : 1D array of the luminosity distance.
        dLmid : parameter or function standing for the dL at which Pdet = 0.5
        gamma : parameter controling the shape of the curve. 
        delta : parameter controling the shape of the curve.
        alpha : parameter controling the shape of the curve. The default is 2.05
        emax : parameter or function controling the maximun search sensitivity

        Returns
        -------
        array of detection probability.

        """
        frac = dL / dLmid
        denom = 1. + frac ** alpha * \
            np.exp(gamma * (frac - 1.) + delta * (frac**2 - 1.))
        return emax / denom
    
    def Dmid_mchirp(self, m1_det, m2_det, cte):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame (our first guess)

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        cte : parameter that we will be optimizing, float
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        M = m1_det + m2_det
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        return cte * Mc**(5/6) 

    def Dmid_mchirp_expansion(self, m1_det, m2_det, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame 

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        cte , a_20, a_01, a_21, a_30, a_10 = params
        
        M = m1_det + m2_det
        eta = m1_det * m2_det / M**2
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        
        pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M )
        
        return pol * Mc**(5/6)
    
    def Dmid_mchirp_expansion_asqrt(self, m1_det, m2_det, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame 

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        cte , a_20, a_01, a_21, a_30, a_sqrt = params
        
        M = m1_det + m2_det
        eta = m1_det * m2_det / M**2
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        
        pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_sqrt * M**(1/2) )
        
        return pol * Mc**(5/6) 
    
    def Dmid_mchirp_expansion_a11(self, m1_det, m2_det, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame 

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """ 
        cte , a_20, a_01, a_21, a_30, a_10, a_11 = params
        
        M = m1_det + m2_det
        eta = m1_det * m2_det / M**2
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        
        pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M + a_11 * M * (1 - 4*eta))
        
        return pol * Mc**(5/6)
    
    
    def Dmid_mchirp_power(self, m1_det, m2_det, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame 
        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        cte , a_20, a_01, a_21, a_30, power_param = params
        
        M = m1_det + m2_det
        eta = m1_det * m2_det / M**2
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        
        pol = cte *(1+ a_20 * M**2 / 2 + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) / 2 + a_30 * M**3 )
        
        return pol * Mc**((5+power_param)/6)
    
    def Dmid_mchirp_expansion_exp(self, m1_det, m2_det, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame 

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        cte , a_20, a_01, a_21, a_30, a_10, a_11, Mstar = params
        
        M = m1_det + m2_det
        eta = m1_det * m2_det / M**2
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        
        pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M + a_11 * M * (1 - 4*eta))
        
        return pol * Mc**(5/6) * np.exp(-M/Mstar)
    
    def Dmid_mchirp_expansion_noa30(self, m1_det, m2_det, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame 

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        cte , a_20, a_01, a_21, a_10, a_11 = params

        M = m1_det + m2_det
        eta = m1_det * m2_det / M**2
        
        Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
        
        pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))
        
        return pol * Mc**(5/6)
    
    def emax_exp(self, m1_det, m2_det, params):
        """
        maximun search sensitivity (emax) as a function of the masses
        in the detector frame

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array

        Returns
        -------
        emax(m1, m2) in the detector frame

        """
        Mtot = m1_det + m2_det
        b_0, b_1, b_2 = params
        return 1 - np.exp(b_0 + b_1 * Mtot + b_2 * Mtot**2)
    
    def emax_sigmoid(self, m1_det, m2_det, params):
        """
        maximun search sensitivity (emax) as a function of the masses
        in the detector frame

        Parameters
        ----------
        m1_det : detector frame mass1, float or 1D array
        m2_det: detector frame mass2, float or 1D array
        params : parameters that we will be optimizing, 1D array

        Returns
        -------
        emax(m1, m2) in the detector frame

        """
        Mtot = m1_det + m2_det
        b_0, k, M_0 = params
        L = 1 - np.exp(b_0)
        return L / (1 + np.exp(k * (Mtot - M_0)))
    
 
    def fun_m_pdf(self, m1, m2):
        """
        Function for the mass pdf aka p(m1,m2)
        
        Parameters
        ----------
        m1 : source frame mass1, float or 1D array
        m2 : source frame mass2, float or 1D array

        Returns
        -------
        float or array

        """
        # if m2 > m1:
        #   return 0
        
        mmin = self.mmin ; mmax = self.mmax
        alpha = self.pow_m1 ; beta = self.pow_m2
        
        m1_norm = (1. + alpha) / (mmax ** (1. + alpha) - mmin ** (1. + alpha))
        m2_norm = (1. + beta) / (m1 ** (1. + beta) - mmin ** (1. + beta))
        
        return m1**alpha * m2**beta * m1_norm * m2_norm * np.heaviside(m1-m2, 1)
    
    def apply_dmid_mtotal_max(self, dmid_values, Mtot_det, max_mtot = None):
        max_mtot = max_mtot if max_mtot != None else self.Mtot_max
        #return np.putmask(dmid_values, Mtot_det > max_mtot, 0.001)
        dmid_values[Mtot_det > max_mtot] = 0.001
        return dmid_values
    
    
    def Nexp(self, dmid_params, shape_params):
        """
        Expected number of found injections, computed as a sum over the 
        probability of detection of every injection

        Parameters
        ----------
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        emax_fun : str, name of the method used for the emax function
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]

        Returns
        -------
        float

        """
        m1_det = self.m1 * (1 + self.z) 
        m2_det = self.m2 * (1 + self.z)
        mtot_det = m1_det + m2_det
        
        dmid_values = self.dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values, mtot_det)
        
        if self.emax_fun is None and self.alpha_vary is None:

            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax, gamma, delta))
            
            
        elif self.emax_fun is None and self.alpha_vary is not None:
            
            gamma, delta, emax, alpha = shape_params[0], shape_params[1], shape_params[2], shape_params[3]
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax, gamma, delta, alpha))
        
        
        elif self.emax_fun is not None and self.alpha_vary is None:
            
            gamma, delta = shape_params[0], shape_params[1]
            emax_params = shape_params[2:]
            
            emax_values = self.emax(self, m1_det, m2_det, emax_params)
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax_values, gamma, delta))
          
            
        else:
            
            gamma, delta, alpha = shape_params[0], shape_params[1], shape_params[2]
            emax_params = shape_params[3:]
            
            emax_values = self.emax(self, m1_det, m2_det, emax_params)
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax_values, gamma, delta, alpha))
        
        return Nexp
        
    def lamda(self, dmid_params, shape_params):
        """
        Number density at found injectionsaka lambda(D,m1,m2)

        Parameters
        ----------
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        emax_fun : str, name of the method used for the emax function
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]

        Returns
        -------
        float

        """
        dL = self.dL[self.found_any]
        dL_pdf = self.dL_pdf[self.found_any]
        m_pdf = self.m_pdf[self.found_any]
        z = self.z[self.found_any]
        m1 = self.m1[self.found_any]
        m2 = self.m2[self.found_any]
        
        m1_det = m1 * (1 + z) 
        m2_det = m2 * (1 + z)
        mtot_det = m1_det + m2_det
        
        dmid_values = self.dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values, mtot_det)
        
        if self.emax_fun is None and self.alpha_vary is None:
            
            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            
            lamda = self.sigmoid(dL, dmid_values, emax, gamma, delta) * m_pdf * dL_pdf * self.Ntotal
            
            
        elif self.emax_fun is None and self.alpha_vary is not None:
            
            gamma, delta, emax, alpha = shape_params[0], shape_params[1], shape_params[2], shape_params[3]
            
            lamda = self.sigmoid(dL, dmid_values, emax, gamma, delta, alpha) * m_pdf * dL_pdf * self.Ntotal
            
        
        elif self.emax_fun is not None and self.alpha_vary is None:
            
            gamma, delta = shape_params[0], shape_params[1]
            emax_params = shape_params[2:]
            
            emax_values = self.emax(self, m1_det, m2_det, emax_params)
            
            lamda = self.sigmoid(dL, dmid_values, emax_values, gamma, delta) * m_pdf * dL_pdf * self.Ntotal


        else:
            
            gamma, delta, alpha = shape_params[0], shape_params[1], shape_params[2]
            emax_params = shape_params[3:]
            
            emax_values = self.emax(self, m1_det, m2_det, emax_params)
            
            lamda = self.sigmoid(dL, dmid_values, emax_values, gamma, delta, alpha) * m_pdf * dL_pdf * self.Ntotal

        return lamda
    
    def logL_dmid(self, dmid_params, shape_params):
        """
        log likelihood of the expected density of found injections

        Parameters
        ----------
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        emax_fun : str, name of the method used for the emax function
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]

        Returns
        -------
        float 

        """
        lnL = -self.Nexp(dmid_params, shape_params) + np.sum(np.log(self.lamda(dmid_params, shape_params)))
        return lnL
    
    def logL_shape(self, dmid_params, shape_params):
        """
        log likelihood of the expected density of found injections

        Parameters
        ----------
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        emax_fun : str, name of the method used for the emax function
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]

        Returns
        -------
        float 

        """
        shape_params[1] = np.exp(shape_params[1])
         
        lnL = -self.Nexp(dmid_params, shape_params) + np.sum(np.log(self.lamda(dmid_params, shape_params)))
        return lnL
        
    
    def MLE_dmid(self, methods):
        """
        minimization of -logL on dmid

        Parameters
        ----------
        methods : str, scipy method used to minimize -logL
        dmid_fun : str, name of the method used for the dmid function
        dmid_params_guess : initial guess for dmid params, 1D array
        shape_params : gamma, delta and emax params, 1D array
        emax_fun : str, name of the method used for the emax function
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]


        Returns
        -------
        params_res : optimized value for Dmid params, 1D array
        -min_likelihood : maximum log likelihood, float

        """
        dmid_params_guess = np.copy(self.dmid_params)
            
        res = opt.minimize(fun=lambda in_param: -self.logL_dmid(in_param, self.shape_params), 
                           x0=np.array([dmid_params_guess]), 
                           args=(), 
                           method=methods)
        opt_params = res.x
        min_likelihood = res.fun   
        self.dmid_params =  opt_params  
          
        return opt_params, -min_likelihood
    
    def MLE_shape(self, methods):
        """
        minimization of -logL on shape params (with emax as a cte)

        Parameters
        ----------
        methods : scipy method used to minimize -logL
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : parameters of the Dmid function, 1D array
        shape_params_guess : initial guess for gamma, delta and emax, 1D array

        Returns
        -------
        gamma: optimized value for gamma, float
        delta: optimized value for delta, float
        emax: optimized value for emax, float
        -min_likelihood : maximum log likelihood, float

        """
    
        shape_params_guess = np.copy(self.shape_params)
        shape_params_guess[1] = np.log(shape_params_guess[1])
        
        res = opt.minimize(fun=lambda in_param: -self.logL_shape(self.dmid_params, in_param), 
                           x0=np.array([shape_params_guess]), 
                           args=(), 
                           method=methods)
        
        opt_params = res.x
        opt_params[1] = np.exp(opt_params[1])
        min_likelihood = res.fun  
        self.shape_params = opt_params
        
        return opt_params, -min_likelihood
    
    def MLE_emax(self, methods):
        """
        minimization of -logL on shape params (with emax as a function of masses)

        Parameters
        ----------
        methods : scipy method used to minimize -logL
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : parameters of the Dmid function, 1D array
        shape_params_guess : initial guess for [gamma, delta, emax_params], 1D array
        emax_fun : str, name of the method used for the emax function

        Returns
        -------
        opt_params: optimized values for [gamma, delta, emax_params], 1D array
        -min_likelihood : maximum log likelihood, float

        """
        shape_params_guess = np.copy(self.shape_params)
        shape_params_guess[1] = np.log(shape_params_guess[1])
        
        res = opt.minimize(fun=lambda in_param: -self.logL_shape(self.dmid_params, in_param), 
                           x0=np.array([shape_params_guess]), 
                           args=(), 
                           method=methods)
        
        opt_params = res.x
        opt_params[1] = np.exp(opt_params[1])
        min_likelihood = res.fun  
        self.shape_params = opt_params

        return opt_params, -min_likelihood
    
    def joint_MLE(self, run_dataset, run_fit, methods, precision = 1e-2):
        '''
        joint optimization of log likelihood, alternating between optimizing dmid params and shape params
        until the difference in the log L is <= precision . Saves the results of each iteration in txt files.

        Parameters
        ----------
        methods : str, scipy method used to minimize -logL
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : 1D array, initial guess for dmid params
        shape_params : 1D array, initial guess for shape params
        emax_fun : str, optional. Name of the method used for the emax function. The default is None.
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]
        precision : float (positive), optional. Tolerance for termination . The default is 1e-2.

        Returns
        -------
        None.

        '''
        
        total_lnL = np.zeros([1])
        all_gamma = []
        all_delta = []
        all_emax = []
        all_alpha = []
        all_dmid_params = np.zeros([1,len(self.dmid_params)])
        
        if self.alpha_vary is None:
            all_emax_params = np.zeros([1,len(self.shape_params[2:])])
            params_emax = None if self.emax is None else np.copy(self.shape_params[2:])
            shape_params_names_file = 'gamma_opt, delta_opt, emax_opt, maxL'
            path = f'{run_fit}/{self.dmid_fun}'
            
        else:
            all_emax_params = np.zeros([1,len(self.shape_params[3:])])
            params_emax = None if self.emax_fun is None else np.copy(self.shape_params[3:])
            
            shape_params_names_file = 'gamma_opt, delta_opt, emax_opt, alpha_opt, maxL'
            path = f'{run_fit}/alpha_vary/{self.dmid_fun}'
        
        for i in range(0, 10000):
            
            dmid_params, maxL_1 = self.MLE_dmid(methods)
            all_dmid_params = np.vstack([all_dmid_params, dmid_params])
            
            if self.emax_fun is None:
                shape_params, maxL_2 = self.MLE_shape(methods)
                gamma_opt, delta_opt, emax_opt = shape_params[:2]
                all_gamma.append(gamma_opt); all_delta.append(delta_opt)
                all_emax.append(emax_opt); total_lnL = np.append(total_lnL, maxL_2)
                
                if self.alpha_vary is not None:
                    alpha_opt = shape_params[3]
                    all_alpha.append(alpha_opt)
                
            
            elif self.emax_fun is not None:
                shape_params, maxL_2 = self.MLE_emax(methods)
                gamma_opt, delta_opt = shape_params[:2]
                
                if self.alpha_vary is not None:
                    alpha_opt = shape_params[2]
                    all_alpha.append(alpha_opt)
                    params_emax = shape_params[3:]
                    
                
                else:
                    params_emax = shape_params[2:]
                    
                all_gamma.append(gamma_opt); all_delta.append(delta_opt)
                all_emax_params = np.vstack([all_emax_params, params_emax])
                total_lnL = np.append(total_lnL, maxL_2)
            
            print('\n', maxL_2)
            print(np.abs( total_lnL[i+1] - total_lnL[i] ))
            
            if np.abs( total_lnL[i+1] - total_lnL[i] ) <= precision : break
        
        print('\nNumber of needed iterations with precision <= %s : %s (+1 since it starts on 0)' %(precision, i))

        total_lnL = np.delete(total_lnL, 0)
        
        if self.emax_fun is None:
            shape_results = np.column_stack((all_gamma, all_delta, all_emax, total_lnL)) if alpha_vary is None else np.column_stack((all_gamma, all_delta, all_emax, all_alpha, total_lnL))
            shape_header = shape_params_names_file
            name_dmid = path + '/joint_fit_dmid.dat'
            np.savetxt(path + '/joint_fit_shape.dat', shape_results, header = shape_header, fmt='%s')
        
        elif self.emax_fun is not None:
            shape_results = np.column_stack((all_gamma, all_delta, np.delete(all_emax_params, 0, axis=0), total_lnL)) if alpha_vary is None else np.column_stack((all_gamma, all_delta, all_alpha, np.delete(all_emax_params, 0, axis=0), total_lnL)) 
            shape_header = f'{self.emax_params_names[self.emax]} , maxL'
            name_dmid = path + f'/{self.emax_fun}/joint_fit_dmid_emaxfun.dat'
            np.savetxt(path + f'/{self.emax_fun}/joint_fit_shape_emaxfun.dat', shape_results, header = shape_header, fmt='%s')
        
        
        all_dmid_params = np.delete(all_dmid_params, 0, axis=0)
        dmid_results = np.column_stack((all_dmid_params, total_lnL))
        dmid_header = f'{self.dmid_params_names[self.dmid_fun]} , maxL'
        np.savetxt(name_dmid, dmid_results, header = dmid_header, fmt='%s')
        
        return


    def cumulative_dist(self, run_dataset, run_fit, var):
        '''
        Saves cumulative distributions plots and prints KS tests for the specified variables  

        Parameters
        ----------
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : 1D array, parameters of Dmid function
        shape_params : 1D array, [gamma, delta, emax] or [gamma, delta, emax_params]
        var : str, variable for the CDFs and KS tests. Options:
            'dL' - luminosity distance
            'Mc' - chirp mass
            'Mtot' - total mass 
            'eta' - symmetric mass ratio
            'Mc_det' - chirp mass in the detector frame
            'Mtot_det' - total mass in the detector frame
        emax_fun : str, optional. Name of the method used for the emax function. The default is None.
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]

        Returns
        -------
        stat : float, statistic from the KStest 
        pvalue : float, pvalue from the KStest 

        '''
        
        self.get_opt_params(run_fit)
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds'}
        
        dic = {'dL': self.dL, 'Mc': self.Mc, 'Mtot': self.Mtot, 'eta': self.eta, 'Mc_det': self.Mc_det, 'Mtot_det': self.Mtot_det}
        path = f'{dmid_fun}' if self.alpha_vary is None else f'alpha_vary/{self.dmid_fun}'
        
        try:
            os.mkdir( path + f'/{emax_dic[self.emax_fun]}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
           
        #cumulative distribution over the desired variable
        indexo = np.argsort(dic[var])
        varo = dic[var][indexo]
        dLo = self.dL[indexo]
        m1o = self.m1[indexo]
        m2o = self.m2[indexo]
        zo = self.z[indexo]
        m1o_det = m1o * (1 + zo) 
        m2o_det = m2o * (1 + zo)
        mtoto_det = m1o_det + m2o_det
        
        dmid_values = self.dmid(self, m1o_det, m2o_det, self.dmid_params)
        self.apply_dmid_mtotal_max(dmid_values, mtoto_det)
        
        if self.emax_fun is not None:
            
            emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
            emax_values = self.emax(self, m1o_det, m2o_det, emax_params)
            
            gamma, delta = np.copy(self.shape_params[:2])
            alpha = 2.05  if alpha_vary is None else self.shape_params[2]
            
            cmd = np.cumsum(self.sigmoid(dLo, dmid_values, emax_values, gamma, delta, alpha))
            
            
        else:
            gamma, delta, emax = np.copy(self.shape_params[:3])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
            
            cmd = np.cumsum(self.sigmoid(dLo, dmid_values, emax, gamma, delta, alpha))
        
        #found injections
        var_found = dic[var][self.found_any]
        indexo_found = np.argsort(var_found)
        var_foundo = var_found[indexo_found]
        real_found_inj = np.arange(len(var_foundo))+1
    
        plt.figure()
        plt.plot(varo, cmd, '.', markersize=2, label='model')
        plt.plot(var_foundo, real_found_inj, '.', markersize=2, label='found injections')
        plt.xlabel(f'{var}$^*$')
        plt.ylabel('Cumulative found injections')
        plt.legend(loc='best')
        name = path + f'/{emax_dic[self.emax_fun]}/{var}_cumulative.png'
        plt.savefig(name, format='png')
        
        #KS test
        
        m1_det = self.m1 * (1 + self.z) 
        m2_det = self.m2 * (1 + self.z)
        mtot_det = m1_det + m2_det
        
        dmid_values = self.dmid(self, m1_det, m2_det, self.dmid_params)
        self.apply_dmid_mtotal_max(dmid_values, mtot_det)
        
        if self.emax_fun is not None:
            emax_values = self.emax(self, m1_det, m2_det, emax_params)
            pdet = self.sigmoid(self.dL, dmid_values, emax_values, gamma, delta, alpha)
            
        else:    
            pdet = self.sigmoid(self.dL, dmid_values, emax, gamma, delta, alpha)
        
        def cdf(x):
            values = [np.sum(pdet[dic[var]<value])/np.sum(pdet) for value in x]
            return np.array(values)
            
        stat, pvalue = kstest(var_foundo, lambda x: cdf(x) )
        
        return stat, pvalue

    
    
    def binned_cumulative_dist(self, run_dataset, run_fit, nbins, var_cmd , var_binned):
        '''
        Saves binned cumulative distributions and prints binned KS tests for the specified variables 

        Parameters
        ----------
        nbins : int, number of bins
        dmid_fun : str, name of the method used for the dmid function
        dmid_params : 1D array, parameters of Dmid function
        shape_params : 1D array, [gamma, delta, emax] or [gamma, delta, emax_params]
        var_cmd : str, variable for the CDFs and KS tests. Options:
            'dL' - luminosity distance
            'Mc' - chirp mass
            'Mtot' - total mass 
            'eta' - symmetric mass ratio
            'Mc_det' - chirp mass in the detector frame
            'Mtot_det' - total mass in the detector frame
        var_binned : str, variable in which we are taking bins. Same options as var_cmd
        emax_fun : str, optional. Name of the method used for the emax function. The default is None.
                   if given None, shape_params should be only 3 numbers
                   if given a method, shape_params should be [gamma, delta, emax_params]

        Returns
        -------
        None.

        '''
        
        self.injection_set(run_dataset)
        
        self.get_opt_params(run_fit)
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds'}
        path = f'{self.dmid_fun}' if self.alpha_vary is None else f'alpha_vary/{self.dmid_fun}'
        
        try:
            os.mkdir( path + f'/{emax_dic[self.emax_fun]}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        try:
            os.mkdir( path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        try:
            os.mkdir( path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        gamma, delta = np.copy(self.shape_params[:2])
        alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
    
        bin_dic = {'dL': self.dL, 'Mc': self.Mc, 'Mtot': self.Mtot, 'eta': self.eta, 'Mc_det': self.Mc_det, 'Mtot_det': self.Mtot_det}
        
        #sort data
        data_not_sorted = bin_dic[var_binned]
        index = np.argsort(data_not_sorted)
        data = data_not_sorted[index]
        
        dLo = self.dL[index]; m1o = self.m1[index]; m2o = self.m2[index]; zo = self.z[index]
        m1o_det = m1o * (1 + zo) 
        m2o_det = m2o * (1 + zo)
        Mco = self.Mc[index]; Mtoto = self.Mtot[index]; etao = self.eta[index]
        Mco_det = self.Mc_det[index]; Mtoto_det = self.Mtot_det[index]
        found_any_o = self.found_any[index]
        
        #create bins with equally amount of data
        def equal_bin(N, m):
            sep = (N.size/float(m))*np.arange(1,m+1)
            idx = sep.searchsorted(np.arange(N.size))
            return idx[N.argsort().argsort()]
        
        index_bins = equal_bin(data, nbins)
        
        print(f'\n{var_binned} bins:\n')
        
        for i in range(nbins):
            #get data in each bin
            data_inbin = data[index_bins==i]
            dL_inbin = dLo[index_bins==i]
            m1_inbin = m1o[index_bins==i]
            m2_inbin = m2o[index_bins==i]
            z_inbin = zo[index_bins==i]
            m1_det_inbin = m1o_det[index_bins==i]
            m2_det_inbin = m2o_det[index_bins==i] 
            
            Mc_inbin = Mco[index_bins==i]
            Mtot_inbin = Mtoto[index_bins==i]
            eta_inbin = etao[index_bins==i]
            Mc_det_inbin = Mco_det[index_bins==i]
            Mtot_det_inbin = Mtoto_det[index_bins==i]
            
            cmd_dic = {'dL': dL_inbin, 'Mc': Mc_inbin, 'Mtot': Mtot_inbin, 'eta': eta_inbin, 'Mc_det': Mc_det_inbin, 'Mtot_det': Mtot_det_inbin}
        
            #cumulative distribution over the desired variable
            indexo = np.argsort(cmd_dic[var_cmd])
            varo = cmd_dic[var_cmd][indexo]
            dL = dL_inbin[indexo]
            m1 = m1_inbin[indexo]
            m2 = m2_inbin[indexo]
            z = z_inbin[indexo]
            m1_det = m1_det_inbin[indexo]
            m2_det = m2_det_inbin[indexo]
            mtot_det = m1_det + m2_det
            
            dmid_values = self.dmid(self, m1_det, m2_det, self.dmid_params)
            self.apply_dmid_mtotal_max(dmid_values, mtot_det)
            
            if self.emax_fun is not None:
                emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
                gamma, delta = np.copy(self.shape_params[:2])
                alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
                emax_values = self.emax(self, m1_det, m2_det, emax_params)
                
                cmd = np.cumsum(self.sigmoid(dL, dmid_values, emax_values, gamma, delta, alpha))
            
            
            else:
                gamma, delta, emax = np.copy(self.shape_params[:3])
                alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
                
                cmd = np.cumsum(self.sigmoid(dL, dmid_values, emax, gamma, delta, alpha))
            
            #found injections
            found_inj_index_inbin = found_any_o[index_bins==i]
            found_inj_inbin = cmd_dic[var_cmd][found_inj_index_inbin]
            indexo_found = np.argsort(found_inj_inbin)
            found_inj_inbin_sorted = found_inj_inbin[indexo_found]
            real_found_inj = np.arange(len(found_inj_inbin_sorted ))+1
        
            plt.figure()
            plt.plot(varo, cmd, '.', markersize=2, label='model')
            plt.plot(found_inj_inbin_sorted, real_found_inj, '.', markersize=2, label='found injections')
            plt.xlabel(f'{var_cmd}$^*$')
            plt.ylabel('Cumulative found injections')
            plt.legend(loc='best')
            plt.title(f'{var_binned} bin {i} : {data_inbin[0]:.6} - {data_inbin[-1]:.6}')
            name = path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/{i}.png'
            plt.savefig(name, format='png')
            plt.close()
            
            #KS test
            
            if self.emax_fun is not None:
                emax_values = self.emax(self, m1_det_inbin, m2_det_inbin, emax_params)
                pdet = self.sigmoid(dL_inbin, dmid_values, emax_values, gamma, delta, alpha)
                
            else:    
                pdet = self.sigmoid(dL_inbin, dmid_values, emax, gamma, delta, alpha)
            
            def cdf(x):
                values = [np.sum(pdet[cmd_dic[var_cmd]<value])/np.sum(pdet) for value in x]
                return np.array(values)
            
            stat, pvalue = kstest(found_inj_inbin_sorted, lambda x: cdf(x) )
            
            print(f'{var_cmd} KStest in {i} bin: statistic = %s , pvalue = %s' %(stat, pvalue))
        
        print('')   
        return
    
    def pdet_only_masses(self, run_dataset, run_fit, m1, m2):
        
        #self.injection_set(run_dataset)
        self.get_opt_params(run_fit)
        
        dmid = lambda dL_int : self.dmid(self, m1*(1 + self.interp_z(dL_int)), m2*(1 + self.interp_z(dL_int)), self.dmid_params)
        mtot = lambda dL_int :m1*(1 + self.interp_z(dL_int)) + m2*(1 + self.interp_z(dL_int))
        
        if self.emax_fun is not None:
            emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
            gamma, delta = np.copy(self.shape_params[:2])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
            
            emax = lambda dL_int : self.emax(self, m1*(1 + self.interp_z(dL_int)), m2*(1 + self.interp_z(dL_int)), emax_params)
            
            quad_fun = lambda dL_int : self.sigmoid(dL_int, self.apply_dmid_mtotal_max(np.array(dmid(dL_int)), np.array(mtot(dL_int))), emax(dL_int) , gamma , delta, alpha) * self.interp_dL_pdf(dL_int)
            
        else:
            gamma, delta, emax = np.copy(self.shape_params[:3])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
        
            quad_fun = lambda dL_int : self.sigmoid(dL_int, self.apply_dmid_mtotal_max(dmid(dL_int), mtot(dL_int)), emax , gamma , delta, alpha) * self.interp_dL_pdf(dL_int)
            
        pdet =  integrate.quad(quad_fun, 0, self.max_dL_inter)[0]
        
        return pdet
    
    def find_proportion_found_inj(self, run_dataset, run_fit):
        
        self.injection_set(run_dataset)
        Nfound = self.found_any.sum()
        print(Nfound)
        
        self.get_opt_params(run_fit)
        
        dmid_params = np.copy(self.dmid_params)
        dmid_params[0] = 1.

        
        m1_det = self.m1 * (1 + self.z)
        m2_det = self.m2 * (1 + self.z)
        mtot_det = m1_det + m2_det
        
        dmid_values = self.dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values, mtot_det)
        
        if self.emax_fun is not None:
            emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
            gamma, delta = np.copy(self.shape_params[:2])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
            
            emax = self.emax(self, m1_det, m2_det, emax_params)
            
        else:
            gamma, delta, emax = np.copy(self.shape_params[:3])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
        
        def find_root(x):
            frac = self.dL / ( x * dmid_values)
            denom = 1. + frac ** alpha * \
                    np.exp(gamma* (frac - 1.) + delta * (frac**2 - 1.))
            return  np.sum( emax / denom )  - Nfound
        
        proportion = fsolve(find_root, [50])
        
        return proportion
    
    
    def predicted_events(self, run_fit):
        
        self.injection_set(run_dataset)
        Nfound = self.found_any.sum()
        print(Nfound)
        
        self.get_opt_params(run_fit)
        
        dmid_params = np.copy(self.dmid_params)
        dmid_params[0] = 1.

        
        m1_det = self.m1 * (1 + self.z)
        m2_det = self.m2 * (1 + self.z)
        mtot_det = m1_det + m2_det
        
        dmid_values = self.dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values, mtot_det)
        
        if self.emax_fun is not None:
            emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
            gamma, delta = np.copy(self.shape_params[:2])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
            
            emax = self.emax(self, m1_det, m2_det, emax_params)
            
        else:
            gamma, delta, emax = np.copy(self.shape_params[:3])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
        
        def manual_found_inj(x):
            frac = self.dL / ( x * dmid_values)
            denom = 1. + frac ** alpha * \
                    np.exp(gamma* (frac - 1.) + delta * (frac**2 - 1.))
            return  np.sum( emax / denom )
        
        o1_inj = manual_found_inj(self.find_proportion_found_inj(self, 'o1', run_fit))
        o2_inj = manual_found_inj(self.find_proportion_found_inj(self, 'o2', run_fit))
        o3_inj = self.injection_set('o3').found_any.sum()
        
        frac1 = o1_inj / o3_inj
        frac2 = o2_inj / o3_inj
        
        pred_rates = { 'o1' : frac1 * self.det_rates['o3'], 'o2' : frac2 * self.det_rates['o3'] }
        pred_nev = { 'o1' : pred_rates['o1'] * self.obs_time['o1'], 'o2' : pred_rates['o2'] * self.obs_time['o2'] }
        
        return pred_nev
    
    def total_pdet(self, dL, m1_det, m2_det):
        
        runs = ['o1', 'o2', 'o3']
        pdet = []
        fracs = np.array(self.prop_obs_time)
        
        for i in range(len(runs)):
            
        
            self.get_opt_params(runs[i])
            
            mtot_det = m1_det + m2_det
            
            dmid_values = self.dmid(self, m1_det, m2_det, self.dmid_params)
            self.apply_dmid_mtotal_max(np.array(dmid_values), mtot_det)
            
            if self.emax_fun is not None:
                emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
                gamma, delta = np.copy(self.shape_params[:2])
                alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
                
                emax = self.emax(self, m1_det, m2_det, emax_params)
                
            else:
                gamma, delta, emax = np.copy(self.shape_params[:3])
                alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
            
            pdet_i = self.sigmoid(dL, dmid_values, emax , gamma , delta , alpha)
            pdet.append(pdet_i)
            
        pdet = np.array(pdet)
        p1 = self.prop_obs_time[0] * pdet[0, :]
        p2 = self.prop_obs_time[1] * pdet[1, :]
        p3 = self.prop_obs_time[2] * pdet[2, :]
            
        return p1 + p2 + p3
        

plt.close('all')

run_fit = 'o3'
run_dataset = 'o3'

# function for dmid and emax we wanna use
dmid_fun = 'Dmid_mchirp_expansion_noa30'
#dmid_fun = 'Dmid_mchirp_expansion_exp'
#dmid_fun = 'Dmid_mchirp_expansion_a11'
#dmid_fun = 'Dmid_mchirp_expansion_asqrt'
#dmid_fun = 'Dmid_mchirp_expansion'
emax_fun = 'emax_exp'
#emax_fun = 'emax_sigmoid'

alpha_vary = None

path = f'{run_dataset}/{dmid_fun}' if alpha_vary is None else f'{run_dataset}/alpha_vary/{dmid_fun}'
if emax_fun is not None:
    path = path + f'/{emax_fun}'

try:
    os.mkdir(f'{run_dataset}')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
if alpha_vary is not None:
    try:
        os.mkdir(f'{run_dataset}/alpha_vary')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
try:
    os.mkdir(path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir(path + f'/{emax_fun}')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
data = Found_injections(dmid_fun, emax_fun, alpha_vary)

data.injection_set(run_dataset)
data.get_opt_params(run_fit)

# index = np.random.choice(np.arange(len(data.dL)), 300, replace=False)
# m1 = data.m1[index]
# m2=data.m2[index]

# pdet = np.array([data.pdet_only_masses(run_dataset, run_fit, m1[i], m2[i]) for i in range(len(m1))])

# plt.figure()
# plt.scatter(m1, m2, s=1, c=pdet)
# plt.xlabel('m1')
# plt.ylabel('m2')
# plt.colorbar(label='Pdet(m1,m2)')
# plt.savefig( path + '/Pdet_m1m2.png')

m1_det = data.m1*(1+data.z)
m2_det = data.m2*(1+data.z)

total_pdet = data.total_pdet(data.dL, m1_det, m2_det)

plt.figure()
plt.scatter(data.dL/data.dmid(data, m1_det, m2_det, data.dmid_params), total_pdet, s=1)
plt.xlabel('dL/dmid')
plt.ylabel('total Pdet')
plt.savefig( path + '/total_pdet.png')



