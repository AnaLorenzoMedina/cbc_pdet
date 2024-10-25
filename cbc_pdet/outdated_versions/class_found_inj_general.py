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

class Found_injections:
    """
    Class for an algorithm of GW detected found injections from signal templates 
    of binary black hole mergers
    
    Input: an h5py file with the parameters of every sample and the threshold for 
    the false alarm rate (FAR). The default value is thr = 1, which means we are 
    considering a signal as detected when FAR <= 1.

    """
    
    def __init__(self, file, thr = 1):
        
        assert isinstance(file, h5py._hl.files.File),\
        "Argument (file) must be an h5py file."
               
        self.data = file
        
        assert isinstance(thr, float) or isinstance(thr, int),\
        "Argument (thr) must be a float or an integrer."
        
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
        
        found_pbbh = self.far_pbbh <= thr
        found_gstlal = self.far_gstlal <= thr
        found_mbta = self.far_mbta <= thr
        found_pfull = self.far_pfull <= thr

        #indexes of the found injections
        self.found_any = found_pbbh | found_gstlal | found_mbta | found_pfull
        print(self.found_any.sum())      
        
        self.pow_m1 = file.attrs['pow_mass1']
        self.pow_m2 = file.attrs['pow_mass2']
        self.mmax = file.attrs['max_mass1']
        self.mmin = file.attrs['min_mass1']
        self.zmax = file.attrs['max_redshift']
        self.max_index = np.argmax(self.dL)
        self.dLmax = self.dL[self.max_index]
        
        self.gamma_opt = -0.7435398875859958 
        self.delta_opt = 0.2511010365079828
        self.emax_opt = 0.79928
        
        self.b0_opt = -5.632796685161368
        self.b1_opt = 0.025575280643270047
        self.b2_opt = -3.1030795279503675e-05
        
        self.cte_opt = 82.86080855190852
        self.a20_opt = -3.843054082930893e-06
        self.a01_opt = -0.4726476591881541
        self.a21_opt = 4.304672026161589e-06
        self.a10_opt = 0.0004746871925697556
        self.a11_opt = -0.0014672425957000098
        self.a30_opt = 0
        self.Mstar = 952.1391466741634
        self.asqrt_opt = 0
        
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
        
        print('finished initializing')
    
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
        
        return m1**alpha * m2**beta * m1_norm * m2_norm
    
    def apply_dmid_mtotal_max(self, dmid_values, max_mtot = None):
        max_mtot = max_mtot if max_mtot != None else self.Mtot_max
        return np.putmask(dmid_values, self.Mtot_det > max_mtot, 0.001)
    
    
    def Nexp(self, dmid_fun, dmid_params, shape_params, alpha_vary = None, emax_fun = None):
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
        
        dmid = getattr(Found_injections, dmid_fun)
        
        m1_det = self.m1 * (1 + self.z) 
        m2_det = self.m2 * (1 + self.z)
        
        dmid_values = dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values)
        
        if emax_fun is None and alpha_vary is None:

            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax, gamma, delta))
            
            
        elif emax_fun is None and alpha_vary is not None:
            
            gamma, delta, emax, alpha = shape_params[0], shape_params[1], shape_params[2], shape_params[3]
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax, gamma, delta, alpha))
        
        
        elif emax_fun is not None and alpha_vary is None:
            
            emax = getattr(Found_injections, emax_fun)
            
            gamma, delta = shape_params[0], shape_params[1]
            emax_params = shape_params[2:]
            
            emax_values = emax(self, m1_det, m2_det, emax_params)
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax_values, gamma, delta))
          
            
        else:
            
            emax = getattr(Found_injections, emax_fun)
            
            gamma, delta, alpha = shape_params[0], shape_params[1], shape_params[2]
            emax_params = shape_params[3:]
            
            emax_values = emax(self, m1_det, m2_det, emax_params)
            
            Nexp = np.sum(self.sigmoid(self.dL, dmid_values, emax_values, gamma, delta, alpha))
        
        return Nexp
        
    def lamda(self, dmid_fun, dmid_params, shape_params = None, alpha_vary = None, emax_fun = None):
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
        
        dmid = getattr(Found_injections, dmid_fun)
        
        dmid_values = dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values)
        
        if emax_fun is None and alpha_vary is None:
            
            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            
            lamda = self.sigmoid(dL, dmid_values, emax, gamma, delta) * m_pdf * dL_pdf * self.Ntotal
            
            
        elif emax_fun is None and alpha_vary is not None:
            
            gamma, delta, emax, alpha = shape_params[0], shape_params[1], shape_params[2], shape_params[3]
            
            lamda = self.sigmoid(dL, dmid_values, emax, gamma, delta, alpha) * m_pdf * dL_pdf * self.Ntotal
            
        
        elif emax_fun is not None and alpha_vary is None:
            
            emax = getattr(Found_injections, emax_fun)
            
            gamma, delta = shape_params[0], shape_params[1]
            emax_params = shape_params[2:]
            
            emax_values = emax(self, m1_det, m2_det, emax_params)
            
            lamda = self.sigmoid(dL, dmid_values, emax_values, gamma, delta) * m_pdf * dL_pdf * self.Ntotal


        else:
            
            emax = getattr(Found_injections, emax_fun)
            
            gamma, delta, alpha = shape_params[0], shape_params[1], shape_params[2]
            emax_params = shape_params[3:]
            
            emax_values = emax(self, m1_det, m2_det, emax_params)
            
            lamda = self.sigmoid(dL, dmid_values, emax_values, gamma, delta, alpha) * m_pdf * dL_pdf * self.Ntotal

        return lamda
    
    # def logL(self, dmid_fun, dmid_params, shape_params = None, alpha_vary = None, emax_fun = None):
    #     """
    #     log likelihood of the expected density of found injections

    #     Parameters
    #     ----------
    #     dmid_fun : str, name of the method used for the dmid function
    #     dmid_params : parameters of the Dmid function, 1D array
    #     shape_params : gamma, delta and emax params, 1D array
    #     emax_fun : str, name of the method used for the emax function
    #                if given None, shape_params should be only 3 numbers
    #                if given a method, shape_params should be [gamma, delta, emax_params]

    #     Returns
    #     -------
    #     float 

    #     """
            
    #     lnL = -self.Nexp(dmid_fun, dmid_params, shape_params, alpha_vary, emax_fun) + np.sum(np.log(self.lamda(dmid_fun, dmid_params, shape_params, alpha_vary, emax_fun)))
    #     return lnL
    
    def logL_dmid(self, dmid_fun, dmid_params, shape_params = None, alpha_vary = None, emax_fun = None):
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
         
        lnL = -self.Nexp(dmid_fun, dmid_params, shape_params, alpha_vary, emax_fun) + np.sum(np.log(self.lamda(dmid_fun, dmid_params, shape_params, alpha_vary, emax_fun)))
        return lnL
    
    def logL_shape(self, dmid_fun, dmid_params, shape_params = None, alpha_vary = None, emax_fun = None):
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
         
        lnL = -self.Nexp(dmid_fun, dmid_params, shape_params, alpha_vary, emax_fun) + np.sum(np.log(self.lamda(dmid_fun, dmid_params, shape_params, alpha_vary, emax_fun)))
        return lnL
        
    
    def MLE_dmid(self, methods, dmid_fun, dmid_params_guess, shape_params, emax_fun = None, alpha_vary = None):
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
        res = opt.minimize(fun=lambda in_param: -self.logL_dmid(dmid_fun, in_param, shape_params, alpha_vary, emax_fun), 
                           x0=np.array([dmid_params_guess]), 
                           args=(), 
                           method=methods)
        
        params_res = res.x
        min_likelihood = res.fun                
        return params_res, -min_likelihood
    
    def MLE_shape(self, methods, dmid_fun, dmid_params, shape_params_guess, alpha_vary = None):
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
        shape_params_guess[1] = np.log(shape_params_guess[1])
        
        res = opt.minimize(fun=lambda in_param: -self.logL_shape(dmid_fun, dmid_params, in_param, alpha_vary = None), 
                           x0=np.array([shape_params_guess]), 
                           args=(), 
                           method=methods)
        
        opt_params = res.x
        opt_params[1] = np.exp(opt_params[1])
        min_likelihood = res.fun  

        return opt_params, -min_likelihood
    
    def MLE_emax(self, methods, dmid_fun, dmid_params, shape_params_guess, emax_fun, alpha_vary = None):
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
        shape_params_guess[1] = np.log(shape_params_guess[1])
        
        res = opt.minimize(fun=lambda in_param: -self.logL_shape(dmid_fun, dmid_params, in_param, alpha_vary, emax_fun), 
                           x0=np.array([shape_params_guess]), 
                           args=(), 
                           method=methods)
        
        opt_params = res.x
        opt_params[1] = np.exp(opt_params[1])
        min_likelihood = res.fun  

        return opt_params, -min_likelihood
    
    # def MLE_difev(self, dmid_fun, emax_fun, bounds, maxiter=50, popsize=100):
        
    #     def lnL(x):
    #         lnL = -self.Nexp(dmid_fun, emax_fun, x) + np.sum(np.log(self.lamda(dmid_fun, emax_fun, x)))
    #         print(lnL)
    #         return lnL
        
    #     result = opt.differential_evolution(lnL, bounds, maxiter=maxiter, popsize=popsize)
    #     params_opt = result.x
    #     maxL = result.fun
    #     return params_opt, maxL
    
    def joint_MLE(self, methods, dmid_fun, dmid_params, shape_params, emax_fun = None, alpha_vary = None, precision = 1e-2):
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
        all_dmid_params = np.zeros([1,len(dmid_params)])
        
        if alpha_vary is None:
            all_emax_params = np.zeros([1,len(shape_params[2:])])
            params_emax = None if emax_fun is None else shape_params[2:]
            shape_params_names_file = 'gamma_opt, delta_opt, emax_opt, maxL'
            path = f'{dmid_fun}'
            
        else:
            all_emax_params = np.zeros([1,len(shape_params[3:])])
            params_emax = None if emax_fun is None else shape_params[3:]
            self.emax_params_names = {'emax_exp' : ['gamma_opt, delta_opt, alpha_opt, b_0, b_1, b_2'],
                                      'emax_sigmoid' : ['gamma_opt, delta_opt, alpha_opt, b_0, k, M_0'],
                                      }
            shape_params_names_file = 'gamma_opt, delta_opt, emax_opt, alpha_opt, maxL'
            path = f'alpha_vary/{dmid_fun}'
        
        for i in range(0, 10000):
            
            dmid_params, maxL_1 = self.MLE_dmid(methods, dmid_fun, dmid_params, shape_params, emax_fun, alpha_vary)
            all_dmid_params = np.vstack([all_dmid_params, dmid_params])
            
            if emax_fun is None:
                shape_params, maxL_2 = self.MLE_shape(methods, dmid_fun, dmid_params, shape_params, alpha_vary)
                gamma_opt, delta_opt, emax_opt = shape_params[:2]
                all_gamma.append(gamma_opt); all_delta.append(delta_opt)
                all_emax.append(emax_opt); total_lnL = np.append(total_lnL, maxL_2)
                
                if alpha_vary is not None:
                    alpha_opt = shape_params[3]
                    all_alpha.append(alpha_opt)
                
            
            elif emax_fun is not None:
                shape_params, maxL_2 = self.MLE_emax(methods, dmid_fun, dmid_params, shape_params, emax_fun, alpha_vary)
                gamma_opt, delta_opt = shape_params[:2]
                
                if alpha_vary is not None:
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
        
        if emax_fun is None:
            shape_results = np.column_stack((all_gamma, all_delta, all_emax, total_lnL)) if alpha_vary is None else np.column_stack((all_gamma, all_delta, all_emax, all_alpha, total_lnL))
            shape_header = shape_params_names_file
            name_dmid = path + '/joint_fit_dmid.dat'
            np.savetxt(path + '/joint_fit_shape.dat', shape_results, header = shape_header, fmt='%s')
        
        elif emax_fun is not None:
            shape_results = np.column_stack((all_gamma, all_delta, np.delete(all_emax_params, 0, axis=0), total_lnL)) if alpha_vary is None else np.column_stack((all_gamma, all_delta, all_alpha, np.delete(all_emax_params, 0, axis=0), total_lnL)) 
            shape_header = f'{self.emax_params_names[emax_fun]} , maxL'
            name_dmid = path + f'/{emax_fun}/joint_fit_dmid_emaxfun.dat'
            np.savetxt(path + f'/{emax_fun}/joint_fit_shape_emaxfun.dat', shape_results, header = shape_header, fmt='%s')
        
        
        all_dmid_params = np.delete(all_dmid_params, 0, axis=0)
        dmid_results = np.column_stack((all_dmid_params, total_lnL))
        dmid_header = f'{self.dmid_params_names[dmid_fun]} , maxL'
        np.savetxt(name_dmid, dmid_results, header = dmid_header, fmt='%s')
        
        return


    def cumulative_dist(self, dmid_fun, dmid_params, shape_params, var , emax_fun = None, alpha_vary = None):
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
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds'}
        
        dmid = getattr(Found_injections, dmid_fun)
        dic = {'dL': self.dL, 'Mc': self.Mc, 'Mtot': self.Mtot, 'eta': self.eta, 'Mc_det': self.Mc_det, 'Mtot_det': self.Mtot_det}
        path = f'{dmid_fun}' if alpha_vary is None else f'alpha_vary/{dmid_fun}'
        
        try:
            os.mkdir( path + f'/{emax_dic[emax_fun]}')
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
        
        dmid_values = dmid(self, m1o_det, m2o_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values)
        
        if emax_fun is not None:
            
            emax = getattr(Found_injections, emax_fun)
            emax_params = shape_params[2:] if alpha_vary is None else shape_params[3:]
            emax_values = emax(self, m1o_det, m2o_det, emax_params)
            
            gamma, delta = shape_params[:2]
            alpha = 2.05  if alpha_vary is None else shape_params[2]
            
            cmd = np.cumsum(self.sigmoid(dLo, dmid_values, emax_values, gamma, delta, alpha))
            
            
        else:
            gamma, delta, emax = shape_params[:3]
            alpha = 2.05  if alpha_vary is None else shape_params[3]
            
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
        name = path + f'/{emax_dic[emax_fun]}/{var}_cumulative.png'
        plt.savefig(name, format='png')
        
        #KS test
        
        m1_det = self.m1 * (1 + self.z) 
        m2_det = self.m2 * (1 + self.z)
        
        dmid_values = dmid(self, m1_det, m2_det, dmid_params)
        self.apply_dmid_mtotal_max(dmid_values)
        
        if emax_fun is not None:
            emax_values = emax(self, m1_det, m2_det, emax_params)
            pdet = self.sigmoid(self.dL, dmid_values, emax_values, gamma, delta, alpha)
            
        else:    
            pdet = self.sigmoid(self.dL, dmid_values, emax, gamma, delta, alpha)
        
        def cdf(x):
            values = [np.sum(pdet[dic[var]<value])/np.sum(pdet) for value in x]
            return np.array(values)
            
        stat, pvalue = kstest(var_foundo, lambda x: cdf(x) )
        
        return stat, pvalue

    
    
    def binned_cumulative_dist(self, nbins, dmid_fun, dmid_params, shape_params, var_cmd , var_binned, emax_fun = None, alpha_vary = None):
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
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds'}
        path = f'{dmid_fun}' if alpha_vary is None else f'alpha_vary/{dmid_fun}'
        
        try:
            os.mkdir( path + f'/{emax_dic[emax_fun]}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        try:
            os.mkdir( path + f'/{emax_dic[emax_fun]}/{var_binned}_bins')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        try:
            os.mkdir( path + f'/{emax_dic[emax_fun]}/{var_binned}_bins/{var_cmd}_cmd')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        gamma, delta = shape_params[:2]
        alpha = 2.05  if alpha_vary is None else shape_params[2]
    
                
        dmid = getattr(Found_injections, dmid_fun)
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
            
            dmid_values = dmid(self, m1_det, m2_det, dmid_params)
            self.apply_dmid_mtotal_max(dmid_values)
            
            if emax_fun is not None:
                emax = getattr(Found_injections, emax_fun)
                emax_params = shape_params[2:] if alpha_vary is None else shape_params[3:]
                gamma, delta = shape_params[:2]
                alpha = 2.05  if alpha_vary is None else shape_params[2]
                emax_values = emax(self, m1_det, m2_det, emax_params)
                
                cmd = np.cumsum(self.sigmoid(dL, dmid_values, emax_values, gamma, delta, alpha))
            
            
            else:
                gamma, delta, emax = shape_params[:3]
                alpha = 2.05  if alpha_vary is None else shape_params[3]
                
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
            name = path + f'/{emax_dic[emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/{i}.png'
            plt.savefig(name, format='png')
            plt.close()
            
            #KS test
            dmid_values = dmid(self, m1_det_inbin, m2_det_inbin, dmid_params)
            self.apply_dmid_mtotal_max(dmid_values)
            
            if emax_fun is not None:
                emax_values = emax(self, m1_det_inbin, m2_det_inbin, emax_params)
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
            
        

plt.close('all')

file = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')


data = Found_injections(file)

# function for dmid and emax we wanna use
dmid_fun = 'Dmid_mchirp_expansion_noa30'
#dmid_fun = 'Dmid_mchirp_expansion_exp'
#dmid_fun = 'Dmid_mchirp_expansion_a11'
#dmid_fun = 'Dmid_mchirp_expansion_asqrt'
#dmid_fun = 'Dmid_mchirp_expansion'
emax_fun = 'emax_exp'
#emax_fun = 'emax_sigmoid'

alpha_vary = False
path = f'{dmid_fun}' if alpha_vary is None else f'alpha_vary/{dmid_fun}'

if alpha_vary is not None:
    try:
        os.mkdir('alpha_vary')
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
        
cte_guess = 99
a20_guess= 0.0001
a01_guess= -0.4
a21_guess = -0.0001
#a22_guess = -0.0002
a30_guess = 0.0001
a10_guess = 0
a11_guess = 0
asqrt_guess = 0
power_guess = 1
Mstar_guess = 200

b0_guess = -3.5
b1_guess = 0
b2_guess = 0

gamma_guess = -0.02726
delta_guess = 0.13166
emax_guess = 0.79928

shape_guess = [gamma_guess, delta_guess, emax_guess]

params_guess = {'Dmid_mchirp': cte_guess, 
                'Dmid_mchirp_expansion': [cte_guess, a20_guess, a01_guess, a21_guess, a30_guess, a10_guess], 
                'Dmid_mchirp_expansion_a11': [cte_guess, a20_guess, a01_guess, a21_guess, a30_guess, a10_guess, a11_guess],
                'Dmid_mchirp_expansion_exp': [cte_guess, a20_guess, a01_guess, a21_guess, a30_guess, a10_guess, a11_guess, Mstar_guess],
                'Dmid_mchirp_expansion_asqrt': [cte_guess, a20_guess, a01_guess, a21_guess, a30_guess, asqrt_guess], 
                'Dmid_mchirp_power': [cte_guess, a20_guess, a01_guess, a21_guess, a30_guess, power_guess]}

params_names = {'Dmid_mchirp': 'cte', 
                'Dmid_mchirp_expansion': ['cte', 'a20', 'a01', 'a21', 'a30', 'a10'], 
                'Dmid_mchirp_expansion_a11': ['cte', 'a20', 'a01', 'a21', 'a30', 'a10','a11'],
                'Dmid_mchirp_expansion_a11': ['cte', 'a20', 'a01', 'a21', 'a30', 'a10','a11', 'Mstar'],
                'Dmid_mchirp_expansion_asqrt': ['cte', 'a20', 'a01', 'a21', 'a30', 'asqrt'], 
                'Dmid_mchirp_power': ['cte', 'a20', 'a01', 'a21', 'a30', 'power_param']}



########## MLE DMID ##########

# params_opt, maxL = data.MLE_dmid('Nelder-Mead', dmid_fun, params_guess[dmid_fun])
# print(params_opt)

# results = np.hstack((params_opt, maxL))
# header = f'{params_names[dmid_fun]} , maxL'
# np.savetxt(f'{dmid_fun}/dmid(m)_results_2method.dat', [results], header = header, fmt='%s')

# params_dmid = np.loadtxt(f'{dmid_fun}/dmid(m)_results_2method.dat')[:-1]

########## MLE SHAPE ##########

# gamma_opt, delta_opt, emax_opt, maxL = data.MLE_shape('Nelder-Mead', dmid_fun, params_dmid, shape_guess, update = True)
# print(gamma_opt, delta_opt, emax_opt)

# results = np.column_stack((gamma_opt, delta_opt, emax_opt, maxL))
# header = 'gamma_opt, delta_opt, emax_opt, maxL'
# np.savetxt(f'{dmid_fun}/shape/opt_shape_params.dat', results, header = header, fmt='%s')

#gamma_opt, delta_opt, emax_opt = np.loadtxt(f'{dmid_fun}/shape/opt_shape_params.dat')[:-1]

########## JOINT FIT #########
#data.joint_MLE('Nelder-Mead', dmid_fun, params_guess[dmid_fun], shape_guess, emax_fun = None, precision = 1e-2)

# # ~~~~~~~~~ THESE LINES ARE NECESSARY TO DO THE JOINT FIT WITH EMAX FUN ~~~~~~~~~

# params_dmid = np.loadtxt(f'{dmid_fun}/joint_fit_dmid.dat')[-1, :-1]
# gamma_opt, delta_opt, emax_opt = np.loadtxt(f'{dmid_fun}/joint_fit_shape.dat')[-1, :-1]
# params_shape = [gamma_opt, delta_opt, emax_opt]

# shape_guess_emax = [gamma_opt, delta_opt, b0_guess, b1_guess, b2_guess]

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

# print('\nparams_dmid:\n', params_dmid)
# print('\ngamma, delta, emax:\n', gamma_opt, delta_opt, emax_opt, '\n')

########## JOINT FIT WITH EMAX FUNCTION ###########

#initial values from joint fit with emax fun and 'cte', 'a20', 'a01', 'a21', 'a30'
#params_dmid = [91.72267028049346, -1.2569702940809277e-05/2, -0.5671077334172103, 9.99097583380413e-06/2, 6.5686541980284065e-09, 0]
#shape_guess_emax  = [-0.5403600227437229, 0.21498422370452497, -2.2413634572753316, -0.014261526887000078, 0.00010764305058111201]

#initial values from joint fit with emax fun and 'cte', 'a20', 'a01', 'a21', 'a30', 'a10'
#params_dmid = [83.6359221338493, -1.7720389246237178e-05, -0.6202851605751052, 5.799384584576895e-06, 2.2901587693040153e-08, 0.0020739921788790177, 0]
#shape_guess_emax  = [-0.5852595282336299, 0.22144333868817984, -3.4645779484229804, 0.01073830230180733, 1.9110955783665044e-06 ]

#initial values fix emax with det mass from joint fit with emax fun and 'cte', 'a20', 'a01', 'a21', 'a30', 'a10'
#params_dmid = [82.785950741024, -1.9044508883488783e-05, -0.41084970235333484, 2.3427319051426644e-05, 2.4624863341861382e-08, 0.0023167923193285157, -0.004968986350895854 ]
#shape_guess_emax  = [-0.5934933873985427, 0.22291710406565518, -3.4645779484229804, -0.001, 0.00001]

#initial values FOR EXP IN DMID from joint fit with emax fun and 'cte', 'a20', 'a01', 'a21', 'a30', 'a10', 'a11', 'Mstar'
# params_dmid = [80.82424577472355, -1.617118775455203e-05, -0.4172004525844498, 2.0548889990723283e-05, 2.0737945867597705e-08, 0.0020630686367234022, -0.00459527905804538, 500]
# shape_guess_emax  = [-0.7354844677038638, 0.24989182786857905 ,-4.296148078417843, 0.014417414295816396, -1.1933745117035866e-05 ]

#initial values FOR EXP IN DMID from joint fit with emax fun and 'cte', 'a20', 'a01', 'a21', 'a10', 'a11'
#params_dmid = [1000.82424577472355, 1.617118775455203e-03, 0.2172004525844498, 2.0548889990723283e-05, 0.0020630686367234022, -0.00459527905804538]
#shape_guess_emax  = [-0.7354844677038638, 0.24989182786857905 ,-4.296148078417843, 0.014417414295816396, -1.1933745117035866e-05 ]

#initial values FOR EXP IN DMID from joint fit with EMAX_SIGMOID fun and 'cte', 'a20', 'a01', 'a21', 'a10', 'a11'
# params_dmid = [1000.82424577472355, 1.617118775455203e-03, 0.2172004525844498, 2.0548889990723283e-05, 0.0020630686367234022, -0.00459527905804538]
# shape_guess_emax  = [-0.7354844677038638, 0.24989182786857905 ,-4.296148078417843, 0.1, 300]

#initial values from fit without a30 allowing alpha to vary: ['cte', 'a20', 'a01', 'a21', 'a10', 'a11'] + ['gamma, delta, alpha, b_0, b_1, b_2'] 
params_dmid = [82.86080855190852, -3.843054082930893e-06, -0.4726476591881541, 4.304672026161589e-06, 0.0004746871925697556, -0.0014672425957000098]
shape_guess_emax  = [-0.7435398875859958, 0.2511010365079828, 2.05, -5.632796685161368, 0.025575280643270047, -3.1030795279503675e-05]


#data.joint_MLE('Nelder-Mead', dmid_fun, params_dmid, shape_guess_emax, emax_fun, alpha_vary)

params_dmid = np.loadtxt( path + f'/{emax_fun}/joint_fit_dmid_emaxfun.dat')[-1, :-1]
params_shape = np.loadtxt( path + f'/{emax_fun}/joint_fit_shape_emaxfun.dat')[-1, :-1]

if alpha_vary is None:
    gamma_opt, delta_opt = params_shape[:2]
    params_emax = params_shape[2:]
    print('\nparams_dmid:\n', params_dmid)
    print('\ngamma, delta, b0, b1, b2:\n', gamma_opt, delta_opt, params_emax, '\n')
    
else: 
    gamma_opt, delta_opt, alpha_opt = params_shape[:3]
    params_emax = params_shape[3:]
    print('\nparams_dmid:\n', params_dmid)
    print('\ngamma, delta, alpha, b0, b1, b2:\n', gamma_opt, delta_opt, alpha_opt, params_emax, '\n')

########## KS TEST ##########

plt.close('all')

# stat_dL, pvalue_dL = data.cumulative_dist(dmid_fun, params_dmid, params_shape, 'dL', emax_fun, alpha_vary)
# print('\ndL KStest: statistic = %s , pvalue = %s' %(stat_dL, pvalue_dL))

# stat_Mtot, pvalue_Mtot = data.cumulative_dist(dmid_fun, params_dmid, params_shape, 'Mtot', emax_fun, alpha_vary)
# print('Mtot KStest: statistic = %s , pvalue = %s' %(stat_Mtot, pvalue_Mtot))

# stat_Mc, pvalue_Mc = data.cumulative_dist(dmid_fun, params_dmid, params_shape, 'Mc', emax_fun, alpha_vary)
# print('Mc KStest: statistic = %s , pvalue = %s' %(stat_Mc, pvalue_Mc))

# stat_Mtot_det, pvalue_Mtot_det = data.cumulative_dist(dmid_fun, params_dmid, params_shape, 'Mtot_det', emax_fun, alpha_vary)
# print('Mtot_det KStest: statistic = %s , pvalue = %s' %(stat_Mtot_det, pvalue_Mtot_det))

# stat_Mc_det, pvalue_Mc_det = data.cumulative_dist(dmid_fun, params_dmid, params_shape, 'Mc_det', emax_fun, alpha_vary)
# print('Mc_det KStest: statistic = %s , pvalue = %s' %(stat_Mc_det, pvalue_Mc_det))

# stat_eta, pvalue_eta = data.cumulative_dist(dmid_fun, params_dmid, params_shape, 'eta', emax_fun, alpha_vary) 
# print('eta KStest: statistic = %s , pvalue = %s' %(stat_eta, pvalue_eta))


# binned cumulative dist analysis
# nbins = 5
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'dL', 'dL', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot', 'dL', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc', 'dL', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot_det', 'dL', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc_det', 'dL', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'eta', 'dL', emax_fun, alpha_vary)

# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'dL', 'Mtot', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot', 'Mtot', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc', 'Mtot', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'eta', 'Mtot', emax_fun, alpha_vary)

# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'dL', 'Mtot_det', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot_det', 'Mtot_det', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc_det', 'Mtot_det', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'eta', 'Mtot_det', emax_fun, alpha_vary)

# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'dL', 'Mc', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot', 'Mc', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc', 'Mc', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'eta', 'Mc', emax_fun, alpha_vary)

# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'dL', 'Mc_det', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot_det', 'Mc_det', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc_det', 'Mc_det', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'eta', 'Mc_det', emax_fun, alpha_vary)

# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'dL', 'eta', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot', 'eta', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc', 'eta', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mtot_det', 'eta', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'Mc_det', 'eta', emax_fun, alpha_vary)
# data.binned_cumulative_dist(nbins, dmid_fun, params_dmid, params_shape, 'eta', 'eta', emax_fun, alpha_vary)

#bounds = [ (50, 150), (-1,1), (-1, 1), (-1, 1), (-1,1) ]

# ddmid = np.loadtxt(f'{dmid_fun}/joint_fit_dmid_emaxfun.dat')
# ddmid[:, 1] = ddmid[:, 1] / 2
# ddmid[:, 3] = ddmid[:, 3] / 2

# name_dmid = f'{dmid_fun}/joint_fit_dmid_emaxfun.dat'
# np.savetxt(name_dmid, ddmid, header = f'{params_names[dmid_fun]}', fmt='%s')

# params_dmid = np.loadtxt(f'{dmid_fun}/joint_fit_dmid_emaxfun.dat')

###### PLOT TO CHECK EPSILON(DL) WITH OPT PARAMETERS #######
#gamma_opt, delta_opt, emax_opt = np.loadtxt(f'{dmid_fun}/joint_fit_shape.dat')[-1, :-1]

dmid = getattr(data, dmid_fun)
emax = getattr(data, emax_fun)

# plt.figure(figsize=(7,6))
# plt.plot(data.dL, data.sigmoid(data.dL, data.Dmid_mchirp_expansion(data.m1, data.m2, data.z, params_dmid), emax_opt, gamma_opt, delta_opt), '.')
# t = ('gamma = %.3f , delta = %.3f , emax = %.3f' %(gamma_opt, delta_opt, emax_opt) )
# plt.title(t)
# plt.xlabel('dL')
# plt.ylabel(r'$\epsilon (dL, dmid(m), \gamma_{opt}, \delta_{opt}, emax_{opt})$')
# plt.show()
# plt.savefig(f'{dmid_fun}/opt_epsilon_plot.png')

###### PLOT TO CHECK EPSILON(DL) WITH OPT PARAMETERS and emax fun #######
#gamma_opt, delta_opt, emax_opt = np.loadtxt(f'{dmid_fun}/joint_fit_shape.dat')[-1, :-1]
'''

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)

plt.figure(figsize=(7,6))
plt.plot(data.dL/dmid(m1_det, m2_det, params_dmid), data.sigmoid(data.dL, dmid(m1_det, m2_det, params_dmid), emax(m1_det, m2_det, params_emax), gamma_opt, delta_opt), '.')
plt.xlabel('D/D_mid')
plt.ylabel('Pdet')
plt.show()
plt.savefig( path + f'/{emax_fun}/opt_epsilon_plot_emaxfun.png')


order=np.argsort(data.dL)
dL=data.dL[order]
m1=data.m1[order]
m2=data.m2[order]
z=data.z[order]

plt.figure(figsize=(7,6))
plt.scatter(m1, m2, s=1, c=dmid(m1*(1+z), m2*(1+z), params_dmid))
plt.xlabel('m1')
plt.ylabel('m2')
plt.colorbar(label='dL_mid')
plt.show()
plt.savefig( path + f'/{emax_fun}/m1m2_dmid.png')

plt.figure(figsize=(7,6))
plt.scatter(m1*(1+z), m2*(1+z), s=1, c=dmid(m1*(1+z), m2*(1+z), params_dmid))
plt.xlabel('m1_det')
plt.ylabel('m2_det')
plt.colorbar(label='dL_mid')
plt.show()
plt.savefig( path + f'/{emax_fun}/m1m2det_dmid.png')

x = np.linspace(min(m1*(1+z)+m2*(1+z)),max(m1*(1+z)+m2*(1+z)), 500)
y = 1 - np.exp(params_emax[0] + params_emax[1] * x + params_emax[2] * x**2)
#y = (1 - np.exp(params_emax[0])) / (1 + np.exp(params_emax[1]*(x-params_emax[2])))
plt.figure()
plt.plot(x, y, '.')
plt.ylim(0, 2)
plt.grid()
plt.xlabel('Mtot_det')
plt.ylabel('emax(m)')
plt.savefig( path + f'/{emax_fun}/emax(m).png')

eta_choice=0.175

M = np.linspace(min(m1*(1+z)+m2*(1+z)),max(m1*(1+z)+m2*(1+z)), 200)
eta = eta_choice*np.ones(len(M))

cte = params_dmid[0] * np.ones(len(M))
a20 = params_dmid[1] * M**2 
a01 = params_dmid[2] * (1 - 4*eta)
a21 = params_dmid[3] * M**2 * (1 - 4*eta) 
#a30 = params_dmid[4] * M**3
a10 = params_dmid[4] * M
#asqrt = params_dmid[5] * M**(1/2)
a11 = params_dmid[5] * M * (1 - 4*eta) 
#Mstar = np.exp(-M/params_dmid[7])

tot = a20 + a01 +a21 + a10 + a11

plt.figure()
#plt.plot(M, np.abs(cte), '-', label='cte')
plt.plot(M, a10, '-', label=r'$a_{10} M$')
#plt.plot(M, np.abs(asqrt), '-', label='a_sqrt * M^(1/2)')
plt.plot(M, a20, '-', label=r'$a_{20} M^2$')
#plt.plot(M, a30, '-', label=r'$a_{30} M^3$')
plt.plot(M, a21, '-', label=r'$a_{21} M^2 (1 - 4\eta)$')
plt.plot(M, a01, '-', label=r'$a_{01} (1 - 4\eta)$')
plt.plot(M, a11, '-', label=r'$a_{11} M (1 - 4\eta)$')
plt.plot(M, tot, '-', label='tot')
plt.semilogx()
plt.ylim(-1, 1)
plt.grid()
plt.title(f'eta = {eta_choice}')
plt.legend()
#plt.semilogy()
plt.xlabel('Mtot_det')
plt.ylabel('contributions to dmid (abs value)*exp(-M/Mstar)')
plt.savefig( path + f'/{emax_fun}/dmid_params_{eta_choice}.png')
'''

'''
plt.figure()
plt.scatter(data.dL, data.m_pdf*data.dL_pdf)
plt.semilogy()
plt.xlabel('dL')
plt.ylabel('p(m)*p(dL)')
plt.title('o3 inj from o3 code')
plt.savefig('pdf_o3')

plt.figure()
plt.scatter(data.dL[np.invert(data.found_any)]/(data.Mc[np.invert(data.found_any)])**(5/6), data.q[np.invert(data.found_any)], label=r'FAR$>$ 1', s=0.8)
plt.scatter(data.dL[data.found_any]/(data.Mc[data.found_any])**(5/6), data.q[data.found_any], label=r'FAR$\leq$ 1', alpha=0.6, s=0.8)
plt.legend()
#plt.loglog()
plt.xlabel(r'dL/Mc$^{5/6}$')
plt.ylabel('q')
plt.title(f'o3 injections')
plt.semilogx()
plt.savefig(f'{dmid_fun}/{emax_fun}/q_vs_chirp_distance.png')

plt.figure()
plt.scatter(data.dL[data.snr < 10]/(data.Mc[data.snr < 10])**(5/6), data.q[data.snr < 10], label=r'SNR$<$ 10', s=0.8)
plt.scatter(data.dL[data.snr>=10]/(data.Mc[data.snr>=10])**(5/6), data.q[data.snr>=10], label=r'SNR$\geq$ 10', alpha=0.6, s=0.8)
plt.legend()
#plt.loglog()
plt.xlabel(r'dL/Mc$^{5/6}$')
plt.ylabel('q')
plt.title(f'o3 injections')
plt.semilogx()
plt.savefig(f'{dmid_fun}/{emax_fun}/q_vs_chirp_distance_SNR.png')
'''

dmid = getattr(data, dmid_fun)
emax = getattr(data, emax_fun)

alpha = 2.05

a_20 = params_dmid[1] 
a_01 = params_dmid[2] 
a_21 = params_dmid[3] 
a_10 = params_dmid[4] 
a_11 = params_dmid[5] 

m1_det = data.m1 * (1 + data.z)
m2_det = data.m2 * (1 + data.z)
M = m1_det + m2_det
eta = m1_det*m2_det / (m1_det+m2_det)**2
Mc = (m1_det * m2_det)**(3/5) / (m1_det + m2_det)**(1/5)

emax_pol = emax(m1_det, m2_det, params_emax)
dmid_pol= (1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))


def manual_found_inj(x):
    frac = data.dL / ( x * dmid_pol * Mc**(5/6) )
    denom = 1. + frac ** alpha * \
            np.exp(gamma_opt * (frac - 1.) + delta_opt * (frac**2 - 1.))
    return np.nansum( emax_pol / denom ) 

o1 = manual_found_inj(36.14178746)
o2 = manual_found_inj(45.66283608)
print('o3 found inj with cte_1 : ', o1)
print('o3 found inj with cte_2 : ', o2)

frac1 = o1 / (78864)
frac2 = o2 / (78864)
frac3 = 78864 / (78864)

print(frac1, frac2)

obs_range = np.array([80, 100, 140]) #Mpc
obs_time = np.array([0.1331507, 0.323288, 0.75435365296528]) #years
n_ev = np.array([3, 7, 59])

det_rates = n_ev / obs_time # 1/yr
print(det_rates)

tot_rate = np.sum(det_rates)
print(tot_rate)
tot_time = np.sum(obs_time)


pred_rates = np.array([frac1*det_rates[2], frac2*det_rates[2]])
pred_nev = pred_rates*obs_time[:2]

nbindL = np.arange(0, max(data.dL)+200, 200)

plt.figure()
plt.hist(data.dL[data.found_any], bins=nbindL, alpha=0.5, label='FAR <= 1')
plt.hist(data.dL[data.snr>=10], bins=nbindL, alpha=0.5, label='SNR >= 10')
plt.xlabel('dL')
plt.legend()
plt.semilogy()
plt.savefig('o3/found_far_vs_snr.png')


