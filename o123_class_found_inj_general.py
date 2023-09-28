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
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import LogNorm

class Found_injections:
    """
    Class for an algorithm of GW detected found injections from signal templates 
    of binary black hole mergers
    
    Input: an h5py file with the parameters of every sample and the threshold for 
    the false alarm rate (FAR). The default value is thr = 1, which means we are 
    considering a signal as detected when FAR <= 1.

    """
    
    def __init__(self, dmid_fun = 'Dmid_mchirp_expansion_noa30', emax_fun = 'emax_exp', alpha_vary = None, thr_far = 1, thr_snr = 10, ini_files = None):
        
        '''
        Argument ini_files must be a list or a numpy array with two elements
        The first one contains the dmid initial values and the second one the shape initial values 
        
        if ini_files = None, it looks for a ini file in a folder called 'ini_values'. The name of the folder is
        selected from the options of emax_fun, dmid_fun and alpha_vary. The method that finds the file is 
        self.get_ini_values()
        
        '''
        # assert isinstance(ini_files, list) or isinstance(ini_files, np.ndarray),\

        assert isinstance(thr_far, float) or isinstance(thr_far, int),\
        "Argument (thr_far) must be a float or an integer."
        
        assert isinstance(thr_snr, float) or isinstance(thr_snr, int),\
        "Argument (thr_snr) must be a float or an integer."
        
        self.thr_far = thr_far
        self.thr_snr = thr_snr
        
        self.dmid_fun = dmid_fun #dmid function name
        self.emax_fun = emax_fun #emax function name
        self.dmid = getattr(Found_injections, dmid_fun) #class method for dmid
        self.emax = getattr(Found_injections, emax_fun) #class method for emax
        self.alpha_vary = alpha_vary
        
        #self.dmid = 
        
        self.H0 = 67.9 #km/sMpc
        self.c = 3e5 #km/s
        self.omega_m = 0.3065
        
        
        self.dmid_ini_values, self.shape_ini_values = ini_files if ini_files is not None else self.get_ini_values()
        
        self.dmid_params = self.dmid_ini_values
        self.shape_params = self.shape_ini_values
        
        if self.alpha_vary is None:
            self.path = f'{run_fit}/{self.dmid_fun}' if self.emax_fun is None else f'{run_fit}/{self.dmid_fun}/{self.emax_fun}'
          
        else: 
            self.path = f'{run_fit}/alpha_vary/{self.dmid_fun}' if self.emax_fun is None else f'{run_fit}/alpha_vary/{self.dmid_fun}/{self.emax_fun}'
        
        self.runs = ['o1', 'o2', 'o3']
        
        self.obs_time = { 'o1' : 0.1331507, 'o2' : 0.323288, 'o3' : 0.75435365296528 } #years
        self.total_obs_time = np.sum(list(self.obs_time.values()))
        self.prop_obs_time = np.array([self.obs_time[i]/self.total_obs_time for i in self.runs])
        
        self.obs_nevents = { 'o1' : 3, 'o2' : 7, 'o3' : 59 }
        
        self.det_rates = { i : self.obs_nevents[i] / self.obs_time[i] for i in self.runs }
        
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
        
        self.shape_params_dict = { 'emax_exp' : 1,
            
                                    }
        
        self.cosmo = FlatLambdaCDM(self.H0, self.omega_m)

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
        
    def read_o1o2_set(self, run_dataset):
        assert run_dataset =='o1' or run_dataset == 'o2',\
        "Argument (run_dataset) must be 'o1' or 'o2'. "
        
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
        
        self.s1x = file["events"][:]["spin1x"]
        self.s1y = file["events"][:]["spin1y"]
        self.s1z = file["events"][:]["spin1z"]
        
        self.s2x = file["events"][:]["spin2x"]
        self.s2y = file["events"][:]["spin2y"]
        self.s2z = file["events"][:]["spin2z"]
        
        self.chieff_d = file["events"][:]["chi_eff"]
        
        #SNR
        self.snr = file["events"][:]["snr_net"]
        found_snr = self.snr >= self.thr_snr
        
        #indexes of the found injections
        self.found_any = found_snr
        print(f'Found inj in {run_dataset} set: ', self.found_any.sum())   
        
        return
       
    def read_o3_set(self):
        assert run_dataset =='o3',\
        "Argument (run_dataset) must be 'o3'. "
        
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
        
        self.s1x = file["injections/spin1x"][:]
        self.s1y = file["injections/spin1y"][:]
        self.s1z = file["injections/spin1z"][:]
        
        self.s2x = file["injections/spin2x"][:]
        self.s2y = file["injections/spin2y"][:]
        self.s2z = file["injections/spin2z"][:]
        
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
        
        return
        
    def load_inj_set(self, run_dataset):
        self.read_o3_set() if run_dataset == 'o3' else self.read_o1o2_set(run_dataset)
        
        A = np.sqrt(self.omega_m * (1 + self.z)**3 + 1 - self.omega_m)
        dL_dif = (self.c * (1 + self.z) / self.H0) * (1/A)
        
        #Luminosity distance sampling pdf values, p(dL), computed for a flat Lambda-Cold Dark Matter cosmology from the z_pdf values
        self.dL_pdf = self.z_pdf / dL_dif
        
        #total mass (m1+m2)
        self.Mtot = self.m1 + self.m2
        self.Mtot_det = self.m1 * (1+self.z) + self.m2 * (1+self.z)
        self.Mtot_max = np.max(self.Mtot_det)
        
        #mass chirp
        self.Mc = (self.m1 * self.m2)**(3/5) / (self.Mtot)**(1/5) 
        self.Mc_det = (self.m1 * self.m2 * (1+self.z)**2 )**(3/5) / self.Mtot**(1/5) 
        
        #eta aka symmetric mass ratio
        mu = (self.m1 * self.m2) / self.Mtot
        self.eta = mu / self.Mtot
        self.q = self.m2 / self.m1
        
        #spin amplitude
        self.a1 = np.sqrt(self.s1x**2 + self.s1y**2 + self.s1z**2)
        self.a2 = np.sqrt(self.s2x**2 + self.s2y**2 + self.s2z**2)
        
        # a1v = np.array([self.s1x , self.s1y , self.s1z])
        # a2v = np.array([self.s1x , self.s1y , self.s1z])
        
        self.chi_eff = (self.s1z * self.m1 + self.s2z* self.m2) / (self.Mtot)
        
        self.max_index = np.argmax(self.dL)
        self.dLmax = self.dL[self.max_index]
        self.zmax = np.max(self.z)
        
        index = np.random.choice(np.arange(len(self.dL)), 200, replace=False)
        if self.max_index not in index:
            index = np.insert(index, -1, self.max_index)
            
        try_dL = self.dL[index]
        try_dLpdf = self.dL_pdf[index]
    
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
    
    def get_opt_params(self, run_fit, rescale_o3 = True):
        '''
        Sets self.dmid_params and self.shape_params as a class attribute (optimal values from some previous fit).

        Parameters
        ----------
        run_fit : str. Observing run from which we want to use the fit. Must be 'o1', 'o2' or 'o3'.

        Returns
        -------
        None

        '''
        
        assert run_fit =='o1' or run_fit == 'o2' or run_fit == 'o3',\
        "Argument (run_fit) must be 'o1' or 'o2' or 'o3'. "
        
        if not rescale_o3: #get separate independent fit files
             run_fit_touse = run_fit
            
        else: #rescale o1 and o2
            run_fit_touse = 'o3'
            
                
        try:
            if self.alpha_vary is None:
                path = f'{run_fit_touse}/{self.dmid_fun}' if self.emax_fun is None else f'{run_fit_touse}/{self.dmid_fun}/{self.emax_fun}'
                    
            else: 
                path = f'{run_fit_touse}/alpha_vary/{self.dmid_fun}' if self.emax_fun is None else f'{run_fit}_touse/alpha_vary/{self.dmid_fun}/{self.emax_fun}'
        
            self.dmid_params = np.loadtxt( path + '/joint_fit_dmid.dat')[-1, :-1]
            self.shape_params = np.loadtxt( path + '/joint_fit_shape.dat')[-1, :-1]
        
        except:
            print('ERROR in self.get_opt_params: There are not such files because there is not a fit yet with these options.')
    
        if rescale_o3 and run_fit != 'o3':
            d0 = self.find_dmid_cte_found_inj(self, run_fit, 'o3')
            self.dmid_params[0] = d0
            
        return
    
    def get_ini_values(self):
        '''
        Gets the dmid and shape initial params for a new optimization

        Returns
        -------
        dmid_ini_values : 1D array. Initial values for the dmid optimization
        shape_ini_values : 1D array. Initial values for the shape optiization
        
        '''
        
        try:
            if self.alpha_vary is None:
                path = f'ini_values/{self.dmid_fun}' if self.emax_fun is None else f'ini_values/{self.dmid_fun}_{self.emax_fun}'
                    
            else: 
                path = f'ini_values/alpha_vary_{self.dmid_fun}' if emax_fun is None else f'ini_values/alpha_vary_{self.dmid_fun}_{self.emax_fun}'
            
            dmid_ini_values = np.loadtxt( path + '_dmid.dat')
            shape_ini_values = np.loadtxt( path + '_shape.dat')
        
        except:
            print('ERROR in self.get_ini_values: Files not found. Please create .dat files with the initial param guesses for this fit.')
    
        return dmid_ini_values, shape_ini_values
    
    def set_shape_params(self):
        '''
        Gets the right shape parameters depending on sigmoid settings (dmid, emax and alpha options we initialise the class with).

        Returns
        -------
        emax_params : 1D array
        gamma : float
        delta : float
        alpha : float

        '''
        
        if self.emax_fun is not None:
            emax_params = np.copy(self.shape_params[2:]) if self.alpha_vary is None else np.copy(self.shape_params[3:])
            gamma, delta = np.copy(self.shape_params[:2])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[2])
            
        else:
            gamma, delta, emax_params = np.copy(self.shape_params[:3])
            alpha = 2.05  if self.alpha_vary is None else np.copy(self.shape_params[3])
        
        return emax_params, gamma, delta, alpha
    
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
        
        dic = {'dL': self.dL, 'Mc': self.Mc, 'Mtot': self.Mtot, 'eta': self.eta, 'Mc_det': self.Mc_det, 'Mtot_det': self.Mtot_det, 'chi_eff': self.chi_eff}
        path = f'{run_fit}/{dmid_fun}' if self.alpha_vary is None else f'alpha_vary/{self.dmid_fun}'
        
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
        
        emax_params, gamma, delta, alpha = self.set_shape_params()
        
        if self.emax_fun is not None:
            emax = self.emax(self, m1o_det, m2o_det, emax_params)
            
        else:
            emax = np.copy(emax_params)
        
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
            emax = self.emax(self, m1_det, m2_det, emax_params)
               
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
        
        self.load_inj_set(run_dataset)
        
        self.get_opt_params(run_fit)
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds'}
        path = f'{run_fit}/{self.dmid_fun}' if self.alpha_vary is None else f'alpha_vary/{self.dmid_fun}'
        
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
        chi_effo = self.chi_eff[index]
        found_any_o = self.found_any[index]
        
        #create bins with equally amount of data
        def equal_bin(N, m):
            sep = (N.size/float(m))*np.arange(1,m+1)
            idx = sep.searchsorted(np.arange(N.size))
            return idx[N.argsort().argsort()]
        
        index_bins = equal_bin(data, nbins)
        
        print(f'\n{var_binned} bins:\n')
        
        chi_eff_params = []
        mid_values = []
        
        for i in range(nbins):
            #get data in each bin
            data_inbin = data[index_bins==i]
            dL_inbin = dLo[index_bins==i]
            m1_det_inbin = m1o_det[index_bins==i]
            m2_det_inbin = m2o_det[index_bins==i] 
            
            Mc_inbin = Mco[index_bins==i]
            Mtot_inbin = Mtoto[index_bins==i]
            eta_inbin = etao[index_bins==i]
            Mc_det_inbin = Mco_det[index_bins==i]
            Mtot_det_inbin = Mtoto_det[index_bins==i]
            chi_eff_inbin = chi_effo[index_bins==i]
            
            cmd_dic = {'dL': dL_inbin, 'Mc': Mc_inbin, 'Mtot': Mtot_inbin, 'eta': eta_inbin, 'Mc_det': Mc_det_inbin, 'Mtot_det': Mtot_det_inbin, 'chi_eff': chi_eff_inbin}
        
            #cumulative distribution over the desired variable
            indexo = np.argsort(cmd_dic[var_cmd])
            varo = cmd_dic[var_cmd][indexo]
            dL = dL_inbin[indexo]
            m1_det = m1_det_inbin[indexo]
            m2_det = m2_det_inbin[indexo]
            mtot_det = m1_det + m2_det
            
            dmid_values = self.dmid(self, m1_det, m2_det, self.dmid_params)
            self.apply_dmid_mtotal_max(dmid_values, mtot_det)
            
            emax_params, gamma, delta, alpha = self.set_shape_params()
            
            if self.emax_fun is not None:
                emax = self.emax(self, m1_det, m2_det, emax_params)
            
            else:
                emax = np.copy(emax_params)
            
            pdet = self.sigmoid(dL, dmid_values, emax, gamma, delta, alpha)
            cmd = np.cumsum(pdet)
            
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
            
            if var_cmd == 'chi_eff':
                
                mid_values.append( (data_inbin[0] + data_inbin[-1]) / 2 )
            
                nbins = np.linspace(-1, 1, 11)
                c1 = 0.7
                def chieff_corr(x, c1):
                    return  np.exp(c1 * x)
                
                x = np.linspace(-1, 1, 1000)
                
                plt.figure()
                nchi, _, _ = plt.hist(varo, bins = nbins, density = True, weights = pdet, histtype = 'step', log = True)
                nfound, _, _, = plt.hist(found_inj_inbin_sorted, bins = nbins, density = True, histtype = 'step', log = True)
                plt.plot(nbins[:-1], nfound/nchi, '-')
                
                #fit
                xfit = nbins[:-1]
                yfit = nfound/nchi
                popt, pcov = opt.curve_fit(chieff_corr, xfit, yfit, p0 = [0.7], absolute_sigma=True)
                yplot = chieff_corr(x, popt)
                
                plt.plot(x, yplot, 'r--')
                plt.hlines(1, -1, 1, linestyle='dashed')
                plt.xlabel(f'{var_cmd}')
                name = path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/hist_{i}.png'
                plt.savefig(name, format='png')
                plt.close()
                
                chi_eff_params.append(popt)
            
            #KS test
            
        #     dmid_values = self.dmid(self, m1_det_inbin, m2_det_inbin, self.dmid_params)
        #     self.apply_dmid_mtotal_max(dmid_values, Mtot_det_inbin)
            
        #     if self.emax_fun is not None:
        #         emax = self.emax(self, m1_det_inbin, m2_det_inbin, emax_params)
                
        #     pdet = self.sigmoid(dL_inbin, dmid_values, emax, gamma, delta, alpha)
            
        #     def cdf(x):
        #         values = [np.sum(pdet[cmd_dic[var_cmd]<value])/np.sum(pdet) for value in x]
        #         return np.array(values)
            
        #     stat, pvalue = kstest(found_inj_inbin_sorted, lambda x: cdf(x) )
            
        #     print(f'{var_cmd} KStest in {i} bin: statistic = %s , pvalue = %s' %(stat, pvalue))
        
        # print('')   
        
        if var_cmd == 'chi_eff':
            name_opt = path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/chi_eff_opt_param'
            np.savetxt(name_opt, chi_eff_params, header = '0, 1, 2, 3, 4', fmt='%s')
            
            name_mid = path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/mid_values'
            np.savetxt(name_mid, mid_values, header = '0, 1, 2, 3, 4', fmt='%s')
        return
    
    def sensitive_volume(self, run_fit, m1, m2):
        
        self.get_opt_params(run_fit)
        
        dmid = lambda dL_int : self.dmid(self, m1*(1 + self.interp_z(dL_int)), m2*(1 + self.interp_z(dL_int)), self.dmid_params)
        mtot = lambda dL_int :m1*(1 + self.interp_z(dL_int)) + m2*(1 + self.interp_z(dL_int))
        
        emax_params, gamma, delta, alpha = self.set_shape_params()
        
        if self.emax_fun is not None:
            emax = lambda dL_int : self.emax(self, m1*(1 + self.interp_z(dL_int)), m2*(1 + self.interp_z(dL_int)), emax_params)
            quad_fun = lambda dL_int : self.sigmoid(dL_int, self.apply_dmid_mtotal_max(np.array(dmid(dL_int)), np.array(mtot(dL_int))), emax(dL_int) , gamma , delta, alpha) * self.interp_dL_pdf(dL_int)
            
        else:
            emax = np.copy(emax_params)
            quad_fun = lambda dL_int : self.sigmoid(dL_int, self.apply_dmid_mtotal_max(dmid(dL_int), mtot(dL_int)), emax , gamma , delta, alpha) * self.interp_dL_pdf(dL_int)
            
        pdet =  integrate.quad(quad_fun, 0, self.dLmax)[0]
        
        vquad = lambda z_int : 4 * np.pi * self.cosmo.differential_comoving_volume(z_int).value / (1 + z_int)
        Vtot = integrate.quad(vquad, 0, self.zmax)[0]
        
        return pdet * Vtot
    
    
    def total_sensitive_volume(self, run_fit, m1, m2):
        
        Vtot = 0
        
        for run, in zip(self.runs):
            
            Vi = self.sensitive_volume(run, m1, m2)
                
            Vi *= self.obs_time[run]
            
            Vtot += Vi
        
        return Vtot
    
    def find_dmid_cte_found_inj(self, run_dataset, run_fit = 'o3'):
        
        try:
        
            self.load_inj_set(run_dataset)
            Nfound = self.found_any.sum()
            print(Nfound)
            
            self.get_opt_params(run_fit, rescale_o3 = False) #we always want 'o3' fit
            
            dmid_params = np.copy(self.dmid_params)
            dmid_params[0] = 1.
    
            m1_det = self.m1 * (1 + self.z)
            m2_det = self.m2 * (1 + self.z)
            mtot_det = m1_det + m2_det
            
            dmid_values = self.dmid(self, m1_det, m2_det, dmid_params)
            #self.apply_dmid_mtotal_max(dmid_values, mtot_det)
            
            emax_params, gamma, delta, alpha = self.set_shape_params()
            
            if self.emax_fun is not None:
                emax = self.emax(self, m1_det, m2_det, emax_params)
                
            else:
                emax = np.copy(emax_params)
            
            def find_root(x):
                new_dmid_values = x * dmid_values
                self.apply_dmid_mtotal_max(new_dmid_values, mtot_det)
                
                frac = self.dL / ( new_dmid_values)
                denom = 1. + frac ** alpha * \
                        np.exp(gamma* (frac - 1.) + delta * (frac**2 - 1.))
                        
                return  np.nansum( emax / denom )  - Nfound
            
            d0 = fsolve(find_root, [50]) #rescaled Dmid cte (it has units of Dmid !!)
            
            return d0
        
        except:
            
            d0 = {'o1' : np.loadtxt('d0.dat')[0], 'o2' : np.loadtxt('d0.dat')[1]}
            
            return d0[run_dataset]
        
        
    
    
    def predicted_events(self, run_fit):
        
        self.load_inj_set(run_dataset)
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
        
        emax_params, gamma, delta, alpha = self.set_shape_params()
        
        if self.emax_fun is not None:
            emax = self.emax(self, m1_det, m2_det, emax_params)
            
        else:
            emax = np.copy(emax_params)
        
        # dmid_check = np.zeros([1,len(dmid_values)])
        # dmid_check = np.vstack([dmid_check, dmid_params])
        
        def manual_found_inj(x):
            frac = self.dL / ( x * dmid_values)
            denom = 1. + frac ** alpha * \
                    np.exp(gamma* (frac - 1.) + delta * (frac**2 - 1.))
            return  np.sum( emax / denom )
        
        o1_inj = manual_found_inj(self.find_dmid_cte_found_inj(self, 'o1', run_fit))
        o2_inj = manual_found_inj(self.find_dmid_cte_found_inj(self, 'o2', run_fit))
        o3_inj = self.load_inj_set('o3').found_any.sum()
        
        frac1 = o1_inj / o3_inj
        frac2 = o2_inj / o3_inj
        
        pred_rates = { 'o1' : frac1 * self.det_rates['o3'], 'o2' : frac2 * self.det_rates['o3'] }
        pred_nev = { 'o1' : pred_rates['o1'] * self.obs_time['o1'], 'o2' : pred_rates['o2'] * self.obs_time['o2'] }
        
        return pred_nev
    
    def run_pdet(self, dL, m1_det, m2_det, run, rescale_o3 = True):
        
        self.get_opt_params(run, rescale_o3) 
        
        mtot_det = m1_det + m2_det
        
        dmid_values = self.dmid(self, m1_det, m2_det, self.dmid_params)
        self.apply_dmid_mtotal_max(np.array(dmid_values), mtot_det)
        
        emax_params, gamma, delta, alpha = self.set_shape_params()
        
        if self.emax_fun is not None:
            emax = self.emax(self, m1_det, m2_det, emax_params)
            
        else:
            emax = np.copy(emax_params)
            
        pdet_i = self.sigmoid(dL, dmid_values, emax , gamma , delta , alpha)
        
        return pdet_i
    
    
    def total_pdet(self, dL, m1_det, m2_det, rescale_o3 = True):
        
        pdet = np.zeros(len(dL))
        
        for run, prop in zip(self.runs, self.prop_obs_time):
            
            pdet_i = self.run_pdet(self, dL, m1_det, m2_det, run, rescale_o3)
            
            pdet += pdet_i * prop
            
        return pdet
        

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

#data.load_inj_set(run_dataset)
#data.get_opt_params(run_fit)

# npoints = 10000
# index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
# m1 = data.m1[index]
# m2=data.m2[index]

# vsensitive = np.array([data.sensitive_volume(run_dataset, run_fit, m1[i], m2[i]) for i in range(len(m1))])

# plt.figure()
# plt.scatter(m1, m2, s=1, c=vsensitive/1e9, norm=LogNorm())
# plt.xlabel('m1')
# plt.ylabel('m2')
# plt.colorbar(label=r'Sensitive volume [Gpc$^3$]')
# plt.savefig( path + f'/Vsensitive_{npoints}.png')

# m1_det = data.m1*(1+data.z)
# m2_det = data.m2*(1+data.z)

# total_pdet = data.total_pdet(data.dL, m1_det, m2_det)

# plt.figure()
# plt.scatter(data.dL/data.dmid(data, m1_det, m2_det, data.dmid_params), total_pdet, s=1)
# plt.xlabel('dL/dmid')
# plt.ylabel('total Pdet')
#plt.savefig( path + '/total_pdet.png')

#data.load_inj_set('o2')

#plt.figure()
#plt.plot(data.dL, data.chi_eff, '.')
#plt.plot(data.dL, data.chieff_d, '.', alpha=0.1)
#plt.plot(data.dL, data.chi_eff - data.chieff_d, '.')

# nbins = 5

# data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'dL')
# data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mtot')
# data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mc')
# data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'Mtot_det')
# data.binned_cumulative_dist(run_dataset, run_fit, nbins, 'chi_eff', 'Mc_det')
# data.binned_cumulative_dist(run_dataset, run_fit, nbins,'chi_eff', 'eta')

# npoints = 10000
# index = np.random.choice(np.arange(len(data.dL)), npoints, replace=False)
# m1 = data.m1[index]
# m2=data.m2[index]

# tot_vsensitive = np.array([data.total_sensitive_volume(run_fit, m1[i], m2[i]) for i in range(len(m1))])

# plt.figure()
# plt.scatter(m1, m2, s=1, c=tot_vsensitive/1e9, norm=LogNorm())
# plt.xlabel('m1')
# plt.ylabel('m2')
# plt.colorbar(label=r'Total sensitive volume [Gpc$^3$]')
# plt.savefig( path + f'/total_Vsensitive_{npoints}.png')

# cmds_bins = ['dL' , 'Mtot', 'Mc', 'Mtot_det', 'Mc_det', 'eta']
# chi_eff_opt_params = {}
# mid_values = {}

# for i in cmds_bins:
#     chi_eff_opt_params[f'{i}_bins'] = np.loadtxt(f'o3/Dmid_mchirp_expansion_noa30/emax_exp_cmds/{i}_bins/chi_eff_cmd/chi_eff_opt_param')
#     mid_values[f'{i}_bins'] = np.loadtxt(f'o3/Dmid_mchirp_expansion_noa30/emax_exp_cmds/{i}_bins/chi_eff_cmd/mid_values')
            
# for i in cmds_bins:
#     plt.figure()
#     plt.plot(mid_values[f'{i}_bins'], chi_eff_opt_params[f'{i}_bins'], 'o')
#     plt.xlabel(f'{i} (middle bin points)')
#     plt.ylabel('c1')
#     plt.title(r'$\chi_{eff}$ correction $ = exp(\chi_{eff} \, c_1)$')
#     plt.savefig(f'o3/Dmid_mchirp_expansion_noa30/emax_exp_cmds/chi_eff_corr_{i}_bins.png')
