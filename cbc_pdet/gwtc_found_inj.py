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
import astropy.cosmology
import astropy.constants as const
from . import fitting_functions as fits #python module which contains the dmid and emax functions

Mtot_max = 510.25378

class Found_injections:
    """
    Class for an algorithm of GW detected found injections from signal templates 
    of binary black hole mergers
    
    Input: an h5py file with the parameters of every sample and the threshold for 
    the false alarm rate (FAR). The default value is thr = 1, which means we are 
    considering a signal as detected when FAR <= 1.
    """

    def __init__(self, dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun = 'emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10, runs = ['o1', 'o2', 'o3', 'o4'], cosmo_parameters = None):
        '''
        Argument ini_files must be a list or a numpy array with two elements
        The first one contains the dmid initial values and the second one the shape initial values 
        
        if ini_files = None, it looks for a ini file in a folder called 'ini_values'. The name of the folder is
        selected from the options of emax_fun, dmid_fun and alpha_vary. The method that finds the file is 
        self.get_ini_values()
        '''
        assert isinstance(thr_far, float) or isinstance(thr_far, int), \
          "Argument (thr_far) must be a float or an integer."
        assert isinstance(thr_snr, float) or isinstance(thr_snr, int), \
          "Argument (thr_snr) must be a float or an integer."
        if isinstance(runs, str):
            runs = [runs]
        assert isinstance(runs, list) or isinstance(runs, tuple),\
          "Argument (runs) must be a list, tuple or string."
          
        allowed_runs = {'o1', 'o2', 'o3', 'o4'}
        assert all(r in allowed_runs for r in runs), \
            f"Invalid run(s) in {runs}. Allowed: {allowed_runs}"
        self.runs = runs
        
        self.thr_far = thr_far
        self.thr_snr = thr_snr
        
        self.dmid_fun = dmid_fun #dmid function name
        self.emax_fun = emax_fun #emax function name
        self.dmid = getattr(fits, dmid_fun) #class method for dmid
        self.emax = getattr(fits, emax_fun) #class method for emax
        self.alpha_vary = alpha_vary
        
        if cosmo_parameters is None:
            cosmo_parameters = {'name': 'FlatLambdaCDM', 'H0': 67.9, 'Om0': 0.3065}
            
        cosmology_class = getattr(astropy.cosmology, cosmo_parameters.pop('name'))
        self.cosmo = cosmology_class(**cosmo_parameters)

        self.Vtot = None  # Slot for total comoving volume up to max z
        self.zinterp_VT = None # Slot for interpolator of z given dL for comoving volume
        
        self.dataset = None
        self.current_pdet = {} #slot for pdet in the optimization
        self.joint_pdfs = {} #slot for pdfs in the optimization
        self.load_all_injections = False
        
        self.dmid_ini_values, self.shape_ini_values = ini_files if ini_files is not None else self.get_ini_values()
        self.dmid_params = self.dmid_ini_values
        self.shape_params = self.shape_ini_values
        
        if self.alpha_vary is None:
            path = f'{self.dmid_fun}' if self.emax_fun is None else f'{self.dmid_fun}/{self.emax_fun}'
        else: 
            path = f'alpha_vary/{self.dmid_fun}' if self.emax_fun is None else f'alpha_vary/{self.dmid_fun}/{self.emax_fun}'
            
        if self.thr_far != 1:
            ifar = int(1 / self.thr_far)
            self.path = f'ifar_{ifar}/' + path
            
        else:
            self.path = path
        
        # Coincident analysis (two or three detectors operating) observing time of O3
        self.coincident_time_o3 = 0.75435365296528  # years
        
        # Total observing time of each run (O3 is different from above because O3 looked at single detector time too)
        self.obs_time = {'o1' : 0.1331507, 'o2' : 0.323288, 'o3' : 0.91101, 'o4' : 0.75564681724846}  # years
        self.total_obs_time = np.sum(list(self.obs_time.values()))
        self.prop_obs_time = np.array([self.obs_time[i]/self.total_obs_time for i in self.runs])
        
        self.obs_nevents = {'o1': 3, 'o2': 7, 'o3': 59, 'o4': 87}

        self.det_rates = {i : self.obs_nevents[i] / self.obs_time[i] for i in self.runs}
        
        self.dmid_params_names = {'Dmid_mchirp': 'D0', 
                                  'Dmid_mchirp_expansion_noa30': 
                                      ['D0', 'a20', 'a01', 'a21', 'a10', 'a11'],
                                  'Dmid_mchirp_fdmid':
                                      ['D0', 'a20', 'a01', 'a21', 'a10', 'a11'], 
                                  'Dmid_mchirp_fdmid_fspin':
                                      ['D0', 'a20', 'a01', 'a21', 'a10', 'a11', 'c1', 'c11'],
                                  'Dmid_mchirp_mixture_logspin_corr':
                                      ['D0', 'B', 'C' , 'mu', 'sigma', 'a_01', 'a_11', 'a_21', 'c_01', 'c_11', 'd_11', 'L'],
                                 }
        
        self.spin_functions = ['Dmid_mchirp_fdmid_fspin', 'Dmid_mchirp_mixture_logspin_corr']
        
        sigmoid_names = ['gamma', 'delta']
        
        if self.emax_fun is None:
            sigmoid_names.append('emax')
            
        if self.alpha_vary is not None:
            sigmoid_names.append('alpha')
        
        self.shape_params_names = {'emax_exp' : sigmoid_names + ['b_0, b_1, b_2'],
                                   'emax_sigmoid' : sigmoid_names + ['b_0, k, M_0'],
                                   'emax_sigmoid_nolog' : sigmoid_names + ['b_0, b_1, b_2'],
                                   'emax_gaussian': sigmoid_names + ['b_0', 'b_1', 'muM', 'sigmaM'],
                                    None : sigmoid_names,
                                  }

        self.sets = {}
        
    def make_folders(self, run, sources):

        if isinstance(sources, str):
            each_source = [source.strip() for source in sources.split(',')] 

        sources_folder = "_".join(sorted(each_source)) 

        try:
            os.mkdir(f'{run}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        try:
            os.mkdir(f'{run}/{sources_folder}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if self.alpha_vary is not None:
            path = f'{run}/{sources_folder}/alpha_vary'
            try:
                os.mkdir(f'{run}/{sources_folder}/alpha_vary')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        else:
            path = f'{run}/{sources_folder}'

        try:
            os.mkdir(path + f'/{self.dmid_fun}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        try:
            os.mkdir(path + f'/{self.dmid_fun}/{self.emax_fun}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        return
        
    def read_o1o2_set(self, run_dataset, source = 'bbh'):
        
        assert run_dataset =='o1' or run_dataset == 'o2', "Argument (run_dataset) must be 'o1' or 'o2'."
        assert source =='bbh' or source == 'bns' or source == 'nsbh' or source == 'imbh', \
          "Argument (source) must be 'bbh' or 'bns' or 'nsbh' or 'imbh'. "
        
        try:
            file = h5py.File(f'{os.path.dirname(__file__)}/{run_dataset}-{source}-IMRPhenomXPHMpseudoFourPN.hdf5', 'r')
        except:
            raise RuntimeError('File with the injection set not found. Please add it to your installation \
                                of cbc_pdet, in the folder where o123_class_found_inj_general.py is. \
                                It can be downloaded from https://dcc.ligo.org/LIGO-T2100280 (currently LVK access)')
                                
        self.sets[source] = {}
                
        atr = dict(file.attrs.items())
        
        # Total number of generated injections
        self.sets[source]['Ntotal'] = atr['total_generated'] 
        
        # Mass 1 and mass 2 values in the source frame in solar units
        self.sets[source]['m1']= file["events"][:]["mass1_source"]
        self.sets[source]['m2'] = file["events"][:]["mass2_source"]

        # Redshift and luminosity distance [Mpc] values 
        self.sets[source]['z'] = file["events"][:]["z"]
        self.sets[source]['dL'] = file["events"][:]["distance"]
      
        # Joint mass sampling pdf (probability density function) values, p(m1,m2)
        self.sets[source]['m1_pdf'] = np.exp(file["events"][:]["logpdraw_mass1_source_GIVEN_z"])
        self.sets[source]['m2_pdf']= np.exp(file["events"][:]["logpdraw_mass2_source_GIVEN_mass1_source"])
        self.sets[source]['m_pdf'] = self.sets[source]['m1_pdf'] * self.sets[source]['m2_pdf']
        
        # Redshift sampling pdf values, p(z), corresponding to a redshift defined by a flat Lambda-Cold Dark Matter cosmology
        self.sets[source]['z_pdf'] = np.exp(file["events"][:]["logpdraw_z"])
        
        self.sets[source]['s1x']= file["events"][:]["spin1x"]
        self.sets[source]['s1y'] = file["events"][:]["spin1y"]
        self.sets[source]['s1z'] = file["events"][:]["spin1z"]
        
        self.sets[source]['s2x'] = file["events"][:]["spin2x"]
        self.sets[source]['s2y'] = file["events"][:]["spin2y"]
        self.sets[source]['s2z'] = file["events"][:]["spin2z"]
        
        self.sets[source]['chieff_d']= file["events"][:]["chi_eff"]
        
        # SNR
        self.sets[source]['snr'] = file["events"][:]["snr_net"]
        found_snr = self.sets[source]['snr'] >= self.thr_snr
        
        # indexes of the found injections
        self.sets[source]['found_any'] = found_snr
        print(f'Found inj in {run_dataset} set: ', self.sets[source]['found_any'].sum())   
        
        return
       
    def read_o3_set(self, source):
        try:
            file = h5py.File(f'{os.path.dirname(__file__)}/endo3_{source}pop-LIGO-T2100113-v12.hdf5', 'r')
        except:
            raise RuntimeError('File with the injection set not found. Please add it to your installation \
                                of cbc_pdet, in the folder where o123_class_found_inj_general.py is. \
                                It can be downloaded from https://zenodo.org/records/7890437')
        
        assert source == 'bbh' or source == 'bns' or source == 'nsbh' or source == 'imbh', \
          "Argument (source) must be 'bbh' or 'bns' or 'nsbh' or 'imbh'. "
            
        self.sets[source] = {}
        
        # Total number of generated injections
        self.sets[source]['Ntotal'] = file.attrs['total_generated'] 
        
        # Mass 1 and mass 2 values in the source frame in solar units
        self.sets[source]['m1'] = file["injections/mass1_source"][:]
        self.sets[source]['m2'] = file["injections/mass2_source"][:]
        
        # Redshift and luminosity distance [Mpc] values 
        self.sets[source]['z'] = file["injections/redshift"][:]
        self.sets[source]['dL'] = file["injections/distance"][:]
      
        # Joint mass sampling pdf (probability density function) values, p(m1,m2)
        self.sets[source]['m_pdf'] = file["injections/mass1_source_mass2_source_sampling_pdf"][:]
        
        # Redshift sampling pdf values, p(z), corresponding to a redshift defined by a flat Lambda-Cold Dark Matter cosmology
        self.sets[source]['z_pdf'] = file["injections/redshift_sampling_pdf"][:]
        
        self.sets[source]['s1x'] = file["injections/spin1x"][:]
        self.sets[source]['s1y'] = file["injections/spin1y"][:]
        self.sets[source]['s1z'] = file["injections/spin1z"][:]
        
        self.sets[source]['s2x'] = file["injections/spin2x"][:]
        self.sets[source]['s2y'] = file["injections/spin2y"][:]
        self.sets[source]['s2z'] = file["injections/spin2z"][:]
        
        # False alarm rate statistics from each pipeline
        self.sets[source]['far_pbbh'] = file["injections/far_pycbc_bbh"][:]
        self.sets[source]['far_gstlal'] = file["injections/far_gstlal"][:]
        self.sets[source]['far_mbta'] = file["injections/far_mbta"][:]
        self.sets[source]['far_pfull'] = file["injections/far_pycbc_hyperbank"][:]
        self.sets[source]['snr'] = file['injections/optimal_snr_net'][:]
        
        found_pbbh = self.sets[source]['far_pbbh'] <= self.thr_far
        found_gstlal = self.sets[source]['far_gstlal'] <= self.thr_far
        found_mbta = self.sets[source]['far_mbta'] <= self.thr_far
        found_pfull = self.sets[source]['far_pfull'] <= self.thr_far

        # Indices of the found injections
        self.sets[source]['found_any'] = found_pbbh | found_gstlal | found_mbta | found_pfull
        print(f'Found inj in o3 set and {source} sources: ', self.sets[source]['found_any'] .sum())  
        
        return
    
    def read_o4_set(self, source = 'all'):
       
       try:
           file = h5py.File(f'{os.path.dirname(__file__)}/samples-rpo4a_v2_20250220153231UTC-1366933504-23846400.hdf', 'r')
       except:
           raise RuntimeError('File with the injection set not found. Please add it to your installation \
                               of cbc_pdet, in the folder where gwtc_found_inj.py is.')
                               
       assert source == 'all', "Argument (source) must be 'all'. " 

       self.sets[source] = {}
       
       # Total number of generated injections
       self.sets[source]['Ntotal'] = file.attrs['total_generated'] 
       
       # Mass 1 and mass 2 values in the source frame in solar units
       self.sets[source]['m1'] = file['events'][:]['mass1_source']
       self.sets[source]['m2'] = file['events'][:]['mass2_source']
       
       # Redshift and luminosity distance [Mpc] values 
       self.sets[source]['z'] = file['events'][:]['z']
       self.sets[source]['dL'] = file['events'][:]['luminosity_distance']
     
       # Joint mass sampling pdf (probability density function) values, p(m1,m2)
       self.sets[source]['m1_pdf'] = np.exp(file["events"][:]["lnpdraw_mass1_source"])
       self.sets[source]['m2_pdf'] = np.exp(file["events"][:]["lnpdraw_mass2_source_GIVEN_mass1_source"])
       self.sets[source]['m_pdf'] = self.sets[source]['m1_pdf'] * self.sets[source]['m2_pdf']
       
       # Redshift sampling pdf values, p(z), corresponding to a redshift defined by a flat Lambda-Cold Dark Matter cosmology
       self.sets[source]['z_pdf'] = np.exp(file["events"][:]["lnpdraw_z"])
       
       self.sets[source]['s1x'] = file["events"][:]["spin1x"]
       self.sets[source]['s1y'] = file["events"][:]["spin1y"]
       self.sets[source]['s1z'] = file["events"][:]["spin1z"]
       
       self.sets[source]['s2x'] = file["events"][:]["spin2x"]
       self.sets[source]['s2y'] = file["events"][:]["spin2y"]
       self.sets[source]['s2z'] = file["events"][:]["spin2z"]
       
       self.sets[source]['a1'] = file["events"][:]["spin1_magnitude"]
       self.sets[source]['a2'] = file["events"][:]["spin2_magnitude"]
       self.sets[source]['theta1'] = file['events'][:]['spin1_polar_angle']
       self.sets[source]['theta2'] = file['events'][:]['spin2_polar_angle']
       
       self.sets[source]['a1_pdf'] = np.exp(file["events"][:]["lnpdraw_spin1_magnitude"])
       self.sets[source]['a2_pdf'] = np.exp(file["events"][:]["lnpdraw_spin2_magnitude"])
       self.sets[source]['theta1_pdf'] = np.exp(file['events'][:]['lnpdraw_spin1_polar_angle'])
       self.sets[source]['theta2_pdf'] = np.exp(file['events'][:]['lnpdraw_spin2_polar_angle'])
       
       self.sets[source]['chi_eff'] = file["events"][:]["chi_eff"]
       
       # False alarm rate statistics from each pipeline
       self.sets[source]['far_pbbh'] = file["events"][:]["pycbc_far"]
       self.sets[source]['far_gstlal'] = file["events"][:]["gstlal_far"]
       self.sets[source]['far_mbta'] = file["events"][:]["mbta_far"]
       self.sets[source]['far_cwb'] = file["events"][:]["cwb-bbh_far"]
       self.sets[source]['snr'] = file["events"][:]["snr_net"]
       
       found_pbbh = self.sets[source]['far_pbbh'] <= self.thr_far
       found_gstlal = self.sets[source]['far_gstlal'] <= self.thr_far
       found_mbta = self.sets[source]['far_mbta'] <= self.thr_far
       found_cwb = self.sets[source]['far_cwb'] <= self.thr_far
       
       # indexes of the found injections
       self.sets[source]['found_any'] = found_pbbh | found_gstlal | found_mbta | found_cwb
       print('Found inj in o4 set: ', self.sets[source]['found_any'] .sum())  
       
       return       
        
    def load_inj_set(self, run_dataset, source = 'all'):
        
        if run_dataset == 'o3':
            self.read_o3_set() 
        
        elif run_dataset == 'o4':
            self.read_o4_set()
            source = 'all'
        
        else:
            self.read_o1o2_set(run_dataset)
        
        self.dataset = run_dataset
        source_data = self.sets[source].copy()
        
        # Luminosity distance sampling pdf values, p(dL), computed for a flat Lambda-Cold Dark Matter cosmology from the z_pdf values
        self.sets[source]['dL_pdf'] = source_data['z_pdf']/ fits.dL_derivative(source_data['z'], source_data['dL'], self.cosmo)
        
        # Total mass (m1+m2)
        Mtot_source = source_data['m1'] + source_data['m2']
        oneplusz = 1 + source_data['z']
        self.sets[source]['m1_det'] = source_data['m1'] * oneplusz
        self.sets[source]['m2_det'] = source_data['m2'] * oneplusz
        self.sets[source]['Mtot'] = Mtot_source
        self.sets[source]['Mtot_det'] = Mtot_source * oneplusz
        
        # Chirp mass
        Mc = fits.Mc(source_data['m1'], source_data['m2'])
        self.sets[source]['Mc'] = Mc
        self.sets[source]['Mc_det'] = Mc * oneplusz
        
        # Eta aka symmetric mass ratio
        mu = (source_data['m1'] * source_data['m2']) / Mtot_source
        self.sets[source]['eta'] = mu / Mtot_source
        self.sets[source]['q'] = source_data['m2'] / source_data['m1']
        
        # Spin amplitude
        self.sets[source]['a1'] = np.sqrt(source_data['s1x']**2 + source_data['s1y']**2 + source_data['s1z']**2)
        self.sets[source]['a2'] = np.sqrt(source_data['s2x']**2 + source_data['s2y']**2 + source_data['s2z']**2)
        
        a1_max = np.max(self.sets[source]['a1'])
        a2_max = np.max(self.sets[source]['a2'])
        self.sets[source]['a1_max'] = a1_max
        self.sets[source]['a2_max'] = a2_max
        
        self.sets[source]['s1z_pdf'] = np.log(a1_max / np.abs(source_data['s1z'])) / (2 * a1_max)
        self.sets[source]['s2z_pdf'] = np.log(a2_max / np.abs(source_data['s2z'])) / (2 * a2_max)
        
        self.sets[source]['chi_eff'] = (source_data['s1z'] * source_data['m1'] + source_data['s2z'] * source_data['m2']) \
                                        / (Mtot_source)

        max_index = np.argmax(source_data['dL'])
        self.sets[source]['dLmax'] = source_data['dL'][max_index]
        self.sets[source]['zmax'] = np.max(source_data['z'])

        index = np.random.choice(np.arange(len(source_data['dL'])), 200, replace=False)
        if max_index not in index:
            index = np.insert(index, -1, max_index)

        try_dL = source_data['dL'][index]
        try_dLpdf = self.sets[source]['dL_pdf'][index]

        # Add a value at 0 for interpolation
        inter_dL = np.insert(try_dL, 0, 0, axis=0)
        inter_dLpdf = np.insert(try_dLpdf, 0, 0, axis=0)
        self.sets[source]['interp_dL_pdf'] = interpolate.interp1d(inter_dL, inter_dLpdf)

        try_z = source_data['z'][index]
        inter_z = np.insert(try_z, 0, 0, axis=0)

        # Add a value for self.zmax
        new_dL = np.insert(inter_dL, -1, self.sets[source]['dLmax'], axis=0)
        new_z = np.insert(inter_z, -1, self.sets[source]['zmax'], axis=0)

        self.sets[source]['interp_z'] = interpolate.interp1d(new_dL, new_z)

        # Distributions for 03 injections
        if source == 'bbh':
            self.sets[source]['mmin'] = 2.
            self.sets[source]['mmax'] = 100.
            
            self.sets[source]['pow_m1'] = -2.35
            self.sets[source]['pow_m2'] = 1.
            
        elif source == 'bns':
            self.sets[source]['mmin'] = 1.
            self.sets[source]['mmax'] = 2.5
            
            self.sets[source]['pow_m1'] = 1.
            self.sets[source]['pow_m2'] = 0
            
        elif source == 'nsbh':
            self.sets[source]['mmin'] = 2.5
            self.sets[source]['mmax'] = 60.
            
            self.sets[source]['pow_m1'] = -2.35
            self.sets[source]['pow_m2'] = 0.
            
        elif source == 'imbh':
            self.sets[source]['mmin'] = 90.
            self.sets[source]['mmax'] = 600.
            
            self.sets[source]['pow_m1'] = -1.
            self.sets[source]['pow_m2'] = -1.
            
        if self.dmid_fun in self.spin_functions:
            
            if self.dataset == 'o4':
                self.joint_pdfs[source] = source_data['m_pdf'] * self.sets[source]['dL_pdf'] * source_data['a1_pdf'] * source_data['a2_pdf'] * source_data['theta1_pdf'] * source_data['theta2_pdf']
            
            else:
                self.sets[source]['a1'] = np.sqrt(source_data['s1x']**2 + source_data['s1y']**2 + source_data['s1z']**2)
                self.sets[source]['a2'] = np.sqrt(source_data['s2x']**2 + source_data['s2y']**2 + source_data['s2z']**2)
                
                self.sets[source]['a1_max'] = np.max(source_data['a1'])
                self.sets[source]['a2_max'] = np.max(source_data['a2'])
                
                self.sets[source]['s1z_pdf'] = np.log(self.sets[source]['a1_max'] / np.abs(source_data['s1z'])) / (2*self.sets[source]['a1_max'])
                self.sets[source]['s2z_pdf'] = np.log(self.sets[source]['a2_max'] / np.abs(source_data['s2z'])) / (2*self.sets[source]['a2_max'])
                
                self.joint_pdfs[source] = source_data['m_pdf'] * source_data['dL_pdf'] * source_data['s1z_pdf'] * source_data['s2z_pdf']
            
        else: 
            self.joint_pdfs[source] = source_data['m_pdf'] * self.sets[source]['dL_pdf']
        print('finished loading inj set')
        return
    
    def load_all_inj_sets(self, run_dataset, sources):
        
        if isinstance(sources, str):
            each_source = [source.strip() for source in sources.split(',')]

        [self.load_inj_set(run_dataset, source) for source in each_source]

        self.Ntotal = sum(self.sets[source]['Ntotal'] for source in each_source)
        self.dL = np.concatenate([self.sets[source]['dL'] for source in each_source])
        self.z = np.concatenate([self.sets[source]['z'] for source in each_source])
        self.m1 = np.concatenate([self.sets[source]['m1'] for source in each_source])
        self.m2 = np.concatenate([self.sets[source]['m2'] for source in each_source])
        self.m1_det = np.concatenate([self.sets[source]['m1_det'] for source in each_source])
        self.m2_det = np.concatenate([self.sets[source]['m2_det'] for source in each_source])
        self.chi_eff = np.concatenate([self.sets[source]['chi_eff'] for source in each_source])
        self.s1x = np.concatenate([self.sets[source]['s1x'] for source in each_source])
        self.s1y = np.concatenate([self.sets[source]['s1y'] for source in each_source])
        self.s1z = np.concatenate([self.sets[source]['s1z'] for source in each_source])
        self.s2x = np.concatenate([self.sets[source]['s2x'] for source in each_source])
        self.s2y = np.concatenate([self.sets[source]['s2y'] for source in each_source])
        self.s2z = np.concatenate([self.sets[source]['s2z'] for source in each_source])
        self.Mtot = np.concatenate([self.sets[source]['Mtot'] for source in each_source])
        self.Mtot_det = np.concatenate([self.sets[source]['Mtot_det'] for source in each_source])
        self.Mc = np.concatenate([self.sets[source]['Mc'] for source in each_source])
        self.Mc_det = np.concatenate([self.sets[source]['Mc_det'] for source in each_source])
        self.eta = np.concatenate([self.sets[source]['eta'] for source in each_source])
        self.q = np.concatenate([self.sets[source]['q'] for source in each_source])
        self.found_any = np.concatenate([self.sets[source]['found_any'] for source in each_source])
        self.total_found_inj = np.sum(self.found_any)
        
        self.dL_pdf = np.concatenate([self.sets[source]['dL_pdf'] for source in each_source])
        self.z_pdf = np.concatenate([self.sets[source]['z_pdf'] for source in each_source])
        self.m_pdf = np.concatenate([self.sets[source]['m_pdf'] for source in each_source])
        self.s1z_pdf = np.concatenate([self.sets[source]['s1z_pdf'] for source in each_source])
        self.s2z_pdf = np.concatenate([self.sets[source]['s2z_pdf'] for source in each_source])
        
        self.load_all_injections = True
        
        return
        
    
    def get_opt_params(self, run_fit, sources, rescale_o3 = True):
        '''
        Sets self.dmid_params and self.shape_params as a class attribute (optimal values from some previous fit).

        Parameters
        ----------
        run_fit : str. Observing run from which we want to use the fit. Must be 'o1', 'o2' or 'o3'.

        Returns
        -------
        None
        '''
        assert run_fit =='o1' or run_fit == 'o2' or run_fit == 'o3' or run_fit == 'o4',\
        "Argument (run_fit) must be 'o1' or 'o2' or 'o3' or 'o4'. "
        
        if isinstance(sources, str):
           each_source = [source.strip() for source in sources.split(',')] 
           
        sources_folder = "_".join(sorted(each_source)) 
       
        if not rescale_o3 or run_fit == 'o4': # get separate independent fit files
             run_fit_touse = run_fit
        else:  # Rescale o1 and o2
            run_fit_touse = 'o3'

        try:
            path = f'{os.path.dirname(__file__)}/{run_fit_touse}/{sources_folder}/' + self.path
            self.dmid_params = np.loadtxt(path + '/joint_fit_dmid.dat')[-1, :-1]
            self.shape_params = np.loadtxt(path + '/joint_fit_shape.dat')[-1, :-1]
        except:
            raise RuntimeError('ERROR in self.get_opt_params: There are not such files because there is not a fit yet with these options.')
    
        if rescale_o3 and run_fit != 'o3' and run_fit != 'o4':
            d0 = self.find_dmid_cte_found_inj(run_fit, 'o3')
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
        if self.alpha_vary is None:
            path = f'{os.path.dirname(__file__)}/ini_values/{self.dmid_fun}' if self.emax_fun is None \
                else f'{os.path.dirname(__file__)}/ini_values/{self.dmid_fun}_{self.emax_fun}'
        else: 
            path = f'{os.path.dirname(__file__)}/ini_values/alpha_vary_{self.dmid_fun}' if self.emax_fun is None \
                else f'{os.path.dirname(__file__)}/ini_values/alpha_vary_{self.dmid_fun}_{self.emax_fun}'

        try:
            dmid_ini_values = np.loadtxt( path + '_dmid.dat')
            shape_ini_values = np.loadtxt( path + '_shape.dat')
        except:
            raise RuntimeError('ERROR in self.get_ini_values: Files not found. Please create .dat files with the initial param guesses for this fit.')

        return dmid_ini_values, shape_ini_values
    
    def get_shape_params(self):
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
            alpha = 2.05 if self.alpha_vary is None else np.copy(self.shape_params[2])

        else:
            gamma, delta, emax_params = np.copy(self.shape_params[:3])
            alpha = 2.05 if self.alpha_vary is None else np.copy(self.shape_params[3])
        
        return emax_params, gamma, delta, alpha
    
    def set_shape_params(self):
        self.emax_params, self.gamma, self.delta, self.alpha = self.get_shape_params()
        
        return
    
    def sigmoid(self, dL, dLmid, emax, gamma, delta, alpha = 2.05):
        """
        Sigmoid function used to estimate the probability of detection of CBC events

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
        Array of detection probability.
        """
        frac = dL / dLmid
        denom = 1. + frac ** alpha * \
            np.exp(gamma * (frac - 1.) + delta * (frac**2 - 1.))
        return emax / denom
 
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
        mmin = self.mmin; mmax = self.mmax
        alpha = self.pow_m1 ; beta = self.pow_m2
        
        m1_norm = (1. + alpha) / (mmax ** (1. + alpha) - mmin ** (1. + alpha))
        m2_norm = (1. + beta) / (m1 ** (1. + beta) - mmin ** (1. + beta))
        
        return m1**alpha * m2**beta * m1_norm * m2_norm * np.heaviside(m1 - m2, 1)

    def apply_dmid_mtotal_max(self, dmid_values, Mtot_det, max_mtot = None):
        '''
        not actually using this at this point, for the recommended functions (i.e the default) it's not necessary
        '''
        max_mtot = max_mtot if max_mtot != None else Mtot_max
        #return np.putmask(dmid_values, Mtot_det > max_mtot, 0.001)
        # If dmid_values is a single float, handle it differently
        if isinstance(dmid_values, (int, float)):
            if np.any(Mtot_det > max_mtot):
                return 0.001  # Return the modified single float
            else:
                return dmid_values  # Return the original value if condition is not met

        # If dmid_values is an array, apply the condition across the array
        dmid_values[Mtot_det > max_mtot] = 0.001

        return dmid_values
    
    def Nexp(self, dmid_params, shape_params, source):
        """
        Expected number of found injections, computed as a sum over the 
        probability of detection of every injection

        Parameters
        ----------
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        
        Returns
        -------
        float
        """
        source_data = self.sets[source].copy()
        
        m1_det = source_data['m1_det']
        m2_det = source_data['m2_det']
        dL = source_data['dL']
        chi_eff = source_data['chi_eff']
        
        if self.dmid_fun in self.spin_functions:
            dmid_values = self.dmid(m1_det, m2_det, chi_eff, dmid_params)
        else: 
            dmid_values = self.dmid(m1_det, m2_det, dmid_params)   
        
        if self.emax_fun is None and self.alpha_vary is None:
            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            sigmoid_args = emax, gamma, delta
            
        elif self.emax_fun is None and self.alpha_vary is not None:
            gamma, delta, emax, alpha = shape_params[0], shape_params[1], shape_params[2], shape_params[3]
            sigmoid_args = emax, gamma, delta, alpha
        
        elif self.emax_fun is not None and self.alpha_vary is None:
            gamma, delta = shape_params[0], shape_params[1]
            emax_params = shape_params[2:]
            emax_values = self.emax(m1_det, m2_det, emax_params)
            sigmoid_args = emax_values, gamma, delta
            
        else:  
            gamma, delta, alpha = shape_params[0], shape_params[1], shape_params[2]
            emax_params = shape_params[3:]
            emax_values = self.emax(m1_det, m2_det, emax_params)
            sigmoid_args = emax_values, gamma, delta, alpha
        
        self.current_pdet[source] = self.sigmoid(dL, dmid_values, *sigmoid_args)
        Nexp = np.sum(self.current_pdet[source])
        return Nexp
    
        return Nexp
        
    def lamda(self, dmid_params, shape_params, source):
        """
        Number density at found injections, aka lambda(D, m1, m2)

        Parameters
        ----------
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        
        Returns
        -------
        float
        """
        source_data = self.sets[source].copy()
        
        found = source_data['found_any']
        pdfs = self.joint_pdfs[source][found]
        
        if source in self.current_pdet:
            pdet = self.current_pdet[source][found]
            
        else:
            dL = source_data['dL'][found]
            m1_det = source_data['m1_det'][found]
            m2_det = source_data['m2_det'][found]
            chieff = source_data['chi_eff'][found]
        
            if self.dmid_fun in self.spin_functions:
                dmid_values = self.dmid(m1_det, m2_det, chieff, dmid_params)
            else: 
                dmid_values = self.dmid(m1_det, m2_det, dmid_params)
                
            #self.apply_dmid_mtotal_max(dmid_values, mtot_det)
            
            if self.emax_fun is None and self.alpha_vary is None:
                gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
                sigmoid_args = emax, gamma, delta
                
            elif self.emax_fun is None and self.alpha_vary is not None:
                gamma, delta, emax, alpha = shape_params[0], shape_params[1], shape_params[2], shape_params[3]
                sigmoid_args = emax, gamma, delta, alpha
    
            elif self.emax_fun is not None and self.alpha_vary is None:
                gamma, delta = shape_params[0], shape_params[1]
                emax_params = shape_params[2:]
                emax_values = self.emax(m1_det, m2_det, emax_params)
                sigmoid_args = emax_values, gamma, delta
    
            else:
                gamma, delta, alpha = shape_params[0], shape_params[1], shape_params[2]
                emax_params = shape_params[3:]
                emax_values = self.emax(m1_det, m2_det, emax_params)
                sigmoid_args = emax_values, gamma, delta, alpha
    
            pdet = self.sigmoid(dL, dmid_values, *sigmoid_args)
            
        return pdet * pdfs * source_data['Ntotal']  # lambda
    
    def logL(self, dmid_params, shape_params, source):
        """
        log likelihood of the expected density of found injections

        Parameters
        ----------
        dmid_params : parameters of the Dmid function, 1D array
        shape_params : gamma, delta and emax params, 1D array
        
        Returns
        -------
        float 
        """
        lnL = -self.Nexp(dmid_params, shape_params, source) + \
            np.sum(np.log(self.lamda(dmid_params, shape_params, source)))
        return lnL
    
    def total_logL(self, dmid_params, shape_params, sources):
       
       if isinstance(sources, str):
           each_source = [source.strip() for source in sources.split(',')]
           
       try: 
           tot_lnL = sum(self.logL(dmid_params, shape_params, source) for source in each_source)
           return tot_lnL
           
       except KeyError as e:
           raise ValueError(f'Unknown source type: {e.args[0]}')
    
    def MLE_dmid(self, methods, sources):
        """
        minimization of -logL on dmid

        Parameters
        ----------
        methods : str, scipy method used to minimize -logL
        
        Returns
        -------
        opt_params : optimized value for Dmid params, 1D array
        -min_likelihood : maximum log likelihood, float

        """
        dmid_params_guess = np.copy(self.dmid_params)
        
        res = opt.minimize(fun=lambda in_param: -self.total_logL(in_param, self.shape_params, sources), 
                           x0=np.array(dmid_params_guess), 
                           args=(), 
                           method=methods)
        opt_params = res.x
        min_likelihood = res.fun   
        self.dmid_params = opt_params  
          
        return opt_params, -min_likelihood
    
    def MLE_shape(self, methods, sources):
        """
        minimization of -logL on shape params (with emax as a cte)

        Parameters
        ----------
        methods : scipy method used to minimize -logL
    
        Returns
        -------
        opt_params: optimized values for [gamma, delta, emax], 1D array
        -min_likelihood : maximum log likelihood, float
        """
        shape_params_guess = np.copy(self.shape_params)
        all_bounds = [(None, None)] * len(shape_params_guess)
        all_bounds[1] = (0, None) #delta
        
        if self.emax_fun == 'emax_gaussian':
            all_bounds[2] = (0, 1) #b0
            all_bounds[3] = (0, 1) #b1
        
        res = opt.minimize(fun=lambda in_param: -self.total_logL(self.dmid_params, in_param, sources), 
                           x0=np.array(shape_params_guess), 
                           args=(), 
                           method=methods,
                           bounds=all_bounds)
        
        opt_params = res.x
        min_likelihood = res.fun  
        self.shape_params = opt_params
        
        return opt_params, -min_likelihood
    
    def joint_MLE(self, run_dataset, sources = 'bbh', methods = 'Nelder-Mead', precision = 1e-2, bootstrap = False):
        '''
        joint optimization of log likelihood, alternating between optimizing dmid params and shape params
        until the difference in the log L is <= precision . Saves the results of each iteration in txt files.

        Parameters
        ----------
        run_dataset : str. Observing run injections that we want to fit. Must be 'o1', 'o2' or 'o3'.
        methods : str, scipy method used to minimize -logL
        precision : float (positive), optional. Tolerance for termination . The default is 1e-2.

        Returns
        -------
        None.
        '''   
        if isinstance(sources, str):
            each_source = [source.strip() for source in sources.split(',')] 
            
        sources_folder = "_".join(sorted(each_source)) 
        
        self.make_folders(run_dataset, sources_folder)
        
        [self.load_inj_set(run_dataset, source) for source in each_source]
        
        total_lnL = np.zeros([1])
        all_gamma = []
        all_delta = []
        all_emax = []
        all_alpha = []
        all_dmid_params = np.zeros([1,len(np.atleast_1d(self.dmid_params))])
        
        path = f'{run_dataset}/{sources_folder}/' + self.path
        
        if self.alpha_vary is None:
            all_emax_params = np.zeros([1,len(np.atleast_1d(self.shape_params[2:]))])
            params_emax = None if self.emax is None else np.copy(self.shape_params[2:])
            
        else:
            all_emax_params = np.zeros([1,len(np.atleast_1d(self.shape_params[3:]))])
            params_emax = None if self.emax_fun is None else np.copy(self.shape_params[3:])
        
        for i in range(0, 10000):
            # Alternate optimizing for d_mid and shape parameters, saving at each iteration
            
            dmid_params, maxL_1 = self.MLE_dmid(methods, sources)
            all_dmid_params = np.vstack([all_dmid_params, dmid_params])

            shape_params, maxL_2 = self.MLE_shape(methods, sources)
            
            if self.emax_fun is None:
                gamma_opt, delta_opt, emax_opt = shape_params[:2]
                all_gamma.append(gamma_opt); all_delta.append(delta_opt)
                all_emax.append(emax_opt); total_lnL = np.append(total_lnL, maxL_2)
                
                if self.alpha_vary is not None:
                    alpha_opt = shape_params[3]
                    all_alpha.append(alpha_opt)
                         
            elif self.emax_fun is not None:
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
                
            lnL_diff = np.abs( total_lnL[i+1] - total_lnL[i] ) 
            
            print('\n', maxL_2)
            print(lnL_diff)
            
            if lnL_diff <= precision : break
        
        print('\nNumber of needed iterations with precision <= %s : %s (+1 since it starts on 0)' %(precision, i))

        total_lnL = np.delete(total_lnL, 0)
        
        shape_header = f'{self.shape_params_names[self.emax_fun]} , maxL'
        
        name_dmid_file = path + '/joint_fit_dmid.dat'
        name_shape_file = path + '/joint_fit_shape.dat'
        
        # Saving opt shape params file
        if self.emax_fun is None:
            shape_results = np.column_stack((all_gamma, all_delta, all_emax, total_lnL)) if self.alpha_vary is None \
                       else np.column_stack((all_gamma, all_delta, all_emax, all_alpha, total_lnL))
        else:
            shape_results = np.column_stack((all_gamma, all_delta, np.delete(all_emax_params, 0, axis=0), total_lnL)) if self.alpha_vary is None \
                       else np.column_stack((all_gamma, all_delta, all_alpha, np.delete(all_emax_params, 0, axis=0), total_lnL)) 
            
        np.savetxt(name_shape_file, shape_results, header = shape_header, fmt='%s') if not bootstrap else None
        
        # Saving opt dmid params file 
        all_dmid_params = np.delete(all_dmid_params, 0, axis=0)
        dmid_results = np.column_stack((all_dmid_params, total_lnL))
        dmid_header = f'{self.dmid_params_names[self.dmid_fun]} , maxL'
        np.savetxt(name_dmid_file, dmid_results, header = dmid_header, fmt='%s') if not bootstrap else None
        
        return shape_results[-1, :-1], dmid_results[-1, :-1]

    def cumulative_dist(self, run_dataset, run_fit, sources, var, ks = False):
        '''
        Saves cumulative distributions plots and prints KS tests for the specified variables  

        Parameters
        ----------
        run_dataset : str. Observing run from which we want the injections. Must be 'o1', 'o2' or 'o3'.
        run_fit : str. Observing run from which we want to use its fit. Must be 'o1', 'o2' or 'o3'.
        var : str, variable for the CDFs and KS tests. Options:
            'dL' - luminosity distance
            'Mc' - chirp mass
            'Mtot' - total mass 
            'eta' - symmetric mass ratio
            'Mc_det' - chirp mass in the detector frame
            'Mtot_det' - total mass in the detector frame

        Returns
        -------
        stat : float, statistic from the KStest 
        pvalue : float, pvalue from the KStest 
        '''
        if isinstance(sources, str):
            each_source = [source.strip() for source in sources.split(',')] 

        sources_folder = "_".join(sorted(each_source)) 
        self.make_folders(run_fit, sources_folder)
        
        self.get_opt_params(run_fit, sources)
        if not self.load_all_injections:
            self.load_all_inj_sets(run_dataset, sources)
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds', 'emax_gaussian' : 'emax_gaussian_cmds'}
        
        dic = {'dL': self.dL,
               'z': self.z,
               'Mc': self.Mc,
               'Mtot': self.Mtot,
               'eta': self.eta,
               'Mc_det': self.Mc_det,
               'Mtot_det': self.Mtot_det,
               'chi_eff': self.chi_eff}
        path = f'{run_fit}/{sources_folder}/{self.dmid_fun}' if self.alpha_vary is None \
            else f'{run_fit}/{sources_folder}/alpha_vary/{self.dmid_fun}'
        names_plotting = {'dL': r'$d_L$',
                          'z': r'$z$',
                          'Mc': r'$\mathcal{M}$',
                          'Mtot': r'$M$',
                          'eta': r'$\eta$',
                          'Mc_det': r'$\mathcal{M}_z$',
                          'Mtot_det': r'$M_z$',
                          'chi_eff': r'$\chi_{eff}$'}
        
        try:
            os.mkdir(path + f'/{emax_dic[self.emax_fun]}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        # Cumulative distribution over the desired variable
        indexo = np.argsort(dic[var])
        varo = dic[var][indexo]
        dLo = self.dL[indexo]
        chieffo = self.chi_eff[indexo]
        m1o_det = self.m1_det[indexo]
        m2o_det = self.m2_det[indexo]
        
        if self.dmid_fun in self.spin_functions:
            dmid_values = self.dmid(m1o_det, m2o_det, chieffo, self.dmid_params)
        else: 
            dmid_values = self.dmid(m1o_det, m2o_det, self.dmid_params)
        
        #self.apply_dmid_mtotal_max(dmid_values, mtoto_det)
        
        emax_params, gamma, delta, alpha = self.get_shape_params()
        
        if self.emax_fun is not None:
            emax = self.emax(m1o_det, m2o_det, emax_params) 
        else:
            emax = np.copy(emax_params)
        
        pdet = self.sigmoid(dLo, dmid_values, emax, gamma, delta, alpha)
        cmd = np.cumsum(pdet)
        
        # Found injections
        var_found = dic[var][self.found_any]
        indexo_found = np.argsort(var_found)
        var_foundo = var_found[indexo_found]
        real_found_inj = np.arange(len(var_foundo))+1
    
        plt.figure()
        plt.scatter(varo, cmd, s=1, label='model', rasterized=True)
        plt.scatter(var_foundo, real_found_inj, s=1, label='found injections', rasterized=True)
        plt.xlabel(names_plotting[var], fontsize = 20)
        plt.ylabel('Cumulative found injections', fontsize = 20)
        plt.legend(loc='best', fontsize = 20)
        name = path + f'/{emax_dic[self.emax_fun]}/{var}_cumulative.png'
        plt.savefig(name, format='png', bbox_inches="tight")
        name = path + f'/{emax_dic[self.emax_fun]}/{var}_cumulative.pdf'
        plt.savefig(name, format='pdf', dpi=150, bbox_inches="tight")
        
        # KS test
        if ks: 
            pdet_weighted = cmd / np.sum(pdet)
            def cdf(x):
                idxs = np.searchsorted(varo, x, side='right')
                return np.where(idxs == 0, 0, pdet_weighted[idxs - 1])
                
            stat, pvalue = kstest(var_foundo, cdf)
            print(f'{var} KStest : statistic = %s , pvalue = %s' %(stat, pvalue))    

        return

    def binned_cumulative_dist(self, run_dataset, run_fit, sources, nbins, var_cmd, var_binned, ks = False):
        '''
        Saves binned cumulative distributions and prints binned KS tests for the specified variables 

        Parameters
        ----------
        run_dataset : str. Observing run from which we want the injections. Must be 'o1', 'o2' or 'o3'.
        run_fit : str. Observing run from which we want to use its fit. Must be 'o1', 'o2' or 'o3'.
        nbins : int, number of bins
        var_cmd : str, variable for the CDFs and KS tests. Options:
            'dL' - luminosity distance
            'Mc' - chirp mass
            'Mtot' - total mass 
            'eta' - symmetric mass ratio
            'Mc_det' - chirp mass in the detector frame
            'Mtot_det' - total mass in the detector frame
        var_binned : str, variable in which we are taking bins. Same options as var_cmd      

        Returns
        -------
        None
        '''        
        if isinstance(sources, str):
            each_source = [source.strip() for source in sources.split(',')] 
        
        sources_folder = "_".join(sorted(each_source)) 
        
        self.get_opt_params(run_fit, sources)
        self.make_folders(run_fit, sources_folder)
        if not self.load_all_injections:
            self.load_all_inj_sets(run_dataset, sources)
        
        emax_dic = {None: 'cmds', 'emax_exp' : 'emax_exp_cmds', 'emax_sigmoid' : 'emax_sigmoid_cmds', 'emax_gaussian' : 'emax_gaussian_cmds'}
        path = f'{run_fit}/{sources_folder}/{self.dmid_fun}' if self.alpha_vary is None else f'{run_fit}/{sources_folder}/alpha_vary/{self.dmid_fun}'
        
        try:
            os.mkdir(path + f'/{emax_dic[self.emax_fun]}')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
     
        try:
            os.mkdir(path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        try:
            os.mkdir(path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        bin_dic = {'dL': self.dL, 'z': self.z, 'Mc': self.Mc, 'Mtot': self.Mtot, 'eta': self.eta, 'Mc_det': self.Mc_det, 'Mtot_det': self.Mtot_det}
        
        # Sort data
        data_not_sorted = bin_dic[var_binned]
        index = np.argsort(data_not_sorted)
        data = data_not_sorted[index]
        
        dLo = self.dL[index]
        m1o_det = self.m1_det[index]
        m2o_det = self.m2_det[index]
        Mco = self.Mc[index]; Mtoto = self.Mtot[index]; etao = self.eta[index]
        Mco_det = self.Mc_det[index]; Mtoto_det = self.Mtot_det[index]
        chi_effo = self.chi_eff[index]
        found_any_o = self.found_any[index]
        
        # Create bins with equal amounts of data
        def equal_bin(N, m):
            sep = (N.size/float(m))*np.arange(1,m+1)
            idx = sep.searchsorted(np.arange(N.size))
            return idx[N.argsort().argsort()]
        
        index_bins = equal_bin(data, nbins)
        
        print(f'\n{var_binned} bins:\n')
        chi_eff_params = []
        mid_values = []
        
        for i in range(nbins):
            # Get data in each bin
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
        
            # Cumulative distribution over the desired variable
            indexo = np.argsort(cmd_dic[var_cmd])
            varo = cmd_dic[var_cmd][indexo]
            dL = dL_inbin[indexo]
            m1_det = m1_det_inbin[indexo]
            m2_det = m2_det_inbin[indexo]
            #mtot_det = m1_det + m2_det
            chi_eff = chi_eff_inbin[indexo]
            
            if self.dmid_fun in self.spin_functions:
                dmid_values = self.dmid(m1_det, m2_det, chi_eff, self.dmid_params)
            else: 
                dmid_values = self.dmid(m1_det, m2_det, self.dmid_params)

            #self.apply_dmid_mtotal_max(dmid_values, mtot_det)
            
            emax_params, gamma, delta, alpha = self.get_shape_params()
            
            if self.emax_fun is not None:
                emax = self.emax(m1_det, m2_det, emax_params)
            else:
                emax = np.copy(emax_params)
            
            pdet = self.sigmoid(dL, dmid_values, emax, gamma, delta, alpha)
            cmd = np.cumsum(pdet)
            
            # Found injections
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
                def chieff_corr(x, c1):
                    return np.exp(c1 * x)

                mid_values.append( (data_inbin[0] + data_inbin[-1]) / 2 )
            
                nbins = np.linspace(-1, 1, 11)
                x = np.linspace(-1, 1, 1000)
                
                plt.figure()
                nchi, _, _ = plt.hist(varo, bins = nbins, density = True, weights = pdet, histtype = 'step', log = True)
                nfound, _, _, = plt.hist(found_inj_inbin_sorted, bins = nbins, density = True, histtype = 'step', log = True)
                plt.plot(nbins[:-1], nfound/nchi, '-')
                
                # Fit
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
            
            if ks: 
                pdet_weighted = cmd / np.sum(pdet)
                def cdf(x):
                    idxs = np.searchsorted(varo, x, side='right')
                    return np.where(idxs == 0, 0, pdet_weighted[idxs - 1])
                    
                stat, pvalue = kstest(found_inj_inbin_sorted, cdf)
                print(f'{var_cmd} KStest in {i} bin: statistic = %s, pvalue = %s' % (stat, pvalue))
        
        print('')   
        
        if var_cmd == 'chi_eff':
            name_opt = path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/chi_eff_opt_param'
            np.savetxt(name_opt, chi_eff_params, header = '0, 1, 2, 3, 4', fmt='%s')
            
            name_mid = path + f'/{emax_dic[self.emax_fun]}/{var_binned}_bins/{var_cmd}_cmd/mid_values'
            np.savetxt(name_mid, mid_values, header = '0, 1, 2, 3, 4', fmt='%s')
        return
    
    def sensitive_volume(self, run_fit, m1, m2, chieff=0., sources='bbh', rescale_o3=True):
        '''
        Sensitive volume for a merger with given masses (m1 and m2), computed from the fit to whichever observed run we want.
        Integrated within the total range of redshift available in the injection's dataset.

        Parameters
        ----------
        run_fit : str. Observing run from which we want to use its fit. Must be 'o1', 'o2' or 'o3'.
        m1 : float. Mass 1 (source)
        m2 : float. Mass 2 (source)
        chieff : float. Effective spin. The default is 0, If you use a fit that includes a dependence on chieff in the dmid function 
                (it has to be on the list of spin functions), it will use chieff. if not, it won't be used for anything.
        rescale_o3 : True or False, optional. The default is True. If True, we use the rescaled fit for o1 and o2. If False, the direct fit.

        Returns
        -------
        pdet * Vtot : float. Sensitive volume
        '''
        self.get_opt_params(run_fit, 'all', rescale_o3) if run_fit == 'o4' else self.get_opt_params(run_fit, sources, rescale_o3)
        source_interp_dL_pdf = 'all' if run_fit == 'o4' else 'bbh'
        self.load_inj_set(run_fit, source_interp_dL_pdf)

        # We compute some values of dl for some z to make later an interpolator
        if self.zinterp_VT is None:
            fun_A = lambda t : np.sqrt(self.cosmo.Om0 * (1 + t)**3 + 1 - self.cosmo.Om0)
            quad_fun_A = lambda t: 1/fun_A(t)

            z = np.linspace(0.002,1.9, 30)
            z0 = np.insert(z, 0, 0, axis = 0)
            dL = np.array([(const.c.value*1e-3 / self.cosmo.H0.value) * (1 + i) * integrate.quad(quad_fun_A, 0, i)[0] for i in z0])

            self.zinterp_VT = interpolate.interp1d(dL, z0)

        m1_det = lambda dL_int : m1 * (1 + self.zinterp_VT(dL_int))
        m2_det = lambda dL_int : m2 * (1 + self.zinterp_VT(dL_int))

        if self.dmid_fun in self.spin_functions:
            dmid = lambda dL_int : self.dmid(m1_det(dL_int), m2_det(dL_int), chieff, self.dmid_params)
        else: 
            dmid = lambda dL_int : self.dmid(m1_det(dL_int), m2_det(dL_int), self.dmid_params)

        emax_params, gamma, delta, alpha = self.get_shape_params()

        if self.emax_fun is not None:
            emax = lambda dL_int : self.emax(m1_det(dL_int), m2_det(dL_int), emax_params)
            quad_fun = lambda dL_int : self.sigmoid(dL_int, dmid(dL_int), emax(dL_int), gamma, delta, alpha) \
                       * self.sets[source_interp_dL_pdf]['interp_dL_pdf'](dL_int)
        else:
            emax = np.copy(emax_params)
            quad_fun = lambda dL_int : self.sigmoid(dL_int, dmid(dL_int), emax, gamma, delta, alpha) \
                       * self.sets[source_interp_dL_pdf]['interp_dL_pdf'](dL_int)

        pdet = integrate.quad(quad_fun, 0, self.sets[source_interp_dL_pdf]['dLmax'])[0]

        if self.Vtot is None:
            # NB the factor of 1/(1+z) for time dilation in the signal rate 
            vquad = lambda z_int : 4 * np.pi * self.cosmo.differential_comoving_volume(z_int).value / (1 + z_int)
            self.Vtot = integrate.quad(vquad, 0, self.sets[source_interp_dL_pdf]['zmax'])[0]

        return pdet * self.Vtot
    
    def total_sensitive_volume(self, m1, m2, chieff=0., sources='bbh', rescale_o3=True):

        '''
        Total sensitive volume computed with the fractions of o1, o2, o3 and o4a observing times and the o1, o2 rescaled fit
        Vtot = V1 * t1_frac + V2 * t2_frac + V3 * t3_frac + V4 * t4_frac

        Parameters
        ----------
        m1 : float. Mass 1 
        m2 : float. Mass 2
        chieff : float. Effective spin. The default is 0, If you use a fit that includes a dependence on chieff in the dmid function 
                (it has to be on the list of spin functions), it will use chieff. if not, it won't be used for anything.
        rescale_o3 : True or False, optional. The default is True. If True, we use the rescaled fit for o1 and o2. If False, the direct fit.

        Returns
        -------
        Vtot : float. Total sensitive volume
        '''
        Vtot = 0
        for run in self.runs:
            Vi = self.sensitive_volume(run, m1, m2, chieff, sources, rescale_o3)
            Vtot += Vi * self.obs_time[run]
        
        return Vtot
    
    def find_dmid_cte_found_inj(self, run_dataset, run_fit='o3'):
        '''
        Method for finding the rescaled factor (d0) for whatever injection set we want (usually o1 or o2) using the run_fit that we want (usually o3)

        Parameters
        ----------
        run_dataset : str. Injection set that we want to obtain the rescaled fit for
        run_fit : str. Observing run that we want to use its fit to rescale the others. The default is 'o3'.

        Returns
        -------
        float. Rescaled constant d0
        '''
        assert run_dataset =='o1' or run_dataset == 'o2', "Argument (run_dataset) must be 'o1' or 'o2'."

        try:
            self.load_all_inj_sets(run_dataset, 'bbh')
            Nfound = self.found_any.sum()
    
            self.get_opt_params(run_fit, 'bbh', rescale_o3 = False) #we always want 'o3' fit
    
            dmid_params = np.copy(self.dmid_params)
            dmid_params[0] = 1.
    
            if self.dmid_fun in self.spin_functions:
                dmid_values = self.dmid(self.m1_det, self.m2_det, self.chi_eff, dmid_params)
            else: 
                dmid_values = self.dmid(self.m1_det, self.m2_det, dmid_params)
            #self.apply_dmid_mtotal_max(dmid_values, mtot_det)

            emax_params, gamma, delta, alpha = self.get_shape_params()

            if self.emax_fun is not None:
                emax = self.emax(self.m1_det, self.m2_det, emax_params)  
            else:
                emax = np.copy(emax_params)

            def find_root(x):
                new_dmid_values = x * dmid_values
                #self.apply_dmid_mtotal_max(new_dmid_values, mtot_det)
                
                frac = self.dL / ( new_dmid_values)
                denom = 1. + frac ** alpha * \
                        np.exp(gamma* (frac - 1.) + delta * (frac**2 - 1.))
      
                return  np.nansum(emax / denom)  -  Nfound

            d0 = fsolve(find_root, [50])  # rescaled Dmid cte (it has units of Dmid !!)
            self.load_all_inj_sets('o3', 'bbh')
            print('Current injection file loaded is o3, to be taken into account.')
            return d0

        except:
            d0 = {'o1' : np.loadtxt(f'{os.path.dirname(__file__)}/d0.dat')[0], 'o2' : np.loadtxt(f'{os.path.dirname(__file__)}/d0.dat')[1]}
            print('Something went wrong computing the cte, using the d0.dat file that already exists instead.')
            return d0[run_dataset]

    def predicted_events(self, run_fit = 'o3', run_dataset='o3'):
        '''
        find the predicted found events in each run using rescaled fits from one injection set

        Parameters
        ----------
        run_fit : str. Observing run that we want to use its fit to rescale the others
        run_dataset : str. Injection set. Must be 'o1' , 'o2' or 'o3'. The default is 'o3'.

        Returns
        -------
        float. predicted found events
        '''
        self.load_all_inj_sets(run_dataset, 'bbh')
        Nfound = self.found_any.sum()

        self.get_opt_params(run_fit, 'bbh')

        dmid_params = np.copy(self.dmid_params)
        dmid_params[0] = 1.

        if self.dmid_fun in self.spin_functions:
            dmid_values = self.dmid(self.m1_det, self.m2_det, self.chi_eff, dmid_params)
        else: 
            dmid_values = self.dmid(self.m1_det, self.m2_det, dmid_params)
        #self.apply_dmid_mtotal_max(dmid_values, mtot_det)

        emax_params, gamma, delta, alpha = self.get_shape_params()

        if self.emax_fun is not None:
            emax = self.emax(self.m1_det, self.m2_det, emax_params)
        else:
            emax = np.copy(emax_params)

        def manual_found_inj(x):
            frac = self.dL / ( x * dmid_values)
            denom = 1. + frac ** alpha * \
                    np.exp(gamma* (frac - 1.) + delta * (frac**2 - 1.))
            return np.sum( emax / denom )

        o1_inj = manual_found_inj(self.find_dmid_cte_found_inj(run_dataset ='o1', run_fit = run_fit))
        o2_inj = manual_found_inj(self.find_dmid_cte_found_inj(run_dataset ='o2', run_fit = run_fit))
        o3_inj = Nfound

        frac1 = o1_inj / o3_inj
        frac2 = o2_inj / o3_inj

        pred_rates = { 'o1' : frac1 * self.det_rates['o3'], 'o2' : frac2 * self.det_rates['o3'] }
        pred_nev = { 'o1' : pred_rates['o1'] * self.obs_time['o1'], 'o2' : pred_rates['o2'] * self.obs_time['o2'] }

        return pred_nev
    
    def run_pdet(self, dL, m1_det, m2_det, chieff = 0., run='o3', sources='bbh', rescale_o3 = True):
        """
        probability of detection for some given masses and distance

        Parameters
        ----------
        dL : float. luminosity distance [Mpc]
        m1_det : float. Mass 1 in the detector's frame masses
        m2_det : float. Mass 2 in the detector's frame masses
        run : str. observing run from which we want the fit. Must be 'o1', 'o2' or 'o3'
        rescale_o3 : True or False, optional. The default is True. If True, we iuse the rescaled fit for o1 and o2. If False, the direct fit.

        Returns
        -------
        pdet_i : probability of detection of a merger with masses m1, m2 and at a distance dL.
        """
        self.get_opt_params(run, sources, rescale_o3) 

        if self.dmid_fun in self.spin_functions:
            dmid_values = self.dmid(m1_det, m2_det, chieff, self.dmid_params)
        else: 
            dmid_values = self.dmid(m1_det, m2_det, self.dmid_params)

        #self.apply_dmid_mtotal_max(np.array(dmid_values), mtot_det)

        emax_params, gamma, delta, alpha = self.get_shape_params()

        if self.emax_fun is not None:
            emax = self.emax(m1_det, m2_det, emax_params)
        else:
            emax = np.copy(emax_params)

        pdet_i = self.sigmoid(dL, dmid_values, emax , gamma , delta , alpha)

        return pdet_i
    
    def total_pdet(self, dL, m1_det, m2_det, chieff = 0., sources='bbh', o4_run=False,rescale_o3 = True):
        '''
        total prob of detection, a combination of the prob of detection with o1, o2 and o3 proportions

        Parameters
        ----------
        dL : float. luminosity distance [Mpc]
        m1_det : float. Mass 1 in the detector's frame masses
        m2_det : float. Mass 2 in the detector's frame masses
        run : str. observing run from which we want the fit. Must be 'o1', 'o2' or 'o3'
        rescale_o3 : True or False, optional. The default is True. If True, we iuse the rescaled fit for o1 and o2. If False, the direct fit.

        Returns
        -------
        pdet : total prob of detection
        '''        
        pdet = np.zeros(len(np.atleast_1d(dL)))

        for run, prop in zip(self.runs, self.prop_obs_time):
            pdet_i = self.run_pdet(dL, m1_det, m2_det, chieff, run, sources, rescale_o3) 
            pdet += pdet_i * prop

        return pdet

    def bootstrap_resampling(self, n_boots, run_dataset, sources):
        if isinstance(sources, str):
            each_source = [source.strip() for source in sources.split(',')] 

        sources_folder = "_".join(sorted(each_source)) 

        self.make_folders(run_dataset, sources_folder)

        [self.load_inj_set(run_dataset, source) for source in each_source]

        all_params = np.zeros([1, len(np.atleast_1d(self.dmid_params)) + len(np.atleast_1d(self.shape_params))])

        for i in range(n_boots):
            for source in each_source:
                self.load_inj_set(run_dataset, source)

                total = len(self.sets[source]['dL'])
                boots = np.random.choice(np.arange(total), total, replace=True)

                self.sets[source]['m1'] = self.sets[source]['m1'][boots]
                self.sets[source]['m2 ']= self.sets[source]['m2'][boots]
                self.sets[source]['z'] = self.sets[source]['z'][boots]
                self.sets[source]['dL'] = self.sets[source]['dL'][boots]
                
                self.sets[source]['Mtot'] = self.sets[source]['Mtot'][boots]
                self.sets[source]['Mtot_det'] = self.sets[source]['Mtot_det'][boots]
                self.sets[source]['Mc'] = self.sets[source]['Mc'][boots]
                self.sets[source]['Mc_det'] = self.sets[source]['Mc_det'][boots]
                self.sets[source]['eta'] = self.sets[source]['eta'][boots]
                self.sets[source]['q'] = self.sets[source]['q'][boots]  
                self.sets[source]['m_pdf'] = self.sets[source]['m_pdf'][boots]
                self.sets[source]['z_pdf'] = self.sets[source]['z_pdf'][boots]
                self.sets[source]['dL_pdf'] = self.sets[source]['dL_pdf'][boots]

                self.sets[source]['s1x'] = self.sets[source]['s1x'][boots]
                self.sets[source]['s1y'] = self.sets[source]['s1y'][boots]
                self.sets[source]['s1z'] = self.sets[source]['s1z'][boots]
                self.sets[source]['s2x'] = self.sets[source]['s2x'][boots]
                self.sets[source]['s2y'] = self.sets[source]['s2y'][boots]
                self.sets[source]['s2z'] = self.sets[source]['s2z'][boots]
                self.sets[source]['s1z_pdf'] = self.sets[source]['s1z_pdf'][boots]
                self.sets[source]['s2z_pdf'] = self.sets[source]['s2z_pdf'][boots]

                self.sets[source]['a1'] = self.sets[source]['a1'][boots]
                self.sets[source]['a2'] = self.sets[source]['a2'][boots]
                self.sets[source]['chi_eff'] = self.sets[source]['chi_eff'][boots]

                self.sets[source]['found_any'] = self.sets[source]['found_any'][boots]

            opt_params_shape, opt_params_dmid = self.joint_MLE(run_dataset, sources, bootstrap=True)
            print(i, 'n boots', opt_params_shape, opt_params_dmid)
            all_params = np.vstack([all_params, np.hstack((opt_params_shape, opt_params_dmid))])

        header = f'{self.shape_params_names[self.emax_fun]}, {self.dmid_params_names[self.dmid_fun]}'
        path = f'{run_dataset}/{sources_folder}/' + self.path
        name_file = path + f'/{n_boots}_boots_opt_params.dat'
        np.savetxt(name_file, all_params, header=header, fmt='%s')

        return
    
    def evaluate(self, dL, m1_det, m2_det, chi_eff, dmid_params, emax_params, gamma, delta):

        dmid_values = self.dmid(m1_det, m2_det, chi_eff, dmid_params)
        emax_values = self.emax(m1_det, m2_det, emax_params)

        pdet = self.sigmoid(dL, dmid_values, emax_values, gamma, delta)

        return pdet
