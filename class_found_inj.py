# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:14:27 2023

@author: Ana
"""

import numpy as np
import h5py
from scipy import interpolate
from scipy import integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt

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
        
        self.Ntotal = file.attrs['total_generated'] 
        """
        Total number of generated injections
        """
        
        self.m1 = file["injections/mass1_source"][:]
        self.m2 = file["injections/mass2_source"][:]
        """
        Mass 1 and mass 2 values in the source frame in solar units
        """
        
        self.z = file["injections/redshift"][:]
        self.dL = file["injections/distance"][:]
        """
        Redshift and luminosity distance [Mpc] values 
        """
        
        self.m_pdf = file["injections/mass1_source_mass2_source_sampling_pdf"][:]
        """
        Joint mass sampling pdf (probability density function) values, p(m1,m2) 
        """
        
        self.z_pdf = file["injections/redshift_sampling_pdf"][:]
        """
        Redshift sampling pdf values, p(z), corresponding to a redshift 
        defined by a flat Lambda-Cold Dark Matter cosmology
        """
        
        H0 = 67.9 #km/sMpc
        c = 3e5 #km/s
        omega_m = 0.3065
        A = np.sqrt(omega_m * (1 + self.z)**3 + 1 - omega_m)
        dL_dif = (c * (1 + self.z) / H0) * (1/A)
        
        self.dL_pdf = self.z_pdf / dL_dif
        """
        Luminosity distance sampling pdf values, p(dL), computed for a 
        flat Lambda-Cold Dark Matter cosmology from the z_pdf values
        """
        
        self.far_pbbh = file["injections/far_pycbc_bbh"][:]
        """
        False alarm rate statistics from the pycbc_bbh search pipeline
        """
        
        self.far_gstlal = file["injections/far_gstlal"][:]
        """
        False alarm rate statistics from the gstlal search pipeline
        """
        
        self.far_mbta = file["injections/far_mbta"][:]
        """
        False alarm rate statistics from the mbta search pipeline
        """
        
        self.far_pfull = file["injections/far_pycbc_hyperbank"][:]
        """
        False alarm rate statistics from the pfull search pipeline
        """
        
        found_pbbh = self.far_pbbh <= thr
        found_gstlal = self.far_gstlal <= thr
        found_mbta = self.far_mbta <= thr
        found_pfull = self.far_pfull <= thr
        self.found_any = found_pbbh | found_gstlal | found_mbta | found_pfull
        
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
    
        #we add 0 value
        inter_dL = np.insert(try_dL, 0, 0, axis=0)
        inter_dLpdf = np.insert(try_dLpdf, 0, 0, axis=0)
        self.interp_dL = interpolate.interp1d(inter_dL, inter_dLpdf)
        
        try_z = self.z[index]
        inter_z = np.insert(try_z, 0, 0, axis=0)
        
        #add a value for self.zmax
        new_dL = np.insert(inter_dL, -1, self.dLmax, axis=0)
        new_z = np.insert(inter_z, -1, self.zmax, axis=0)
        
        self.interp_z = interpolate.interp1d(new_dL, new_z)
        
        print('finished initializing')
    
    #now we define methods for this class
    
    def Dmid_mchirp(self, m1, m2, z, cte):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame (our first guess)

        Parameters
        ----------
        m1 : mass1 
        m2: mass2
        z : redshift
        cte : parameter that we will be optimizing
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        m1_det = m1 * (1 + z) 
        m2_det = m2 * (1 + z)
        
        Mc = (m1_det * m2_det)**(3/5) / (m1_det + m2_det)**(1/5)
        
        return cte * Mc**(5/6)
    
    def Dmid_inter(self, m1, m2, dL, cte):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame (our first guess)
        
        We are writing it in terms of dL, and then compute the redshift by interpolating from dL

        Parameters
        ----------
        m1 : mass1 
        m2: mass2
        dL : luminosity distance
        cte : parameter that we will be optimizing

        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        z = self.interp_z(dL)
        return self.Dmid_mchirp(m1,m2,z,cte)
    
    
    def sigmoid(self, dL, dLmid, gamma = -0.18395, delta = 0.1146989, alpha = 2.05, emax = 0.967):
        """
        Sigmoid function used to estime the probability of detection of bbh events

        Parameters
        ----------
        dL : 1D array of the luminosity distance.
        dLmid : dL at which Pdet = 0.5.
        gamma : parameter controling the shape of the curve. The default is -0.18395.
        delta : parameter controling the shape of the curve. The default is 0.1146989.
        alpha : parameter controling the shape of the curve. The default is 2.05.
        emax : parameter controling the shape of the curve. The default is 0.967.

        Returns
        -------
        array of detection probability.

        """
        frac = dL / dLmid
        denom = 1. + frac ** alpha * \
            np.exp(gamma * (frac - 1.) + delta * (frac**2 - 1.))
        return emax / denom
    
 
    def fun_m_pdf(self, m1, m2):
        """
        Function for the mass pdf aka p(m1,m2)

        Returns
        -------
        A continuous function which describes p(m1,m2)

        """
        if m2 > m1:
            return 0
        
        mmin = self.mmin ; mmax = self.mmax
        alpha = self.pow_m1 ; beta = self.pow_m2
        
        m1_norm = (1. + alpha) / (mmax ** (1. + alpha) - mmin ** (1. + alpha))
        m2_norm = (1. + beta) / (m1 ** (1. + beta) - mmin ** (1. + beta))
        
        return m1**alpha * m2**beta * m1_norm * m2_norm
    
    
    def Nexp(self, params):
        """
        Expected number of found injections, computed as a 
        triple integral of p(dL)*p(m1,m2)*sigmoid( dL, Dmid(m1,m2, dL, cte) )*Ntotal
        
        Note that in order to make the integral, we had to use an interpolation for the redshift, just so we can
        write Dmid as a function of dL and then integrate over d(dL)

        Parameters
        ----------
        params : parameters of the Dmid function

        Returns
        -------
        Expected number of found injections

        """
        
        quad_fun = lambda m1, m2, dL_int: self.Ntotal * self.fun_m_pdf(m1, m2) *  \
            self.interp_dL(dL_int) * self.sigmoid(dL_int, self.Dmid_inter(m1, m2, dL_int, params)) 
        
        lim_m2 = lambda m1: [self.mmin, m1]
        return integrate.nquad( quad_fun, [[self.mmin, self.mmax], lim_m2, [0, self.dLmax]])[0]
        
        
    def Lambda(self, params):
        """
        Number density at found injections

        Parameters
        ----------
        params : parameters of the Dmid function

        Returns
        -------
        Number density at found injections aka lambda(D,m1,m2)

        """
        dL = self.dL[self.found_any]
        dL_pdf = self.dL_pdf[self.found_any]
        m_pdf = self.m_pdf[self.found_any]
        z = self.z[self.found_any]
        m1 = self.m1[self.found_any]
        m2 = self.m2[self.found_any]
        
        # print(self.sigmoid(dL, self.Dmid_mchirp(m1, m2, z, cte)))
        return self.sigmoid(dL, self.Dmid_mchirp(m1, m2, z, params)) * m_pdf * dL_pdf * self.Ntotal
         
    
    def logL(self, in_param):
        """
        log likelihood of the expected density of found injections

        Parameters
        ----------
        in_param : parameters that will be optimized. It should be cte of self.Dmid(cte).
        We will use exp(cte) in the minimization, so we have to remember then to take log(opt_cte).

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        cte = np.exp(in_param[0]) 
        lnL = -self.Nexp(cte) + np.sum(np.log(self.Lambda(cte)))
        print(lnL)
        # print(-self.Nexp(cte))
        # print(np.sum(np.log(self.Lambda(cte))))
        return lnL
        
    
    def MLE(self, cte_guess, methods):
        """
        minimization of -logL 

        Parameters
        ----------
        cte_guess : initial guess value for cte of Dmid
        methods : scipy method used to minimize -logL

        Returns
        -------
        cte_res : optimized value for cte of Dmid.
        -min_likelihood : maximum log likelihood. 

        """
        res = opt.minimize(fun=lambda in_param: -self.logL(in_param), 
                           x0=np.array([np.log(cte_guess)]), 
                           args=(), 
                           method=methods)
        
        cte_res = np.exp(res.x) 
        min_likelihood = res.fun                
        return cte_res, -min_likelihood

file = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')

data = Found_injections(file)

cte_guess = 70

cte_opt, maxL = data.MLE(cte_guess, methods='Nelder-Mead')
    
results = np.column_stack((cte_opt, maxL))
header = "cte_opt, maxL"
np.savetxt('dmid(m)_results.dat', results, header = header)

# nbin1 = 14
# nbin2 = 14

# m1_bin = np.round(np.logspace(np.log10(data.mmin), np.log10(data.mmax), nbin1+1), 1)
# m2_bin = np.round(np.logspace(np.log10(data.mmin), np.log10(data.mmax), nbin2+1), 1)

# mid_1 = (m1_bin[:-1]+ m1_bin[1:])/2
# mid_2 = (m2_bin[:-1]+ m2_bin[1:])/2

# Mc = np.array ([[(mid_1[i] * mid_2[j])**(3/5) / (mid_1[i] + mid_2[j])**(1/5) for j in range(len(mid_2))] for i in range(len(mid_1))] )

# k=42
# dmid = np.loadtxt(f'dL_joint_fit_results_emax/dLmid/dLmid_{k}.dat')
# toplot = np.nonzero(dmid)

# Mc_plot = Mc[toplot]
# dmid_plot = dmid[toplot]

# plt.figure()
# plt.loglog(Mc_plot.flatten(), dmid_plot.flatten(), '.')
# plt.xlabel('Mc(m1,m2)')
# plt.ylabel('dL_mid')
# plt.grid(True, which='both')
# plt.loglog()

# plt.plot(Mc_plot.flatten(), 70*(Mc_plot.flatten())**(5/6), 'r-', label='cte = 70')
# plt.legend()

