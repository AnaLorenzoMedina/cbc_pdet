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
from scipy.stats import uniform

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
        
        #total mass (m1+m2)
        self.Mtot = self.m1 + self.m2
        
        #eta aka symmetric mass ratio
        mu = (self.m1 * self.m2) / (self.m1 + self.m2)
        self.eta = mu / self.Mtot
        
        #False alarm rate statistics from each pipeline
        self.far_pbbh = file["injections/far_pycbc_bbh"][:]
        self.far_gstlal = file["injections/far_gstlal"][:]
        self.far_mbta = file["injections/far_mbta"][:]
        self.far_pfull = file["injections/far_pycbc_hyperbank"][:]
        
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
    
    def sigmoid(self, dL, dLmid, gamma = -0.02726, delta = 0.13166, emax = 0.79928, alpha = 2.05):
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
    
    def Dmid_mchirp_expansion(self, m1, m2, z, params):
        """
        Dmid values (distance where Pdet = 0.5) as a function of the masses 
        in the detector frame (our first guess)

        Parameters
        ----------
        m1 : mass1 
        m2: mass2
        z : redshift
        params : parameters that we will be optimizing
        
        Returns
        -------
        Dmid(m1,m2) in the detector's frame

        """
        cte , a_20, a_01, a_22 = params[0], params[1], params[2], params[3]
        
        m1_det = m1 * (1 + z) 
        m2_det = m2 * (1 + z)
        M = m1_det + m2_det
        eta = m1*m2 / (m1+m2)**2
        
        Mc = (m1_det * m2_det)**(3/5) / (m1_det + m2_det)**(1/5)
        
        pol = cte *(1+ a_20 * M**2 / 2 + a_01 * (1 - 4*eta) + a_22 * M**2 * (1 - 4*eta)**2 / 4)
        
        return pol * Mc**(5/6)
    
 
    def fun_m_pdf(self, m1, m2):
        """
        Function for the mass pdf aka p(m1,m2)

        Returns
        -------
        A continuous function which describes p(m1,m2)

        """
        # if m2 > m1:
        #   return 0
        
        mmin = self.mmin ; mmax = self.mmax
        alpha = self.pow_m1 ; beta = self.pow_m2
        
        m1_norm = (1. + alpha) / (mmax ** (1. + alpha) - mmin ** (1. + alpha))
        m2_norm = (1. + beta) / (m1 ** (1. + beta) - mmin ** (1. + beta))
        
        return m1**alpha * m2**beta * m1_norm * m2_norm
    
    
    def Nexp(self, dmid_fun, params, shape_params = None):
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
        # quad_fun = lambda m1, m2, dL_int: self.Ntotal * self.fun_m_pdf(m1, m2) *  \
        #     self.interp_dL(dL_int) * self.sigmoid(dL_int, self.Dmid_inter(m1, m2, dL_int, params)) 
        
        # lim_m2 = lambda m1: [self.mmin, m1]
        # return integrate.nquad( quad_fun, [[self.mmin, self.mmax], lim_m2, [0, self.dLmax]], full_output=True)[0]
        
        # we try this sencond method for Nexp
        
        DMID = getattr(Found_injections, dmid_fun)
        
        if shape_params is not None:
            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            Nexp = np.sum(self.sigmoid(self.dL, DMID(self, self.m1, self.m2, self.z, params), gamma, delta, emax))
            
        else:
            Nexp = np.sum(self.sigmoid(self.dL, DMID(self, self.m1, self.m2, self.z, params)))
            
        print(Nexp)
        return Nexp
        
    def Lambda(self, dmid_fun, params, shape_params = None):
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
        
        DMID = getattr(Found_injections, dmid_fun)
        
        if shape_params is not None:
            gamma, delta, emax = shape_params[0], shape_params[1], shape_params[2]
            Lambda = self.sigmoid(dL, DMID(self, m1, m2, z, params), gamma, delta, emax) * m_pdf * dL_pdf * self.Ntotal
        
        else:
            Lambda = self.sigmoid(dL, DMID(self, m1, m2, z, params)) * m_pdf * dL_pdf * self.Ntotal
        
        return Lambda
    
    def logL(self, dmid_fun, dmid_params, shape_params = None):
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
        params = np.exp(dmid_params) 
        
        if shape_params is not None:
            shape_params = [shape_params[0], shape_params[1], shape_params[2]]
            
        lnL = -self.Nexp(dmid_fun, params, shape_params) + np.sum(np.log(self.Lambda(dmid_fun, params, shape_params)))
        print(lnL)
        return lnL
        
    
    def MLE_dmid(self, dmid_fun, params_guess, methods):
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
        res = opt.minimize(fun=lambda in_param: -self.logL(dmid_fun, in_param), 
                           x0=np.array([np.log(params_guess)]), 
                           args=(), 
                           method=methods)
        
        params_res = np.exp(res.x) 
        min_likelihood = res.fun                
        return params_res, -min_likelihood
    
    def MLE_shape(self, dmid_fun, params_dmid, params_shape_guess, methods):
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
        gamma_guess, delta_guess, emax_guess = params_shape_guess
        
        res = opt.minimize(fun=lambda in_param: -self.logL(dmid_fun, params_dmid, in_param), 
                           x0=np.array([gamma_guess, delta_guess, emax_guess]), 
                           args=(), 
                           method=methods)
        
        gamma, delta, emax = res.x 
        min_likelihood = res.fun                
        return gamma, delta, emax, -min_likelihood
    
    def cumulative_dist(self, dmid_fun, params, var = 'dL'):
        #params = self.MLE(cte_guess, a20_guess, a01_guess, methods='Nelder-Mead')[:-1]
        #params = 79.70666689915684
        
        DMID = getattr(Found_injections, dmid_fun)
        dic = {'dL': self.dL, 'Mc': self.Mc, 'Mtot': self.Mtot, 'eta': self.eta}
           
        #cumulative distribution over the desired variable
        indexo = np.argsort(dic[var])
        varo = dic[var][indexo]
        dLo = self.dL[indexo]
        m1o = self.m1[indexo]
        m2o = self.m2[indexo]
        zo = self.z[indexo]
        cmd = np.cumsum(self.sigmoid(dLo, DMID(self, m1o, m2o, zo, params)))
        
        var_found = dic[var][self.found_any]
        indexo_found = np.argsort(var_found)
        var_foundo = var_found[indexo_found]
        
        real_found_inj = np.arange(len(var_foundo))+1
        
        cdf = lambda var_i : np.cumsum(self.sigmoid(self.dL[var_i], DMID(self, self.m1[var_i], self.m2[var_i], self.z[var_i], params)))
        
        plt.figure()
        plt.plot(varo, cmd, '.', markersize=2, label='model')
        plt.plot(var_foundo, real_found_inj, '.', markersize=2, label='found injections')
        #plt.plot(varo, cdf(indexo))
        plt.xlabel(f'${var}^*$')
        plt.ylabel('Cumulative found injections')
        plt.legend(loc='best')
        name=f'{dmid_fun}/{var}_cumulative.png'
        plt.savefig(name, format='png')
        
        
        # if var=='dL':
        #     plt.figure()
        #     plt.plot(var_foundo, real_found_inj/len(var_foundo))
        #     plt.plot(varo, cdf(indexo)/len(varo))
            
        return kstest(real_found_inj, cdf(indexo))

plt.close('all')

file = h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')


data = Found_injections(file)

# function for dmid we wanna use
dmid_fun = 'Dmid_mchirp_expansion'

try:
    os.mkdir(f'{dmid_fun}')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.mkdir(f'{dmid_fun}/shape')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

cte_guess = 70
a20_guess= 0.05
a01_guess= 0.05
a22_guess = 0.5

gamma_guess = -0.1
delta_guess = 0.1
emax_guess = 1

shape_guess = [gamma_guess, delta_guess, emax_guess]


params_guess = {'Dmid_mchirp': cte_guess, 'Dmid_mchirp_expansion': [cte_guess, a20_guess, a01_guess, a22_guess]}
params_names = {'Dmid_mchirp': 'cte', 'Dmid_mchirp_expansion': ['cte', 'a20', 'a01', 'a22']}

# params_opt, maxL = data.MLE_dmid(dmid_fun, params_guess[dmid_fun], methods='Nelder-Mead')
# print(params_opt)

# #%%
# results = np.hstack((params_opt, maxL))
# header = f'{params_names[dmid_fun]} , maxL'
# np.savetxt(f'{dmid_fun}/dmid(m)_results_2method.dat', [results], header = header, fmt='%s')

params_dmid = np.loadtxt(f'{dmid_fun}/dmid(m)_results_2method.dat')[:-1]

gamma_opt, delta_opt, emax_opt, maxL = data.MLE_shape(dmid_fun, params_dmid, shape_guess, methods='Nelder-Mead')
print(gamma_opt, delta_opt, emax_opt)

results = np.column_stack((gamma_opt, delta_opt, emax_opt, maxL))
header = 'gamma_opt, delta_opt, emax_opt, maxL'
np.savetxt(f'{dmid_fun}/shape/opt_shape_params.dat', results, header = header, fmt='%s')

# nexp = data.Nexp_MC(dmid_fun, 1000000, params_dmid)
# print(nexp)

# stat_Mtot, pvalue_Mtot = data.cumulative_dist(dmid_fun, params_dmid, 'Mtot')
# stat_Mc, pvalue_Mc = data.cumulative_dist(dmid_fun, params_dmid, 'Mc')
# stat_eta, pvalue_eta = data.cumulative_dist(dmid_fun, params_dmid, 'eta') 
# stat_dL, pvalue_dL = data.cumulative_dist(dmid_fun, params_dmid, 'dL')

# print('dL KStest: statistic = %s , pvalue = %s' %(stat_dL, pvalue_dL))
# print('Mtot KStest: statistic = %s , pvalue = %s' %(stat_Mtot, pvalue_Mtot))
# print('Mc KStest: statistic = %s , pvalue = %s' %(stat_Mc, pvalue_Mc))
# print('eta KStest: statistic = %s , pvalue = %s' %(stat_eta, pvalue_eta))

# #%%
# nbin1 = 14
# nbin2 = 14

# m1_bin = np.round(np.logspace(np.log10(data.mmin), np.log10(data.mmax), nbin1+1), 1)
# m2_bin = np.round(np.logspace(np.log10(data.mmin), np.log10(data.mmax), nbin2+1), 1)

# mid_1 = (m1_bin[:-1]+ m1_bin[1:])/2
# mid_2 = (m2_bin[:-1]+ m2_bin[1:])/2

# Mc = np.array ([[(mid_1[i] * mid_2[j])**(3/5) / (mid_1[i] + mid_2[j])**(1/5) for j in range(len(mid_2))] for i in range(len(mid_1))] )
# Mtot = np.array ([[(mid_1[i] + mid_2[j]) for j in range(len(mid_2))] for i in range(len(mid_1))] )
# q = np.array ([[(mid_2[j] / mid_1[i]) for j in range(len(mid_2))] for i in range(len(mid_1))] )
# mu = np.array ([[(mid_1[i] * mid_2[j]) / (mid_1[i] + mid_2[j]) for j in range(len(mid_2))] for i in range(len(mid_1))] ) 

# k=42
# dmid = np.loadtxt(f'dL_joint_fit_results_emax/dLmid/dLmid_{k}.dat')
# toplot = np.nonzero(dmid)

# Mc_plot = Mc[toplot]
# dmid_plot = dmid[toplot]
# Mtot_plot = Mtot[toplot]
# q_plot = q[toplot]
# mu_plot = mu[toplot]
# eta_plot = mu_plot / Mtot_plot

# plt.figure()
# plt.scatter(Mtot_plot, dmid_plot/(Mc_plot)**(5/6), c=eta_plot)
# plt.colorbar(label=r'$\eta$')
# plt.xlabel('Mtot = m1 + m2')
# plt.ylabel('dL_mid / Mc^(5/6)')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax/Mtot.png"
# plt.savefig(name, format='png', dpi=1000)


# #plt.plot(Mc_plot.flatten(), 70*(Mc_plot.flatten())**(5/6), 'r-', label='cte = 70')
# #plt.legend()


# plt.figure()
# plt.loglog(q_plot, dmid_plot/(Mc_plot)**(5/6), '.')
# plt.xlabel('q = m2 / m1')
# plt.ylabel('dL_mid / Mc^(5/6)')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax/q.png"
# plt.savefig(name, format='png', dpi=1000)

# plt.figure()
# plt.scatter(eta_plot, dmid_plot/(Mc_plot)**(5/6), c=Mtot_plot)
# plt.colorbar(label='Mtot')
# plt.xlabel(r'$\eta = \mu / Mtot $')
# plt.ylabel('dL_mid / Mc^(5/6)')
# plt.grid(True, which='both')
# name="dL_joint_fit_results_emax/eta.png"
# plt.savefig(name, format='png', dpi=1000)

#%%
plt.close('all')
plt.figure(figsize=(7,6))
plt.plot(data.dL, data.sigmoid(data.dL, data.Dmid_mchirp_expansion(data.m1, data.m2, data.z, params_dmid), gamma_opt, delta_opt, emax_opt), '.')
t = ('gamma = %.3f , delta = %.3f , emax = %.3f' %(gamma_opt, delta_opt, emax_opt) )
plt.title(t)
plt.xlabel('dL')
plt.ylabel(r'$\epsilon (dL, dmid(m), \gamma_{opt}, \delta_{opt}, emax_{opt})$')
plt.show()
plt.savefig(f'{dmid_fun}/shape/opt_epsilon_plot.png')

