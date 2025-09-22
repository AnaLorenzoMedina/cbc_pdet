#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:58:56 2023

@author: ana
"""

import numpy as np
import astropy.constants as const

def Mc(m1, m2):
    M = m1 + m2
    return (m1 * m2)**(3/5) / M**(1/5)

def Dmid_mchirp(m1_det, m2_det, cte):
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
    Mc_det = Mc(m1_det + m2_det)
    return cte * Mc_det**(5/6) 

def Dmid_mchirp_expansion(m1_det, m2_det, params):
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
    Mc_det = Mc(m1_det + m2_det)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M )
    
    return pol * Mc_det**(5/6)

def Dmid_mchirp_expansion_asqrt(m1_det, m2_det, params):
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
    Mc_det = Mc(m1_det + m2_det)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_sqrt * M**(1/2) )
    
    return pol * Mc_det**(5/6) 

def Dmid_mchirp_expansion_a11(m1_det, m2_det, params):
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
    Mc_det = Mc(m1_det + m2_det)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return pol * Mc_det**(5/6)


def Dmid_mchirp_power(m1_det, m2_det, params):
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
    Mc_det = Mc(m1_det + m2_det)
    
    pol = cte *(1+ a_20 * M**2 / 2 + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) / 2 + a_30 * M**3 )
    
    return pol * Mc_det**((5+power_param)/6)

def Dmid_mchirp_expansion_exp(m1_det, m2_det, params):
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
    Mc_det = Mc(m1_det + m2_det)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return pol * Mc_det**(5/6) * np.exp(-M/Mstar)

def Dmid_mchirp_expansion_noa30(m1_det, m2_det, params):
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
    
    Mc_det = Mc(m1_det + m2_det)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return pol * Mc_det**(5/6)

def emax_exp(m1_det, m2_det, params):
    """
    maximum search sensitivity (emax) as a function of the masses
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

def emax_sigmoid(m1_det, m2_det, params):
    """
    maximum search sensitivity (emax) as a function of the masses
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

def emax_sigmoid_nolog(m1_det, m2_det, params):
    """
    maximum search sensitivity (emax) as a function of the masses
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
    return 1 / (1 + np.exp(b_0 + b_1 * Mtot + b_2 * Mtot**2))

def emax_gaussian(m1_det, m2_det, params):
    """
    maximum search sensitivity (emax) as a function of the masses
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
    b_0, b_1, muM, sigmaM = params
    return (1 - b_0) * (1 - (b_1 * np.exp(-(np.log(Mtot) - np.log(muM))**2 / (2 *sigmaM**2))))


def Dmid_mchirp_fdmid(m1_det, m2_det, params):
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
    D0 , a_20, a_01, a_21, a_10, a_11 = params

    M = m1_det + m2_det
    eta = m1_det * m2_det / M**2
    
    Mc_det = Mc(m1_det + m2_det)
    
    f_dmid = (a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return  Mc_det**(5/6) * D0 * np.exp(f_dmid)


def Dmid_mchirp_fdmid_fspin(m1_det, m2_det, chi_eff, params):
    """
    Dmid values (distance where Pdet = 0.5) as a function of the masses 
    in the detector frame and the spins (chi effective)

    Parameters
    ----------
    m1_det : detector frame mass1, float or 1D array
    m2_det: detector frame mass2, float or 1D array
    chi_eff: chi effective, float or 1D array
    params : parameters that we will be optimizing, 1D array
    
    Returns
    -------
    Dmid(m1,m2, chi_eff) in the detector's frame

    """
    D0 , a_20, a_01, a_21, a_10, a_11, c_1, c_11 = params

    M = m1_det + m2_det
    eta = m1_det * m2_det / M**2
    
    Mc_det = Mc(m1_det + m2_det)
    
    f_dmid = (a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))
    f_as = (c_1 + c_11 * M) * chi_eff
    
    return  Mc_det**(5/6) * D0 * np.exp(f_dmid) * np.exp(f_as)

def Dmid_mchirp_fdmid_fspin_c21(m1_det, m2_det, chi_eff, params):
    """
    Dmid values (distance where Pdet = 0.5) as a function of the masses 
    in the detector frame and the spins (chi effective)

    Parameters
    ----------
    m1_det : detector frame mass1, float or 1D array
    m2_det: detector frame mass2, float or 1D array
    chi_eff: chi effective, float or 1D array
    params : parameters that we will be optimizing, 1D array
    
    Returns
    -------
    Dmid(m1,m2, chi_eff) in the detector's frame

    """
    D0 , a_20, a_01, a_21, a_10, a_11, c_1, c_11, c_21 = params

    M = m1_det + m2_det
    eta = m1_det * m2_det / M**2
    
    Mc_det = Mc(m1_det + m2_det)
    
    f_dmid = (a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))
    f_as = (c_1 + c_11 * M + c_21 * M**2) * chi_eff
    
    return  Mc_det**(5/6) * D0 * np.exp(f_dmid) * np.exp(f_as)

def Dmid_mchirp_mixture_logspin_corr(m1_det, m2_det, chi_eff, params):
    """
    Dmid values (distance where Pdet = 0.5) as a function of the masses 
    in the detector frame and the spins (chi effective)

    Parameters
    ----------
    m1_det : detector frame mass1, float or 1D array
    m2_det: detector frame mass2, float or 1D array
    chi_eff: chi effective, float or 1D array
    params : parameters that we will be optimizing, 1D array
    
    Returns
    -------
    Dmid(m1,m2, chi_eff) in the detector's frame

    """
    D0, B, C , mu, sigma, a_01, a_11, a_21, c_01, c_11, d_11, L = params

    M = m1_det + m2_det
    eta = m1_det * m2_det / M**2
    
    Mc_det = Mc(m1_det + m2_det)
    
    fexp = np.exp(-B * M - L * np.log(M))
    fgauss = np.exp(-(np.log(M)-np.log(mu))**2 / (2*sigma**2))
    
    f_M = D0 * (fexp + C * fgauss)
    
    f_eta =  a_01 * (1 - 4*eta)  + a_11 * M * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)
    f_as = (c_01 + c_11 * M + d_11 * np.log(M)) * chi_eff
    
    return  Mc_det**(5/6) * f_M * np.exp(f_eta) * np.exp(f_as)


def dL_derivative(z, dL, cosmo):
    #denominator in comoving distance integral 
    A = np.sqrt(cosmo.Om0 * (1 + z)**3 + 1 - cosmo.Om0)
    #derivative of comoving distance
    dC_dif = (const.c.value*1e-3 / cosmo.H0.value) / A
    #second term is equal to comoving distance 
    dL_dif = (1 + z) * dC_dif + dL / (1 + z)
    
    return dL_dif

