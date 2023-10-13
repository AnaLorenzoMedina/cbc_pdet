#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:58:56 2023

@author: ana
"""

import numpy as np

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
    M = m1_det + m2_det
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    return cte * Mc**(5/6) 

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
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M )
    
    return pol * Mc**(5/6)

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
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_sqrt * M**(1/2) )
    
    return pol * Mc**(5/6) 

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
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return pol * Mc**(5/6)


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
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    
    pol = cte *(1+ a_20 * M**2 / 2 + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) / 2 + a_30 * M**3 )
    
    return pol * Mc**((5+power_param)/6)

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
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta)  + a_30 * M**3 + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return pol * Mc**(5/6) * np.exp(-M/Mstar)

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
    
    Mc = (m1_det * m2_det)**(3/5) / M**(1/5)
    
    pol = cte *(1+ a_20 * M**2  + a_01 * (1 - 4*eta) + a_21 * M**2 * (1 - 4*eta) + a_10 * M + a_11 * M * (1 - 4*eta))
    
    return pol * Mc**(5/6)

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