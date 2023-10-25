#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 20:24:55 2023

@author: ana
"""

def Nexp_MC(self, dmid_fun, n, params):
    #random values of m1, m2 and dL
    DMID = getattr(Found_injections, dmid_fun)
    N=int(1e6)
    dim = 3
    ns = 0
    ymin = 0
    ymax = 0.79928*np.max(self.m_pdf)*np.max(self.dL_pdf)
    #ymax = 2
    xmax = np.array((self.dLmax, self.mmax, self.mmax))
    xmin = np.array((0, np.min(self.mmin), np.min(self.mmin)))
    A = ymax*np.prod(xmax - xmin)
    
    mu = 500
    qx = uniform
    
    def f(x):
        # print(self.interp_dL(x[0]))
        # print(self.fun_m_pdf(x[1], x[2]))
        # print(self.sigmoid(x[0], DMID(self, x[1], x[2], self.interp_z(x[0]), params)))
        p = self.interp_dL(x[0]) * self.fun_m_pdf(x[1], x[2]) 
        return self.sigmoid(x[0], DMID(self, x[1], x[2], self.interp_z(x[0]), params)) * p
    
    def p(x):
        return self.interp_dL(x[0]) * self.fun_m_pdf(x[1], x[2]) 
    
    # for i in range(n): 
    #     dLr = (self.dLmax - 0)*np.random.random() + 0
    #     m1r = (self.mmax - self.mmin)*np.random.random() + self.mmin
    #     m2r = (m1r - self.mmin)*np.random.random()+ self.mmin
    #     #print(dLr, m1r, m2r)
    #     # dmidr = DMID(self, m1r, m2r, self.interp_z(dLr), params)
    #     # dLr = (dmidr*1.5 - 0)*np.random.random() + 0
    #     # # pm = self.fun_m_pdf(m1r, m2r)
    #     # pdL = self.interp_dL(dLr)
    #     # p = pm*pdL
        
        
    #     xr = np.array((dLr, m1r, m2r))
    #     yr = (ymax - ymin)*np.random.random() + ymin
    #     #yr = np.random.random() 
        
    #     print(yr)
    #     print(f(xr))
    #     if yr<f(xr):
    #         ns += 1
            
     
        
    dLr = (self.dLmax - 0)*np.random.random(n) + 0
    m1r = (self.mmax - self.mmin)*np.random.random(n) + self.mmin
    m2r = (m1r - self.mmin)*np.random.random(n)+ self.mmin
    xr = np.array((dLr, m1r, m2r))
    suma = np.sum(f(xr))/n
    print(ns)
    return self.Ntotal*suma

def Dmid_inter(self, dmid_fun, m1, m2, dL, params):
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
    DMID = getattr(Found_injections, dmid_fun)
    z = self.interp_z(dL)
    return DMID(self, m1, m2, z, params)