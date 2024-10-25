# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:07:03 2022

@author: Ana
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def estat(y, uy, yest, par):
    r=y-yest
    SS=sum(r**2)
    chi2=sum((r/uy)**2)
    chi2r=chi2/(len(y)-par)
    SStot=sum((y-np.mean(y))**2)
    r2=1-SS/SStot
    return SS, chi2, chi2r, r2

  
plt.close('all')

from matplotlib import rc
rc('text', usetex=True)

np.random.seed(42)



######### Finn/Chernoff paper 

def sigmoid(d,a,b):
    return e_fmax/(1+(d/dmid)**(a*(1+b*np.tanh(d/dmid))))



dmid = 0.327
e_fmax = 1

P = np.array([1,0.9,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.005,0.004,0.003,0.002,0.001])
x2 = np.array([0,0.240,0.542,0.707,0.878,1.25,1.709,2.283,3.020,3.485,4.063,6.144,6.471,6.832,7.239,7.701,8.233,8.857,9.614,10.589,11.985,13.054,13.35,13.682,14.091,14.284])

d = np.sqrt(x2)/4
sP = np.ones(len(P))*10**-4

x0 = [1,1] #estimation of a,b
y_est, mcov = opt.curve_fit(sigmoid, d, P, x0)

a_est,b_est = y_est
sa,sb = np.sqrt(np.diag(mcov))

d_est = np.linspace(min(d), max(d), 1000)

# plt.figure()
# plt.plot(d,P, 'k.')
# plt.plot(d_est, sigmoid(d_est,a_est,b_est), '-', linewidth=1.2, label=r'fit to $\varepsilon_1$')
# #plt.plot(d_est, sigmoid(d_est,a_est,b_est+0.2), 'b--')
# plt.xlabel(r'$d_L / d_{hor}$', fontsize=14)
# plt.ylabel(r'$P(\Theta > x^2)$', fontsize=14)
# #plt.legend(fontsize=14)


print(a_est,b_est)

_,chi2,chi2r,_ = estat(P,sP,sigmoid(d,a_est,b_est), 2)

print('\nFun 1 chi2=', chi2, 'chi2r=', chi2r)

'''
plt.figure()
plt.plot(d[10:], 1/P[10:]-1, '.')
plt.plot(d_est[550:],1/sigmoid(d_est[550:],a_est,b_est)-1, 'r-')
plt.semilogy()

plt.figure()
plt.plot(d[:10], 1/P[:10]-1, '.')
plt.plot(d_est[:550],1/sigmoid(d_est[:550],a_est,b_est)-1, 'r-')
plt.ylim(0,3)

'''


#def new_sigmoid(d,delta, gamma, alpha, emax=1):
#    return emax/(1+(d/dmid)**alpha*np.exp(delta*(d**2-dmid**2)+gamma*(d-dmid)))

def new_sigmoid(d,delta, gamma, alpha, emax=1):
    return emax/(1+(d/dmid)**alpha*np.exp(delta*(d**2/dmid**2 - 1)+gamma*(d/dmid -1)))


x0_new = [1,1,1] #estimation of delta, gamma, alpha
y_est_new, mcov_new = opt.curve_fit(new_sigmoid, d, P, x0_new)

delta,gamma,alpha = y_est_new
s_delta,s_gamma,s_alpha = np.sqrt(np.diag(mcov_new))

_,chi2,chi2r,_ = estat(P,sP,new_sigmoid(d,delta,gamma, alpha), 3)
print('\nFun2 chi2=', chi2, 'chi2r=', chi2r)

plt.figure()
plt.plot(d,P, 'k.', label='Finn \& Chernoff paper')
plt.plot(d_est, sigmoid(d_est,a_est,b_est), '-', linewidth=1.2, label=r'fit to $\varepsilon_1$')
plt.xlabel(r'$d_L / d_{hor}$', fontsize=14)
plt.ylabel(r'$P(\Theta > x^2)$', fontsize=14)
plt.plot(d_est, new_sigmoid(d_est,delta,gamma, alpha), '-', linewidth=1.2, label=r'fit to $\varepsilon_2$')
plt.legend(fontsize=14)
name="general_plots_dL/FC.png"
plt.savefig(name, format='png', dpi=1000)

plt.figure()
plt.plot(d,P, 'ko', label='Finn \& Chernoff, PRD 1993')
plt.plot(d_est, sigmoid(d_est,a_est,b_est), 'C0', linewidth=1.2, label=r'fit to $\varepsilon_1$')
plt.hlines(0.5,-0.03, 0.33, color='tab:red', linestyle='dashed')
plt.vlines(0.33, -0.03, 0.5, color='tab:red', linestyle='dashed')
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize=18)
plt.ylabel(r'$P(\Theta > x^2)$', fontsize=18)
plt.xlim(-0.03, 0.99)
plt.ylim(-0.03, 1.05)
plt.legend(fontsize=15)
plt.savefig("general_plots_dL/FC_e1.png", format='png', dpi=300)
plt.savefig("general_plots_dL/FC_e1.pdf", format='pdf', dpi=150, bbox_inches="tight")


plt.figure()
plt.plot(d,P, 'ko', label='Finn \& Chernoff, PRD 1993')
plt.plot(d_est, new_sigmoid(d_est,delta,gamma, alpha), 'C1', linewidth=1.2, label=r'fit to $\varepsilon_2$')
plt.hlines(0.5,-0.03, 0.33, color='tab:red', linestyle='dashed')
plt.vlines(0.33, -0.03, 0.5, color='tab:red', linestyle='dashed')
plt.xlabel(r'$d_L / d_\mathrm{mid}$', fontsize=18)
plt.ylabel(r'$P(\Theta > x^2)$', fontsize=18)
plt.xlim(-0.03, 0.99)
plt.ylim(-0.03, 1.05)
plt.legend(fontsize=15)
plt.savefig("general_plots_dL/FC_e2.png", format='png', dpi=300)
plt.savefig("general_plots_dL/FC_e2.pdf", format='pdf', dpi=150, bbox_inches="tight")


print('delta, gamma, alpha = ',delta,gamma, alpha)

# plt.figure()
# plt.plot(d, 1/P-1, '.')
# plt.semilogy()
# plt.xlabel(r'$d_L / d_{hor}$', fontsize=14)
# plt.ylabel(r'$1/P(\Theta > x^2) \, -1$', fontsize=14)



'''

plt.figure()
plt.plot(d[10:], 1/P[10:]-1, '.')
plt.plot(d_est[550:],1/new_sigmoid(d_est[550:],delta,gamma, alpha)-1, 'r-')
plt.semilogy()

plt.figure()
plt.plot(d[:10], 1/P[:10]-1, '.')
plt.plot(d_est[:550],1/new_sigmoid(d_est[:550],delta,gamma, alpha)-1, 'r-')
plt.ylim(0,3)


#fixing alpha=1

x0_fix = [1,1] #estimation of a,b
y_est_fix, mcov_fix = opt.curve_fit(new_sigmoid, d, P, x0_fix)

delta_fix,gamma_fix = y_est_fix
s_delta_fix,s_gamma_fix = np.sqrt(np.diag(mcov_fix))

_,chi2_fix,chi2r_fix,_ = estat(P,sP,new_sigmoid(d,delta_fix,gamma_fix), 2)
print('\nFixing alpha=1, chi2=', chi2_fix, 'chi2r=', chi2r_fix)
'''