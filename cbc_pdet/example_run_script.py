#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:42:57 2024

@author: ana
"""

#example script on how to initialise the class 
import cbc_pdet 
from cbc_pdet.o123_class_found_inj_general import Found_injections

#set the ini variables, dmid_fun is the function you are using for dmid, 
#emax_fun is the function for emax (max search efficiency)

dmid_fun = 'Dmid_mchirp_fdmid_fspin' #the final one including spins effect
emax_fun = 'emax_exp'

data = Found_injections(dmid_fun, emax_fun) #initialise class

#set these two variables for the results 
run_fit = 'o3'   #the fit you want the optimal parameters from (in case you want to use o3 fit with o2 dataset, for example)
run_dataset = 'o3'  #the dataset you want to use (either for the fit or to use for extra results), can be o1, o2 or o3

#create folders to save results
data.make_folders(run_fit)

#load the injection set (i.e. the data)
data.load_inj_set(run_dataset)

#fetch the optimal parameters for the sigmoid (Pdet) function from the fit you have made
data.get_opt_params(run_fit)

#compute pdet for some luminosity distance, masses and spins values (effective spin)

dL = 100 #Mpc
m1_det = 60 #solar masses in the detector's frame
m2_det = 45 #solar masses in the detector's frame
chieff = 0.75


#dmid values
dmid_values = data.dmid(m1_det, m2_det, chieff, data.dmid_params)

#get correct shape params
data.set_shape_params()

#emax values
emax_values =  data.emax(m1_det, m2_det, data.emax_params)

#compute P_det specifying gamma and delta (fit parameters outside dmid and emax)
#here we use the class variables for delta and gamma (from the fit), which are set when running
#data.shape_params(), but you have an option to use another values if you'd like
pdet1 = data.sigmoid(dL,dmid_values, emax_values, data.gamma, data.delta)

#another way to get Pdet. Here, run_fit is the oberserving run from which we want the fit
#you can't give it different values of params than the ones from the fit
pdet2 = data.run_pdet(dL, m1_det, m2_det, run_fit, chieff)

#you can check both methods return the same value :)
print(pdet1, pdet2)




