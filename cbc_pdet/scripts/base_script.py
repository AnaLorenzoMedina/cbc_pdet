#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:55:11 2024

@author: ana
"""
import sys
import os

# Save the current working directory
original_working_directory = os.getcwd()

try:
    # Change the current working directory to the parent directory
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Import the class from the module
    from o123_class_found_inj_general import Found_injections

    run_fit = 'o3'
    run_dataset = 'o3'
    dmid_fun = 'Dmid_mchirp_fdmid_fspin'
    emax_fun = 'emax_exp'
    alpha_vary = None
    
    data = Found_injections(dmid_fun, emax_fun, alpha_vary)
    path = f'{run_dataset}/' + data.path
    
    data.make_folders(run_fit)
    
    data.load_inj_set(run_dataset)
    data.get_opt_params('o3')

finally:
    # Restore the original working directory
    os.chdir(original_working_directory)

'''

# Add the parent directory (one level up from 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from o123_class_found_inj_general import Found_injections

run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin'
emax_fun = 'emax_exp'
alpha_vary = None

data = Found_injections(dmid_fun, emax_fun, alpha_vary)
path = f'{run_dataset}/' + data.path

data.make_folders(run_fit)

data.load_inj_set(run_dataset)
data.get_opt_params('o3')
'''