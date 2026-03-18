This repository contains code for a compact binary mergers selection function for gravitational wave (GW) population analysis, used in the following paper: 
**"A physically modelled selection function for compact binary mergers in the LIGO-Virgo O3 run and beyond"**  
*Authors: Ana Lorenzo-Medina, Thomas Dent*  
DOI: [https://doi.org/10.48550/arXiv.2408.13383](https://doi.org/10.48550/arXiv.2408.13383)

The main branch contains  a python package with only the necessary files to run it. There is an example_run_script.py with the code to initialize the class Found_injections() to be able to use its methods. The class is in the o123_class_found_inj_general.py file, and the functions for dmid and emax are in fitting_functions.py. The o1 o2 and o3 folders contain the optimal values of the parameters for every combination of dmid/emax functions we have tried. They are not strictly needed to initialize the class but they are to be able to use a lot of the methods, if not given, you would need to make a fit to have them. The files that are necessary to initialize the class are the .dat files in ini_values, since they are initial points for the maximization algorithm of the likelihood. There is an option to give the class your own arrays of ini_values, if you don't want to use the ones we provide here. 

In the branch legacy we have everything produced for the paper mentioned before, as well as all the extra tests and studies we performed. 

If you find this code useful in your research, please consider citing our paper:
@misc{lorenzomedina2024physicallymodelledselectionfunction,
      title={A physically modelled selection function for compact binary mergers in the LIGO-Virgo O3 run and beyond}, 
      author={Ana Lorenzo-Medina and Thomas Dent},
      year={2024},
      eprint={2408.13383},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2408.13383}, 
}