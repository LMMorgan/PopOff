import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import glob
import re
import numpy as np
from vasppy.outcar import forces_from_outcar


    
    
    
    

def pd_obj_to_nparray(pd_object):
    pd_object = pd_object.replace('[','').replace(']','').split('\n')
    pd_object = [( re.sub("\s+", ",", s.strip()) ) for s in pd_object]
    pd_object = [l.split(',') for l in pd_object]
    pd_object = np.array([np.array(np.float_(sub)) for sub in pd_object])
    return pd_object

def get_lammps_data():
    forces_list = []
    for i, outcar in enumerate(glob.glob('outcars/OUTCAR*')):
        forces = forces_from_outcar('outcars/OUTCAR{}'.format(i+1))[-1]
        forces_list.append(forces)
    return forces_list    
      
def forces_plots(forces_filename, i):
    direction = ['x', 'y', 'z']
    forces_data = pd.read_csv(forces_filename)
    in_forces = get_lammps_data()
    out_forces = []
    for forces in forces_data['forces']:
        forces_list = pd_obj_to_nparray(forces)
        out_forces.append(forces_list)
        
    for a, direct in enumerate(direction):
        plt.plot(out_forces[i-1][:,a], label='calculated_forces')
        plt.plot(in_forces[0][:,a], label='expected_forces')
        plt.legend()
        plt.xlabel('atom number')
        plt.ylabel('force in {}'.format(direct))
        plt.savefig('plots/force_num_{}_{}.png'.format(direct, i),dpi=500, bbox_inches = "tight")
        plt.close()

    for a, direct in enumerate(direction):
        plt.plot(out_forces[i-1][:,a], in_forces[0][:,a], '.')
        plt.xlabel('calculated forces in {}'.format(direct))
        plt.ylabel('expected forces in {}'.format(direct))
        plt.savefig('plots/force_vs_expected_{}_{}.png'.format(direct, i),dpi=500, bbox_inches = "tight")
        plt.close()