#! /usr/bin/env python3

import os
from fitting import FitModel
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import input_checker as ic
import json

def create_directory(head_directory_name, structure_number):
    directory = os.path.join(head_directory_name, str(structure_number))
    os.makedirs(directory)
    return directory

def setup_error_checks(include_labels, bounds_list, fit_data, params):
        if len(include_labels) != len(bounds_list):
            raise IndexError('include_labels and bounds_list are not of equal length. Check there are bounds associated with each label with the correct bound values.')
        for label, bounds in zip(include_labels, bounds_list):
            if label.startswith('dq_'):
                ic.check_coreshell(label, bounds, fit_data)
            elif label == 'q_scaling':
                ic.check_scaling_limits(label, bounds)
            elif '-' in label:
                ic.check_spring(label, bounds, params)
            elif '_a' in label or '_rho' in label or '_c' in label:
                ic.check_buckingham(label, bounds, params)
            else:
                raise TypeError('Label {} is not a valid label type'.format(label))

def extract_stresses_and_forces(fit_data, values, args):
    fit_data.init_potential(values, args)
    ip_forces, ip_stresses = fit_data.get_forces_and_stresses()
    dft_forces = fit_data.expected_forces()
    dft_stresses = fit_data.expected_stresses()
    return dft_forces, ip_forces, dft_stresses, ip_stresses                

if __name__ == '__main__':

    params = {}
    params['core_shell'] = { 'Li': False, 'Ni': False, 'O': False}
    params['charges'] = {'Li': +1.0,
                         'Ni': +3.0,
                         'O': -2.0}
    params['masses'] = {'Li': 6.941,
                        'Ni': 58.6934,
                        'O': 15.9991}

    distribution = {}
    distribution['Li-O'] = {'bpp' : [663.111, 0.119, 0.0],
                            'sd' : [80, 0.01, 0.01]}
    distribution['Ni-O'] = {'bpp' : [1393.540, 0.218, 0.000],
                            'sd'  : [80, 0.01, 0.01]}
    distribution['O-O'] = {'bpp' : [25804.807, 0.284, 0.0],
                           'sd'  : [200, 0.01, 5]}

    include_labels = [ 'q_scaling', 'Li_O_a',     'Li_O_rho', 'Ni_O_a',      'Ni_O_rho', 'O_O_a',        'O_O_rho']
    bounds_list = [ (0.3,1.0),  (100.0,50000.0),(0.01,1.0), (100.0,50000.0),(0.01,1.0), (150.0,50000.0),(0.01,1.0)]

    tot_num_structures = 15
    num_struct_to_fit = 5
    num_of_fits = 1
    head_directory_name = 'final_potentials/partial+ri'.format(num_struct_to_fit)

    sets_of_structures = []
    while len(sets_of_structures) < num_of_fits:
        struct_set = np.sort(np.random.randint(0,tot_num_structures, size=num_struct_to_fit), axis=0)
        if len(set(struct_set)) != num_struct_to_fit:
            continue
        if not any(np.array_equiv(struct_set, x) for x in sets_of_structures):
            sets_of_structures.append(struct_set) 
    sets_of_structures = np.array(sets_of_structures)

    poscar_directory = os.path.join('poscars','thermos')
    outcar_directory = os.path.join('outcars','thermos')
    for fit, structs in enumerate(sets_of_structures): 
        for struct_num, struct in enumerate(structs):
            os.system('cp {}/POSCAR{} {}/POSCAR{}'.format(poscar_directory, struct+1, 'poscars', struct_num+1))
            os.system('cp {}/OUTCAR{} {}/OUTCAR{}'.format(outcar_directory, struct+1, 'outcars', struct_num+1))
        fit_data = FitModel.collect_info(params, distribution, supercell=[2,2,2])
        setup_error_checks(include_labels, bounds_list, fit_data, params)
        s = optimize.differential_evolution(fit_data.chi_squared_error, bounds=bounds_list, popsize=25,
                                            args=([include_labels]), maxiter=2000,
                                            disp=True, init='latinhypercube', workers=-1)
        dft_f, ip_f, dft_s, ip_s = extract_stresses_and_forces(fit_data, s.x, include_labels)
        local_struct_dir = '-'.join([ '{}'.format(struct+1) for struct in structs])
        struct_directory = create_directory(head_directory_name, local_struct_dir)
        np.savetxt('{}/dft_forces.dat'.format(struct_directory), dft_f, fmt='%.10e', delimiter=' ')
        np.savetxt('{}/ip_forces.dat'.format(struct_directory), ip_f, fmt='%.10e', delimiter=' ')
        np.savetxt('{}/dft_stresses.dat'.format(struct_directory), dft_s, fmt='%.10e', delimiter=' ')
        np.savetxt('{}/ip_stresses.dat'.format(struct_directory), ip_s, fmt='%.10e', delimiter=' ')
        with open('{}/error.dat'.format(struct_directory), 'w') as f:
            f.write(str(s.fun))
        potential_dict = {k:v for k, v in zip(include_labels, s.x)}
        with open('{}/potentials.json'.format(struct_directory), 'w') as f:
            json.dump(potential_dict, f)
