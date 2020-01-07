#! /usr/bin/env python3

from fitting import FitModel
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import input_checker as ic
import json

def get_forces(fit_data, values, args):
    fit_data.init_potential(values, args)
    ip_forces = np.concatenate(fit_data.get_forces(), axis=0)
    dft_forces = np.concatenate(fit_data.expected_forces(), axis=0)
    return dft_forces, ip_forces

if __name__ == '__main__':

    params = {}
    params['core_shell'] = { 'Li': False, 'Ni': False, 'O': True }
    params['charges'] = {'Li': +1.0,
                         'Ni': +3.0,
                         'O': {'core':  -2.0, #+0.960,
                               'shell': 0.0}} #-2.960}}
    params['masses'] = {'Li': 6.941,
                        'Ni': 58.6934,
                        'O': {'core': 14.3991,
                              'shell': 1.5999} }
    params['cs_springs'] = {'O-O' : [20.0, 0.0]}


    distribution = {}
    distribution['Li-O'] = {'bpp' : [663.111, 0.119, 0.0],
                            'sd' : [80, 0.01, 0.01]}
    distribution['Ni-O'] = {'bpp' : [1393.540, 0.218, 0.000],
                            'sd'  : [80, 0.01, 0.01]}
    distribution['O-O'] = {'bpp' : [25804.807, 0.284, 0.0],
                           'sd'  : [200, 0.01, 5]}

    fit_data = FitModel.collect_info(params, distribution, supercell=[2,2,2])

    include_labels = ['dq_O', 'q_scaling', 'O-O spring', 'Li_O_a',     'Li_O_rho', 'Ni_O_a',      'Ni_O_rho', 'O_O_a',        'O_O_rho']
    bounds_list = [(0.01, 4), (0.3,1.0),   (10.0,150.0), (100.0,50000.0),(0.01,1.0), (100.0,50000.0),(0.01,1.0), (150.0,50000.0),(0.01,1.0)]

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

    s = optimize.differential_evolution(fit_data.chi_squared_error,
                                        bounds=bounds_list,
                                        popsize=25,
#                                         tol=0.0001,
                                        args=([include_labels]),
                                        maxiter=2000,
                                        disp=True,
                                        init='latinhypercube',
                                        workers=-1)

    print(s)


    
    dft_forces, ip_forces = get_forces(fit_data, s.x, include_labels)

    np.savetxt('dft_forces.dat', dft_list, fmt='%.10e', delimiter=' ')
    np.savetxt('ip_forces.dat', ip_list, fmt='%.10e', delimiter=' ')

    np.savetxt('error.dat', s.fun, fmt='%.10')
    potential_dict = {k:v for k, v in zip(include_labels, s.x)}
    with open('potentials.json', 'w') as f:
        json.dump(potential_dict, f)
    
    
    plt.plot(ip_list, label='ip')
    plt.plot(dft_list, label='dft', alpha=0.6)
    plt.ylabel('force')
    plt.xlabel('index')
    plt.legend(loc='upper right')
    plt.text(500,4.5, 'error: {0:.5f}'.format(np.sum(((dft_list-ip_list)**2)/ip_list.size)))
    plt.text(500,4.0, 'scaling factor: {:.4f}'.format(s.x[1]))
    plt.text(500,3.5, 'potential values: {:.4f} {:.4f}'.format(*include_values[:2]))
    plt.text(500,3.0, '{:.4f} {:.4f} {:.4f} {:.4f}'.format(*include_values[2:]))
    plt.text(500,2.5, 'spring constant: {:.4f}'.format(s.x[2]))
    plt.text(500,2.0, 'O-O dQ: {:.4f}'.format(s.x[0]))
    plt.savefig('q+buck+cs+spring+ratio_3thermo.png',dpi=500, bbox_inches = "tight")
    plt.show()

