#! /usr/bin/env python3

import arviz as az
import matplotlib.pyplot as plt
from fitting import FitModel
import pymc3 as pm

if __name__ == '__main__':

    params = {}
    params['core_shell'] = { 'Li': False, 'Ni': False, 'O': True }
    params['charges'] = {'Li': +1.0,
                         'Ni': +3.0,
                         'O': {'core':  +0.960,
                               'shell': -2.960}}
    params['masses'] = {'Li': 6.941,
                        'Ni': 58.6934,
                        'O': {'core': 14.3991,
                              'shell': 1.5999} }

    params['cs_springs'] = {'O' : [100.0, 0.0]}

    distribution = {}
    distribution['Li-O'] = {'bpp' : [691.229, 0.269, 0.0],
                            'sd' : [80, 0.01, 0.01]}

    distribution['Ni-O'] = {'bpp' : [591.665, 0.382, 0.000],
                            'sd'  : [80, 0.01, 0.01]}

    distribution['O-O'] = {'bpp' : [22739.211, 0.146, 67.764],
                           'sd'  : [200, 0.01, 5]}

    excude_from_fit = [] # string of atom1_atom2_param. Example of format = 'O_O_rho'

    fit_data = FitModel.collect_info(params, distribution,  supercell=[1,1,1])

    #dist_func must be `sum_of_squared_distance` or `absolute_error`
    trace = fit_data.run_fit(excude_from_fit=excude_from_fit, epsilon=1.0, draws=500, dist_func='sum_of_squared_distance')

    az.style.use('arviz-darkgrid')
    az.plot_trace(trace)
    plt.savefig('test_trace.png',dpi=500, bbox_inches = "tight")

    az.plot_posterior(trace, round_to = 3, point_estimate = 'mode')
    plt.savefig('test_mode.png',dpi=500, bbox_inches = "tight")

    filename = 'summary.txt'
    with open('summary.txt', 'a') as file:
        pm.summary(trace).to_csv(filename, index=False)
