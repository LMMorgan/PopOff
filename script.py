#! /usr/bin/env python3

from fitting import FitModel
from databases import initiate_dbs, print_summary, print_forces
from plotters import plotter
from iteration_functions import iter_fitting, update_potentials
from convergence_checker import get_modes, converge_check
from scipy import stats

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

    distribution['O-O'] = {'bpp' : [22739.211, 0.146, 20.7],
                           'sd'  : [200, 0.01, 5]}

    excude_from_fit = [] # string of atom1_atom2_param. Example of format = 'O_O_rho'
    
    def mode_potentials(trace):
        potential_dict = {}
        for var in trace.varnames:
            potential_dict['{}'.format(var)] = float(stats.mode(trace.get_values(var))[0])
        return potential_dict
    
    i=1
    prev_modes = None
    converge = False
    summary_filename = 'summary.csv'
    forces_filename = 'forces.csv'
    initiate_dbs(summary_filename, forces_filename)

    while converge is False:

        #Runs FitModel and trace, finds mode, and updates the potentials in the distribution dictionary with mean values
        trace, fit_data = iter_fitting(params, distribution, excude_from_fit)
        modes = get_modes(trace)
        distribution = update_potentials(trace, modes, distribution)

        #Runs with mode potential and returns forces
        kwargs = mode_potentials(trace)
        mode_forces = fit_data.get_forces(**kwargs)

        #Fills the databases
        print_forces(i, mode_forces, forces_filename)
        print_summary(i, trace, summary_filename)

        #Plots the distributions
        plotter(trace, i)

        #Checks convergence, sets modes to prev_moves
        converge = converge_check(modes, distribution, prev_modes)
        prev_modes = modes
        i+=1
