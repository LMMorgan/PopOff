#! /usr/bin/env python3

from buckfit.fitting_code import FitModel
import buckfit.fitting_output as output
from buckfit.input_checker import setup_error_checks
from scipy import optimize
import numpy as np

def random_set_of_structures(fits, structures, structures_to_fit, seed=False):
    """
    Randomly selects structures up to the number of structures to include in the fit, checks there are no repeats and returns the structure numbers of those to be included in the fit.
    Args:
        fits (int): Number of fits to run.
        structures (int): Total number of structures in the training set.
        structures_to_fit (int): Number of structures to fit to.
        seed (optional: int or 1-d array_like): Seed for RandomState. Must be convertible to 32 bit unsigned integers. Default=False
    Returns:
        (np.array): Structure numbers of structures in training set to be fitted to.
    """
 
    for x in [fits, structures, structures_to_fit]:
        if isinstance(x, int) == False:
            raise TypeError('Input values must be integers.')
        if x <=0:
            raise ValueError('Integer values must be positive.')

    if seed is not False:
        if isinstance(seed, int) == False and isinstance(seed, np.ndarray) == False:
            raise TypeError('seed must be an integer, 1-d array, or False')
        else:
            np.random.seed(seed)
            sets_of_structures = []
            while len(sets_of_structures) < fits:
                struct_set = np.sort(np.random.randint(0, structures, size=structures_to_fit), axis=0)
                if len(set(struct_set)) != structures_to_fit:
                    continue
                if not any(np.array_equiv(struct_set, x) for x in sets_of_structures):
                    sets_of_structures.append(struct_set) 
            return np.array(sets_of_structures)

def run_fit(sets_of_structures, params, labels, bounds, supercell=None, seed=None):
    """
    Collates the structures to be fitted into the working directory, creates the lammps inputs and runs the optimiser to fit to the designated parameters. Calls another function to save the data in the appropriate output directory.
    Args:
        sets_of_structures (np.array): Structure numbers of structures in training set to be fitted to.
        params (dict(dict)): Setup dictionary containing the inputs for coreshell, charges, masses, potentials, and core-shell springs.
        labels (list(str)): List of parameters to be fitted.
        bounds (list(tuple(float))): List of lower and upper bound tuples associated with each parameter.
        supercell (optional:list(int) or list(list(int))): 3 integers defining the cell increase in x, y, and z for all structures, or a list of lists where each list is 3 integers defining the cell increase in x, y, z, for each individual structure in the fitting process. i.e. all increase by the same amount, or each structure increased but different amounts. Default=None.
        seed (optional: int or np.random.RandomState): If seed is not specified the np.RandomState singleton is used. If seed is an int, a new np.random.RandomState instance is used, seeded with seed. If seed is already a np.random.RandomState instance, then that np.random.RandomState instance is used. Specify seed for repeatable minimizations.
    Returns:
        None
    """
    
    for fit, structs in enumerate(sets_of_structures):
        fit_data = FitModel.collect_info(params, structs, supercell=supercell)
        setup_error_checks(labels, bounds, fit_data, params)

        fit_output = optimize.differential_evolution(fit_data.chi_squared_error, bounds=bounds, popsize=25,
                                           args=([labels]), maxiter=1000, updating='deferred',
                                           disp=True, init='latinhypercube', workers=16, seed=seed)
        dft_forces, ip_forces, dft_stresses, ip_stresses = output.extract_stresses_and_forces(fit_data, fit_output.x, labels)
        local_struct_dir = '-'.join([ '{}'.format(struct+1) for struct in structs])
        struct_directory = output.create_directory(head_directory_name, local_struct_dir)
        output.save_data(struct_directory, labels, fit_output, dft_forces, ip_forces, dft_stresses, ip_stresses)
    fit_data.reset_directories()

if __name__ == '__main__':

    params = {}
    params['core_shell'] = { 'Na': False, 'O': True, 'Cl': True }
    params['charges'] = {'Na': +1.0,
                         'O': {'core':  -2.0, 'shell': 0.0},
                         'Cl': {'core':  -1.0, 'shell': 0.0}}
    params['masses'] = {'Na': 22.9898,
                        'O': {'core': 14.3991, 'shell': 1.5999},
                        'Cl': {'core': 31.905, 'shell': 3.545}}
    params['potentials'] = {'Na-Na': [1788.19, 0.159, 0.0],
                            'Na-Cl': [1170.41, 0.315, 0.0],
                            'Na-O': [588.38, 0.338, 0.0],
                            'Cl-Cl': [1227.20, 0.3210, 14.53],
                            'Cl-O': [8286.91, 0.2590, 62.20],
                            'O-O': [22764.30, 0.1490, 13.19]}
    params['cs_springs'] = {'Cl-Cl' : [593.716, 0.0],
                            'O-O': [29.38, 0.0]}

    labels = ['q_scaling', 'dq_Cl', 'dq_O', 'Cl-Cl spring', 'O-O spring', 'Na_Na_a', 'Na_Na_rho', 'Na_Cl_a', 'Na_Cl_rho', 'Na_O_a', 'Na_O_rho', 'Cl_Cl_a', 'Cl_Cl_rho', 'Cl_O_a', 'Cl_O_rho', 
              'O_O_a', 'O_O_rho']
    bounds = [(0.3,1.0), (0.01, 2), (0.01,4), (10.0,100.0), (10.0,100.0), (100.0,50000.0), (0.01,1.0), (100.0,50000.0), (0.01,1.0), (100.0,50000.0), (0.01,1.0), (100.0,50000.0),
              (0.01,1.0), (100.0,50000.0), (0.01,1.0), (100.0,50000.0), (0.01,1.0)]

    structures = 7 #Total number of structures in the training set
    structures_to_fit = 7 #Number of structures you wish to fit to
    fits = 1 #Number of fits to run
    head_directory_name = 'results/Na3OCl/partial_charge'.format(structures_to_fit)

    sets_of_structures = random_set_of_structures(fits, structures, structures_to_fit, seed=7)

    run_fit(sets_of_structures, params, labels, bounds, supercell=[2,2,2], seed=7)
