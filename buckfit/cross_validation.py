from buckfit.fitting_code import FitModel
import numpy as np
import os
import json
import glob
import buckfit.fitting_output as fit_out
import matplotlib.pyplot as plt

def validation_sets(fits, structures, structures_in_fit, structure_nums, seed=False):
    """
    Randomly selects structures up to the number of structures to include in the cross-validation, checks there are no repeats and no structures used in the fit. The number of structure sets is equal to the number of cross-validation fits to be performed. Note: check you have enough structures avaliable (not including those in the fit) to make a complete set, and to the number of sets you want.
    Args:
        fits (int): Number of fits to run.
        structures (int): Total number of structures in the training set.
        structures_in_fit (int): Number of structures to fit to.
        structure_nums (np.array): Structure numbers of structures in the fit which is being validated.
    Returns:
        (np.array): Sets of structure numbers of structures in training set to be in validation sets.
    """
    if seed is not False:
        np.random.seed(seed)
    sets_of_structures = []
    while len(sets_of_structures) < fits:
        struct_set = np.sort(np.random.randint(1,structures+1, size=structures_in_fit), axis=0)
        if len(set(struct_set)) != structures_in_fit:
            continue
        if any(x in struct_set for x in structure_nums):
            continue
        if not any(np.array_equiv(struct_set, x) for x in sets_of_structures):
            sets_of_structures.append(struct_set)
    return np.array(sets_of_structures)

def chi_squared_error(dft_forces, ip_forces, dft_stresses, ip_stresses):
    """
    Calculates a chi squared error between the dft and ip forces and stress tensors.
    Args:
        dft_forces (np.array): 3D array of dft forces in x,y,z for each atom in each structure.
        ip_forces (np.array): 3D array of fitted interatomic potential forces in x,y,z for each atom in each structure.
        dft_stresses (np.array): 3D array of dft stress tensors in each structure.
        ip_stresses (np.array): 3D array of fitted interatomic potential stress tensors in each structure.
     Returns:
         error (float): The chi squared error calculated between dft forces and the MD forces under the given potential and the DFT and MD stress tensors.
     """
    force_diff = np.sum((dft_forces - ip_forces)**2)/ dft_forces.size
    stess_diff = np.sum((dft_stresses - ip_stresses)**2)/6
    return force_diff + (stess_diff*0.001)

def save_cv_data(output_directory, structs, error, dft_forces, ip_forces, dft_stresses, ip_stresses):
    """
    Collates the output data from the fit and saves it to different files within the designated output directory.
    Args:
        output_directory (str): directory pathway to local output directory.
        structs (np.array): Structure numbers of structures in the validation set.
        error (float): The chi squared error calculated between dft forces and the MD forces under the given potential and the DFT and MD stress tensors.
        dft_forces (np.array): 3D array of dft forces in x,y,z for each atom in each structure.
        ip_forces (np.array): 3D array of fitted interatomic potential forces in x,y,z for each atom in each structure.
        dft_stresses (np.array): 3D array of dft stress tensors in each structure.
        ip_stresses (np.array): 3D array of fitted interatomic potential stress tensors in each structure.
     Returns:
         None
     """
    np.savetxt('{}/s{}_dft_forces.dat'.format(output_directory, '-'.join([str(num) for num in structs])), dft_forces, fmt='%.10e', delimiter=' ')
    np.savetxt('{}/s{}_ip_forces.dat'.format(output_directory, '-'.join([str(num) for num in structs])), ip_forces, fmt='%.10e', delimiter=' ')
    np.savetxt('{}/s{}_dft_stresses.dat'.format(output_directory, '-'.join([str(num) for num in structs])), dft_stresses, fmt='%.10e', delimiter=' ')
    np.savetxt('{}/s{}_ip_stresses.dat'.format(output_directory, '-'.join([str(num) for num in structs])), ip_stresses, fmt='%.10e', delimiter=' ')
    with open('{}/s{}_error.dat'.format(output_directory, '-'.join([str(num) for num in structs])), 'w') as f:
        f.write(str(error))
        
def run_cross_validation(fits, structures, structures_in_fit, head_directory_name, head_output_directory, params, supercell=None, seed=False):
    """
    Collates the structures to be fitted into the working directory, creates the lammps inputs and runs the optimiser to fit to the designated parameters. Calls another function to save the data in the appropriate output directory.
    Args:
        fits (int): Number of fits to run.
        structures (int): Total number of structures in the training set.
        structures_in_fit (int): Number of structures to fit to.
        head_directory_name (str): Name of the main results directory.
        head_output_directory (str): Name of the main output directory for the cross-validation.
        params (dict(dict)): Setup dictionary containing the inputs for coreshell, charges, masses, potentials, and core-shell springs.
        supercell (optional:list(int) or list(list(int))): 3 integers defining the cell increase in x, y, and z for all structures, or a list of lists where each list is 3 integers defining the cell increase in x, y, z, for each individual structure in the fitting process. i.e. all increase by the same amount, or each structure increased but different amounts. Default=None.
        
    Returns:
        None
    """
    for potential_file in sorted(glob.glob('{}/*/potentials.json'.format(head_directory_name))):
        with open(potential_file, 'r') as f:
            potentials = json.load(f)
        structure_nums = potential_file.replace('/potentials.json', '').replace('{}/'.format(head_directory_name),'')
        structure_nums = np.array([int(num) for num in structure_nums.split('-')])
        include_labels = list(potentials.keys())
        include_values = list(potentials.values())
        indv_output_directory = fit_out.create_directory(head_output_directory, 'p{}'.format('-'.join([str(num) for num in structure_nums])))
        sets_of_structures = validation_sets(fits, structures, structures_in_fit, structure_nums, seed=False)
        for structs in sets_of_structures:
            fit_data = FitModel.collect_info(params, structs, supercell=supercell)
            dft_f, ip_f, dft_s, ip_s = fit_out.extract_stresses_and_forces(fit_data, include_values, include_labels)
            error = chi_squared_error(dft_f, ip_f, dft_s, ip_s)
            save_cv_data(indv_output_directory, structs, error, dft_f, ip_f, dft_s, ip_s)
            fit_data.reset_directories()
            
def setup_error_dict(cv_directory):
    """
    Returns a dictionary of the chi squared errors associated with each fit in the group.
    Args:
        cv_directory (str): Name of the individual cross-validation output directory. 
    Returns:
        error_dict (dict): keys are the structurs in the fit separated by a dash, i.e. '1-2-5', and values are the chi squared error of that fit.
    """
    error_dict = {}
    for error_file in sorted(glob.glob('{}/*error.dat'.format(cv_directory))):
        cv_struct_nums = error_file.replace(cv_directory, '').replace('/s','').replace('_error.dat', '')
        error = float(np.loadtxt(error_file))
        error_dict.update( {cv_struct_nums:error})
    return error_dict



def plot_cross_validation(error_dict, cv_directory, head_output_directory, xlabel_rotation=50, title='default', save=True):
    """
    Plots the chi squared errors for each fit in a sequence of fits, with the x-axis being the fit (labeled with the structure numbers in the fit) and the y-axis being the chi squared error.
    Args:
        error_dict (dict): Keys are the structures in the fit separated by a dash, i.e. '1-2-5', and values are the chi squared error of that fit.
        cv_directory (str): The cross-validation directory.
        head_output_directory (str): Directory pathway to main output directory.
        xlabel_rotation (optional: int): Rotation applied to the x-axis labels. Default=50.
        title (optional: str): plot title, default='{} structure fit errors ($\chi^2$)'.format(structures_in_fit).
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    plt.scatter(*zip(*sorted(error_dict.items())))
    plt.xticks(rotation=xlabel_rotation)
    plt.xlabel('cross-validation structure numbers')
    plt.ylabel('$\chi^2$ error')
    
    if title is 'default':
        plt.title('Potential {} cross-validation fit errors ($\chi^2$)'.format(cv_directory.replace('{}/p'.format(head_output_directory),'')))
    elif title is not None and title is not 'default':
        plt.title('{}'.format(title))
        
    if save is True:
        plt.savefig('{}/p{}_cv_fit_errors.png'.format(head_output_directory,cv_directory.replace('{}/p'.format(head_output_directory),'')),dpi=500, bbox_inches = "tight")
    plt.show()
