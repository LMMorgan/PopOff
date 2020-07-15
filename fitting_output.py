import os
import numpy as np
import json

def create_directory(head_directory_name, local_struct_dir):
    """
    Returns a string of the joint file path to the output directory and creates the directory pathway.
    Args:
        head_directory_name (str): Name of the main output directory. 
        local_struct_dir (str): fitted structure numbers joined by dashes.
    Returns:
        directory (str): directory pathway to output directory.
    """
    directory = os.path.join(head_directory_name, local_struct_dir)
    if os.path.isdir(directory):
        raise FileExistsError('Results directory already exists. Please either delete/rename the directory or rename the output location.')
    else:
        os.makedirs(directory)
        return directory
    
def extract_stresses_and_forces(fit_data, values, args):
    """
    Returns the extracted forces and stress tensors for the dft training structures and then runs a shell relaxation (if core-shell) and single point canculation to get the forces and stress tensors using the fitted potential.
    Args:
        fit_data (obj(FitModel)): all structural data and associated properties defined, with methods for implementing the fitting process using LAMMPS. 
        args (list(str)): Keys relating to the fitting parameters for the system, such as charge, springs, and buckingham parameter
        values (list(float)): Values relating to the fitting arguments passes in.
    Returns:
        dft_forces (np.array): 3D array of dft forces in x,y,z for each atom in each structure.
        ip_forces (np.array): 3D array of fitted interatomic potential forces in x,y,z for each atom in each structure.
        dft_stresses (np.array): 3D array of dft stress tensors in each structure.
        ip_stresses (np.array): 3D array of fitted interatomic potential stress tensors in each structure.
    """
    fit_data.init_potential(values, args)
    ip_forces, ip_stresses = fit_data.get_forces_and_stresses()
    dft_forces = fit_data.expected_forces()
    dft_stresses = fit_data.expected_stresses()
    return dft_forces, ip_forces, dft_stresses, ip_stresses

def save_data(struct_directory, labels, fit_output, dft_forces, ip_forces, dft_stresses, ip_stresses):
    """
    Collates the output data from the fit and saves it to different files within the designated output directory.
    Args:
        struct_directory (str): directory pathway to local output directory.
        labels (list(str)): List of parameters to be fitted.
        fit_output (OptimizeResult): resulting values and fitting data from scipy. Full type is scipy.optimize.optimize.OptimizeResult.
        dft_forces (np.array): 3D array of dft forces in x,y,z for each atom in each structure.
        ip_forces (np.array): 3D array of fitted interatomic potential forces in x,y,z for each atom in each structure.
        dft_stresses (np.array): 3D array of dft stress tensors in each structure.
        ip_stresses (np.array): 3D array of fitted interatomic potential stress tensors in each structure.
    Returns:
        None
    """
    np.savetxt('{}/dft_forces.dat'.format(struct_directory), dft_forces, fmt='%.10e', delimiter=' ')
    np.savetxt('{}/ip_forces.dat'.format(struct_directory), ip_forces, fmt='%.10e', delimiter=' ')
    np.savetxt('{}/dft_stresses.dat'.format(struct_directory), dft_stresses, fmt='%.10e', delimiter=' ')
    np.savetxt('{}/ip_stresses.dat'.format(struct_directory), ip_stresses, fmt='%.10e', delimiter=' ')
    with open('{}/error.dat'.format(struct_directory), 'w') as f:
        f.write(str(fit_output.fun))
    potential_dict = {k:v for k, v in zip(labels, fit_output.x)}
    with open('{}/potentials.json'.format(struct_directory), 'w') as f:
        json.dump(potential_dict, f)  
        
