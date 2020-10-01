import json
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def setup_error_dict(head_directory_name):
    """
    Returns a dictionary of the chi squared errors associated with each fit in the group.
    Args:
        head_directory_name (str): Name of the main output directory. 
    Returns:
        error_dict (dict): keys are the structurs in the fit separated by a dash, i.e. '1-2-5', and values are the chi squared error of that fit.
    """
    error_dict = {}
    for error_file in sorted(glob.glob('{}/*/error.dat'.format(head_directory_name))):
        error = float(np.loadtxt(error_file))
        structure_num = error_file.replace('/error.dat', '').replace('{}/'.format(head_directory_name), '')
        error_dict.update( {structure_num:error})
    return error_dict

def setup_potentials_dict(head_directory_name):
    """
    Returns a dictionary of the chi squared errors associated with each fit in the group.
    Args:
        head_directory_name (str): Name of the main output directory. 
    Returns:
        potential_dicts (list(dict)): keys are the potential parameter labels (str) and values are a tuple of the fitted structure numbers (str) and the associated parameter value (float).
    """
    list_of_potential_dicts = []
    for potential_file in sorted(glob.glob('{}/*/potentials.json'.format(head_directory_name))):
        with open(potential_file, 'r') as f:
            potentials = json.load(f)
        structure_num = potential_file.replace('/potentials.json', '').replace('{}/'.format(head_directory_name), '')
        potentials.update((k, (structure_num,v) ) for k,v in potentials.items()) #Remove int if directory naming system changes to include strings. This will change the order of the structures though.
        list_of_potential_dicts.append(potentials)
    potentials_dict = {}
    for key in list_of_potential_dicts[0].keys():
        potentials_dict[key] = [potentials_dict[key] for potentials_dict in list_of_potential_dicts]
    return potentials_dict
    
def plot_errors(error_dict, output_directory, xlabel_rotation=50, title='default', save=True):
    """
    Plots the chi squared errors for each fit in a sequence of fits, with the x-axis being the fit (labeled with the structure numbers in the fit) and the y-axis being the chi squared error.
    Args:
        error_dict (dict): Keys are the structurs in the fit separated by a dash, i.e. '1-2-5', and values are the chi squared error of that fit.
        output_directory (str): Directory pathway to output directory.
        xlabel_rotation (optional: int): Rotation applied to the x-axis labels. Default=50.
        title (optional: str): plot title, default='fitted errors ($\chi^2$)'.
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    plt.scatter(*zip(*sorted(error_dict.items())))
    plt.xticks(rotation=xlabel_rotation)
    plt.xlabel('fitted structure numbers')
    plt.ylabel('$\chi^2$ error')
    
    if title is 'default':
        plt.title('fit errors ($\chi^2$)')
    elif title is not None and title is not 'default':
        plt.title('{}'.format(title))
        
    if save is True:
        plt.savefig('{}/fit_errors.png'.format(output_directory),dpi=500, bbox_inches = "tight")
    plt.show()

def plot_parameters(potentials_dict, output_directory, xlabel_rotation=50, title='default', save=True):
    """
    Plots the potential parameters for each fit in a sequence of fits, with the x-axis being the fit (labeled with the structure numbers in the fit) and the y-axis being the parameter value.
    Args:
        potentials_dict (dict): Keys are the potential parameter labels (str) and values are a tuple of the fitted structure numbers (str) and the associated parameter value (float).
        output_directory (str): Directory pathway to output directory.
        xlabel_rotation (optional: int): Rotation applied to the x-axis labels. Default=50
        title (optional: str): Plot title, default=''{} fitted parameter'.format(k)'.
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    for k,v in potentials_dict.items():
        x_val = [x[0] for x in v]
        y_val = [y[1] for y in v]
        plt.scatter(x_val,y_val)
        plt.xticks(rotation=xlabel_rotation)
        plt.xlabel('fitted structure numbers')
        plt.ylabel('{} value'.format(k))
    
        if title is 'default':
            plt.title('{} fitted parameter'.format(k))
        elif title is not None and title is not 'default':
            plt.title('{}'.format(title))
        
        if save is True:
            plt.savefig('{}/fitted_{}.png'.format(output_directory, k) ,dpi=500, bbox_inches = "tight")
        
        plt.show()
            
def plot_forces(dft_forces, ip_forces, output_directory, local_directory, alpha=0.02, save=True):
    """
    Plots the 
    Args:
        dft_forces (np.array): dft forces associated with each atom (x,y,and z) in each structure fitted.
        ip_forces (np.array): Fitted forces associated with each atom (x,y,and z) in each structure fitted.
        output_directory (str): Directory pathway to output directory.
        local_directory (Str): The individual fit directory in a series of fits or the singular fit directory.
        alpha (optional: float): Degree of transprancy for the plot series. Default=0.02.
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    x = np.arange(0, len(dft_forces.flatten()))
    plt.scatter(x, dft_forces.flatten(), label='dft', alpha=alpha)
    plt.scatter(x, ip_forces.flatten(), label='ip', alpha=alpha)
    plt.legend()
    plt.xlabel('atom number')
    plt.ylabel('force')
    # plt.text(0,4.5, 'error: {0:.5f}'.format(np.sum((dft_forces - ip_forces)**2)/ dft_forces.size))
    if save is True:
        plt.savefig('{}/{}_forces.png'.format(output_directory,local_directory),dpi=500, bbox_inches = "tight")
    plt.show()
    
def plot_stresses(dft_stresses, ip_stresses, output_directory, local_directory, save=True):
    """
    Plots the 
    Args:
        dft_stresses (np.array): dft stress tensors associated with each structure fitted.
        ip_stressed (np.array): Fitted stress tensors associated with each structure fitted.
        output_directory (str): Directory pathway to output directory.
        local_directory (Str): The individual fit directory in a series of fits or the singular fit directory.
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    x= ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'ZX']
    dft_y = dft_stresses.reshape(5, 6)
    ip_y = ip_stresses.reshape(5,6)
    for dy, iy in zip(dft_y, ip_y):
        plt.scatter(x, dy, label='dft', color='tab:blue')
        plt.scatter(x, iy , label='ip', color='tab:orange')
    plt.ylabel('stress tensor')
    plt.text(1.5,90, 'stress tensor error: {0:.5f}'.format(np.sum((dft_stresses - ip_stresses)**2)/ 6))
    plt.text(4.5,90, 'DFT', color='tab:blue')
    plt.text(4.5,75, 'IP', color='tab:orange')
    if save is True:
        plt.savefig('{}/{}_stresses.png'.format(output_directory,local_directory),dpi=500, bbox_inches = "tight")
    plt.show()
