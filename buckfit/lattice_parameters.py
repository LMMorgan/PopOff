from buckfit.fitting_code import FitModel
import numpy as np
import os
import json
import glob
import lammps
import buckfit.fitting_output as fit_out
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

def get_lattice(fit_data, values, args):
    """
    Returns relaxed structures with the parameters from the fitted potential implemented.
    Args:
        fit_data (obj(FitModel)): all structural data and associated properties defined, with methods for implementing the fitting process using LAMMPS. 
        values (list(float)): Values relating to the fitting arguments passes in.
        args (list(str)): Keys relating to the fitting parameters for the system, such as charge, springs, and buckingham parameters.
    Returns:
        lmp (obj): Lammps object with structure and specified commands implemented, after MD minimisation and relaxation.
    """
    fit_data.init_potential(values, args)
    lmp = fit_data.get_lattice_params()
    return lmp

def differences(lattice_params, ref):
    """
    Returns relaxed structures with the parameters from the fitted potential implemented.
    Args:
        lattice_params (np.array): lattice parameters (a,b,c) and volume of fitted MD parameters.
        ref (np.array): lattice parameters (a,b,c) and volume of DFT reference values.
    Returns:
        (np.array): Percentage differences between MD (lattice_params) and DFT (ref) lattice parameters.
    """
    return ((lattice_params-ref)/ref)*100


def run_relaxation(structures, directory, output_directory, params, labels, ref_DFT, supercell=None):
    """
    For each potential in the subdirectories of the working directory, the potential is taken and used to relax every structure in the training set (independently), returning the resulting lattice parameters for each. The output in saved in datafiles in the 'lattice_parameters' subdirectory.
    Args:
        structures (int): Total number of structures in the training set.
        directory (str): Name of the main results directory.
        output_directory (str): Name of the main output directory for the lattice_parameters.
        params (dict(dict)): Setup dictionary containing the inputs for coreshell, charges, masses, potentials, and core-shell springs.
        labels (list(str)): lattice parameter labels and units.
        ref_DFT (np.array): 1D array of reference lattice parameter values in same order as labels.
        supercell (optional:list(int) or list(list(int))): 3 integers defining the cell increase in x, y, and z for all structures, or a list of lists where each list is 3 integers defining the cell increase in x, y, z, for each individual structure in the fitting process. i.e. all increase by the same amount, or each structure increased but different amounts. Default=None.
    Returns:
        percent_difference (np.array): percentage difference for each lattice parameter in each strucutre of the training set, from the reference lattice parameters.
    """    
    for potential_file in sorted(glob.glob('{}/*/potentials.json'.format(directory))):
        with open(potential_file, 'r') as f:
            potentials = json.load(f)
        pot_structures = potential_file.replace('/potentials.json', '').replace('{}/'.format(directory),'')
        calculated_parameters = []
        percent_difference = []
        for structure in range(structures):
            fit_data = FitModel.collect_info(params, [structure], supercell=supercell)
            lmp = get_lattice(fit_data, potentials.values(), potentials.keys())
            lammps = lmp[0]
            lattice_params = np.array([lammps.box.lengths[0], lammps.box.lengths[1], lammps.box.lengths[2], lammps.box.volume])
            calculated_parameters.append(lattice_params)
            diffs = differences(lattice_params, ref_DFT)
            percent_difference.append(diffs)
            fit_data.reset_directories()
        np.savetxt('{}/{}_lattice_values.dat'.format(output_directory, pot_structures), calculated_parameters, header = ' '.join(labels))
        np.savetxt('{}/{}_lattice_diffs.dat'.format(output_directory, pot_structures), percent_difference, header = ' '.join(labels))
    return np.array(percent_difference)


def plot_lattice_params(labels, calculated_parameters, ref_DFT, output_directory, pot_structures, save=True):
    """
    Plots the lattice parameters for each structure in the trainning set with the given potential along with the reference lattice parameters.
    Args:
        labels (list(str)): lattice parameter labels and units.
        calculated_parameters (np.array): lattice parameters for each strucutre of the training set using the given potential.
        ref_DFT (np.array): 1D array of reference lattice parameter values in same order as labels.
        output_directory (str): Name of the main output directory for the lattice_parameters.
        pot_structures (str): The structures in the fitted potential separated by a dash, i.e. '1-2-5'
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    fig, axs = plt.subplots(1, len(labels), figsize=(10,8))
    plt.subplots_adjust(wspace=0.8)
    for i, label in enumerate(labels):
        x = np.random.rand(len(calculated_parameters))
        axs[i].scatter(x, calculated_parameters.transpose()[i])
        axs[i].scatter(0.5, ref_DFT[i])
        axs[i].set_xlim(left=-1.5, right=2.5)
        axs[i].set_xticks([])
        axs[i].set_title('{}'.format(label))
    if save is True:
        plt.savefig('{}/{}_lattice_comparison.png'.format(output_directory, pot_structures),dpi=500, bbox_inches = "tight")
    plt.show()    

def _scatter_plot(x, y, ref_DFT, label, gs):
    """
    Sets the conditions and plots the scatter plot.
    Args:
        x (np.array): random values between 0 and 1 up to len(calculated_parameters) to give a spread, making each point easier to see.
        y (np.array): lattice parameters for each strucutre of the training set using the given potential.
        ref_DFT (np.array): 1D array of reference lattice parameter values in same order as labels.
        labels (list(str)): lattice parameter labels and units.
        gs (gridspec.GridSpec): 1 by 2 gridspec for formatting of plot.
    Returns:
        axs (np.array): array of matplotlib subplots (AxesSubplot) objects.
    """    
    axs = plt.subplot(gs)
    axs.scatter(x, y)
    axs.scatter(0.5, ref_DFT)
    axs.set_xlim(left=-1.5, right=2.5)
    axs.set_xticks([])
    axs.set_title('{}'.format(label))
    plt.subplots_adjust(wspace=0)
    return axs

def _distribution_plot(y, gs):
    """
    Sets the conditions and plots the distribution plot (kdeplot).
    Args:
        y (np.array): lattice parameters for each strucutre of the training set using the given potential.
        gs (gridspec.GridSpec): 1 by 2 gridspec for formatting of plot.
    Returns:
        axs (np.array): array of matplotlib subplots (AxesSubplot) objects.
    """ 
    axs = plt.subplot(gs)
    axs.axis('off')
    axs = sns.kdeplot(y, vertical=True, shade=True)
    return axs

def _y_limits(ref_DFT, y):
    """
    Sets the y_limits of each plot to encompass both the data points and reference data, making sure the distributions are to the same scale.
    Args:
        ref_DFT (np.array): 1D array of reference lattice parameter values in same order as labels.
        y (np.array): lattice parameters for each strucutre of the training set using the given potential.
    Returns:
        ylims (tuple): Contains the minimum and maximum limits for the y-axis.
    """ 
    if ref_DFT <= y.min():
        ylims = (ref_DFT-(ref_DFT*0.002), y.max()+(y.max()*0.002))
    elif ref_DFT >= y.max():
        ylims = (y.min()-(y.min()*0.002), ref_DFT+(ref_DFT*0.002))
    else:
        ylims = (y.min()-(y.min()*0.002), y.max()+(y.max()*0.002))
    return ylims    

    
def plot_lattice_params_with_distributions(labels, calculated_parameters, ref_DFT, output_directory, pot_structures, save=True):
    """
    Plots the lattice parameters for each structure in the trainning set with the given potential along with the reference lattice parameters. This also has the distribution plotted along the y-axis.
    Args:
        labels (list(str)): lattice parameter labels and units.
        calculated_parameters (np.array): lattice parameters for each strucutre of the training set using the given potential.
        ref_DFT (np.array): 1D array of reference lattice parameter values in same order as labels.
        output_directory (str): Name of the main output directory for the lattice_parameters.
        pot_structures (str): The structures in the fitted potential separated by a dash, i.e. '1-2-5'
        save (optional: bool): True to save the plot, Flase to not save. Default=True.
    Returns:
        None
    """
    short_labels = ['a', 'b', 'c', 'volume']
    for i in range(len(labels)):
        fig, axs = plt.subplots(1, 2, figsize=(4,8), sharey=True)
        plt.subplots_adjust(wspace=0)
        x = np.random.rand(len(calculated_parameters))
        y = calculated_parameters.T[i]
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        axs[0] = _scatter_plot(x,y,ref_DFT[i],labels[i],gs[0])
        axs[1] = _distribution_plot(y, gs[1])
        ylims = _y_limits(ref_DFT[i], y)
        axs[0].set_ylim(ylims)
        axs[1].set(ylim=ylims)
        if save is True:
            plt.savefig('{}/{}_lattices_distribution_{}.png'.format(output_directory, pot_structures, short_labels[i]),dpi=500, bbox_inches = "tight")
        plt.show()