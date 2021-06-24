#! /usr/bin/env python3
import pytest
import numpy as np
from lammps import lammps
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import gridspec
from mock import Mock, patch, call, mock_open, MagicMock

from popoff.lattice_parameters import (get_lattice, differences, run_relaxation,
                                       plot_lattice_params, _scatter_plot, _distribution_plot,
                                       _y_limits, plot_lattice_params_with_distributions)

@patch('popoff.lattice_parameters.FitModel.get_lattice_params') 
@patch('popoff.lattice_parameters.FitModel.init_potential')   
def test_get_lattice_in_lattice_parameters(fit_init, lat_params, fit_data, labels, values):
    lat_params.return_value = 'foo'
    lmp = get_lattice(fit_data, values, labels)
    fit_init.assert_called_with(values, labels)
    lat_params.assert_called_with()
    assert lmp == 'foo'

def test_differences_in_lattice_parameters():
    lattice_params = np.array([10.0, 10.0, 10.0, 1000.0])
    ref = np.array([10.1, 10.0, 10.2, 1030.2])
    diff = differences(lattice_params, ref)
    np.testing.assert_almost_equal(diff, np.array([-0.99009901,0.0,-1.96078431, -2.93146962]))

@patch('popoff.lattice_parameters.np.savetxt')
@patch('popoff.lattice_parameters.differences')
@patch('popoff.lattice_parameters.get_lattice')
@patch('popoff.lattice_parameters.FitModel')
def test_run_relaxation_in_lattice_parameters(fitmodel, get_lattice, diffs, savetxt, fit_data, params, labels):
    head_dir = 'test_files/test_outputs'
    head_out = 'test_files/test_outputs/outputs'   
    ref = np.array([10.1, 10.0, 10.2, 1030.2])
    fitmodel.collect_info = MagicMock(return_value=fit_data)
    lmp = Mock(lammps())
    lmp.extract_box = MagicMock(return_value = ([0.0,0.0,0.0], [10.0, 10.0, 10.0], 0.0, 0.0, 0.0, [1,1,1], 0)
    lmp.get_thermo("vol") = MagicMock(return_value= 1000.0)


    
    savetxt.return_value = None
    get_lattice.return_value = [lmp]
    diffs.return_value = np.zeros((4))
    fit_data.reset_directories = MagicMock(return_value=None)
    per_diff = run_relaxation(15, head_dir, head_out, params, labels, ref, supercell=None)
    fitmodel_calls = [call.collect_info(params,[0],supercell=None), call.collect_info(params,[1],supercell=None),
                      call.collect_info(params,[2],supercell=None), call.collect_info(params,[3],supercell=None),
                      call.collect_info(params,[4],supercell=None), call.collect_info(params,[5],supercell=None),
                      call.collect_info(params,[6],supercell=None), call.collect_info(params,[7],supercell=None),
                      call.collect_info(params,[8],supercell=None), call.collect_info(params,[9],supercell=None),
                      call.collect_info(params,[10],supercell=None), call.collect_info(params,[11],supercell=None),
                      call.collect_info(params,[12],supercell=None), call.collect_info(params,[13],supercell=None),
                      call.collect_info(params,[14],supercell=None)]
    fitmodel.assert_has_calls(fitmodel_calls)
    np.testing.assert_almost_equal(per_diff,np.zeros((15,4)))

@patch('popoff.lattice_parameters.np.random.rand')
@patch('popoff.lattice_parameters.plt')
def test_plot_lattice_params_in_lattice_parameters(mock_plt, rand):
    labels = ['A ang', 'rho ang', 'C ang']
    calc_params = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ref = np.array([0.1, 0.2, 0.3])
    head_dir = 'test_files/test_outputs'
    rand.return_value = [0.4, 0.6]
    axs = [Mock(spec=Axes)]*3

    mock_plt.subplots = MagicMock(return_value= (Mock(spec=Figure),axs))
    plot_lattice_params(labels, calc_params, ref, head_dir, '1-2', save=True)
    calls_command = [call.subplots(1, 3, figsize=(10,8)),
                     call.subplots_adjust(wspace=0.8),
                     call.savefig(f'{head_dir}/1-2_lattice_comparison.png',dpi=500, bbox_inches = "tight"),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)

@patch('popoff.lattice_parameters.plt.subplots_adjust')     
@patch('popoff.lattice_parameters.plt.subplot')    
def test_scatter_plot_in_lattice_parameters(mock_subplot, mock_adjust):
    gs = Mock(spec=gridspec.GridSpec)
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([10.0, 10.1, 10.3])
    ref_DFT = np.array([10.0, 10.0, 10.0])
    label = ['a', 'rho', 'c']
    output = _scatter_plot(x,y,ref_DFT,label,gs)
    calls_command = [call(gs),
                     call().scatter(x,y),
                     call().scatter(0.5, ref_DFT),
                     call().set_xlim(left=-1.5, right=2.5),
                     call().set_xticks([]),
                     call().set_title(f"{label}")]
    mock_subplot.assert_has_calls(calls_command)
    mock_adjust.assert_called_with(wspace=0)
         
@patch('popoff.lattice_parameters.plt.subplot')    
def test_distribution_plot_in_lattice_parameters(mock_subplot):
    gs = Mock(spec=gridspec.GridSpec)
    y = np.array([10.0, 10.1, 10.3])
    output = _distribution_plot(y,gs)
    calls_command = [call(gs),
                     call().axis('off')]
    mock_subplot.assert_has_calls(calls_command) 

def test_y_limits_in_lattice_parameters():
    ref_DFT = 10.0
    y_list = [np.array([10.2, 10.2, 10.2]), np.array([8.0, 8.0, 8.0]), np.array([8.0, 10.2, 10.0])]
    output = [(9.98, 10.2204),(7.984, 10.02),(7.984, 10.2204)]
    for i, y in enumerate(y_list):
        ylims = _y_limits(ref_DFT, y)
        assert ylims == output[i]


@patch('popoff.lattice_parameters._y_limits')        
@patch('popoff.lattice_parameters._distribution_plot')
@patch('popoff.lattice_parameters._scatter_plot')         
@patch('popoff.lattice_parameters.gridspec.GridSpec')         
@patch('popoff.lattice_parameters.np.random.rand')  
@patch('popoff.lattice_parameters.plt')    
def test_plot_lattice_params_with_distributions_in_lattice_parameters(mock_plt, rand, gridspec, scatter, distribution, ylims):
    labels = ['A ang', 'rho ang']
    calc_params = np.array([[0.1, 0.2], [0.5, 0.6]])
    ref_DFT = np.array([0.1, 0.2])
    head_dir = 'test_files/test_outputs'
    out_title = '1-2_lattices_distribution'
    rand.return_value = [0.4, 0.6]
    axs = [Mock(spec=Axes)]*3
    mock_plt.subplots = MagicMock(return_value= (Mock(spec=Figure),axs))
    plot_lattice_params_with_distributions(labels, calc_params, ref_DFT, head_dir, '1-2', save=True)  
    calls_command = [call.subplots(1, 2, figsize=(4, 8), sharey=True), call.subplots_adjust(wspace=0),
                     call.savefig(f'{head_dir}/{out_title}_a.png', bbox_inches='tight', dpi=500), call.show(),
                     call.subplots(1, 2, figsize=(4, 8), sharey=True), call.subplots_adjust(wspace=0),
                     call.savefig(f'{head_dir}/{out_title}_b.png', bbox_inches='tight', dpi=500), call.show()]
    mock_plt.assert_has_calls(calls_command)
    gridspec.assert_called_with(1, 2, width_ratios=[2, 1])
