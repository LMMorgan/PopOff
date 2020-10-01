#! /usr/bin/env python3
import pytest
import numpy as np
import lammps
from mock import Mock, patch, call, mock_open, MagicMock
# from lammps_potenial_fitting.fitting_code import FitModel
from lammps_potenial_fitting.lattice_parameters import get_lattice, differences, run_relaxation,plot_lattice_params 

@patch('lammps_potenial_fitting.lattice_parameters.FitModel.get_lattice_params') 
@patch('lammps_potenial_fitting.lattice_parameters.FitModel.init_potential')   
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

# @patch('lammps_potenial_fitting.lattice_parameters.differences')    
# @patch('lammps_potenial_fitting.lattice_parameters.get_lattice')
# @patch('lammps_potenial_fitting.lattice_parameters.FitModel.reset_directories')
# @patch('lammps_potenial_fitting.lattice_parameters.FitModel.collect_info')
# def test_run_relaxation_in_lattice_parameters(fit_info, fit_reset, get_lattice, differences, fit_data, params, labels):
#     head_dir = 'test_files/test_outputs'
#     head_out = 'test_files/test_outputs/outputs'
#     ref = np.array([10.1, 10.0, 10.2, 1030.2])
#     fit_info.return_value = fit_data
#     lammps_lengths = Mock(spec=lammps.Lammps)
#     lammps_lengths.box.lengths = [10.0, 10.0, 10.0]
#     lammps_lengths.box.volume = 1000.0
#     get_lattice.return_value = [lammps_lengths]
#     per_diff = run_relaxation(15, head_dir, head_out, params, labels, ref, supercell=None)
#     fit_info.assert_called_with(params, [14], supercell=None)
#     fit_reset.assert_called_with()

@patch("lammps_potenial_fitting.lattice_parameters.np.random.rand")
@patch("lammps_potenial_fitting.lattice_parameters.plt")
def test_plot_lattice_params_in_lattice_parameters(mock_plt, rand):
    labels = ['A ang', 'rho ang', 'C ang']
    calc_params = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ref = np.array([0.1, 0.2, 0.3])
    head_dir = 'test_files/test_outputs'
    rand.return_value = [0.4, 0.6]
    plot_lattice_params(labels, calc_params, ref, head_dir, '1-2', save=True)
    calls_command = [call.subplots(1, 3, figsize=(10,8)),
                     call.subplots_adjust(wspace=0.8),
                     call.scatter([0.4, 0.6], [0.1,0.4]),
                     call.scatter(0.5, 0.1),
                     call.set_xlim(left=-1.5, right=2.5),
                     call.set_xticks([]),
                     call.set_title('{}'.format(label)),
                     call.savefig(f'{head_dir}/1-2_lattice_comparison.png',dpi=500, bbox_inches = "tight"),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)
    

    
    
    
