#! /usr/bin/env python3
import pytest
import numpy as np
from mock import patch, call
from popoff.plotting import (setup_error_dict, setup_potentials_dict, plot_errors,
                             plot_parameters, plot_forces, plot_stresses)

def test_setup_error_dict_in_plotting():
    head_dir = 'test_files/test_outputs'
    error_dict = setup_error_dict(head_dir)
    assert error_dict == {'1': 0.1, '2': 0.2}
    
def test_setup_potentials_dict_in_plotting():
    head_dir = 'test_files/test_outputs'
    pot_dict = setup_potentials_dict(head_dir)
    assert pot_dict == {'q_scaling': [('1', 0.5), ('2', 0.6)], 'Li_O_a': [('1', 1.0), ('2', 2.0)], 'Li_O_rho': [('1', 0.1), ('2', 0.2)]}

@patch('popoff.plotting.plt')
def test_plot_errors_in_plotting(mock_plt):
    error_dict = {'1': 0.1, '2': 0.2}
    plot_errors(error_dict, 'test', xlabel_rotation=50, title='default', save=True)
    calls_command = [call.scatter(*zip(*sorted(error_dict.items()))),
                     call.xticks(rotation=50),
                     call.xlabel('fitted structure numbers'),
                     call.ylabel('$\chi^2$ error'),
                     call.title('fit errors ($\chi^2$)'),
                     call.savefig('test/fit_errors.png',dpi=500, bbox_inches = "tight"),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)
    plot_errors(error_dict, 'test', xlabel_rotation=50, title='testing', save=False)
    calls_command = [call.scatter(*zip(*sorted(error_dict.items()))),
                     call.xticks(rotation=50),
                     call.xlabel('fitted structure numbers'),
                     call.ylabel('$\chi^2$ error'),
                     call.title('testing'),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)
    
@patch('popoff.plotting.plt')
def test_plot_parameters_in_plotting(mock_plt):
    pot_dict = {'q_scaling': [('1', 0.5), ('2', 0.6)], 'Li_O_a': [('1', 1.0), ('2', 2.0)], 'Li_O_rho': [('1', 0.1), ('2', 0.2)]}
    plot_parameters(pot_dict, 'test', xlabel_rotation=50, title='default', save=True)
    calls_command = [call.scatter(['1','2'],[0.5,0.6]),
                     call.xticks(rotation=50),
                     call.xlabel('fitted structure numbers'),
                     call.ylabel('q_scaling value'),
                     call.title('q_scaling fitted parameter'),
                     call.savefig('test/fitted_q_scaling.png',dpi=500, bbox_inches = "tight"),
                     call.show(),
                     call.scatter(['1', '2'], [1.0, 2.0]),
                     call.xticks(rotation=50),
                     call.xlabel('fitted structure numbers'),
                     call.ylabel('Li_O_a value'),
                     call.title('Li_O_a fitted parameter'),
                     call.savefig('test/fitted_Li_O_a.png', dpi=500, bbox_inches='tight'),
                     call.show(),
                     call.scatter(['1', '2'], [0.1, 0.2]),
                     call.xticks(rotation=50),
                     call.xlabel('fitted structure numbers'),
                     call.ylabel('Li_O_rho value'),
                     call.title('Li_O_rho fitted parameter'),
                     call.savefig('test/fitted_Li_O_rho.png', dpi=500, bbox_inches='tight'),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)

@patch('popoff.plotting.plt')
def test_plot_forces_in_plotting(mock_plt):
    dft_forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ip_forces = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    x = np.arange(0, len(dft_forces.flatten()))
    plot_forces(dft_forces, ip_forces,'test', 'testing', alpha=0.02, save=True)
    np.testing.assert_array_equal(x, np.array([0,1,2,3,4,5]))
    np.testing.assert_array_equal(dft_forces.flatten(), np.array([0.1,0.2,0.3,0.4,0.5,0.6]))
    np.testing.assert_array_equal(ip_forces.flatten(), np.array([0.2,0.3,0.4,0.5,0.6,0.7]))
    calls_command = [call.legend(),
                     call.xlabel('atom number'),
                     call.ylabel('force'),
                     call.savefig('test/testing_forces.png',dpi=500, bbox_inches = "tight"),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)

@patch('popoff.plotting.plt')
def test_plot_stresses_in_plotting(mock_plt):
    dft_stresses = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                             1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
                             2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0])
    ip_stresses = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                             1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
                             2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0])
    dft_y = dft_stresses.reshape(5, 6)
    ip_y = ip_stresses.reshape(5,6) 
    x= ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'ZX']    
    plot_stresses(dft_stresses, ip_stresses,'test', 'testing', save=True)    
    calls_command = [call.ylabel('stress tensor'),
                     call.text(1.5, 90, 'stress tensor error: 0.00000'),
                     call.text(4.5, 90, 'DFT', color='tab:blue'),
                     call.text(4.5, 75, 'IP', color='tab:orange'),
                     call.savefig('test/testing_stresses.png', dpi=500, bbox_inches='tight'),
                     call.show()]
    mock_plt.assert_has_calls(calls_command)
    i = 0
    for dy, iy in zip(dft_y, ip_y):
        np.testing.assert_array_equal(dy, dft_stresses[i:i+6])
        np.testing.assert_array_equal(iy, ip_stresses[i:i+6])
        i+=6
