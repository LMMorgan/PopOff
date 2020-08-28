#! /usr/bin/env python3
import pytest
import numpy as np
import json
from scipy import optimize
from mock import Mock, patch, call, mock_open
from lammps_potenial_fitting.fitting_code import FitModel
from lammps_potenial_fitting.fitting_output import (create_directory,
                                                    extract_stresses_and_forces,
                                                    save_data)

@pytest.fixture
def mock_forces():
    ip_forces = np.ones((10,3))
    dft_forces = np.zeros((10,3))
    return ip_forces, dft_forces

@pytest.fixture
def mock_stresses():
    ip_stresses = np.array([[0,1,2],[3,4,5],[6,7,8]])
    dft_stresses = np.array([[1,2,3],[4,5,6],[7,8,9]])
    return ip_stresses, dft_stresses


def test_fileexistserror_for_directory_in_create_directory():
    head_directory_name = 'test_files'
    local_struct_dir = ""
    with pytest.raises(FileExistsError):
        create_directory(head_directory_name, local_struct_dir)

@patch('lammps_potenial_fitting.fitting_output.os.path.isdir')
@patch('lammps_potenial_fitting.fitting_output.os.makedirs')
def test_output_for_directory_in_create_directory(mock_makedirs, mock_isdir):
    head_directory_name = 'test_files'
    local_struct_dir = "test2"
    mock_isdir.return_value = False
    create_directory(head_directory_name, local_struct_dir)
    mock_makedirs.assert_called_with('test_files/test2')
    
def test_asserts_for_extract_stresses_and_forces(labels, mock_forces, mock_stresses):
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mock_fit_data = Mock(spec=FitModel)
    mock_fit_data.get_forces_and_stresses = Mock()
    mock_fit_data.get_forces_and_stresses.return_value = [mock_forces[0], mock_stresses[0]]
    mock_fit_data.expected_forces.return_value = mock_forces[1]
    mock_fit_data.expected_stresses.return_value = mock_stresses[1]
    dft_forces, ip_forces, dft_stresses, ip_stresses = extract_stresses_and_forces(mock_fit_data, values, labels)
    assert np.allclose(dft_forces, np.zeros((10,3)))
    assert np.allclose(ip_forces, np.ones((10,3)))
    assert np.allclose(dft_stresses, np.array([[1,2,3],[4,5,6],[7,8,9]]))
    assert np.allclose(ip_stresses, np.array([[0,1,2],[3,4,5],[6,7,8]]))


@patch('lammps_potenial_fitting.fitting_output.json.dump')
@patch("builtins.open", new_callable=mock_open)
@patch('lammps_potenial_fitting.fitting_output.np.savetxt')
def test_saves_for_save_data(mock_savetxt, mock_open, mock_json, labels, mock_forces, mock_stresses):
    struct_dir = 'test_files'
    fit_output = Mock(spec=optimize.OptimizeResult)
    fit_output.fun = 1.0
    fit_output.x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    potential_dict = {k:v for k, v in zip(labels, fit_output.x)}
    save_data(struct_dir, labels, fit_output, mock_forces[1], mock_forces[0],
              mock_stresses[1], mock_stresses[0])
    calls_save = [call('test_files/dft_forces.dat', mock_forces[1], fmt='%.10e', delimiter=' '),
                  call('test_files/ip_forces.dat', mock_forces[0], fmt='%.10e', delimiter=' '),
                  call('test_files/dft_stresses.dat', mock_stresses[1], fmt='%.10e', delimiter=' '),
                  call('test_files/ip_stresses.dat', mock_stresses[0], fmt='%.10e', delimiter=' ')]
    mock_savetxt.assert_has_calls(calls_save)

    calls_open = [call('test_files/error.dat', 'w'),
                  call().__enter__(),
                  call().write(str(fit_output.fun)),
                  call().__exit__(None, None, None)]
    mock_open.assert_has_calls(calls_open)

    mock_open.assert_called_with('test_files/potentials.json', 'w')
    mock_json.assert_called_with(potential_dict, mock_open.return_value.__enter__.return_value)