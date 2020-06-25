#! /usr/bin/env python3
import pytest
import numpy as np
import mock
from fitting_code import FitModel
from lammps_data import LammpsData
from atom_types import AtomType
from input_checker import (check_coreshell, check_scaling_limits, check_spring,
                           check_buckingham, setup_error_checks)

@pytest.fixture
def params():
    params = {}
    params['core_shell'] = { 'Li': False, 'Ni': False, 'O': True }
    params['charges'] = {'Li': +1.0,
                         'Ni': +3.0,
                         'O': {'core':  -2.0,
                               'shell': 0.0}} 
    params['masses'] = {'Li': 6.941,
                        'Ni': 58.6934,
                        'O': {'core': 14.3991,
                              'shell': 1.5999} }
    params['potentials'] = {'Li-O': [100.00, 0.1, 0.0],
                            'Ni-O': [1000.00, 0.2, 0.0],
                            'O-O':  [10000.00, 0.3, 0.0]}
    params['cs_springs'] = {'O-O' : [10.0, 0.0]}
    return params

@pytest.fixture
def labels():
    return ['dq_O', 'q_scaling', 'O-O spring', 'Li_O_a', 'Li_O_rho', 'Ni_O_a', 'Ni_O_rho', 'O_O_a', 'O_O_rho']

@pytest.fixture
def bounds():
    return [(0.01, 4), (0.3,1.0), (10.0,150.0), (100.0,50000.0), (0.01,1.0), (100.0,50000.0), (0.01,1.0), (150.0,50000.0), (0.01,1.0)]

@pytest.fixture
def atom_types():
    atom_types = []
    atom_types.append(AtomType(1, 'Li', 'Li', 1.555, 1.0))
    atom_types.append(AtomType(2, 'Ni', 'Ni', 56.111, 3.0))
    atom_types.append(AtomType(3, 'O core', 'O', 14.001, -0.00, 'core'))
    atom_types.append(AtomType(3, 'O shell', 'O', 1.998, -2.00, 'shell'))
    return atom_types

@pytest.fixture
def lammps_data(atom_types):
    lammps_data = mock.Mock(LammpsData)
    lammps_data.atom_types = atom_types
    return lammps_data

@pytest.fixture
def fit_data(lammps_data):
    fit_data = mock.Mock(FitModel)
    fit_data.lammps_data = [lammps_data]
    return fit_data
    
@pytest.mark.parametrize(  'label', [('dq_Li'),('dq_Ba')])
def test_typeerror_for_label_in_check_coreshell(label, bounds, fit_data):
    with pytest.raises(TypeError):
        check_coreshell(label, bounds, fit_data) 
        
@pytest.mark.parametrize( 'bounds', [((-0.9,0.9)),((0,-0.9)),((0.9,1.1)),((1.1,7)),((0.9,0.8))])
def test_valueerror_for_bounds_in_check_coreshell(bounds, fit_data):
    with pytest.raises(ValueError):
        check_coreshell("dq_O", bounds, fit_data)            

@pytest.mark.parametrize( 'bounds', [((-0.9,0.9)),((0,-0.9)),((0.9,1.1)),((1.1,1.1)),((0.9,0.8))])
def test_valueerror_for_bounds_in_check_scaling_limits(bounds):
    with pytest.raises(ValueError):
        check_scaling_limits(bounds) 

@pytest.mark.parametrize( 'bounds', [((-0.9,0.9)),((0,-0.9)),((0.9,0.8))])
def test_valueerror_for_bounds_in_check_spring(bounds, params):
    with pytest.raises(ValueError):
        check_spring('O-O spring', bounds, params)

@pytest.mark.parametrize( 'label', [('Ni-Ni spring'),('Li-Li spring'),('Li-O spring')])
def test_typeerror_for_label_in_check_spring(label, bounds, params):
    with pytest.raises(TypeError):
        check_spring(label, bounds, params)


@pytest.mark.parametrize( 'bounds', [((-0.9,0.9)),((0,-0.9)),((0.9,0.8))])
def test_valueerror_for_bounds_in_check_buckingham(bounds, params):
    with pytest.raises(ValueError):
        check_buckingham('Li_O_a', bounds, params)
        
@pytest.mark.parametrize( 'label', [('Ni_O_a'),('Li_B_a'),('Li_O_aa')])
def test_typeerror_for_bounds_in_check_buckingham(label, bounds, params):
    with pytest.raises(TypeError):
        check_buckingham(label, bounds, params)
        
def test_indexerror_in_setup_error_checks(labels, bounds, params):
    fit_data = mock.Mock(FitModel)
    labels = labels[:-1]
    with pytest.raises(IndexError):
        setup_error_checks(labels, bounds, fit_data, params)
        
def test_typeerror_in_setup_error_checks(labels, bounds, params):
    fit_data = mock.Mock(FitModel)
    labels[0] = 'test'
    with pytest.raises(TypeError):
        setup_error_checks(labels, bounds, fit_data, params)