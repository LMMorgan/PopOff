#! /usr/bin/env python3
import pytest
import numpy as np
from unittest import mock
from fitting_code import FitModel
from lammps_data import LammpsData
from atom_types import AtomType
from input_checker import (check_coreshell, check_scaling_limits, check_spring,
                           check_buckingham, setup_error_checks)

@pytest.fixture
def fit_data(mock_lammps_data):
    fit_data = mock.Mock(FitModel)
    fit_data.lammps_data = [mock_lammps_data]
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