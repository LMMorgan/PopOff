#! /usr/bin/env python3
from potential_parameters import BuckinghamParameter, buckingham_parameters
import pytest
import numpy as np

@pytest.fixture
def potential_parameters():
    return BuckinghamParameter('Li_O_a', 'a', 1000.0)

@pytest.fixture
def potentials():
    potentials = {'Li-O':[10.00, 0.1, 0.0],
                  'Ni-O':[100.00, 0.1, 0.0],
                  'O-O':[1000.00, 1.0, 1.0]}
    return potentials

def test_assert_potential_parameters(potential_parameters):
    assert potential_parameters.label_string == 'Li_O_a'
    assert potential_parameters.param_type == 'a'
    assert potential_parameters.value == 1000.0


@pytest.mark.parametrize( 'label_string', [(1),(1.0),(True), (['Li','O','a']) ])
def test_typeerror_for_label_string_in_potential_parameters(label_string, potential_parameters):
    with pytest.raises(TypeError):
        BuckinghamParameter(label_string, potential_parameters.param_type, potential_parameters.value)
        
@pytest.mark.parametrize( 'param_type', [(1),(1.0),(True), (['Li','O','a']) ])
def test_typeerror_for_param_type_in_potential_parameters(param_type, potential_parameters):
    with pytest.raises(TypeError):
        BuckinghamParameter(potential_parameters.label_string, param_type, potential_parameters.value)
        
@pytest.mark.parametrize( 'value', [(1),('Li'),(True), (['Li','O','a']) ])
def test_typeerror_for_value_in_potential_parameters(value, potential_parameters):
    with pytest.raises(TypeError):
        BuckinghamParameter(potential_parameters.label_string, potential_parameters.param_type, value)

@pytest.mark.parametrize( 'label_string', [('LiOa'),('LiO_a'),('L_i_O_a') ])
def test_valueerror_for_label_string_in_potential_parameters(label_string, potential_parameters):
    with pytest.raises(ValueError):
        BuckinghamParameter(label_string, potential_parameters.param_type, potential_parameters.value)        
        
@pytest.mark.parametrize( 'param_type', [('arhoc'),('a '),('Li_O_a') ])
def test_valueerror_for_param_type_in_potential_parameters(param_type, potential_parameters):
    with pytest.raises(ValueError):
        BuckinghamParameter(potential_parameters.label_string, param_type, potential_parameters.value)

        

def test_assert_buckingham_parameters(potentials):
    assert potentials.keys() == {'Li-O', 'Ni-O', 'O-O'}
    assert potentials['Li-O'] == [10.00, 0.1, 0.0]
    assert potentials['Ni-O'] == [100.00, 0.1, 0.0]
    assert potentials['O-O'] == [1000.00, 1.0, 1.0]
    
    
@pytest.mark.parametrize( 'keys', [(1),(1.0),(True)])
def test_typeerror_for_keys_in_potential_parameters(keys, potentials):
    potentials[keys] = potentials.pop('Li-O')
    for key, pot in potentials.items():
        with pytest.raises(TypeError):
            buckingham_parameters(potentials)
                       
@pytest.mark.parametrize( 'values', [(1),(1.0),(True), ('test')])
def test_typeerror_for_values_in_potential_parameters(values, potentials):
    potentials['Li-O'] = values
    for key, pot in potentials.items():
        with pytest.raises(TypeError):
            buckingham_parameters(potentials)                       
            
@pytest.mark.parametrize( 'keys', [('arhoc'),('a '),('Li_O_a'),('Li_O'),('Li-O-O') ])
def test_valueerror_for_keys_in_potential_parameters(keys, potentials):
    potentials[keys] = potentials.pop('Li-O')
    for key, pot in potentials.items():
        with pytest.raises(ValueError):
            buckingham_parameters(potentials)
            
@pytest.mark.parametrize( 'values', [(1),(True), ('test')])
def test_typeerror_for_values_in_list_in_potential_parameters(values, potentials):
    for key, pot in potentials.items():
        pot[0]=values
        potentials[key] = pot 
        with pytest.raises(TypeError):
            buckingham_parameters(potentials)