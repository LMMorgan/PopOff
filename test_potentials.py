#! /usr/bin/env python3
from potentials import BuckinghamPotential, buckingham_potentials
from atom_types import AtomType
from potential_parameters import BuckinghamParameter
import pytest
import numpy as np

def test_assert_buckingham_potential(buckingham_potential, pot_params):
    assert buckingham_potential.labels == ['Li','O']
    assert buckingham_potential.atype_index == [1,3]
    assert type(buckingham_potential.a) == type(pot_params[0])  #only compares the type, not the content.
    assert type(buckingham_potential.rho) == type(pot_params[1])  #only compares the type, not the content.
    assert type(buckingham_potential.c) == type(pot_params[2])  #only compares the type, not the content.
        
@pytest.mark.parametrize( 'labels', [(1),(1.0),(True), ([1,'a']), (['Li','O','a']) ])
def test_typeerror_for_labels_in_buckingham_potential(labels, buckingham_potential):
    with pytest.raises(TypeError):
        BuckinghamPotential(labels, buckingham_potential.atype_index, buckingham_potential.a, buckingham_potential.rho, buckingham_potential.c)
        
@pytest.mark.parametrize( 'atype_index', [(1),(1.0),(True), ([1,'a']), (['Li','O','a']) ])
def test_typeerror_for_atype_index_in_buckingham_potential(atype_index, buckingham_potential):
    with pytest.raises(TypeError):
        BuckinghamPotential(buckingham_potential.labels, atype_index, buckingham_potential.a, buckingham_potential.rho, buckingham_potential.c)
        
        
@pytest.mark.parametrize( 'a', [('test'),(1.0),(True),([1,2,3]),
                                (1),(AtomType(1, 'Li', 'Li', 1.555, 1.0))])
def test_typeerror_for_a_in_buckingham_potential(a, buckingham_potential):
    with pytest.raises(TypeError):
        BuckinghamPotential(buckingham_potential.labels, buckingham_potential.atype_index, a, buckingham_potential.rho, buckingham_potential.c)       

@pytest.mark.parametrize( 'rho', [('test'),(1.0),(True),([1,2,3]),
                                  (1),(AtomType(1, 'Li', 'Li', 1.555, 1.0))])
def test_typeerror_for_rho_in_buckingham_potential(rho, buckingham_potential):
    with pytest.raises(TypeError):
        BuckinghamPotential(buckingham_potential.labels, buckingham_potential.atype_index, buckingham_potential.a, rho, buckingham_potential.c)        

@pytest.mark.parametrize( 'c', [('test'),(1.0),(True),([1,2,3]),
                                (1),(AtomType(1, 'Li', 'Li', 1.555, 1.0))])
def test_typeerror_for_c_in_buckingham_potential(c, buckingham_potential):
    with pytest.raises(TypeError):
        BuckinghamPotential(buckingham_potential.labels, buckingham_potential.atype_index, buckingham_potential.a, buckingham_potential.rho, c)  
        

def test_assert_potentials_in_buckingham_potentials(potentials, atom_types, pot_params):
    assert potentials.keys() == {'Li-O', 'Ni-O', 'O-O'}
    assert potentials['Li-O'] == [10.00, 0.1, 0.0]
    assert potentials['Ni-O'] == [100.00, 0.1, 0.0]
    assert potentials['O-O'] == [1000.00, 1.0, 1.0]

def test_assert_atom_types_0_in_buckingham_potentials(atom_types):
    assert atom_types[0].atom_type_index == 1
    assert atom_types[0].label == 'Li'
    assert atom_types[0].element_type == 'Li'
    assert atom_types[0].mass == 1.555
    assert atom_types[0].charge == 1.0
    assert atom_types[0].formal_charge == 1.0
    assert atom_types[0].core_shell == None
    
def test_assert_atom_types_1_in_buckingham_potentials(atom_types):
    assert atom_types[1].atom_type_index == 2
    assert atom_types[1].label == 'Ni'
    assert atom_types[1].element_type == 'Ni'
    assert atom_types[1].mass == 56.111
    assert atom_types[1].charge == 3.0
    assert atom_types[1].formal_charge == 3.0
    assert atom_types[1].core_shell == None

def test_assert_atom_types_2_in_buckingham_potentials(atom_types):
    assert atom_types[2].atom_type_index == 3
    assert atom_types[2].label == 'O core'
    assert atom_types[2].element_type == 'O'
    assert atom_types[2].mass == 14.001
    assert atom_types[2].charge == -0.00
    assert atom_types[2].formal_charge == -0.00
    assert atom_types[2].core_shell == 'core'
    
def test_assert_atom_types_3_in_buckingham_potentials(atom_types):
    assert atom_types[3].atom_type_index == 3
    assert atom_types[3].label == 'O shell'
    assert atom_types[3].element_type == 'O'
    assert atom_types[3].mass == 1.998
    assert atom_types[3].charge == -2.0
    assert atom_types[3].formal_charge == -2.0
    assert atom_types[3].core_shell == 'shell'
    
def test_assert_pot_params_0_in_buckingham_potentials(pot_params):
    assert pot_params[0].label_string == 'Li_O_a'
    assert pot_params[0].param_type == 'a'
    assert pot_params[0].value == 1

def test_assert_pot_params_1_in_buckingham_potentials(pot_params):
    assert pot_params[1].label_string == 'Li_O_rho'
    assert pot_params[1].param_type == 'rho'
    assert pot_params[1].value == 0.1
    
def test_assert_pot_params_2_in_buckingham_potentials(pot_params):
    assert pot_params[2].label_string == 'Li_O_c'
    assert pot_params[2].param_type == 'c'
    assert pot_params[2].value == 0.0
    
    
@pytest.mark.parametrize( 'potentials', [(1),(1.0),(True), (['test', 1])])
def test_typeerror_for_potentials_in_buckingham_potentials(potentials, atom_types, pot_params):
    with pytest.raises(TypeError):
        buckingham_potentials(potentials, atom_types, pot_params)

@pytest.mark.parametrize( 'atom_types', [(1),(1.0),(True), (['test',AtomType(1, 'Li', 'Li', 1.555, 1.0)])])
def test_typeerror_for_atom_types_in_buckingham_potentials(potentials, atom_types, pot_params):
    with pytest.raises(TypeError):
        buckingham_potentials(potentials, atom_types, pot_params)
        
@pytest.mark.parametrize( 'pot_params', [(1),(1.0),(True), ([AtomType(1, 'Li', 'Li', 1.555, 1.0),AtomType(1, 'Li', 'Li', 1.555, 1.0)])])
def test_typeerror_for_pot_params_in_buckingham_potentials(pot_params, potentials, atom_types):
    with pytest.raises(TypeError):
        buckingham_potentials(potentials, atom_types, pot_params)
        
def test_output_in_buckingham_potentials(atom_types, pot_params):
    output = buckingham_potentials({'Li-O':[10.00, 0.1, 0.0]}, atom_types, pot_params)
    assert type(output) == list
    assert output[0].labels == ['Li', 'O']
    assert output[0].atype_index == [1, 3]
    assert type(output[0].a) == BuckinghamParameter
    assert type(output[0].rho) == BuckinghamParameter
    assert type(output[0].c) == BuckinghamParameter