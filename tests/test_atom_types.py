#! /usr/bin/env python3
from buckfit.atom_types import AtomType
import pytest
from mock import Mock

@pytest.fixture
def atom_type():
    return AtomType(1, 'O core', 'O', 1.555, 1.0)

def test_assert_atom_type(atom_type):
    assert atom_type.atom_type_index == 1
    assert atom_type.label == 'O core'
    assert atom_type.element_type == 'O'
    assert atom_type.mass == 1.555
    assert atom_type.charge == 1.0
    assert atom_type.core_shell == None

@pytest.mark.parametrize( 'atom_type_index', [('test'), (1.0), (True)])
def test_typeerror_for_atom_type_index_in_atom_type(atom_type_index, atom_type):
    with pytest.raises(TypeError):
        AtomType(atom_type_index, atom_type.label, atom_type.element_type, atom_type.mass, atom_type.charge)
                      
@pytest.mark.parametrize( 'label', [(1), (1.0), (True)])
def test_typeerror_for_label_in_atom_type(label, atom_type):
    with pytest.raises(TypeError):
        AtomType(atom_type.atom_type_index, label, atom_type.element_type, atom_type.mass, atom_type.charge)
        
@pytest.mark.parametrize( 'element_type', [(1), (1.0), (True)])
def test_typeerror_for_element_type_in_atom_type(element_type, atom_type):
    with pytest.raises(TypeError):
        AtomType(atom_type.atom_type_index, atom_type.label, element_type, atom_type.mass, atom_type.charge)
        
@pytest.mark.parametrize( 'mass', [(1), ('test'), (True)])
def test_typeerror_for_mass_in_atom_type(mass, atom_type):
    with pytest.raises(TypeError):
        AtomType(atom_type.atom_type_index, atom_type.label, atom_type.element_type, mass, atom_type.charge)        
@pytest.mark.parametrize( 'charge', [(1), ('test'), (True)])
def test_typeerror_for_charge_in_atom_type(charge, atom_type):
    with pytest.raises(TypeError):
        AtomType(atom_type.atom_type_index, atom_type.label, atom_type.element_type, atom_type.mass, charge)

@pytest.mark.parametrize( 'core_shell', [('test'), (1.0), (1), (True)])
def test_valueerror_for_core_shell_in_atom_type(core_shell, atom_type):
    with pytest.raises(ValueError):
        AtomType(atom_type.atom_type_index, atom_type.label, atom_type.element_type, atom_type.mass, atom_type.charge, core_shell)

def test_core_shell_string_in_atom_type():
    core_shells = ['core', 'shell', None]
    outputs = ['core', 'shell', '']
    for cs, output in zip(core_shells, outputs):
        atom_type = AtomType(1, 'Li', 'Li', 1.555, 1.0, core_shell=cs)
        assert atom_type.core_shell_string == output
