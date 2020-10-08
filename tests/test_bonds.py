#! /usr/bin/env python3
from lammps_potenial_fitting.atom_types import AtomType
from lammps_potenial_fitting.bond_types import BondType
from lammps_potenial_fitting.bonds import Bond
import pytest
import numpy as np

@pytest.fixture
def bond():
    return Bond(1, [1,2], BondType(1, 'Li-O', 65.0))

def test_assert_bond(bond):
    assert bond.bond_index == 1
    assert bond.atom_indices == [1,2]
    assert type(bond.bond_type) == type(BondType(1, 'Li-O', 65.0))  #only compares the type, not the content.


@pytest.mark.parametrize( 'bond_index', [('test'),(1.0),(True)])
def test_typeerror_for_bond_index_in_bond(bond_index, bond):
    with pytest.raises(TypeError):
        Bond(bond_index, bond.atom_indices, bond.bond_type)

@pytest.mark.parametrize( 'atom_indices', [('test'),(1.0),(True), ([1,2,3])])
def test_typeerror_for_atom_indices_in_bond(atom_indices, bond):
    with pytest.raises(TypeError):
        Bond(bond.bond_index, atom_indices, bond.bond_type)
        
@pytest.mark.parametrize( 'bond_type', [('test'),(1.0),(True),([1,2,3]),
                                        (1),(AtomType(1, 'Li', 'Li', 1.555, 1.0))])
def test_typeerror_for_bond_type_in_bond(bond_type, bond):
    with pytest.raises(TypeError):
        Bond(bond.bond_index, bond.atom_indices, bond_type) 

@pytest.mark.parametrize( 'atom_indices', [([0.1, 1]),(['test', 1]),([True, 1]), ([[1,2], 1])])
def test_valueerror_for_atom_indices_in_bond(atom_indices, bond):
    with pytest.raises(ValueError):
        Bond(bond.bond_index, atom_indices, bond.bond_type)
        
def test_input_string_in_atom(bond):
    assert bond.input_string() == '1    1    1    2   '