#! /usr/bin/env python3
from popoff.atom_types import AtomType
from popoff.bond_types import BondType
from popoff.atoms import Atom
import pytest
import numpy as np

@pytest.fixture
def testing_arrays():
    array_1 = np.array([0.01, 0.02, 0.03, 0.04])
    array_2 = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
    array_3 = np.array([[0.01], [0.02], [0.03], [0.04], [0.05]])
    return [array_1, array_2, array_3]

@pytest.fixture
def atom():
    coords = np.array([0.1, 0.2, 0.3])
    atom_forces = np.array([0.01, 0.02, 0.03])
    atom_type = AtomType(1, 'Li', 'Li', 1.555, 1.0)
    return Atom(1, 2, coords, atom_forces, atom_type)

def test_assert_atom(atom):
    assert atom.atom_index == 1
    assert atom.molecule_index == 2
    assert np.allclose(atom.coords,np.array([0.1, 0.2, 0.3]))
    assert np.allclose(atom.forces, np.array([0.01, 0.02, 0.03]))
    assert type(atom.atom_type) == type(AtomType(1, 'Li', 'Li', 1.555, 1.0))  #only compares the type, not the content.

@pytest.mark.parametrize( 'atom_index', [('test'),(1.0),(True)])
def test_typeerror_for_atom_index_in_atom(atom_index, atom):
    with pytest.raises(TypeError):
        Atom(atom_index, atom.molecule_index, atom.coords, atom.forces, atom.atom_type)

@pytest.mark.parametrize( 'molecule_index', [('test'),(1.0),(True)])
def test_typeerror_for_molecule_index_in_atom(molecule_index, atom):
    with pytest.raises(TypeError):
        Atom(atom.atom_index, molecule_index, atom.coords, atom.forces, atom.atom_type)  
        
@pytest.mark.parametrize( 'coords', [('test'),(1.0),(True),([1,2,3]),(1)])
def test_typeerror_for_coords_in_atom(coords, atom):
    with pytest.raises(TypeError):
        Atom(atom.atom_index, atom.molecule_index, coords, atom.forces, atom.atom_type)
        
@pytest.mark.parametrize( 'atom_forces', [('test'),(1.0),(True),([1,2,3]),(1)])
def test_typeerror_for_atom_forces_in_atom(atom_forces, atom):
    with pytest.raises(TypeError):
        Atom(atom.atom_index, atom.molecule_index, atom.coords, atom_forces, atom.atom_type)
        
@pytest.mark.parametrize( 'atom_type', [('test'),(1.0),(True),
                                        ([1,2,3]),(1),(BondType(1, 'Li-O', 1.555, 1.0))])
def test_typeerror_for_atom_type_in_atom(atom_type, atom):
    with pytest.raises(TypeError):
        Atom(atom.atom_index, atom.molecule_index, atom.coords, atom.forces, atom_type)  
                           
def test_valueerror_for_coords_in_atom(testing_arrays, atom):
    for coords in testing_arrays:
        with pytest.raises(ValueError):
            Atom(atom.atom_index, atom.molecule_index, coords, atom.forces, atom.atom_type)
            
def test_valueerror_for_atom_forces_in_atom(testing_arrays, atom):
    for atom_forces in testing_arrays:
        with pytest.raises(ValueError):
            Atom(atom.atom_index, atom.molecule_index, atom.coords, atom_forces, atom.atom_type)

def test_input_string_in_atom(atom):
    assert atom.input_string() == '1    2    1     1.0000   0.100000   0.200000   0.300000'
