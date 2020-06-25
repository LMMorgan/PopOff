#! /usr/bin/env python3
from from_structure import types_from_structure, atoms_and_bonds_from_structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core.composition import Composition
from atom_types import AtomType
from bond_types import BondType
import pytest
import numpy as np

@pytest.fixture
def structure():
    vasprun = Vasprun('test_files/test_vasprun.xml')
    structure = vasprun.ionic_steps[0]['structure']
    structure.add_site_property('forces', np.array(vasprun.ionic_steps[0]['forces']))
    return structure


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
    params['cs_springs'] = {'O-O' : [10.0, 0.0]}
    return params

@pytest.fixture        
def types_output(structure, params):
    atom_types, bond_types = types_from_structure( structure, params['core_shell'], params['charges'], params['masses'], params['cs_springs'] )
    return atom_types, bond_types

@pytest.fixture
def atoms_and_bonds(structure, types_output):
    return atoms_and_bonds_from_structure(structure, types_output[0], types_output[1])


def test_assert_structure_in_types_from_structure(structure):
    assert structure.composition == Composition("Li48Ni48O96")
    assert structure.num_sites == 192
    
def test_assert_core_shell_in_types_from_structure(params):
    assert params['core_shell']['Li'] == False
    assert params['core_shell']['Ni'] == False
    assert params['core_shell']['O'] == True
    
def test_assert_charges_in_types_from_structure(params):
    assert params['charges']['Li'] == 1.0
    assert params['charges']['Ni'] == 3.0
    assert params['charges']['O'] == {'core':  -2.0, 'shell': 0.0}
    
def test_assert_masses_in_types_from_structure(params):
    assert params['masses']['Li'] == 6.941
    assert params['masses']['Ni'] == 58.6934
    assert params['masses']['O'] == {'core': 14.3991, 'shell': 1.5999}
    
def test_assert_cs_springs_in_types_from_structure(params):
    assert params['cs_springs']['O-O'] == [10.0, 0.0]

@pytest.mark.parametrize( 'core_shell', [('Co') ])
def test_valueerror_for_core_shell_in_types_from_structure(core_shell, structure, params):
    with pytest.raises(ValueError):
        types_from_structure(structure, core_shell, params['charges'], params['masses'])
        
@pytest.mark.parametrize( 'charges', [('Co') ])
def test_valueerror_for_charges_in_types_from_structure(charges, structure, params):
    with pytest.raises(ValueError):
        types_from_structure(structure, params['core_shell'], charges, params['masses'])
        
@pytest.mark.parametrize( 'masses', [('Co') ])
def test_valueerror_for_masses_in_types_from_structure(masses, structure, params):
    with pytest.raises(ValueError):
        types_from_structure(structure, params['core_shell'], params['charges'], masses)  
           
def test_assert_atom_types_0_in_types_from_structure(types_output):
    atom_types = types_output[0]
    assert type(atom_types[0]) == AtomType
    assert atom_types[0].atom_type_index == 1
    assert atom_types[0].label == 'Li'
    assert atom_types[0].element_type == 'Li'
    assert atom_types[0].mass == 6.941
    assert atom_types[0].charge == 1.0
    assert atom_types[0].formal_charge == 1.0
    assert atom_types[0].core_shell == None

def test_assert_atom_types_1_in_types_from_structure(types_output):
    atom_types = types_output[0]
    assert atom_types[1].atom_type_index == 2
    assert atom_types[1].label == 'Ni'
    assert atom_types[1].element_type == 'Ni'
    assert atom_types[1].mass == 58.6934
    assert atom_types[1].charge == 3.0
    assert atom_types[1].formal_charge == 3.0
    assert atom_types[1].core_shell == None

def test_assert_atom_types_2_in_types_from_structure(types_output):
    atom_types = types_output[0]
    assert atom_types[2].atom_type_index == 3
    assert atom_types[2].label == 'O core'
    assert atom_types[2].element_type == 'O'
    assert atom_types[2].mass == 14.3991
    assert atom_types[2].charge == -2.0
    assert atom_types[2].formal_charge == -2.0
    assert atom_types[2].core_shell == 'core'
    
def test_assert_atom_types_3_in_types_from_structure(types_output):
    atom_types = types_output[0]
    assert atom_types[3].atom_type_index == 4
    assert atom_types[3].label == 'O shell'
    assert atom_types[3].element_type == 'O'
    assert atom_types[3].mass == 1.5999
    assert atom_types[3].charge == 0.0
    assert atom_types[3].formal_charge == 0.0
    assert atom_types[3].core_shell == 'shell'

def test_assert_bond_types_in_types_from_structure(types_output):
    bond_types = types_output[1]
    assert type(bond_types) == list
    assert bond_types[0].bond_type_index == 1
    assert bond_types[0].label == 'O-O spring'
    assert bond_types[0].spring_coeff_1 == 10.0
    assert bond_types[0].spring_coeff_2 == 0.0

def test_assert_atoms_in_atoms_and_bonds_from_structure(atoms_and_bonds):
    atoms = atoms_and_bonds[0]
    assert len(atoms) == 288
    assert atoms[0].atom_index == 1
    assert atoms[0].molecule_index == 1
    assert np.allclose(atoms[0].coords, np.array([2.74535186, 1.41748673, 2.44928821]))
    assert np.allclose(atoms[0].forces, np.array([-0.01149628, -0.36440729, -0.0674791]))
    assert type(atoms[0].atom_type) == AtomType

def test_assert_bonds_in_atoms_and_bonds_from_structure(atoms_and_bonds):
    bonds = atoms_and_bonds[1]
    assert len(bonds) == 96
    assert bonds[0].bond_index == 1
    assert bonds[0].atom_indices == [97,98]
    assert type(bonds[0].bond_type) == BondType