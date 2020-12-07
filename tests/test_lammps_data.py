#! /usr/bin/env python3
import pytest
import numpy as np
from mock import Mock, patch, call, mock_open
from buckfit.lammps_data import (LammpsData, abc_matrix, new_basis,
                                 apply_new_basis, lammps_lattice)
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

@patch('buckfit.lammps_data.LammpsData.write_lammps_files')
def test_init_in_LammpsData(mock_write, atom_types, mock_atoms, mock_bonds, mock_bt):
    cell_lengths = [10.1, 10.2, 10.3]
    tilt_factors = [0.0, 0.0, 0.0]
    file_name = 'test_files/vasprun_small.xml'
    expected_st = np.array([1,2,3,4,5,6])
    lammps_data = LammpsData(atom_types, mock_bt, mock_atoms, mock_bonds, cell_lengths, tilt_factors, file_name, expected_st)
    mock_write.assert_called_with()
    assert lammps_data.atoms == mock_atoms
    assert lammps_data.bonds == mock_bonds
    assert lammps_data.atom_types == atom_types
    assert lammps_data.bond_types == mock_bt
    assert lammps_data.cell_lengths == [10.1, 10.2, 10.3]
    assert lammps_data.tilt_factors == [0.0, 0.0, 0.0]
    assert lammps_data.file_name == 'test_files/vasprun_small.xml'
    assert np.allclose(lammps_data.expected_stress_tensors, expected_st)


@patch('buckfit.lammps_data.lammps_lattice', autospec=True)    
@patch('buckfit.lammps_data.types_from_structure', autospec=True)
@patch('buckfit.lammps_data.atoms_and_bonds_from_structure', autospec=True)
@patch('buckfit.lammps_data.LammpsData.write_lammps_files')
def test_from_structure_in_LammpsData(mock_write, mock_atoms_and_bonds, mock_types, mock_lattice,
                                      mock_lammps_data, params, atom_types, mock_atoms, mock_bonds,
                                      mock_bt): 
    cell_lengths = [10.1, 10.2, 10.3]
    tilt_factors = [0.0, 0.0, 0.0]
    mock_structure = Mock(spec=Structure)
    mock_types.return_value = atom_types, mock_bt
    mock_atoms_and_bonds.return_value = mock_atoms, mock_bonds
    mock_lattice.side_effect = cell_lengths, tilt_factors, mock_structure
    struct_data = LammpsData.from_structure(mock_structure, params, 0, np.array([1,2,3,4,5,6]))
    mock_write.assert_called_with()
    assert struct_data.atoms == mock_atoms
    assert struct_data.bonds == mock_bonds
    assert struct_data.atom_types == atom_types
    assert struct_data.bond_types == mock_bt
    assert struct_data.file_name == 'lammps/coords1.lmp'
    assert np.allclose(mock_lammps_data.expected_stress_tensors, np.array([1,2,3,4,5,6]))    
    
def test_header_string_in_LammpsData(lammps_data):
    head_string = lammps_data._header_string()
    assert head_string == 'title\n\n4   atoms\n1   bonds\n\n4   atom types\n1   bond types\n\n\n'

def test_cell_dimensions_string_in_LammpsData(lammps_data):
    cell_string = lammps_data._cell_dimensions_string()
    assert cell_string == '0.0 10.100000 xlo xhi\n0.0 10.200000 ylo yhi\n0.0 10.300000 zlo zhi\n\n0.00000 0.00000 0.00000 xy xz yz \n\n'

def test_masses_string_in_LammpsData(lammps_data):
    mass_string = lammps_data._masses_string()
    assert mass_string == 'Masses\n\n1   1.55500 # Li\n2  56.11100 # Ni\n3  14.00100 # O core\n3   1.99800 # O shell\n\n'

def test_atoms_string_in_LammpsData(lammps_data):
    atoms_string = lammps_data._atoms_string()
    assert atoms_string == 'Atoms\n\nfoo0\nfoo1\nfoo2\nfoo3\n\n'
    
def test_bonds_string_in_LammpsData(lammps_data):
    bonds_string = lammps_data._bonds_string()
    assert bonds_string == 'Bonds\n\nfoo\n\n'
    
def test_input_string_in_LammpsData(lammps_data):
    input_string = lammps_data.input_string()
    return_str = ''
    return_str += lammps_data._header_string()
    return_str += lammps_data._cell_dimensions_string()
    return_str += lammps_data._masses_string()
    return_str += lammps_data._atoms_string()
    return_str += lammps_data._bonds_string()
    assert input_string == return_str

def test_core_mask_in_LammpsData(lammps_data):
    core_mask = lammps_data.core_mask()
    assert core_mask == [True, True, True, False]
    
def test_type_core_in_LammpsData(lammps_data):
    type_core = lammps_data.type_core()
    assert type_core == '1 2 3'
    
def test_type_shell_in_LammpsData(lammps_data):
    type_shell = lammps_data.type_shell()
    assert type_shell == '3'

@patch("builtins.open", new_callable=mock_open)
def test_write_lammps_files_in_LammpsData(mock_open, lammps_data):
    lammps_data.write_lammps_files()
    calls_open = [call('test_files/test_coords.lmp', 'w'),
                  call().__enter__(),
                  call().write(lammps_data.input_string()),
                  call().__exit__(None, None, None)]
    mock_open.assert_has_calls(calls_open)

@patch('buckfit.lammps_data.lammps.Lammps')
def test_initiate_lmp_in_LammpsData(mock_lammps, mock_bt, lammps_data, params):
    bonds_string = lammps_data._bonds_string()
    lammps_data.initiate_lmp(params['cs_springs'])
    calls_command = [call(units='metal', style='full', args=['-log', 'none', '-screen', 'none']),
                     call().command(f'read_data {lammps_data.file_name}'),
                     call().command(f'group cores type {lammps_data.type_core()}'),
                     call().command(f'group shells type {lammps_data.type_shell()}'),
                     call().command('pair_style buck/coul/long/cs 10.0'),
                     call().command('pair_coeff * * 0 1 0'),
                     call().command('bond_style harmonic'),
                     call().command('bond_coeff foo'),
                     call().command('kspace_style ewald 1e-6'),
                     call().command('min_style cg')]
    mock_lammps.assert_has_calls(calls_command)

def test_abc_matrix():
    a = np.array([10, 0.1, 0.2])
    b = np.array([0.1, 10, 0.2])
    c = np.array([0.1, 0.2, 10])
    abc_mat = abc_matrix(a, b, c)
    assert np.allclose(abc_mat, np.array([[10.00249969,  0.20394902,  0.30192453],
                                          [ 0.00000000, 10.00042023,  0.39482569],
                                          [ 0.00000000,  0.00000000,  9.99014285]]))

def test_new_basis():
    lattice = Lattice(np.array([[10, 0.1, 0.2],[0.1, 10, 0.2],[0.1, 0.2, 10]]))
    abc = np.array([[10.00249969,  0.20394902,  0.30192453],
                    [ 0.00000000, 10.00042023,  0.39482569],
                    [ 0.00000000,  0.00000000,  9.99014285]])
    new_base = new_basis(abc, lattice)
    assert np.allclose(new_base, np.array([[ 1.00054425, -0.00980926, -0.00980926],
                                           [ 0.01079383,  1.00033638, -0.02011467],
                                           [ 0.01002782,  0.01941178,  0.99852577]]))
    

def test_apply_new_basis():
    new_base = np.array([[ 1.00054425, -0.00980926, -0.00980926],
                         [ 0.01079383,  1.00033638, -0.02011467],
                         [ 0.01002782,  0.01941178,  0.99852577]])
    struct_cart_coords = np.array([[0.1, 0.2, 0.3],
                                   [0.4, 0.5, 0.6],
                                   [0.7, 0.8, 0.9]])
    new_coords = apply_new_basis(new_base, struct_cart_coords)
    assert np.allclose(new_coords, np.array([[0.08926424, 0.38713367, 0.70773553],
                                             [0.18735681, 0.48623522, 0.81053207],
                                             [0.28544938, 0.58533677, 0.91332861]]))
    

@patch('buckfit.lammps_data.Structure', return_value='foo')
@patch('buckfit.lammps_data.apply_new_basis')    
@patch('buckfit.lammps_data.new_basis')    
@patch('buckfit.lammps_data.abc_matrix', return_value = np.array([[10, 0.1, 0.2],[0.1, 10, 0.2],[0.1, 0.2, 10]]))
def test_lammps_lattice(mock_matrix, mock_basis, mock_apply, mock_struct):
    structure = Mock(spec=Structure)
    structure.lattice.matrix = np.array([[10, 0.1, 0.2],[0.1, 10, 0.2],[0.1, 0.2, 10]])
    structure.lattice = Lattice(np.array([[10, 0.1, 0.2],[0.1, 10, 0.2],[0.1, 0.2, 10]]))
    structure.site_properties = {'forces': np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9]])}
    cell_lengths, tilt_factors, new_structure = lammps_lattice(structure)
    assert np.allclose(cell_lengths, np.array([10., 10., 10.]))
    assert np.allclose(tilt_factors, np.array([0.1, 0.2, 0.2]))
    assert new_structure == 'foo'