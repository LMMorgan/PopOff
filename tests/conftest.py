#! /usr/bin/env python3
import pytest
import numpy as np
from mock import Mock, patch
from lammps_potenial_fitting.potentials import BuckinghamPotential
from lammps_potenial_fitting.atom_types import AtomType
from lammps_potenial_fitting.bond_types import BondType
from lammps_potenial_fitting.atoms import Atom
from lammps_potenial_fitting.bonds import Bond
from lammps_potenial_fitting.potential_parameters import BuckinghamParameter
from lammps_potenial_fitting.lammps_data import LammpsData
from lammps_potenial_fitting.fitting_code import FitModel
from pymatgen.io.vasp.outputs import Vasprun

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
def values():
    return [0.5, 0.6, 20.0, 40000.0, 0.5, 700.0, 0.3, 10000, 0.1]

@pytest.fixture
def potentials():
    potentials = {'Li-O':[10.00, 0.1, 0.0],
                  'Ni-O':[100.00, 0.1, 0.0],
                  'O-O':[1000.00, 1.0, 1.0]}
    return potentials

@pytest.fixture
def structure():
    vasprun = Vasprun('test_files/test_vasprun.xml')
    structure = vasprun.ionic_steps[0]['structure']
    structure.add_site_property('forces', np.array(vasprun.ionic_steps[0]['forces']))
    return structure

@pytest.fixture
def atom_types():
    atom_types = []
    atom_types.append(AtomType(1, 'Li', 'Li', 1.555, 1.0))
    atom_types.append(AtomType(2, 'Ni', 'Ni', 56.111, 3.0))
    atom_types.append(AtomType(3, 'O core', 'O', 14.001, -0.00, 'core'))
    atom_types.append(AtomType(3, 'O shell', 'O', 1.998, -2.00, 'shell'))
    return atom_types

@pytest.fixture
def pot_params():
    parameters = []
    parameters.append(BuckinghamParameter('Li_O_a', 'a', 1.0))
    parameters.append(BuckinghamParameter('Li_O_rho', 'rho', 0.1))
    parameters.append(BuckinghamParameter('Li_O_c', 'c', 0.0))
    return parameters

@pytest.fixture
def buckingham_potential(pot_params):
    return BuckinghamPotential(['Li','O'], [1,3], pot_params[0], pot_params[1], pot_params[2])

@pytest.fixture
def mock_atoms(atom_types):
    atoms =[]
    for i, atom_type in enumerate(atom_types):
        atom = Mock(spec=Atom)
        atom.atom_index = i+1
        atom.molecule_index = atom_type.atom_type_index
        atom.coords = np.array([float(i), float(i+0.1), float(i+0.2)])
        atom.forces = np.array([float(i+0.1), float(i+0.2), float(i+0.3)])
        atom.atom_type = atom_type
        atom.input_string = Mock(return_value=f'foo{i}')
        atoms.append(atom)
    return atoms

@pytest.fixture
def mock_bt():
    bond_types = Mock(spec=BondType)
    bond_types.bond_type_index = 1
    bond_types.label = 'Li-O'
    bond_types.spring_coeff_1 = 65.0
    bond_types.spring_coeff_2 = 0.0
    bond_types.bond_string = Mock(return_value='bond_coeff foo')
    return [bond_types]

@pytest.fixture
def mock_bonds(mock_bt):
    bond = Mock(spec=Bond)
    bond.bond_index = 1
    bond.atom_indices = [3,4]
    bond.bond_type = mock_bt
    bond.input_string = Mock(return_value='foo')
    return [bond]

@pytest.fixture
def mock_lammps_data(atom_types, mock_atoms, mock_bonds, mock_bt):
    lammps_data = Mock(spec=LammpsData)
    lammps_data.atoms = mock_atoms
    lammps_data.bonds = mock_bonds
    lammps_data.atom_types = atom_types
    lammps_data.bond_types = mock_bt
    lammps_data.cell_lengths = [10.1, 10.2, 10.3]
    lammps_data.tilt_factors = [0.0, 0.0, 0.0]
    lammps_data.file_name = 'test_files/vasprun_small.xml'
    lammps_data.expected_stress_tensors = np.array([1,2,3,4,5,6])
    return lammps_data

@pytest.fixture
@patch('lammps_potenial_fitting.lammps_data.LammpsData.write_lammps_files')
def lammps_data(mock_write, atom_types, mock_atoms, mock_bonds, mock_bt):
    cell_lengths = [10.1, 10.2, 10.3]
    tilt_factors = [0.0, 0.0, 0.0]
    file_name = 'test_files/vasprun_small.xml'
    expected_st = np.array([1,2,3,4,5,6])
    lammps_data = LammpsData(atom_types, mock_bt, mock_atoms, mock_bonds,
                             cell_lengths, tilt_factors, file_name, expected_st)
    mock_write.assert_called_with()
    return lammps_data

@pytest.fixture
def fit_data(buckingham_potential, lammps_data):
    fit_data = FitModel([buckingham_potential], [lammps_data], cs_springs=None)
    return fit_data