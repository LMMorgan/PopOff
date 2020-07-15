#! /usr/bin/env python3
import pytest
import numpy as np
from unittest import mock
from potentials import BuckinghamPotential
from atom_types import AtomType
from potential_parameters import BuckinghamParameter
from lammps_data import LammpsData
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
