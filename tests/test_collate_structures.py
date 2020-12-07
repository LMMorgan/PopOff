#! /usr/bin/env python3
import pytest
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from mock import Mock, patch
from buckfit.lammps_data import LammpsData
from buckfit.atom_types import AtomType
from buckfit.collate_structures import (collate_structural_data, data_from_vasprun)

@pytest.mark.parametrize('supercell', [(['test_string', 2,2]),(1.0),(10),
                                       (np.array([1.0,1.0,1.0])), ([1.0,2,2]),
                                       ([2,2,2,2])])
def test_typeerror_for_supercell_in_data_from_vasprun(supercell, structure, params):
    with pytest.raises(TypeError):
        data_from_vasprun(params, 'test_files/vasprun_small.xml', 0, supercell)

@patch('buckfit.lammps_data.LammpsData.write_lammps_files')
def test_asserts_for_data_from_vasprun(mock_write, params):
    struct_data = data_from_vasprun(params, 'test_files/vasprun_small.xml', 0, supercell=None)
    assert type(struct_data) == LammpsData
    assert struct_data.file_name == 'lammps/coords1.lmp'
    assert np.allclose(struct_data.tilt_factors, np.array([0.,0.,0.]))
    assert type(struct_data.atom_types) == list
    assert len(struct_data.atom_types) == 4
    assert len(struct_data.atoms) == 36
    assert np.allclose(struct_data.atoms[0].coords, np.array([2.72774735, 1.43820405, 2.34930819]))
    assert np.allclose(struct_data.atoms[0].forces, np.array([-0.00435576, -0.01932167,  0.01268978]))
    assert struct_data.bonds[0].atom_indices == [13,14]

@patch('buckfit.collate_structures.data_from_vasprun', autospec=True)
def test_output_in_collate_structural_data(mock_data_from_vasprun, params, atom_types):
    mock_lammps_data = [Mock(spec=LammpsData), Mock(spec=LammpsData)]
    mock_lammps_data[0].atom_types = atom_types
    mock_lammps_data[1].atom_types = atom_types
    mock_lammps_data[0].file_name = 'test_files/vasprun_small_1.xml'
    mock_lammps_data[1].file_name = 'test_files/vasprun_small_2.xml'
    mock_data_from_vasprun.side_effect = mock_lammps_data
    output = collate_structural_data(params, [0,1])
#     output should contain [mock_lammps_data[0], mock_lammps_data[1]]
    assert [d.atom_types for d in output] == [atom_types, atom_types]
    assert [d.file_name for d in output] == ['test_files/vasprun_small_1.xml', 'test_files/vasprun_small_2.xml']
    assert output[0].atom_types[0].atom_type_index == 1
    assert output[0].atom_types[0].label == 'Li'
    assert output[1].atom_types[3].atom_type_index == 3
    assert output[1].atom_types[3].label == 'O shell'