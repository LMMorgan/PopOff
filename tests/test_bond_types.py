#! /usr/bin/env python3
from popoff.bond_types import BondType
import pytest

@pytest.fixture
def bond_type():
    return BondType(1, 'Li-O', 65.0)

def test_assert_bond_type(bond_type):
    assert bond_type.bond_type_index == 1
    assert bond_type.label == 'Li-O'
    assert bond_type.spring_coeff_1 == 65.0
    assert bond_type.spring_coeff_2 == 0.0


@pytest.mark.parametrize( 'bond_type_index', [('test'), (1.0), (True)])
def test_typeerror_for_bond_type_index_in_bond_type(bond_type_index, bond_type):
    with pytest.raises(TypeError):
        BondType(bond_type_index, bond_type.label, bond_type.spring_coeff_1, bond_type.spring_coeff_2)

@pytest.mark.parametrize( 'label', [(1), (1.0), (True)])
def test_typeerror_for_label_in_bond_type(label, bond_type):
    with pytest.raises(TypeError):
        BondType(bond_type.bond_type_index, label, bond_type.spring_coeff_1, bond_type.spring_coeff_2)

@pytest.mark.parametrize( 'spring_coeff_1', [(1), ('test'), (True)])
def test_typeerror_for_spring_coeff_1_in_bond_type(spring_coeff_1, bond_type):
    with pytest.raises(TypeError):
        BondType(bond_type.bond_type_index, bond_type.label, spring_coeff_1, bond_type.spring_coeff_2)

@pytest.mark.parametrize( 'spring_coeff_2', [(1), ('test'), (True)])
def test_typeerror_for_spring_coeff_2_in_bond_type(spring_coeff_2, bond_type):
    with pytest.raises(TypeError):
        BondType(bond_type.bond_type_index, bond_type.label, bond_type.spring_coeff_1, spring_coeff_2)

@pytest.mark.parametrize( 'label', [('LiO'), ('Li_O'), ('Li O'), ('something')])
def test_valueerror_for_label_in_bond_type(label, bond_type):
    with pytest.raises(ValueError):
        BondType(bond_type.bond_type_index, label, bond_type.spring_coeff_1, bond_type.spring_coeff_2)
        
def test_bond_string_in_bond_type(bond_type):
    assert bond_type.bond_string() == 'bond_coeff 1  65.00    0.0'
