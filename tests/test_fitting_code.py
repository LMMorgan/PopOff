#! /usr/bin/env python3
import pytest
import numpy as np
from mock import Mock, patch, call, mock_open, MagicMock
from lammps_potenial_fitting.fitting_code import FitModel
from lammps_potenial_fitting.lammps_data import LammpsData

def test_init_in_FitModel(mock_lammps_data, buckingham_potential):
    fit_data = FitModel([buckingham_potential], mock_lammps_data)
    assert fit_data.potentials == [buckingham_potential]
    assert fit_data.lammps_data == mock_lammps_data
    assert fit_data.cs_springs == None

@patch('lammps_potenial_fitting.fitting_code.pot', autospec=True) 
@patch('lammps_potenial_fitting.fitting_code.collate_structural_data', autospec=True)   
def test_collect_info_with_cs_in_FitModel(mock_collate, mock_pot, params, buckingham_potential,
                                  pot_params): 
    structs = np.array([0, 1, 2])
    mock_pot.buckingham_parameters = Mock(return_value = pot_params)
    mock_pot.buckingham_potentials = Mock(return_value = [buckingham_potential])
    fit_data = FitModel.collect_info(params, structs)    
    assert fit_data.potentials == [buckingham_potential]
    assert fit_data.cs_springs == {'O-O': [10.0, 0.0]}

@patch('lammps_potenial_fitting.fitting_code.pot', autospec=True) 
@patch('lammps_potenial_fitting.fitting_code.collate_structural_data', autospec=True)   
def test_collect_info_no_cs_in_FitModel(mock_collate, mock_pot, buckingham_potential,
                                  pot_params, mock_lammps_data): 
    structs = np.array([0, 1, 2])
    params = {}
    params['core_shell'] = { 'Li': False, 'Ni': False, 'O': False }
    params['charges'] = {'Li': +1.0, 'Ni': +3.0, 'O': -2.0} 
    params['masses'] = {'Li': 6.941, 'Ni': 58.6934, 'O': 15.999}
    params['potentials'] = {'Li-O': [100.00, 0.1, 0.0],
                            'Ni-O': [1000.00, 0.2, 0.0],
                            'O-O':  [10000.00, 0.3, 0.0]}
    mock_pot.buckingham_parameters = Mock(return_value = pot_params)
    mock_pot.buckingham_potentials = Mock(return_value = [buckingham_potential])
    fit_data = FitModel.collect_info(params, structs)    
    assert fit_data.potentials == [buckingham_potential]
    assert fit_data.cs_springs == None
    assert fit_data.lammps_data is not None
    
@patch('lammps_potenial_fitting.lammps_data.LammpsData.core_mask', autospec=True) 
def test_expected_forces_in_FitModel(mock_core_mask, fit_data):
    mock_core_mask.return_value = [True, True, True, False]
    expected_forces = fit_data.expected_forces()
    assert np.allclose(expected_forces, np.array([[0.1,0.2,0.3],
                                                  [1.1,1.2,1.3],
                                                  [2.1,2.2,2.3]]))

def test_expected_stresses_in_FitModel(fit_data):
    expected_stresses = fit_data.expected_stresses()
    assert np.allclose(expected_stresses, np.array([1,2,3,4,5,6]))
    
@patch("lammps_potenial_fitting.lammps_data.lammps.Lammps")
def test_set_charges_in_FitModel(mock_lammps, fit_data):
    fit_data._set_charges(mock_lammps)
    calls_command = [call.command('set type 1 charge 1.000000'),
                     call.command('set type 2 charge 3.000000'),
                     call.command('set type 3 charge -0.000000'),
                     call.command('set type 3 charge -2.000000')]
    mock_lammps.assert_has_calls(calls_command)
                     
@patch("lammps_potenial_fitting.lammps_data.lammps.Lammps")
def test_set_springs_in_FitModel(mock_lammps, fit_data):
    fit_data._set_springs(mock_lammps)
    calls_command = [call.command('bond_coeff foo')]
    mock_lammps.assert_has_calls(calls_command)
    
@patch("lammps_potenial_fitting.lammps_data.lammps.Lammps")
def test_set_potentials_in_FitModel(mock_lammps, fit_data):
    fit_data._set_potentials(mock_lammps)
    calls_command = [call.command('pair_coeff 1 3 1.0000 0.1000 0.0000')]
    mock_lammps.assert_has_calls(calls_command)
    
@patch("lammps_potenial_fitting.lammps_data.lammps.Lammps")
def test_calls_for_get_forces_and_stresses_in_FitModel(mock_lammps, fit_data):
    fit_data.get_forces_and_stresses()
    calls_command = [call(units='metal',style='full',args=['-log','none','-screen','none']),
                     call().command('read_data test_files/vasprun_small.xml'),
                     call().command('group cores type 1 2 3'),
                     call().command('group shells type 3'),
                     call().command('pair_style buck/coul/long 10.0'),
                     call().command('pair_coeff * * 0 1 0'),
                     call().command('kspace_style ewald 1e-6'),
                     call().command('min_style cg'),
                     call().command('pair_coeff 1 3 1.0000 0.1000 0.0000'),
                     call().command('set type 1 charge 1.000000'),
                     call().command('set type 2 charge 3.000000'),
                     call().command('set type 3 charge -0.000000'),
                     call().command('set type 3 charge -2.000000'),
                     call().run(0),
                     call().system.forces.__getitem__([True, True, True, False]),
                     call().thermo.computes.__getitem__('thermo_press')]
    mock_lammps.assert_has_calls(calls_command)

@patch("lammps_potenial_fitting.lammps_data.lammps.Lammps")
def test_calls_for_convert_stresses_to_vasp_in_FitModel(mock_lammps, fit_data):
    instances = [lmp.initiate_lmp(fit_data.cs_springs) for lmp in fit_data.lammps_data]
    ip_stresses = fit_data.convert_stresses_to_vasp(instances[0])
    calls_command = [call(units='metal',style='full',args=['-log','none','-screen','none']),
                     call().command('read_data test_files/vasprun_small.xml'),
                     call().command('group cores type 1 2 3'),
                     call().command('group shells type 3'),
                     call().command('pair_style buck/coul/long 10.0'),
                     call().command('pair_coeff * * 0 1 0'),
                     call().command('kspace_style ewald 1e-6'),
                     call().command('min_style cg'),
                     call().thermo.computes.__getitem__('thermo_press'),
                     call().thermo.computes.__getitem__().vector.__truediv__(1000)]
    mock_lammps.assert_has_calls(calls_command)
    
def test_charge_reset_in_FitModel(fit_data):
    fit_data._charge_reset()
    for data in fit_data.lammps_data:
        for at in data.atom_types:
            assert at.charge == at.formal_charge
            
def test_update_q_ratio_in_FitModel(fit_data):
    fitting_parameters = {'dq_O': 0.2, 'q_scaling':0.6, 'O-O spring':10.5}
    for data in fit_data.lammps_data:
        for at in data.atom_types:
            if at.label == 'Li':
                assert at.charge == 1.0
            elif at.label == 'Ni':
                assert at.charge == 3.0
            elif at.label == "O core":
                assert at.charge == 0.0
            elif at.label == 'O shell':
                assert at.charge == -2.0
    fit_data._update_q_ratio(fitting_parameters)
    for data in fit_data.lammps_data:
        for at in data.atom_types:
            if at.label == 'Li':
                assert at.charge == 1.0
            elif at.label == 'Ni':
                assert at.charge == 3.0
            elif at.label == "O core":
                assert at.charge == 0.2
            elif at.label == 'O shell':
                assert at.charge == -2.2
                
def test_update_springs_in_FitModel(fit_data):
    fitting_parameters = {'dq_O': 0.2, 'q_scaling':0.6, 'Li-O':10.5}
    for arg, value in fitting_parameters.items():
        for data in fit_data.lammps_data:
            for bt in data.bond_types:
                if bt.label == arg:
                    assert bt.spring_coeff_1 == 65.0    
    fit_data._update_springs(fitting_parameters)
    for arg, value in fitting_parameters.items():
        for data in fit_data.lammps_data:
            for bt in data.bond_types:
                if bt.label == arg:
                    assert bt.spring_coeff_1 == 10.5
                                                     
def test_update_potentials_in_FitModel(fit_data):
    fitting_parameters = {'dq_O': 0.2, 'q_scaling':0.6, 'Li-O':10.5, 'Li_O_a':10.0, 'Li_O_rho':0.11}
    for arg, value in fitting_parameters.items():
        for pot in fit_data.potentials:
            if arg == pot.a.label_string:
                assert pot.a.value == 1.0
            if arg == pot.rho.label_string:
                assert pot.rho.value == 0.1                   
    fit_data._update_potentials(fitting_parameters)
    for arg, value in fitting_parameters.items():
        for pot in fit_data.potentials:
            if arg == pot.a.label_string:
                assert pot.a.value == 10.0
            if arg == pot.rho.label_string:
                assert pot.rho.value == 0.11  

def test_update_charge_scaling_in_FitModel(fit_data):
    fitting_parameters = {'dq_O': 0.2, 'q_scaling':0.6, 'Li-O':10.5, 'Li_O_a':10.0, 'Li_O_rho':0.11}
    for arg, value in fitting_parameters.items():
        if arg == 'q_scaling':
            for data in fit_data.lammps_data:
                for at in data.atom_types:
                    if at.label == 'Li':
                        assert at.charge == 1.0
                    elif at.label == 'Ni':
                        assert at.charge == 3.0
                    elif at.label == "O core":
                        assert at.charge == 0.0
                    elif at.label == 'O shell':
                        assert at.charge == -2.0               
    fit_data._update_charge_scaling(fitting_parameters)
    for arg, value in fitting_parameters.items():
        if arg == 'q_scaling':
            for data in fit_data.lammps_data:
                for at in data.atom_types:
                    if at.label == 'Li':
                        assert at.charge == 0.6
                    elif at.label == 'Ni':
                        np.testing.assert_almost_equal(at.charge, 1.8)
                    elif at.label == "O core":
                        np.testing.assert_almost_equal(at.charge, 0.0)
                    elif at.label == 'O shell':
                        np.testing.assert_almost_equal(at.charge, -1.2)

@patch('lammps_potenial_fitting.fitting_code.FitModel._update_charge_scaling')
@patch('lammps_potenial_fitting.fitting_code.FitModel._update_potentials')                        
@patch('lammps_potenial_fitting.fitting_code.FitModel._update_springs')                        
@patch('lammps_potenial_fitting.fitting_code.FitModel._update_q_ratio') 
@patch('lammps_potenial_fitting.fitting_code.FitModel._charge_reset')    
def test_init_potential_in_FitModel(q_reset, q_ratio, springs, pot, q_scaling, fit_data, labels, values):
    fitting_parameters = dict(zip(labels, values))
    fit_data.init_potential(values, labels)
    q_reset.assert_called_with()
    q_ratio.assert_called_with(fitting_parameters)
    springs.assert_called_with(fitting_parameters)
    pot.assert_called_with(fitting_parameters)
    q_scaling.assert_called_with(fitting_parameters)

@patch('lammps_potenial_fitting.fitting_code.FitModel.expected_stresses') 
@patch('lammps_potenial_fitting.fitting_code.FitModel.expected_forces')     
@patch('lammps_potenial_fitting.fitting_code.FitModel.get_forces_and_stresses')  
@patch('lammps_potenial_fitting.fitting_code.FitModel.init_potential')     
def test_chi_squared_error(init_pot, get_fs, expect_f, expect_s,fit_data, labels, values):
    ip_forces = np.array([0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5])
    ip_stresses = np.array([0.1, 0.2, 0.3, 0.2, 0.3, 0.4])
    dft_forces = np.array([0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.5])
    dft_stresses = np.array([0.1, 0.3, 0.4, 0.2, 0.3, 0.2])
    get_fs.return_value = ip_forces, ip_stresses
    expect_f.return_value = dft_forces
    expect_s.return_value = dft_stresses 
    chi = fit_data.chi_squared_error(values, labels)
    init_pot.assert_called_with(values, labels)
    get_fs.assert_called_with()
    np.testing.assert_almost_equal(chi, 0.00223222222)
     
@patch('lammps_potenial_fitting.fitting_code.FitModel._set_charges') 
@patch('lammps_potenial_fitting.fitting_code.FitModel._set_potentials')          
@patch("lammps_potenial_fitting.lammps_data.lammps.Lammps")
def test_get_lattice_params_in_FitModel(mock_lammps, set_pot, set_q, fit_data):
    instances = [lmp.initiate_lmp(fit_data.cs_springs) for lmp in fit_data.lammps_data]
    fit_data.get_lattice_params()
    calls_command = [call(units='metal', style='full', args=['-log', 'none', '-screen', 'none']),
                     call().command('read_data test_files/vasprun_small.xml'),
                     call().command('group cores type 1 2 3'),
                     call().command('group shells type 3'),
                     call().command('pair_style buck/coul/long 10.0'),
                     call().command('pair_coeff * * 0 1 0'),
                     call().command('kspace_style ewald 1e-6'),
                     call().command('min_style cg'),
                     call().command('reset_timestep 0'),
                     call().command('timestep 0.1'),
                     call().command('fix 2 all box/relax aniso 1.0 vmax 0.0005'),
                     call().command('min_style cg'),
                     call().command('minimize 1e-25 1e-25 1000 5000'),
                     call().command('unfix 2')]
    set_pot.assert_called_with(instances[0])
    set_q.assert_called_with(instances[0])
    mock_lammps.assert_has_calls(calls_command)
    
@patch('lammps_potenial_fitting.fitting_code.os.system')
def test_reset_directories(mock_os, fit_data):
    fit_data.reset_directories()
    mock_os.assert_called_with('rm lammps/coords*')