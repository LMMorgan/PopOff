#! /usr/bin/env python3
import pytest
import numpy as np
from mock import patch, call, Mock, mock_open
from lammps_potenial_fitting.fitting_code import FitModel
from lammps_potenial_fitting.collate_structures import data_from_vasprun
from lammps_potenial_fitting.cross_validation import (validation_sets, chi_squared_error,
                                                      save_cv_data, run_cross_validation,
                                                      setup_error_dict, plot_cross_validation)

def test_validation_sets_in_cross_validation():
    sets_of_structs = validation_sets(2, 15, 5, np.array([2,4,6,8,10]), seed=7)
    np.testing.assert_array_equal(sets_of_structs, np.array([[3,7,9,11,12],[1,7,11,14,15]]))
    
def test_chi_squared_error_in_cross_validation(force_stress):
    chi = chi_squared_error(force_stress[2], force_stress[0], force_stress[3], force_stress[1])
    np.testing.assert_almost_equal(chi, 0.00223222222)    

@patch("builtins.open", new_callable=mock_open)
@patch('lammps_potenial_fitting.cross_validation.np.savetxt')
def test_save_cv_data_in_cross_validation(mock_savetxt, mock_open, force_stress):
    save_cv_data('test_files', np.array([1,2]), 0.00223222222,
                 force_stress[2],force_stress[0],force_stress[3],force_stress[1])
    calls_save = [call('test_files/s1-2_dft_forces.dat', force_stress[2], fmt='%.10e', delimiter=' '),
                  call('test_files/s1-2_ip_forces.dat', force_stress[0], fmt='%.10e', delimiter=' '),
                  call('test_files/s1-2_dft_stresses.dat', force_stress[3], fmt='%.10e', delimiter=' '),
                  call('test_files/s1-2_ip_stresses.dat', force_stress[1], fmt='%.10e', delimiter=' ')]
    mock_savetxt.assert_has_calls(calls_save)
    calls_open = [call('test_files/s1-2_error.dat', 'w'),
                  call().__enter__(),
                  call().write(str(0.00223222222)),
                  call().__exit__(None, None, None)]
    mock_open.assert_has_calls(calls_open)

@patch('lammps_potenial_fitting.cross_validation.save_cv_data')
@patch('lammps_potenial_fitting.cross_validation.chi_squared_error')
@patch('lammps_potenial_fitting.cross_validation.FitModel.reset_directories')
@patch('lammps_potenial_fitting.cross_validation.FitModel.collect_info')
@patch('lammps_potenial_fitting.cross_validation.validation_sets')
@patch('lammps_potenial_fitting.cross_validation.fit_out')    
def test_run_cross_validation_in_cross_validation(fit_out, val_set, fit_collect, fit_reset, chi, save, params, force_stress, fit_data):
    head_dir = 'test_files/test_outputs'
    head_out = 'test_files/test_outputs/outputs'
    labels = ['q_scaling', 'Li_O_a', 'Li_O_rho']
    val_set.return_value = [[1,2,3], [4,5,6]]
    temp = Mock(spec=FitModel)
    temp.reset_directories = fit_reset
    fit_collect.return_value = temp
    fit_out.create_directory.return_value = 'test2'
    fit_out.extract_stresses_and_forces.return_value = force_stress

    run_cross_validation(1, 15, 5, head_dir, head_out, params, supercell=None, seed=7)
    
    fit_out_calls = [call.create_directory('test_files/test_outputs/outputs', 'p1'),
                     call.extract_stresses_and_forces(temp, [0.5, 1.0, 0.1], labels),
                     call.extract_stresses_and_forces(temp, [0.5, 1.0, 0.1], labels),
                     call.create_directory('test_files/test_outputs/outputs', 'p2'),
                     call.extract_stresses_and_forces(temp, [0.6, 2.0, 0.2], labels),
                     call.extract_stresses_and_forces(temp, [0.6, 2.0, 0.2], labels)]
    fit_out.assert_has_calls(fit_out_calls)
    val_set.assert_called_with(1, 15, 5, np.array([2]), seed=False)
    fit_collect.assert_called_with(params, [4, 5, 6], supercell=None)
    fit_reset.assert_called_with()
    
#     np.testing.assert_almost_equal(chi(force_stress[2], force_stress[0], force_stress[3], force_stress[1]), 0.00223222222)
#     chi.assert_called_with(force_stress[2], force_stress[0], force_stress[3], force_stress[1])
#     save.assert_called_with('test2', [4,5,6], chi, force_stress[2], force_stress[0], force_stress[3], force_stress[1])




# # @patch('lammps_potenial_fitting.cross_validation.save_cv_data')
# @patch('lammps_potenial_fitting.cross_validation.FitModel.collect_info')
# @patch('lammps_potenial_fitting.lammps_data.LammpsData.write_lammps_files')
# @patch('lammps_potenial_fitting.collate_structures.collate_structural_data')
# @patch('lammps_potenial_fitting.cross_validation.validation_sets')
# @patch('lammps_potenial_fitting.cross_validation.fit_out.create_directory') 
# def test_run_cross_validation_in_cross_validation(create_dir, val_set, struct_dat, mock_write, fit_info, buckingham_potential, params, fit_data):
#     head_dir = 'test_files/test_outputs'
#     head_out = 'test_files/test_outputs/outputs'
#     labels = ['q_scaling', 'Li_O_a', 'Li_O_rho']
#     val_set.return_value = [1]
#     struct_dat.return_value = [data_from_vasprun(params, 'test_files/vasprun_small.xml', 1, supercell=None)]
#     mock_write.assert_called_with()
#     fit_data = FitModel([buckingham_potential], struct_dat, cs_springs=None)
#     fit_info.return_value = fit_data
#     s = run_cross_validation(1, 15, 5, head_dir, head_out, params, supercell=None, seed=7)

# #     print(dft_f, ip_f, dft_s, ip_s)
# #     dftf, ipf, dfts,ips = run_cross_validation(1, 15, 5, head_dir, head_out, params, supercell=None, seed=7)
# #     create_dir_calls = [call(head_out, 'p1'), call(head_out, 'p2')]
# #     create_dir.assert_has_calls(create_dir_calls)
# #     val_set.assert_called_with(1, 15, 5, np.array([1]), seed=False)











    
    
    
    
    
    
    
    
    
# def test_setup_error_dict_in_cross_validation():
#     head_dir = 'test_files/test_cv'
#     error_dict = setup_error_dict(head_dir)
#     assert error_dict == {'1': 0.1}    

# @patch("lammps_potenial_fitting.cross_validation.plt")
# def test_plot_cross_validation_in_cross_validation(mock_plt):
#     error_dict = {'1': 0.1, '2': 0.2}
#     plot_cross_validation(error_dict, 'test/p1', 'test', xlabel_rotation=50, title='default', save=True)
#     calls_command = [call.scatter(*zip(*sorted(error_dict.items()))),
#                      call.xticks(rotation=50),
#                      call.xlabel('cross-validation structure numbers'),
#                      call.ylabel('$\chi^2$ error'),
#                      call.title('Potential 1 cross-validation fit errors ($\chi^2$)'),
#                      call.savefig('test/p1_cv_fit_errors.png',dpi=500, bbox_inches = "tight"),
#                      call.show()]
#     mock_plt.assert_has_calls(calls_command)