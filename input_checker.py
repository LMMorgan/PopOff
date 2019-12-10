#! /usr/bin/env python3

def check_coreshell(label, fit_data, bounds):
    for data in fit_data.lammps_data:
        for at in data.atom_types:
            if label.endswith(at.element_type) and at.core_shell == None:
                raise TypeError('"dq_" arguments are for core-shell atoms. This has been implemented on {}, which is a rigid ion atom.'.format(at.element_type))
            elif label.endswith(at.element_type) and at.core_shell == 'core':
                charge = at.formal_charge
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('dq bounds must be greater than zero.')
    if bounds[1] > abs(charge)*2:
        raise ValueError('dq can not be greater than the absolute atom formal charge. Check upper bound.')
    if bounds[0] > bounds[1]:
        raise ValueError('dq lower bound larger than upper bound.')    
            
def check_scaling_limits(label, bounds):
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('Charge scaling bounds must be greater than zero.')
    if bounds[1] > 1:
        raise ValueError('Charge scaling can not be greater than 1. Check upper bound.')
    if bounds[0] > bounds[1]:
        raise ValueError('Charge scalling lower bound larger than upper bound.')
        
def check_spring(label, bounds):
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('{} spring bounds must be greater than zero.'.format(label))