#! /usr/bin/env python3

def check_coreshell(label, bounds, fit_data):
    """
    Checks core-shell inputs into the fitting fuctions are correct. This includes checking the charge ratio between the core and shell is applied to a core-shell atom, that the atoms exist within the system, the upper bound doesn't exceed 2*formal charge, neither bound is <=0, and the lower bound is smaller than the upper bound.
     Args:
        label (str): dQ parameter key, relating to the dQ to be applied to the stated element.
        bounds (tuple(float)): Lower and upper bounds associated with the dQ parameter.
        fit_data (obj(FitModel)): all structural data and associated properties defined, with methods for implementing the fitting process using LAMMPS.
    Returns:
        None
    """
    element = label.replace('dq_', '')   
    for data in fit_data.lammps_data:
        for at in data.atom_types:
            if label.endswith(at.element_type) and at.core_shell == None:
                raise TypeError('"dq_" arguments are for core-shell atoms. This has been implemented on {}, which is a rigid ion atom.'.format(at.element_type))
            elif label.endswith(at.element_type) and at.core_shell == 'core':
                charge = at.formal_charge
        if element not in [at.element_type for at in data.atom_types]:
            raise TypeError('"dq_" arguments are for core-shell atoms. This has been implemented on {}, an atom which does not exist in the system.'.format(element))         
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('dq bounds must be greater than zero.')
    if bounds[1] > abs(charge)*2:
        raise ValueError('dq can not be greater than the absolute atom formal charge. Check upper bound.')
    if bounds[0] > bounds[1]:
        raise ValueError('dq lower bound larger than upper bound.')    
            
def check_scaling_limits(bounds):
    """
    Checks scaling inputs into the fitting fuctions are correct. This includes checking the upper bound doesn't exceed 1.0, neither bound is <=0, and the lower bound is smaller than the upper bound.
    Args:
        bounds (tuple(float)): Lower and upper bounds associated with the scaling parameter.
    Returns:
        None
    """
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('Charge scaling bounds must be greater than zero.')
    if bounds[0] > 1 or bounds[1] > 1:
        raise ValueError('Charge scaling can not be greater than 1. Check bounds.')
    if bounds[0] > bounds[1]:
        raise ValueError('Charge scalling lower bound larger than upper bound.')
        
def check_spring(label, bounds, params):
    """
    Checks core-shell spring inputs into the fitting fuctions are correct. This includes checking the spring is applied to a core-shell atom, that the atoms exist within the system, the core and shell belong to the same element, neither bound is <=0, and the lower bound is smaller than the upper bound.
    Args:
        label (str): core-shell spring parameter key, relating to the spring to be applied between the core and shell of an element.
        bounds (tuple(float)): Lower and upper bounds associated with the spring parameter.
        params (dict(dict)): Setup dictionary containing the inputs for coreshell, charges, masses, potentials, and core-shell springs.
    Returns:
        None
    """
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('{} spring bounds must be greater than zero.'.format(label))
    if bounds[0] > bounds[1]:
        raise ValueError('{} spring lower bound larger than upper bound.'. format(label))
    split_label = label.replace('-', ' ').split()
    for element in split_label[:2]:
        if element not in params['core_shell'].keys():
            raise TypeError('Element {} in label {} not found in structure. Please check your labels and params.'.format(element, label))
        for key, value in params['core_shell'].items():
            if key == element and value == False:
                raise TypeError('Label {} is a spring for a coreshell atom. This is not a coreshell atom. Please check your labels and params.'.format(label))
    if split_label[0] != split_label[1]:
        raise TypeError("In label {} that atoms don't match. A coreshell spring must be between the core and shell of the same atom type.".format(label))
        
def check_buckingham(label, bounds, params):
    """
    Checks the buckingham parameter input into the fitting fuction is correct. This includes checking the parameter is correctly formatted for a buckingham potential, that the elements exist within the system, neither bound is <=0, and the lower bound is smaller than the upper bound.
    Args:
        label (str): buckingham parameter key, relating to the buckingham parameter to be fitted.
        bounds (tuple(float)): Lower and upper bounds associated with the buckingham parameter.
        params (dict(dict)): Setup dictionary containing the inputs for coreshell, charges, masses, potentials, and core-shell springs.
    Returns:
        None
    """
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise ValueError('{} bounds must be greater than zero.'.format(label))    
    if bounds[0] > bounds[1]:
        raise ValueError('{} lower bound larger than upper bound.'. format(label))    
    individual_labels = label.split('_')
    for element in individual_labels[:2]:
        if element not in params['core_shell'].keys():
            raise TypeError('Element {} in label {} not found in structure. Please check your labels and params.'.format(element, label))
    for element in individual_labels[2:]:
        if element not in ['a', 'rho', 'c']:
            raise TypeError('Parameter {} in label {} is not a buckingham parameter. Please check your labels and params.'.format(element, label))       
            
def setup_error_checks(include_labels, bounds_list, fit_data, params):
    """
    Checks the labels list and bounds list are the same length, then iterated through each item to run specific checks ensuring the labels and associated bounds are appropriate.
    Args:
        include_labels (list(str)): List of parameters to be fitted.
        bounds_list (list(tuple(float))): List of lower and upper bound tuples associated with each parameter.
        fit_data (obj(FitModel)): all structural data and associated properties defined, with methods for implementing the fitting process using LAMMPS. 
        params (dict(dict)): Setup dictionary containing the inputs for coreshell, charges, masses, potentials, and core-shell springs.
    Returns:
        None
    """ 
    if len(include_labels) != len(bounds_list):
        raise IndexError('include_labels and bounds_list are not of equal length. Check there are bounds associated with each label with the correct bound values.')
    for label, bounds in zip(include_labels, bounds_list):
        if label.startswith('dq_'):
            check_coreshell(label, bounds, fit_data)
        elif label == 'q_scaling':
            check_scaling_limits(bounds)
        elif '-' in label:
            check_spring(label, bounds, params)
        elif '_a' in label or '_rho' in label or '_c' in label:
            check_buckingham(label, bounds, params)
        else:
            raise TypeError('Label {} is not a valid label type'.format(label))