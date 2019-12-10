#! /usr/bin/env python3
            
def generate_include_list(fitting_list, params, interacting_atoms=None):
    include_list = []
    for item in fitting_list:
        if item == 'q':
            for key, value in params['core_shell'].items():
                if value == True:
                    include_list.append('dq_{}'.format(key))
            include_list.append('q_scaling')
        elif item == 'coreshell':
            for key, value in params['cs_springs'].items():
                include_list.append('{} spring'.format(key))
        elif item is interacting_atoms:
            for i, j in interacting_atoms:
                [include_list.append('{}_{}_{}'.format(i,j,x)) for x in ['a', 'rho']]
    return include_list
