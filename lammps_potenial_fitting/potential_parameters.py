class BuckinghamParameter():
    """
    Class that contains the information for each parameter in the buckingham potentials.
    """
    def __init__(self, label_string, param_type, value):
        """
        Initialises each parameter in all buckingham potentials.

        Args:
            label_string (str): The label defining the two atoms and parameter in the form 'atomtype1_atomtype2_parameter'. Example: 'Li_O_rho'.
            param_type (str): The label for the specific parameter type i.e. a, rho, or c.
            value (float): Value of the parameter.
                
        Returns:
            None
        """
        
        if not isinstance(label_string, str):
            raise TypeError('The label must be of type string.')        
        if not isinstance(param_type, str):
            raise TypeError('The param_type must be of type string. Example: "a","rho","c".')
        if not isinstance(value, float):
            raise TypeError('The value must be of type float.')
        if '_' not in label_string or len(label_string.split('_')) != 3:
            raise ValueError('Label string must be in format "atomtype1_atomtype2_parameter". Example: "Li_O_rho".')
        if param_type not in ['a', 'rho', 'c']:
            raise ValueError('param_type argument should be "a","rho", or "c".')
            
            
        self.label_string = label_string
        self.param_type = param_type
        self.value = value
            
    
def buckingham_parameters(potentials):
    """
    Extracts and defines each buckingham parameter in the buckingham potentials.
    
    Args:
        potentials(dict): Contains buckingham potentials list(float), where the potentials keys are atom label pairs (str), example: 'Li-O'.
    
    Returns:
        parameters (list(obj)): BuckinghamParameter objects including label_string (str), param_type (str), and value (float).
        
    """
    if not isinstance(potentials, dict):
        raise TypeError('Potentials should be stored in a dictionary. The keys should be at the atom pairs as a string (i.e. "Li-O") and the values a list of buckingham a, rho, c values.')

    for key, pot in potentials.items():
        if not isinstance(key, str):
            raise TypeError('Potentials keys should be at the atom pairs as a string (i.e. "Li-O").')
        if '-' not in key or len(key.split('-')) != 2:
            raise ValueError('Potentials keys should be at the atom pairs as a string (i.e. "Li-O").') 
        if not isinstance(pot, list):
            raise TypeError('Potentials values should be a list of buckingham a, rho, c values.')
        for parameter in pot:
            if not isinstance(parameter, float):
                raise TypeError('Potentials values must be of type float.')
                
    parameters = []
    
    for key, value in potentials.items():

        at1, at2 = key.split('-') #at is atom_type
        parameters.append(BuckinghamParameter(label_string = "{}_{}_{}".format(at1, at2, 'a'),
                                              param_type = 'a',
                                              value = value[0]))
        parameters.append(BuckinghamParameter(label_string = "{}_{}_{}".format(at1, at2, 'rho'),
                                              param_type = 'rho',
                                              value = value[1]))
        parameters.append(BuckinghamParameter(label_string = "{}_{}_{}".format(at1, at2, 'c'),
                                              param_type = 'c',
                                              value = value[2]))
    return parameters
