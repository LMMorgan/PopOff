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
