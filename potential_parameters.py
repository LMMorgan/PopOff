import pymc3 as pm

class BuckinghamParameter():
    """
    Class that contains the information for each parameter in the buckingham potentials.
    """
    def __init__(self, parameter_type, label_string, value, sd):
        """
        Initialises each parameter in all buckingham potentials.

        Args:
            parameter_type (str): The label for the specific parameter type i.e. a, rho, or c.
            parameter_string (str): The label defining the two atoms and parameter in the form 'atomtype1_atomtype2_parameter'. Example: 'Li_O_rho'.
            value (float): Value of the parameter.
            sd (float): Standard deviation used in pymc3.
                
        Returns:
            None
        """  
        self.type = parameter_type
        self.label_string = label_string
        self.value = value
        self.sd = sd
        
    def distribution(self):
        """
        Creates pymc3.model.FreeRV object for each parameter. This function must be called within pm.Model. Where pm is from `import pymc3 as pm`.

        Args:
            None
                
        Returns:
            distribution (obj): pymc3.model.FreeRV object for the parameter.
        """
        if self.sd is None:
            raise ValueError('No value given for the standard deviation')
        else:
            distribution = pm.Normal('{}'.format(self.label_string), mu = self.value, sd = self.sd)
            
        return distribution
            
    
def buckingham_parameters(params):
    """
    Extracts and defines each buckingham parameter in the buckingham potentials.
    
    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
    
    Returns:
        parameters (list(obj)): BuckinghamParameter objects including parameter_type (str), label_string (str), value (float), and sd (float).
        
    """
    parameters = []
    for (key1,value1), (key2,value2) in zip(params['bpp'].items(), params['sd'].items()):
        atom_name_1, atom_name_2 = key1.split('-')
        
        parameters.append(BuckinghamParameter(parameter_type = 'a',
                                              label_string = "{}_{}_{}".format(atom_name_1, atom_name_2, 'a'),
                                              value = value1[0],
                                              sd = value2[0]))
        parameters.append(BuckinghamParameter(parameter_type = 'rho',
                                              label_string = "{}_{}_{}".format(atom_name_1, atom_name_2, 'rho'),
                                              value = value1[1],
                                              sd = value2[1]))
        parameters.append(BuckinghamParameter(parameter_type = 'c',
                                              label_string = "{}_{}_{}".format(atom_name_1, atom_name_2, 'c'),
                                              value = value1[2],
                                              sd = value2[2]))
    return parameters