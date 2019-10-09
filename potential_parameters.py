import pymc3 as pm

class BuckinghamParameter():
    """
    Class that contains the information for each parameter in the buckingham potentials.
    """
    def __init__(self, label_string, param_type, value, sd):
        """
        Initialises each parameter in all buckingham potentials.

        Args:
            label_string (str): The label defining the two atoms and parameter in the form 'atomtype1_atomtype2_parameter'. Example: 'Li_O_rho'.
            param_type (str): The label for the specific parameter type i.e. a, rho, or c.
            value (float): Value of the parameter.
            sd (float): Standard deviation used in pymc3 for Normal distribution.
                
        Returns:
            None
        """  
        self.label_string = label_string
        self.param_type = param_type
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
            raise ValueError('No value given for the standard deviation. Required for Normal distribution')
        else:
            distribution = pm.Normal('{}'.format(self.label_string), mu = self.value, sd = self.sd)
                
        return distribution
            
    
def buckingham_parameters(distribution):
    """
    Extracts and defines each buckingham parameter in the buckingham potentials.
    
    Args:
        distribution(dict(dict)): Contains buckingham potential, 'bpp':list(float), and 'sd':list(float) dictionaries where the distribution keys are atom label pairs (str), example: 'Li-O'.
    
    Returns:
        parameters (list(obj)): BuckinghamParameter objects including label_string (str), param_type (str), value (float), and sd (float).
        
    """
    parameters = []
    
    for key, value in distribution.items():
#         try:
#             value['sd']
#         except KeyError:
#             raise ValueError('No value given for the standard deviation. Required for Normal distribution')
        
        at1, at2 = key.split('-') #at is atom_type
        parameters.append(BuckinghamParameter(label_string = "{}_{}_{}".format(at1, at2, 'a'),
                                              param_type = 'a',
                                              value = value['bpp'][0],
                                              sd = value['sd'][0]))
        parameters.append(BuckinghamParameter(label_string = "{}_{}_{}".format(at1, at2, 'rho'),
                                              param_type = 'rho',
                                              value = value['bpp'][1],
                                              sd = value['sd'][1]))
        parameters.append(BuckinghamParameter(label_string = "{}_{}_{}".format(at1, at2, 'c'),
                                              param_type = 'c',
                                              value = value['bpp'][2],
                                              sd = value['sd'][2]))
    return parameters
