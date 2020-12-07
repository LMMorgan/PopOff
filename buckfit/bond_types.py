class BondType():
    """
    Class for each bond type.
    """
    
    def __init__(self, bond_type_index, label, spring_coeff_1, spring_coeff_2=0.0):
        """
        Initialise an instance for each bond type in the structure.

        Args:
            bond_type_index (int): Integer value given for the bond type. 
            label (str): Identity of the bond atoms format "element_1_index-element_2_index spring", where element_1 is the core, and element_2 is the shell.
            spring_coeff_1 (float): Spiring coefficient value
            spring_coeff_2 (float): Spring coefficient value, default=0.0.
                
        Returns:
            None
        """  
        
        if not isinstance(bond_type_index, int) or  isinstance(bond_type_index, bool):
            raise TypeError('The bond type index must be an integer.')
        if not isinstance(label, str):
            raise TypeError('The label must be of type string.')
        if not isinstance(spring_coeff_1, float) or not isinstance(spring_coeff_2, float):
            raise TypeError('The spring coeffients must be of type float.')
            
        if '-' not in label:
            raise ValueError('Label string must be in format "element_1_index-element_2_index spring", where element_1 is the core, and element_2 is the shell.')        
        

        self.bond_type_index = bond_type_index
        self.label = label
        self.spring_coeff_1 = spring_coeff_1
        self.spring_coeff_2 = spring_coeff_2
        
    def bond_string(self):
        return_str = 'bond_coeff {} {:6.2f} {:6.2}'.format(self.bond_type_index,
                                                           self.spring_coeff_1,
                                                           self.spring_coeff_2)
        return return_str   