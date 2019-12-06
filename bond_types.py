class BondType():
    """
    Class for each bond type.
    """
    
    def __init__(self, bond_type_index, label, spring_coeff_1, spring_coeff_2):
        """
        Initialise an instance for each bond type in the structure.

        Args:
            bond_type_index (int): Integer value given for the bond type. 
            label (str): Identity of the bond atoms format "element_1_index-element_2_index spring", where element_1 is the core, and element_2 is the shell.
                
        Returns:
            None
        """  
        self.bond_type_index = bond_type_index
        self.label = label
        self.spring_coeff_1 = spring_coeff_1
        self.spring_coeff_2 = spring_coeff_2
        
    def bond_string(self):
        return_str = 'bond_coeff {} {:6.2f} {:6.2}'.format(self.bond_type_index,
                                                           self.spring_coeff_1,
                                                           self.spring_coeff_2)
        return return_str   