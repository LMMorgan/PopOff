class BondType():
    """
    Class for each bond type.
    """
    
    def __init__(self, bond_type_index, label):
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