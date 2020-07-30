class AtomType():
    """
    Class for each atom type.
    """
    
    def __init__( self, atom_type_index, label, element_type, mass, charge, core_shell=None ):
        """
        Initialise an instance for each atom type in the structure.

        Args:
            atom_type_index (int): Integer index for this atom type.
            label (str): Label used to identify this atom type.
            element_type (str): Elemental symbol for atom type.
            mass (float): Mass of the atom type.
            charge(float): Charge of the atom type.
            core_shell (optional:str): 'core' or 'shell'. Default is None.
                
        Returns:
            None
        """
        if not isinstance(atom_type_index, int) or isinstance(atom_type_index, bool):
            raise TypeError('The atom type index must be an integer.')
        if not isinstance(label, str):
            raise TypeError('The label must be of type string.')
        if not isinstance(element_type, str):
            raise TypeError('The element type must be of type string.')
        if not isinstance(mass, float):
            raise TypeError('The mass must be a float.')
        if not isinstance(charge, float):
            raise TypeError('The charge must be a float.')
        if core_shell not in ['core', 'shell', None]:
            raise ValueError('core_shell argument should be "core" or "shell"')
            
        self.atom_type_index = atom_type_index
        self.label = label
        self.element_type = element_type
        self.mass = mass
        self.charge = charge
        self.formal_charge = charge
        self.core_shell = core_shell

    @property
    def core_shell_string(self):
        """
        Defines a string for a comment in a lammps input file format labelling cores/shells.

        Args:
            None
                
        Returns:
            core_shell (str): Either 'core', 'shell', or '' if core_shell is None. 
        """  
        if self.core_shell is None:
            return ''
        return self.core_shell