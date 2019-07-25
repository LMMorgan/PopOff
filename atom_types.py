class AtomType():
    """
    Class for each atom type.
    """
    
    def __init__( self, atom_type_index, label, mass, charge, core_shell=None ):
        """
        Initialise an instance for each atom type in the structure.

        Args:
            atom_type_index (int): Integer index for this atom type.
            label (str): Label used to identify this atom type.
            mass (float): Mass of the atom type.
            charge(float): Charge of the atom type.
            core_shell (optional:str): 'core' or 'shell'. Default is None.
                
        Returns:
            None
        """        
        self.atom_type_index = atom_type_index
        self.label = label
        self.mass = mass
        self.charge = charge
        if core_shell not in ['core', 'shell', None]:
            raise ValueError('core_shell argument should be "core" or "shell"')
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