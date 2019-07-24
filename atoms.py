class Atom():
    """
    Class for each atom.
    """
    
    def __init__(self, atom_index, molecule_index, coords, atom_forces, atom_type):
        """
        Initialise an instance for each atom in the structure.

        Args:
            atom_index (int): Individual atom number.
            molecule_index (int): Index of the molecule the atom belongs to i.e. core and shell will be the same molecule, but indivdual atoms will be separate.
            atom_type_id (int): Integer value given for the atom type.
            coords (np.array): x, y, and z positions of the atom.
            atom_forces (np.array): x, y, and z forces of the atom.
            atom_type (obj): AtomType object including atom_type_index (int), label (str), mass (float),
                             charge (float), and core_shell (str).
                
        Returns:
            None
        """
        self.atom_index = atom_index
        self.molecule_index = molecule_index
        self.coords = coords
        self.forces = atom_forces
        self.atom_type = atom_type
        
    def input_string(self):
        """
        Defines the a string in a lammps input file format for the atom data.

        Args:
            None
                
        Returns:
            (str): contains atom_index (int), molecule (int), atom_type_id (int), charge(float),
                   and coordinates (list(float)) for lammps file.
        """
        return '{:<4} {:<4} {:<4} {: 1.4f}  {: 2.6f}  {: 2.6f}  {: 2.6f}'.format( 
            self.atom_index, self.molecule_index, self.atom_type.atom_type_index, self.atom_type.charge, *self.coords )