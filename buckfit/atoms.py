import numpy as np
from buckfit.atom_types import AtomType

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
            coords (np.array): x, y, and z positions of the atom.
            atom_forces (np.array): x, y, and z forces of the atom.
            atom_type (obj): AtomType object including atom_type_index (int), label (str), mass (float), charge (float), formal_charge (float), and core_shell (str).
                
        Returns:
            None
        """
        if not isinstance(atom_index, int) or  isinstance(atom_index, bool):
            raise TypeError('The atom_index must be an integer.')
        if not isinstance(molecule_index, int) or  isinstance(molecule_index, bool):
            raise TypeError('The molecule_index must be an integer.')
        if not isinstance(coords, np.ndarray):
            raise TypeError('coords must be a numpy.array.')
        if np.shape(coords) != (3,):
            raise ValueError('coords np.array must be 1-d with 3 components containing the x y z coordinates, respectively.')
        if not isinstance(atom_forces, np.ndarray):
            raise TypeError('atom_forces must be a numpy.array.')
        if np.shape(atom_forces) != (3,):
            raise ValueError('atom_forces np.array must be 1-d with 3 components containing the x y z coordinates, respectively.')
        if not isinstance(atom_type, AtomType):
            raise TypeError('The atom_type must be a AtomType object.')
        
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
