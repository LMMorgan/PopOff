from buckfit.bond_types import BondType

class Bond():
    """
    Class for each bond present between core-shell atoms.
    """
    
    def __init__(self, bond_index, atom_indices, bond_type):
        """
        Initialise an instance for each bond in the structure.

        Args:
            bond_index (int): Individual bond number.
            atom_indices (list(int)): Index numbers of the 2 atoms in the bond.
            bond_type (obj): BondType object including bond_type_index (int) and label (str).
                
        Returns:
            None
        """
        if not isinstance(bond_index, int) or  isinstance(bond_index, bool):
            raise TypeError('The bond_index must be an integer.')
        if not isinstance(atom_indices, list) or len(atom_indices) != 2:
            raise TypeError('The atom_indices must be a list of length 2.')
        for item in atom_indices:
            if not isinstance(item, int) or isinstance(item,bool):
                raise ValueError('Both items in the atom_indices list must be integers.')
        if not isinstance(bond_type, BondType):
            raise TypeError('The bond_type must be a BondType object.')            
            
        self.bond_index = bond_index
        self.atom_indices = atom_indices
        self.bond_type = bond_type
        
    def input_string(self):
        """
        Defines the a string in a lammps input file format for the bond data.

        Args:
            None
                
        Returns:
            (str): Containing bond_index (int), bond_type.bond_type_index (int), and
                   atom_indices (list(int)) for lammps file.
        """       
        return '{:<4} {:<4} {:<4} {:<4}'.format( self.bond_index, self.bond_type.bond_type_index, *self.atom_indices )