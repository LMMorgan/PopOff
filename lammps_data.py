import numpy as np
from from_structure import types_from_structure, atoms_and_bonds_from_structure


class LammpsData():
    """
    Class that collates all structural information for outputing a Lammps format.
    """
    def __init__(self, atom_types, bond_types, atoms, bonds, cell_lengths, tilt_factor, file_name):
        """
        Initialise an instance for all information relating to the pysical and electronic structure needed for the Lammps input.

        Args:
            atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float), charge (float), and core_shell (str).
            bond_types (list(obj)): BondType objects including bond_type_index (int) and label (str).
            atoms (list(obj)): Atom objects including atom_index (int), molecule_index (int), coords (np.array), forces (np.array), and atom_type (obj:AtomType).
            bonds (list(obj)): Bond objects including bond_index (int), atom_indices (list(int)), and bond_type (obj:BondType).
            cell_lengths (list(float)): Lengths of each cell direction.
            tilt_factor (list(float)): Tilt factors of the cell.
            file_name (str): Name of lammps formatted file to be written.
                
        Returns:
            None
        """  
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.atoms = atoms
        self.bonds = bonds
        self.cell_lengths = cell_lengths
        self.tilt_factor = tilt_factor
        self.file_name = file_name
        
    @classmethod
    def from_structure(cls, structure, params, forces, i):
        """
        Collects information from initial structure, params, forces, and an index.

        Args:
            structure (obj): A pymatgen structural object created from a POSCAR.
            params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
            forces (np.array): 2D numpy array containing the x,y,z forces acting on each atom in the given POSCAR (without coreshell)
            i (int): index identifier of the file in the list of files, i.e. 1 for POSCAR1 and OUTCAR1.
        Returns:
            (LammpsData):  LammpsData object containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), and file_name (str).          
        """  
        atom_types, bond_types = types_from_structure( structure=structure, 
                                       core_shell=params['core_shell'], 
                                       charges=params['charges'], 
                                       masses=params['masses'], verbose=True )
        atoms, bonds = atoms_and_bonds_from_structure( structure, forces, atom_types, bond_types )
        cell_lengths, tilt_factor = lammps_matrix(structure)
        file_name = 'lammps/coords{}.lmp'.format(i+1)

        return cls( atom_types, bond_types, atoms, bonds, cell_lengths, tilt_factor, file_name)
    
    def header_string( self, title='title' ):
        """
        Prints the top part for the lammps input file.

        Args:
            title (optional:str): Title for lammps file, default = 'title'.
                
        Returns:
            return_str (str): Contains title, atoms, bonds, atom types, and bond types information for the lammps input file.
        """  
        return_str = ''
        return_str += '{}\n\n'.format( title )
        return_str += '{}   atoms\n'.format( len(self.atoms) )
        return_str += '{}   bonds\n\n'.format( len(self.bonds) )
        return_str += '{}   atom types\n'.format( len(self.atom_types ) )
        return_str += '{}   bond types\n\n'.format( len(self.bond_types ) )
        return return_str
        
    def cell_dimensions_string(self):
        """
        Prints the cell dimensions for the lammps file.
        
        Args:
            None
        
        Returns:
            return_str (str): Contains cell dimensions, tilt factors, and related labels for the lammps input file.
        """
        return_str = ''
        return_str += '0.0 {:2.6f} xlo xhi\n0.0 {:2.6f} ylo yhi\n0.0 {:2.6f} zlo zhi\n\n'.format(*self.cell_lengths)
        return_str += '{:2.5f} {:2.5f} {:2.5f} xy xz yz \n\n'.format(*self.tilt_factor)
        
        return return_str
        
    def masses_string(self):
        """
        Prints the mass information for each species for the lammps file.
        
        Args:
            None
            
        Returns:
            return_str (str): Contains atom types, masses, and commented atom type label for the lammps input file.
        """
        return_str = 'Masses\n\n'
        for at in self.atom_types:
            return_str += '{} {:9.5f} # {}\n'.format( at.atom_type_index, float(at.mass), at.label)
        return_str += '\n'
        return return_str
    
    def atoms_string(self):
        """
        Prints the atoms information for the lammps file.
        
        Args:
            None
            
        Returns:
            return_str (str): Contains atom information for the lammps input file as defined by the Atom class.
        """
        return_str = 'Atoms\n\n'
        for atom in self.atoms:
            return_str += '{}\n'.format(atom.input_string())
        return_str += '\n'
        return return_str
    
    def bonds_string(self):
        """
        Prints the bonds information for the lammps file.
        
        Args:
            None
            
        Returns:
            return_str (str): Contains bond information for the lammps input file as defined by the Bond class.
        """
        return_str = 'Bonds\n\n'
        for bond in self.bonds:
            return_str += '{}\n'.format(bond.input_string())
        return_str += '\n'
        return return_str
        
    def input_string(self, title='title'):
        """
        Prints all information for the lammps file.
        
        Args:
            title (optional:str): Title for lammps file, default = 'title'.
            
        Returns:
            return_str (str): Contains all information for the lammps input file as defined by the class methods above.
        """
        return_str = ''
        return_str += self.header_string(title)
        return_str += self.cell_dimensions_string()
        return_str += self.masses_string()
        return_str += self.atoms_string()
        return_str += self.bonds_string()
        return return_str

    def core_mask(self):
        """
        Creates a boolean mask, returing True for all but shell atoms.
        
        Args:
            None
            
        Returns:
            mask (list(bool)): Returns False for shell atoms and True for all others.
        """
        mask = []
        for atom in self.atoms:
            if "shell" not in atom.atom_type.label:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
    def type_core(self):
        """
        Returns a string containing the atom type index of all non-shell atoms.
        
        Args:
            None
            
        Returns:
            type_core (str): Atom type index of all non-shell atoms as a string separated by a single space.
        """
        type_core = ' '.join(['{}'.format(atom.atom_type_index) for atom in self.atom_types
                              if 'shell' not in atom.label])
        return type_core
    
    def type_shell(self):
        """
        Returns a string containing the atom type index of all shell atoms.
        
        Args:
            None
            
        Returns:
            type_shell (str): Atom type index of all shell atoms as a string separated by a single space.
        """
        type_shell = ' '.join(['{}'.format(atom.atom_type_index) for atom in self.atom_types
                               if 'shell' in atom.label])
        return type_shell
    
def lammps_matrix(structure):
    """
    Defines the cell dimensions of the system, imposing a transformation for left/right-hand basis if needed.
    
    Args:
        structure (obj): A pymatgen structural object created from a POSCAR.
    
    Returns:
        cell_lengths (list(float)): Lengths of each cell direction.
        tilt_factor (list(float)): Tilt factors of the cell.
        
    """
    a, b, c = structure.lattice.lengths
    alpha, beta, gamma = np.deg2rad(structure.lattice.angles)
    ax = a
    bx = b*np.cos(gamma)
    by = b*np.sin(gamma)
    cx = c*np.cos(beta)
    cy = ( np.dot(b,c) - (bx*cx) )/by
    cz = np.sqrt(c**2-cx**2-cy**2)
    cell_lengths = [ax, by, cy]
    tilt_factor = [bx, cx, cz]
    return cell_lengths, tilt_factor    