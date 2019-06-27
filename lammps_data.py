from collections import Counter
from itertools import count

class Atom():
    
    def __init__(self, atom_index, molecule, atom_type_id, charge, coords):
        self.atom_index = atom_index
        self.molecule = molecule
        self.atom_type_id = atom_type_id
        self.charge = charge
        self.coords = coords
        
    def input_string(self):
        return '{:<4} {:<4} {:<4} {: 1.4f}  {: 2.6f}  {: 2.6f}  {: 2.6f}'.format( 
            self.atom_index, self.molecule, self.atom_type_id, self.charge, *self.coords )
    
class Bond():
    
    def __init__(self, bond_index, bond_type_id, atom_indices):
        self.bond_index = bond_index
        self.bond_type_id = bond_type_id
        self.atom_indices = atom_indices
        
    def input_string(self):
        return '{:<4} {:<4} {:<4} {:<4}'.format( self.bond_index, self.bond_type_id, *self.atom_indices )
        
class BondType():
    
    def __init__(self, bond_type_index, label):
        self.bond_type_index = bond_type_index
        self.label = label

class AtomType():
    """
    Class for each atom type.
    """
    
    def __init__( self, atom_type_index, label, 
                  mass, charge, core_shell=None ):
        """
        Initialise an AtomType instance.

        Args:
            atom_type_index (int): Integer index for this atom type.
            label (str): Label used to identify this atom type.
            mass (float): Mass of the atom type.
            charge(float): Charge of the atom type.
            core_shell (optional:str):  'core' or 'shell'. 
                Default is None.
                
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
        Defines a string for a comment in lammps output file labelling cores/shells.

        Args:
            None
                
        Returns:
            core_shell(str): Either 'core', 'shell', or '' if core_shell is None. 
        """  
        if self.core_shell is None:
            return ''
        return self.core_shell
    
def types_from_structure( structure, core_shell, charges, masses, verbose=True ):
    atom_types = []
    bond_types = []
    atom_type_index = 0
    bond_type_index = 0
    elements =list(Counter(structure.species).keys())
    if verbose:
        print( "Found elements: {}".format( [e.name for e in elements]))
    for e in elements:
        if e.name not in core_shell:
            raise ValueError( '{} not in core_shell dictionary'.format(e.name) )
        if e.name not in charges:
            raise ValueError( '{} not in charges dictionary'.format(e.name) )
        if e.name not in masses:
            raise ValueError( '{} not in masses dictionary'.format(e.name) )
    for e in elements:
        if core_shell[e.name]: # Create two atom_tyoes for core + shell
            atom_type_index += 1
            atom_types.append( AtomType(atom_type_index=atom_type_index,
                                        label='{} core'.format(e.name),
                                        mass=masses[e.name]['core'],
                                        charge=charges[e.name]['core'],
                                        core_shell='core') )
            atom_type_index += 1
            atom_types.append( AtomType(atom_type_index=atom_type_index,
                            label='{} shell'.format(e.name),
                            mass=masses[e.name]['shell'],
                            charge=charges[e.name]['shell'],
                            core_shell='shell') )
            bond_type_index += 1
            bond_types.append( BondType(bond_type_index=bond_type_index,
                                       label='{}-{} spring'.format(e.name, e.name)))
        else:
            atom_type_index += 1
            atom_types.append( AtomType(atom_type_index=atom_type_index,
                            label='{}'.format(e.name),
                            mass=masses[e.name],
                            charge=charges[e.name] ) )
    return atom_types, bond_types

def atoms_and_bonds_from_structure( structure, atom_types, bond_types ):
    atoms = []
    bonds = []
    atom_types_dict = {}
    for at in atom_types:
        atom_types_dict[at.label] = at
    bond_types_dict = {}
    for bt in bond_types:
        bond_types_dict[bt.label] = bt
    atom_index = 0
    bond_index = 0
    molecule_index = 0
    for site in structure.sites:
        molecule_index += 1
        if site.species_string in atom_types_dict: # not core-shell atom
            atom_index += 1
            atom_type = atom_types_dict[site.species_string]
            atoms.append( Atom(atom_index=atom_index,
                               molecule=molecule_index,
                               atom_type_id=atom_type.atom_type_index,
                               charge=atom_type.charge,
                               coords=site.coords ) )
        else: # need to handle core + shell
            atom_index += 1
            atom_type = atom_types_dict[site.species_string + ' core']
            atoms.append( Atom(atom_index=atom_index,
                               molecule=molecule_index,
                               atom_type_id=atom_type.atom_type_index,
                               charge=atom_type.charge,
                               coords=site.coords ) )
            atom_index += 1
            atom_type = atom_types_dict[site.species_string + ' shell']
            atoms.append( Atom(atom_index=atom_index,
                               molecule=molecule_index,
                               atom_type_id=atom_type.atom_type_index,
                               charge=atom_type.charge,
                               coords=site.coords ) )
            bond_index += 1
            bond_type = bond_types_dict['{}-{} spring'.format( site.species_string, site.species_string )]
            bonds.append( Bond(bond_index=bond_index,
                               bond_type_id=bond_type.bond_type_index,
                               atom_indices = [atom_index-1, atom_index]))
    return atoms, bonds

class LammpsData():
    
    def __init__(self, atom_types, bond_types, atoms, bonds, cell_matrix):
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.atoms = atoms
        self.bonds = bonds
        self.cell_matrix = cell_matrix
        
    @classmethod
    def from_structure(cls, structure, params):
        atom_types, bond_types = types_from_structure( structure=structure, 
                                       core_shell=params['core_shell'], 
                                       charges=params['charges'], 
                                       masses=params['masses'], verbose=True )
        atoms, bonds = atoms_and_bonds_from_structure( structure, atom_types, bond_types )
        cell_matrix = structure.lattice.matrix
        return cls( atom_types, bond_types, atoms, bonds, cell_matrix )
    
    def header_string( self, title='title' ):
        """
        Information for the top part of a LAMMPS file.
        """
        return_str = ''
        return_str+='{}\n\n'.format( title )
        return_str += '{}   atoms\n'.format( len(self.atoms) )
        return_str += '{}   bonds\n\n'.format( len(self.bonds) )
        return_str += '{}   atom types\n'.format( len(self.atom_types ) )
        return_str += '{}   bond types\n\n'.format( len(self.bond_types ) )
        return return_str
        
    def cell_dimensions_string(self):
        """
        Prints the cell dimensions for the lammps file.
        
        !! Orthorhombic cells only !!
        """    
        return '0.0 {:2.6f} xlo xhi\n0.0 {:2.6f} ylo yhi\n0.0 {:2.6f} zlo zhi\n\n'.format(*self.cell_matrix.diagonal())
        
    def masses_string(self):
        """
        Prints the mass information for each species for the lammps file.
        """
        return_str = 'Masses\n\n'
        for at in self.atom_types:
            return_str += '{} {:9.5f} # {} {}\n'.format( at.atom_type_index, 
                                                        float(at.mass), at.label, at.core_shell_string)
        return_str += '\n'
        return return_str
    
    def atoms_string(self):
        return_str = 'Atoms\n\n'
        for atom in self.atoms:
            return_str += '{}\n'.format(atom.input_string())
        return_str += '\n'
        return return_str
    
    def bonds_string(self):
        return_str = 'Bonds\n\n'
        for bond in self.bonds:
            return_str += '{}\n'.format(bond.input_string())
        return_str += '\n'
        return return_str
        
    def input_string(self, title='title'):
        return_str = ''
        return_str += self.header_string(title)
        return_str += self.cell_dimensions_string()
        return_str += self.masses_string()
        return_str += self.atoms_string()
        return_str += self.bonds_string()
        return return_str
