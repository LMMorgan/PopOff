from collections import Counter
import numpy as np
import pymc3 as pm

class Atom():
    """
    Class for each atom.
    """
    
    def __init__(self, atom_index, molecule, coords, atom_type):
        """
        Initialise an instance for each atom in the structure.

        Args:
            atom_index (int): Individual atom number.
            molecule (int): Index of the molecule the atom belongs to i.e. core and shell will be the
                            same molecule, but indivdual atoms will be separate.
            atom_type_id (int): Integer value given for the atom type.
            coords (list(float)): x, y, and z positions of the atom.
            atom_type (obj): AtomType object including atom_type_index (int), label (str), mass (float),
                             charge (float), and core_shell (str).
                
        Returns:
            None
        """
        self.atom_index = atom_index
        self.molecule = molecule
        self.coords = coords
        self.atom_type = atom_type
        
    def input_string(self):
        """
        Defines a string format in lammps output file for the atom data.

        Args:
            None
                
        Returns:
            (str): contains atom_index (int), molecule (int), atom_type_id (int), charge(float),
                   and coordinates (list(float)) for lammps file.
        """
        return '{:<4} {:<4} {:<4} {: 1.4f}  {: 2.6f}  {: 2.6f}  {: 2.6f}'.format( 
            self.atom_index, self.molecule, self.atom_type.atom_type_index, self.atom_type.charge, *self.coords )
    
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
        self.bond_index = bond_index
        self.atom_indices = atom_indices
        self.bond_type = bond_type
        
    def input_string(self):
        """
        Defines a string format in lammps output file for the bond data.

        Args:
            None
                
        Returns:
            (str): Containing bond_index (int), bond_type.bond_type_index (int), and
                   atom_indices (list(int)) for lammps file.
        """       
        return '{:<4} {:<4} {:<4} {:<4}'.format( self.bond_index, self.bond_type.bond_type_index, *self.atom_indices )
        
class BondType():
    """
    Class for each bond type.
    """
    
    def __init__(self, bond_type_index, label):
        """
        Initialise an instance for each bond type in the structure.

        Args:
            bond_type_index (int): Integer value given for the bond type. 
            label (str): Identity of the bond atoms format "element_1_index-element_2_index spring",
                         where element_1 is the core, and element_2 is the shell.
                
        Returns:
            None
        """  
        self.bond_type_index = bond_type_index
        self.label = label

class AtomType():
    """
    Class for each atom type.
    """
    
    def __init__( self, atom_type_index, label, 
                  mass, charge, core_shell=None ):
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
        Defines a string for a comment in lammps output file labelling cores/shells.

        Args:
            None
                
        Returns:
            core_shell (str): Either 'core', 'shell', or '' if core_shell is None. 
        """  
        if self.core_shell is None:
            return ''
        return self.core_shell
    
def types_from_structure( structure, core_shell, charges, masses, verbose=True ):
    """
    Defines the atom and bond types from the structure and given information.
    
    Args:
        structure (obj): A pymatgen structural object created from a POSCAR.
        core_shell (dict): A dictionary of booleans stating if any atoms should be made core-shell.
        charges (dict): A dictionary of charges for each atom type. Key = atom label(str),
                        value = charge(float)/sub_dict(dict). If atom is core-shell a sub dictionary
                        will be the value, where sub_key = 'core' or 'shell' and sub_value = charge(float).
        masses (dict): A dictionary of masses for each atom type. Key = atom label (str),
                       value = mass(float)/sub_dict(dict). If atom is core-shell a sub dictionary
                       will be the value, where sub_key = 'core' or 'shell' and sub_value = mass(float).
        verbose (optional:bool): Print verbose output. Default = True.

    Returns:
        atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float),
                                charge (float), and core_shell (str).
        bond_types (list(obj)): BondType objects including bond_type_index (int) and label (str).
    """
    atom_types = []
    bond_types = []
    atom_type_index = 0
    bond_type_index = 0
    elements = Counter(structure.species)
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
    """
    Defines the atoms and bonds from the structure and given information.
    
    Args:
        atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float),
                                charge (float), and core_shell (str).
        bond_types (list(obj)): BondType objects including bond_type_index (int) and label (str).

    Returns:
        atoms (list(obj)): Atoms objects including atom_index (int), molecule (int), atom_type_id (int),
                           charge (float), coordinates (list(float)), and atom_type (obj).
        bonds (list(obj)): Bonds objects including bond_index (int), bond_type (obj), atom_indices (list(int)).
    """
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
                               coords=site.coords,
                               atom_type = atom_type) )
        else: # need to handle core + shell
            atom_index += 1
            atom_type = atom_types_dict[site.species_string + ' core']
            atoms.append( Atom(atom_index=atom_index,
                               molecule=molecule_index,
                               coords=site.coords,
                               atom_type = atom_type) )
            atom_index += 1
            atom_type = atom_types_dict[site.species_string + ' shell']
            atoms.append( Atom(atom_index=atom_index,
                               molecule=molecule_index,
                               coords=site.coords,
                               atom_type = atom_type) )
            bond_index += 1
            bond_type = bond_types_dict['{}-{} spring'.format( site.species_string, site.species_string )]
            bonds.append( Bond(bond_index=bond_index,
                               atom_indices = [atom_index-1, atom_index],
                               bond_type=bond_type))
    return atoms, bonds

def lammps_matrix(structure):
    """
    Defines the atoms and bonds from the structure and given information.
    
    Args:
        structure (obj): A pymatgen structural object created from a POSCAR.
    
    Returns:
        cell_lengths (list(floats)): Lengths of each cell direction.
        tilt_factor (list(floats)): Tilt factors of the cell.
        
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


class LammpsData():
    """
    Class that collates all structural information for outputing a Lammps format.
    """
    def __init__(self, atom_types, bond_types, atoms, bonds, cell_lengths, tilt_factor, potentials):
        """
        Initialise a Lammps instance.

        Args:
            atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float),
                                    charge (float), and core_shell (str).
            bond_types (list(obj)): BondType objects including bond_type (int) and label (str).
            atoms (list(obj)): Atoms objects including atom_index (int), molecule (int), atom_type_id (int),
                               charge (float), coordinates (list(float)), and atom_type (obj).
            bonds (list(obj)): Bonds objects including bond_index (int), bond_type (obj), and atom_indices (list(int)).
            cell_lengths (list(floats)): Lengths of each cell direction.
            tilt_factor (list(floats)): Tilt factors of the cell.
                
        Returns:
            None
        """  
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.atoms = atoms
        self.bonds = bonds
        self.cell_lengths = cell_lengths
        self.tilt_factor = tilt_factor
        self.potentials = potentials
        
    @classmethod
    def from_structure(cls, structure, params):
        """
        Collects information from initial structure and parameters.

        Args:
            structure (obj): A pymatgen structural object created from a POSCAR.
            params (dict(dict)): Contains core_shell(bool), charges(float), and masses(float) dictionaries
                                 where the keys are atom label (str). Also contains bpp(list(float))
                                 dictionary where the keys are atom label pairs (str), example: 'Li-O'.
                
        Returns:
            (LammpsData):             
        """  
        atom_types, bond_types = types_from_structure( structure=structure, 
                                       core_shell=params['core_shell'], 
                                       charges=params['charges'], 
                                       masses=params['masses'], verbose=True )
        atoms, bonds = atoms_and_bonds_from_structure( structure, atom_types, bond_types )
        parameters = buckingham_parameters(params)
        potentials = buckingham_potentials(params, atom_types, parameters)
        cell_lengths, tilt_factor = lammps_matrix(structure)
        
        return cls( atom_types, bond_types, atoms, bonds, cell_lengths, tilt_factor, potentials)
    
    def header_string( self, title='title' ):
        """
        Information for the top part of a LAMMPS file.

        Args:
            title (optional:str): Title for lammps file, default = 'title'.
                
        Returns:
            return_str (str): title, atoms, bonds, atom types, and bond types information for lammps format.
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
            return_str (str): cell dimensions, tilt factors, and related labels for lammps file.
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
            return_str (str): atom types, masses, and commented atom type label for lammps file.
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
            return_str (str): atom information for lammps file as defined by Atoms class.
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
            return_str (str): bond information for lammps file as defined by Bonds class.
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
            return_str (str): all information for lammps file as defined by class methods above.
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
            mask (list(bool)): returns False for shell atoms and True for all others.
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
        Returns a string containing the atom index of all non-shell atoms.
        
        Args:
            None
            
        Returns:
            type_core (str): atom index of all non-shell atoms as string separated by a single space.
        """
        type_core = ' '.join(['{}'.format(atom.atom_type_index) for atom in self.atom_types
                              if 'shell' not in atom.label])
        return type_core
    
    def type_shell(self):
        """
        Returns a string containing the atom index of all shell atoms.
        
        Args:
            None
            
        Returns:
            type_shell (str): atom index of all shell atoms as string separated by a single space.
        """
        type_shell = ' '.join(['{}'.format(atom.atom_type_index) for atom in self.atom_types
                               if 'shell' in atom.label])
        return type_shell 
    
    
class BuckinghamParameter():
    """
    Class that contains the information for each parameter in the buckingham potential.
    """
    def __init__(self, parameter_type, parameter_string, value):
        """
        Initialise each parameter in all buckingham potentials.

        Args:
            parameter_type (str): The label for the specific parameter type i.e. a, rho, or c.
            parameter_string (str): The label defining the two atoms and parameter in the form
                                    'atomtype1_atomtype2_parameter'. Example: 'Li_O_rho'
            value (float): Value of the parameter.
                
        Returns:
            None
        """  
        self.type = parameter_type
        self.label_string = parameter_string
        self.value = value
        
    def distribution(self, stand_dev=None):
        """
        Creates pymc3.model.FreeRV for each parameter. This function must be called within pm.Model.
        Where pm is from `import pymc3 as pm`.

        Args:
            stand_dev (float): Value for the standard deviation in the parameter distribution. Default=None.
                
        Returns:
            distribution (obj): pymc3.model.FreeRV object for the parameter.
        """
        if stand_dev is None:
            raise ValueError('No value given for the standard deviation')
        else:
            distribution = pm.Normal('{}'.format(self.label_string), mu = self.value, sd = stand_dev)
            
        return distribution
            
    
def buckingham_parameters(params):
    """
    Defines each buckingham parameter.
    
    Args:
        params (dict(dict)): Contains core_shell(bool), charges(float), and masses(float) dictionaries
                             where the keys are atom label (str). Also contains bpp(list(float))
                             dictionary where the keys are atom label pairs (str), example: 'Li-O'.
    
    Returns:
        parameters (list(obj)): BuckinghamParameter objects including parameter_type (str),
                                parameter_string (str), and value (float).
        
    """
    parameters = []
    for key, item in params['bpp'].items():
        atom_name_1, atom_name_2 = key.split('-')                
        parameters.append(BuckinghamParameter(parameter_type = 'a',
                                              parameter_string = "{}_{}_{}".format(atom_name_1, atom_name_2, 'a'),
                                              value = item[0]))
        parameters.append(BuckinghamParameter(parameter_type = 'rho',
                                              parameter_string = "{}_{}_{}".format(atom_name_1, atom_name_2, 'rho'),
                                              value = item[1]))
        parameters.append(BuckinghamParameter(parameter_type = 'c',
                                              parameter_string = "{}_{}_{}".format(atom_name_1, atom_name_2, 'c'),
                                              value = item[2]))
    return parameters     


class BuckinghamPotential():
    """
    Class that contains the information for each buckingham potential, where parameters are
    BuckinghamParameter(obj).
    """
    def __init__(self, labels, atom_type_index, a, rho, c):
        """
        Initialise each parameter in all buckingham potentials.

        Args:
            labels (list(str)):
            atom_type_index (list(int)):
            a (obj): BuckinghamParameter objects including parameter_type (str), parameter_string (str),
                     and value (float).
            rho (obj): BuckinghamParameter objects including parameter_type (str), parameter_string (str),
                       and value (float).
            c (obj): BuckinghamParameter objects including parameter_type (str), parameter_string (str),
                     and value (float).
                
        Returns:
            None
        """ 
        self.labels = labels
        self.atype_index = atom_type_index
        self.a = a
        self.rho = rho
        self.c = c
        
    def potential_string(self):
        """
        Prints potential string for lammps pair_coeff command.
        
        Args:
            None
            
        Returns:
            return_str (str): atype_index for atom pairs, and buchingham potential parameter values
                              formatted for lammps command.
        """
        return_str = 'pair_coeff {} {} {} {} {}'.format(self.atype_index[0],
                                                        self.atype_index[1],
                                                        self.a.value,
                                                        self.rho.value,
                                                        self.c.value)
        return return_str
    

def buckingham_potentials(params, atom_types, parameters):
    """
    Defines the buckingham potential for each given atom pair.
    
    Args:
        params (dict(dict)): Contains core_shell(bool), charges(float), and masses(float) dictionaries
                             where the keys are atom label (str). Also contains bpp(list(float))
                             dictionary where the keys are atom label pairs (str), example: 'Li-O'.
        atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float),
                                charge (float), and core_shell (str).                     
        parameters (list(obj)): BuckinghamParameter objects including parameter_type (str),
                                parameter_string (str), and value (float).                     
    
    Returns:
        potentials (list(obj)): BuckinghamPotential objects including labels (list(str)), atom_type_index (list(int)),
                                a (obj), rho (obj), and c (obj). Each object is a BuckinghamParameter object.
        
    """
    i = 0 #parameter_conter
    potentials = []
    for key, item in params['bpp'].items():
        atom_name_1, atom_name_2 = key.split('-')
        for atom in atom_types:
            if atom_name_1 in atom.label and 'core' not in atom.label:
                atom_type_index_1 = atom.atom_type_index
            if atom_name_2 in atom.label and 'core' not in atom.label:
                atom_type_index_2 = atom.atom_type_index
                
        potentials.append(BuckinghamPotential(labels = [atom_name_1,atom_name_2],
                                               atom_type_index = [atom_type_index_1, atom_type_index_2],
                                               a = parameters[i],
                                               rho=parameters[i+1],
                                               c=parameters[i+2]))
        i+=3
        
    return potentials 


