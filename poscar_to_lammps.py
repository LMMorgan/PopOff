from pymatgen.io.vasp import Poscar
import numpy as np
from collections import Counter
from itertools import count

class Species():
    """
    Class for each species type.
    """
    
    #Initiates count when initialise() isn't called
    _core_shell_count = count(1)
    
    def __init__( self, label, mass, charge, core_shell=False, shell_mass=None ):
        """
        Initialise a Species instance.

        Args:
            label (str): A string if the species symbol/label.
            mass (float): Mass of the species.
            charge(float): Charge of the species.
            core_shell (optional:bool): True if core_shell, False if no core_shell. Default = False.
            shell_mass (optional:float): A string of characters, either 'nvt' or 'npt' to enter into the CONTROL file. Default = None.
                
        Returns:
            None
        """
        self.label = label
        self.mass = mass
        self.core_shell = core_shell
        if core_shell:
            self.core_shell_index = next(self._core_shell_count)
            if not shell_mass:
                shell_mass = self.mass * 0.1
            self.core_mass = self.mass - shell_mass
            self.shell_mass = shell_mass
            self.types = [ AtomType( mass=self.core_mass, core_shell='core', charge=charge['core'] ), 
                           AtomType( mass=self.shell_mass, core_shell='shell', charge=charge['shell'] ) ]
        else:
            self.types = [ AtomType( mass=self.mass, charge=charge ) ]
    
    @classmethod
    def initialise(cls):
        """
        Initialise a counter used in __init__.

        Args:
            None
                
        Returns:
            None
        """        
        Species._core_shell_count = count(1)

class AtomType():
    """
    Class for each atom type.
    """
    
    #Initiates count when initialise() isn't called
    _atom_type_count = count(1)
    
    def __init__( self, mass, charge, core_shell=None ):
        """
        Initialise an AtomType instance.

        Args:
            mass (float): Mass of the atom type.
            charge(float): Charge of the atom type.
            core_shell (optional:str): Either 'core' or 'shell'. Default = None.
                
        Returns:
            None
        """        
        self.mass = mass
        self.charge = charge
        self.index = next(self._atom_type_count)
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

    @classmethod
    def initialise(cls):
        """
        Initialise a counter used in __init__.

        Args:
            None
                
        Returns:
            None
        """  
        AtomType._atom_type_count = count(1)
    

# print functions
def print_header( p, species ):
    """
    Prints the information for the top part of the lammps file.
    
    Args:
        p (obj): A pymatgen structural object created from a POSCAR.
        species (dict): A dictionary of Species classes.

    Returns:
        None.
    """
    title = p.comment
    n_atoms = sum( [ n * ( 2 if species[symbol].core_shell else 1 )
                    for symbol, n in zip( p.site_symbols, p.natoms ) ] )
    syms = [site.specie.symbol for site in p.structure]
    n_bonds = len([1.0 for s in syms if species[s].core_shell])
    n_atom_types = len([item for sublist in species.values() for item in sublist.types]) 
    n_bond_types = len([ 1.0 for s in species.values() if s.core_shell ])
    print( '{}'.format( title ) )
    print()
    print( '{}   atoms'.format( n_atoms ) )
    print( '{}   bonds'.format( n_bonds ) )
    print()
    print( '{}   atom types'.format( n_atom_types ) )
    print( '{}   bond types'.format( n_bond_types ) )
    print()

def print_cell_dimensions(poscar):
    """
    Prints the cell dimensions for the lammps file.
    
    Args:
        poscar (obj): A pymatgen structural object created from a POSCAR.

    Returns:
        None.
    """    
    print( '0.0 {:2.6f} xlo xhi\n0.0 {:2.6f} ylo yhi\n0.0 {:2.6f} zlo zhi'.format(*poscar.structure.lattice.abc))
    print()
    
def print_masses( species ):
    """
    Prints the mass information for each species for the lammps file.
    
    Args:
        species (dict): A dictionary of Species classes.

    Returns:
        None.
    """
    print('Masses\n')
    for s in species.values():
        for t in s.types:
            print('{} {:9.5f} # {} {}'.format( t.index, float(t.mass), s.label, t.core_shell_string))
    print()
    
def print_atoms( poscar, species ):
    """
    Prints the individual atom information for the lammps file.
    
    Args:
        poscar (obj): A pymatgen structural object created from a POSCAR.
        species (dict): A dictionary of Species classes.

    Returns:
        None.
    """
    print('Atoms\n')
    i = 0 # counter for atom index
    m = 0 # counter for molecule index
    for s in poscar.structure.sites:
        m += 1
        spec = species[s.specie.name]
        for t in spec.types:
            i += 1
            print( '{:>3}  {:>3}  {:>3}   {: 1.3f}  {: 2.6f}  {: 2.6f}  {: 2.6f}'.format( i, m, t.index, t.charge, *s.coords ) )
    print()
    
def print_bonds( poscar, species ):
    """
    Prints the bond information for any core-shell atoms for the lammps file.
    
    Args:
        poscar (obj): A pymatgen structural object created from a POSCAR.
        species (dict): A dictionary of Species classes.

    Returns:
        None.
    """
    print('Bonds\n')
    i = 0 # counter for atom index
    m = 0 # counter for molecule index
    b = 0 # counter for bonds
    for s in poscar.structure.sites:
        m += 1
        spec = species[s.specie.name]
        if spec.core_shell:
            b += 1
            i += 2
            print( '{:>4} {:>4} {:>4} {:>4}'.format( b, spec.core_shell_index, i-1, i ) )
        else:
            i += 1
    print()

def poscar_to_lammps( poscar, core_shell, charges ):
    """
    Executes classes and print functions to create the lammps input file.
    
    Args:
        poscar (obj): A pymatgen structural object created from a POSCAR.
        core_shell (dict): A dictionary of booleans stating if any atoms should be made core-shell.
        charges (dict): A dictionary of charges for each atom type. Key = Atom label(str),
                        value = charge(float)/sub_dict(dict). If atom is core-shell a sub dictionary
                        will be the value, where sub_key = 'core' or 'shell' and sub_value = charge(float).
                        
    Note:
        The two .initialise() commands are used to reset the calss index counters.
        They are required if called from a notebook but redundant if called from the commandline.
        If calling more than once from a notebook, the indexing will be incorrect without restarting the kernal.

    Returns:
        None.
    """
    Species.initialise() #Required when using a notebook, not if command line called.
    AtomType.initialise() #Required when using a notebook, not if command line called.
    elements =list(Counter(poscar.structure.species).keys())
    species = { e.name: Species( label=e.name,
                             mass=e.atomic_mass,
                             charge=charges[e.name],
                             core_shell=core_shell[e.name] ) 
                for e in elements }
    print_header( poscar, species )
    print_cell_dimensions( poscar )
    print_masses( species )
    print_atoms( poscar, species )
    if any(core_shell.values()): #Only prints bonds section if at least one atom is core-shell.
        print_bonds( poscar, species )