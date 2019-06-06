from pymatgen.io.vasp import Poscar
import numpy as np
from collections import Counter

class Species():
    
    def __init__( self, label, mass, charge, core_shell=False, shell_mass=None ):
        self.label = label
        self.mass = mass
        self.core_shell = core_shell
        if core_shell:
            Species.core_shell_index += 1
            self.core_shell_index = Species.core_shell_index
            if not shell_mass:
                shell_mass = self.mass * 0.1
            self.core_mass = self.mass - shell_mass
            self.shell_mass = shell_mass
            self.types = [ AtomType( mass=self.core_mass, core_shell='core', charge=charge['core'] ), 
                           AtomType( mass=self.shell_mass, core_shell='shell', charge=charge['shell'] ) ]
        else:
            self.types = [ AtomType( mass=self.mass, charge=charge ) ]

class AtomType():
    
    def __init__( self, mass, charge, core_shell=None ):
        self.mass = mass
        self.charge = charge
        AtomType.atom_type_index += 1
        self.index = AtomType.atom_type_index
        self.core_shell = core_shell

    @property
    def core_shell_string(self):
        if self.core_shell is None:
            return ''
        return self.core_shell

# print functions
def print_header( p, species ):
    title = p.comment
    n_atoms = len(p.structure) #INCORRECT AS LENGTH INCREASES WITH CORESHELL ADDITION, NEED TO UPDATE
    syms = [site.specie.symbol for site in p.structure]
    n_bonds = len([1.0 for s in syms if species[s].core_shell])
    n_atom_types = len([item for sublist in species.values() for item in sublist.types]) 
    n_bond_types = len([ 1.0 for s in species.values() if s.core_shell ]) #HOW DOES THIS WORK WITH 1.0????
    print( '{}'.format( title ) )
    print()
    print( '{}   atoms'.format( n_atoms ) )
    print( '{}   bonds'.format( n_bonds ) )
    print()
    print( '{}   atom types'.format( n_atom_types ) )
    print( '{}   bond types'.format( n_bond_types ))
    print()
    
def print_masses( species ):
    print('Masses\n')
    for s in species.values():
        for t in s.types:
            print('{} {:9.5f} # {} {}'.format( t.index, float(t.mass), s.label, t.core_shell_string))
    print()
    
def print_cell_dimensions(poscar):
    print( '0.0 {:2.6f} xlo xhi\n0.0 {:2.6f} ylo yhi\n0.0 {:2.6f} zlo zhi'.format(*poscar.structure.lattice.abc))
    print()
    
def print_atoms( poscar, species ):
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
    Species.core_shell_index = 0
    AtomType.atom_type_index = 0
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
    print_bonds( poscar, species ) 
