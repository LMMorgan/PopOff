from collections import Counter
import numpy as np
from buckfit.bonds import Bond
from buckfit.atoms import Atom
from buckfit.atom_types import AtomType
from buckfit.bond_types import BondType

    
def types_from_structure( structure, core_shell, charges, masses, cs_spring=None, verbose=True ):
    """
    Defines the atom types and bond types from the structure and given information from params.
    
    Args:
        structure (obj): A pymatgen structural object created from the transformed matrix structure, with forces included as site properties.
        core_shell (dict): A dictionary of booleans stating if any atoms should be made core-shell.
        charges (dict): A dictionary of charges for each atom type. Key = atom label (str), value = charge(float)/sub_dict(dict). If atom is core-shell a sub dictionary will be the value, where sub_key = 'core' or 'shell' (str) and sub_value = charge (float).
        masses (dict): A dictionary of masses for each atom type. Key = atom label (str), value = mass(float)/sub_dict(dict). If atom is core-shell a sub dictionary will be the value, where sub_key = 'core' or 'shell' and sub_value = mass (float).
        cs_spring (dict): A dictionary of the cor-shell spring values. Key = atom labels separated by "-" (str), value = list of two values (float) for the spring values.
        verbose (optional:bool): Print verbose output. Default = True.

    Returns:
        atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float), charge (float), and core_shell (str).
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
        if core_shell[e.name]: # Create two atom_types for core + shell
            atom_type_index += 1
            atom_types.append( AtomType(atom_type_index=atom_type_index,
                                        label='{} core'.format(e.name),
                                        element_type = e.name,
                                        mass=masses[e.name]['core'],
                                        charge=charges[e.name]['core'],
                                        core_shell='core') )
            atom_type_index += 1
            atom_types.append( AtomType(atom_type_index=atom_type_index,
                                        label='{} shell'.format(e.name),
                                        element_type = e.name,
                                        mass=masses[e.name]['shell'],
                                        charge=charges[e.name]['shell'],
                                        core_shell='shell') )
            bond_type_index += 1
            bond_types.append( BondType(bond_type_index=bond_type_index,
                                        label='{}-{} spring'.format(e.name, e.name),
                                        spring_coeff_1=cs_spring['{}-{}'.format(e.name, e.name)][0],
                                        spring_coeff_2=cs_spring['{}-{}'.format(e.name, e.name)][1]))
        else:
            atom_type_index += 1
            atom_types.append( AtomType(atom_type_index=atom_type_index,
                                        label='{}'.format(e.name),
                                        element_type = e.name,
                                        mass=masses[e.name],
                                        charge=charges[e.name] ) )
    return atom_types, bond_types

def atoms_and_bonds_from_structure( structure, atom_types, bond_types ):
    """
    Defines the atoms and bonds from the structure and given information from params.
    
    Args:
        structure (obj): A pymatgen structural object created from the transformed matrix structure, with forces included as site properties.
        atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float), charge (float), and core_shell (str).
        bond_types (list(obj)): BondType objects including bond_type_index (int) and label (str).

    Returns:
        atoms (list(obj)): Atom objects including atom_index (int), molecule_index (int), coords (np.array), forces (np.array), and atom_type (obj:AtomType).
        bonds (list(obj)): Bond objects including bond_index (int), atom_indices (list(int)), and bond_type (obj:BondType).
    """
    atoms = []
    bonds = []
    atom_types_dict = {}
    atom_types_dict = {at.label: at for at in atom_types}
    bond_types_dict = {}
    bond_types_dict = {bt.label: bt for bt in bond_types}
    atom_index = 0
    bond_index = 0
    molecule_index = 0
    for site in structure:
        molecule_index += 1
        if site.species_string in atom_types_dict: # not core-shell atom
            atom_index += 1
            atom_type = atom_types_dict[site.species_string]
            atoms.append( Atom(atom_index=atom_index,
                               molecule_index=molecule_index,
                               coords=site.coords,
                               atom_forces=site.properties['forces'],
                               atom_type=atom_type) )
        else: # need to handle core + shell
            atom_index += 1
            atom_type = atom_types_dict[site.species_string + ' core']
            atoms.append( Atom(atom_index=atom_index,
                               molecule_index=molecule_index,
                               coords=site.coords,
                               atom_forces=site.properties['forces'],
                               atom_type=atom_type) )
            atom_index += 1
            atom_type = atom_types_dict[site.species_string + ' shell']
            atoms.append( Atom(atom_index=atom_index,
                               molecule_index=molecule_index,
                               coords=site.coords,
                               atom_forces=np.array([0.0, 0.0, 0.0]),
                               atom_type=atom_type) )
            bond_index += 1
            bond_type = bond_types_dict['{}-{} spring'.format( site.species_string, site.species_string )]
            bonds.append( Bond(bond_index=bond_index,
                               atom_indices = [atom_index-1, atom_index],
                               bond_type=bond_type))
    return atoms, bonds 
