from pymatgen.io.vasp.outputs import Vasprun
from buckfit.lammps_data import LammpsData
import numpy as np

def collate_structural_data(params, structs, supercell=None):
    """
    Creates a list of directory paths for the vasprun.xml files and passes them and the params data to data_from_vasprun to collect the lammps_data objects. These are returned into a list of lammps_data objects for each structure.
    
    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
        structs (np.array): An array containing the list of structure numbers to fit to. Note: this starts from 0 not 1 so check your vasprun.xml numbering.
        supercell (list(int)): 3 integers defining the cell increase in x, y, and z. Default=None if called directly.
        
    Returns:
        lammps_data (list(obj)):  LammpsData objects containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), file_name (str), and expected_stress_tensors (np.array).
        """
    vaspruns = [f'vaspruns/vasprun{i}.xml' for i in structs]
    lammps_data = [data_from_vasprun(params, v, i, supercell) for i, v in enumerate(vaspruns)]
    return lammps_data


def data_from_vasprun(params, filename, i, supercell):
    """
    Collects the information needed for the lammps data inputs from vasprun.xml files with additional information provided by params.
    
    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
        filename (str): directory path and filename of the vasprun.xml file to be read.
        i (int): Counter for the structures. Used in naming the lammps input files.
        supercell (list(int)): 3 integers defining the cell increase in x, y, and z. Default=None if called directly.
        
    Returns:
        struct_data (obj:LammpsData):  LammpsData objects containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), file_name (str), and expected_stress_tensors (np.array).
        """
    vasprun = Vasprun(filename)
    structure = vasprun.ionic_steps[0]['structure']
    structure.add_site_property('forces', np.array(vasprun.ionic_steps[0]['forces']))
    stressT = vasprun.ionic_steps[0]['stress']
    stresses = np.array([stressT[0][0], stressT[1][1], stressT[2][2],
                         stressT[0][1], stressT[2][1], stressT[0][2]])
    
    if supercell is not None:
        if all(isinstance(i, int) for i in supercell) and len(supercell) == 3:
            structure = structure*supercell
        else:
            raise TypeError('Incorrect dimensions for supercell. Requires x,y,z expansion (i.e. list of 3 integers).')

    struct_data = LammpsData.from_structure(structure, params, i, stresses)
    return struct_data