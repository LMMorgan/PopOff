import numpy as np
import pymc3 as pm
import glob
from vasppy.outcar import forces_from_outcar
from pymatgen.io.vasp import Poscar
import lammps
import potentials as pot
from lammps_data import LammpsData
import pandas as pd

class FitModel():
    """
    Class that collates all fitting information and runs the fitting process using PyMC3 and LAMMPS.
    """
    def __init__(self, potentials, lammps_data, cs_springs):
        """
        Initialise an instance for all information relating to the pysical and electronic structure needed for the Lammps input.
        Args:
            potentials (list(obj)): BuckinghamPotential objects including labels (list(str)), atom_type_index (list(int)), a (obj), rho (obj), and c (obj). Each object is a BuckinghamParameter object.
            lammps_data (list(obj)):  LammpsData objects containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), and file_name (str).
            cs_springs (dict): The key is the atom label (str) and the value the spring values (list(float)).
        Returns:
            None
        """  
        self.potentials = potentials
        self.lammps_data = lammps_data
        self.cs_springs = cs_springs

    @classmethod
    def collect_info(cls, params, distribution, supercell=None):
        """
        Collects information from other classes relating to the potentials and lammps_data using params input information.
        Args:
            params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str).
            distribution(dict(dict)): Contains buckingham potential, 'bpp':list(float), and 'sd':list(float) dictionaries where the distribution keys are atom label pairs (str), example: 'Li-O'.
            supercell (optional:list(int)): 3 integers defining the cell increase in x, y, and z. Default=None.
        Returns:
            (FitModel):  FitModel object containing potentials (list(obj:BuckinghamPotential)), lammps_data (obj:LammpsData), and cs_spring (dict).      
        """  
        lammps_data = get_lammps_data(params, supercell)
        parameters = pot.buckingham_parameters(distribution)
        potentials = pot.buckingham_potentials(distribution, lammps_data[0].atom_types, parameters) #REWRITE TO READ ATOM_TYPES WITHOUT [0] 
        cs_springs =  params['cs_springs']
        return cls(potentials, lammps_data, cs_springs)
    
    def expected_forces(self):
        """
        Collates the expected forces on all atoms, in all structures to a flattened 2D array, where axis 0 is the list of atoms and axis 1 is the x,y,z forces.

        Args:
            None
                
        Returns:
            expected_forces (np.array): 2D array of all atoms forces in all structures.
        """
        forces_data = []
        for structure in self.lammps_data:
            atom_forces = []
            for atoms in structure.atoms:
                atom_forces.append(atoms.forces)
            forces = np.stack(atom_forces, axis=0)
            core_mask = structure.core_mask()
            forces_data.append(forces[core_mask])
        expected_forces = np.concatenate(forces_data, axis=0)
        return expected_forces
    
    def _update_potentials(self, **kwargs):
        """
        Unpdates the potentials set by pymc3 into the dictionary for the fitting process.

        Args:
            **kwargs: The parameters to be updated in the fitting process as set with pm.Model.

        Returns:
            None.
        """
        for key, value in kwargs.items():
            for pot in self.potentials:
                if key is pot.a.label_string:
                    pot.a.value = value
                if key is pot.rho.label_string:
                    pot.rho.value = value
                if key is pot.c.label_string:
                    pot.c.value = value
                    
    def _set_potentials(self, lmp):
        """
        Sets the potential for the sepecified Lammps system (changes for each iteration of the potential fit).

        Args:
            lmp (obj): Lammps object with structure and specified commands implemented.

        Returns:
            None
        """
        for pot in self.potentials:
            lmp.command('{}'.format(pot.potential_string()))
        
    def _simfunc(self, **kwargs):
        """
        Runs a minimization and zero step run for the instance and returns the forces.
        Args:
            **kwargs: Contain data for type of fitting and to what parameters as set with pm.Model.
        Returns:
            out (np.array): x,y,z forces on each atom.
        """
        instances = [lmp.initiate_lmp(self.cs_springs) for lmp in self.lammps_data]

        if min(kwargs.values()) > 0:
            self._update_potentials(**kwargs)
            out = np.zeros(self.expected_forces().shape)
            
            structure_forces = []
            for ld, instance in zip(self.lammps_data, instances):
                self._set_potentials(instance)
                instance.command('fix 1 cores setforce 0.0 0.0 0.0')
                instance.command('minimize 1e-25 1e-25 5000 10000')
                instance.command('unfix 1')
                instance.run(0)
                structure_forces.append(instance.system.forces[ld.core_mask()])
                
            out = np.concatenate(structure_forces, axis=0)
            

        else: out = np.ones(self.expected_forces().shape)*999999999 # ThisAlgorithmBecomingSkynetCost

            
        return out

                    
    def run_fit(self, excude_from_fit=None, epsilon=0.1, draws=1000, dist_func='absolute_error'):
        """
        Runs the PyMC3 fitting process. Initiating a pm.Model, applying the potentials and distributions, and running the simulator that calls Lammps.

        Args:
            excude_from_fit (list(str)): Label of the parameters not wishing to fit to. Example 'Li_O_c'. Default=None.
            epsilion (float): Designates the value of epsilon. Default=0.1.
            draws (int): Designates the number of draws per step for the fitting. Default=1000.
            dist_func (optional:str): Name of distance calculation to perform in pm.SMC. Default='absolute_error'.

        Returns:
            trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3. This can be read by arviz to be plotted.
        """
        
        with pm.Model() as model:
            my_dict = {}
            for pot in self.potentials:
                name = '{}'.format(pot.a.label_string)
                if name not in excude_from_fit:
                    my_dict[name] = pot.a.distribution()
                name = '{}'.format(pot.rho.label_string)
                if name not in excude_from_fit:
                    my_dict[name] = pot.rho.distribution()
                name = '{}'.format(pot.c.label_string)
                if name not in excude_from_fit:
                    my_dict[name] = pot.c.distribution()

            simulator = pm.Simulator('simulator', self._simfunc, observed=self.expected_forces())
            trace = pm.sample(step=pm.SMC(ABC=True, epsilon=epsilon), dis_func='absolute_error', draws=draws)
        return trace
    
    
    
    def get_forces(self, **kwargs):
        """
        Runs a minimization and zero step run for the instance and returns the forces.
        Args:
            **kwargs: Dictionary containing potential paramenters (value) with the parameter label (key).
        Returns:
            out (np.array): x,y,z forces on each atom.
        """
        instances = [lmp.initiate_lmp(self.cs_springs) for lmp in self.lammps_data]

        self._update_potentials(**kwargs)
        out = np.zeros(self.expected_forces().shape)
            
        structure_forces = []
        for ld, instance in zip(self.lammps_data, instances):
            self._set_potentials(instance)
            instance.command('fix 1 cores setforce 0.0 0.0 0.0')
            instance.command('minimize 1e-25 1e-25 5000 10000')
            instance.command('unfix 1')
            instance.run(0)
            structure_forces.append(instance.system.forces[ld.core_mask()])
                
        out = np.concatenate(structure_forces, axis=0)
                           
        return out
    
    
          
def get_lammps_data(params, supercell=None):
    """
    Collects the information needed for the lammps data inputs from the POSCARs and OUTCARs with additional information provided by params.

    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
        supercell (list(int)): 3 integers defining the cell increase in x, y, and z. Default=None if called directly.

    Returns:
        lammps_data (list(obj)):  LammpsData objects containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), and file_name (str).
        """
    if supercell is not None and len(supercell) is not 3:
            raise ValueError('Incorrect dimensions for supercell. Requires x,y,z expansion, i.e. list of 3 integers')
            
    lammps_data = []
    for i, pos in enumerate(glob.glob('poscars/POSCAR*')):
        structure = Poscar.from_file(pos).structure
        forces = forces_from_outcar('outcars/OUTCAR{}'.format(i+1))[-1]
        structure.add_site_property('forces', forces)
        if supercell is not None:
            structure = structure*supercell
        struct_data = LammpsData.from_structure(structure, params, i)
        lammps_data.append(struct_data)  
    return lammps_data
