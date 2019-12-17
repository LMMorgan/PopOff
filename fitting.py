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
            supercell (optional:list(int) or list(list(int))): 3 integers defining the cell increase in x, y, and z for all structures, or a list of lists where each list is 3 integers defining the cell increase in x, y, z, for each individual structure in the fitting process. i.e. all increase by the same amount, or each structure increased but different amounts. Default=None.
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

    def _set_charges(self, lmp):
        """
        Sets the charge on each atom in the structure by type for the specified Lammps system (changes for each iteration of the potential fit).
        Args:
            lmp (obj): Lammps object with structure and specified commands implemented.
        Returns:
            None
        """
        for data in self.lammps_data:
            for atom_type in data.atom_types:
                lmp.command('set type %d charge %f' % (atom_type.atom_type_index, atom_type.charge))
    
    def _set_springs(self, lmp):
        """
        Sets the spring constant for the core-shell bonds for the specified Lammps system (changes for each iteration of the potential fit).
        Args:
            lmp (obj): Lammps object with structure and specified commands implemented.
        Returns:
            None
        """
        for data in self.lammps_data:
            for bond_type in data.bond_types:
                    lmp.command(bond_type.bond_string())            
                    
    def _set_potentials(self, lmp):
        """
        Sets the potential for the specified Lammps system (changes for each iteration of the potential fit).
        Args:
            lmp (obj): Lammps object with structure and specified commands implemented.
        Returns:
            None
        """
        for pot in self.potentials:
            lmp.command('{}'.format(pot.potential_string()))    
    
    def get_forces(self):
        """
        Runs a minimization (if core-shell atoms present) and zero step run for each instance and returns the forces. If any core-shell atoms are present the output forces are masked, so only the forces acting on the rigid ion and core atoms are returned for direct comparison with the relating dft forces.
        Args:
            None
        Returns:
            out (np.array): x,y,z forces on each atom for each instance (each structure).
        """        
        instances = [lmp.initiate_lmp(self.cs_springs) for lmp in self.lammps_data]
        ip_forces = np.zeros(self.expected_forces().shape)
        structure_forces = []
        for ld, instance in zip(self.lammps_data, instances):
            self._set_potentials(instance)
            self._set_charges(instance)
            if self.cs_springs: #If coreshell do the minimisation otherwise do run(0) only
                self._set_springs(instance)
                instance.command('fix 1 cores setforce 0.0 0.0 0.0')
                instance.command('minimize 1e-25 1e-3 3000 10000')
                instance.command('unfix 1')
            instance.run(0)
            structure_forces.append(instance.system.forces[ld.core_mask()])
        ip_forces = np.concatenate(structure_forces, axis=0)
        return ip_forces  
    
    def _charge_reset(self):
        """
        Resets charge (the working charge) to the formal charge for each atom type.
        Args:
            None
        Returns:
            None
        """
        for data in self.lammps_data:
            for at in data.atom_types:
                at.charge = at.formal_charge
       
    def _update_q_ratio(self, fitting_parameters):
        """
        Updates the charge ratio between a given core-shell pair. This is done by passing a factor dQ, which is added to the core charge and sutracted from the shell charge.
        Args:
            fitting_parameters (dict): The keys relate to the parameters being fitted, and the values are the corresponding values.
        Returns:
            None
        """
        for arg, value in fitting_parameters.items():
            if arg.startswith('dq_'):
                for data in self.lammps_data:
                    for at in data.atom_types:
                        if arg.endswith(at.element_type) and at.core_shell == 'core':
                            at.charge += value
                        if arg.endswith(at.element_type) and at.core_shell == 'shell':
                            at.charge -= value    
       
    def _update_springs(self, fitting_parameters):
        """
        Updates the spring constant for the given core-shell atom.
        Args:
            fitting_parameters (dict): The keys relate to the parameters being fitted, and the values are the corresponding values.
        Returns:
            None
        """
        for arg, value in fitting_parameters.items():
            for data in self.lammps_data:
                for bt in data.bond_types:
                    if bt.label == arg:
                        bt.spring_coeff_1 = value
                                             
    def _update_potentials(self, fitting_parameters):
        """
        Updates the buckingham potential for all parameters being fitted. If a specific parameter isn't updated, the current default value will be used and not varied.
        Args:
            fitting_parameters (dict): The keys relate to the parameters being fitted, and the values are the corresponding values.
        Returns:
            None
        """
        for arg, value in fitting_parameters.items():
            for pot in self.potentials:
                if arg == pot.a.label_string:
                    pot.a.value = value
                if arg == pot.rho.label_string:
                    pot.rho.value = value
                if arg == pot.c.label_string:
                    pot.c.value = value                                                
                            
    def _update_charge_scaling(self, fitting_parameters):
        """
        Updates the charge scaling for all atoms. This is done by passing a scaling factor between 0 and 1, which is multiplied by the charge. Example Li = +1 charge, with a scaling factor of 0.5 is 1*0.5, giving a scaled charge of 0.5.
        Args:
            fitting_parameters (dict): The keys relate to the parameters being fitted, and the values are the corresponding values.
        Returns:
            None
        """
        for arg, value in fitting_parameters.items():
            if arg == 'q_scaling':
                for data in self.lammps_data:
                    for at in data.atom_types:
                        at.charge *= value
                    
    def fit_error(self, values, args):
        """
        Takes a list of fitting arguments and their corresponding values, resets the charges on the atoms, then applies a series of updates relating to the given arguements. Following these updates, the forces are extracted and compared to the relating dft forces for the structure(/s). A sum of squared errors is calculated and returned.
        Args:
            values (list(float)): Values relating to the fitting arguments passes in.
            args (list(str)): Keys relating to the fitting parameters for the system, such as charge, springs, and buckingham parameters.
        Returns:
            error (float): The sum of squared errors calculated between dft forces and the MD forces under the given conditions.
        """
        self._charge_reset()
        fitting_parameters = dict(zip(args, values))
        self._update_q_ratio(fitting_parameters)
        self._update_springs(fitting_parameters)
        self._update_potentials(fitting_parameters)        
        self._update_charge_scaling(fitting_parameters)        
        ip_forces = self.get_forces()
        error = np.sum((self.expected_forces() - ip_forces)**2)/ ip_forces.size
        return error

    
    
    
    
    
    def get_lattice_params(self):
        instances = [lmp.initiate_lmp(self.cs_springs) for lmp in self.lammps_data]  
        for ld, instance in zip(self.lammps_data, instances):
            self._set_potentials(instance)
            self._set_charges(instance)
            if self.cs_springs: #If coreshell do the minimisation otherwise do run(0) only
                self._set_springs(instance)
                instance.command('reset_timestep 0')
                instance.command('min_style fire')
                instance.command('fix 1 cores setforce 0.0 0.0 0.0')
                instance.command('minimize 1e-25 1e-3 3000 10000')
                instance.command('unfix 1')
            
            instance.command('reset_timestep 0')
            instance.command('timestep 0.002')
            instance.command('fix 2 all box/relax aniso 1.0 vmax 0.0005')
            instance.command('min_style cg')
            instance.command('minimize 1e-25 1e-25 50000 10000')
            instance.command('unfix 2')        
        return instances        
        
  




#     def _simfunc(self, **kwargs):
#         """
#         Runs a minimization and zero step run for the instance and returns the forces.
#         Args:
#             **kwargs: Contain data for type of fitting and to what parameters as set with pm.Model.
#         Returns:
#             out (np.array): x,y,z forces on each atom.
#         """
#         instances = [lmp.initiate_lmp(self.cs_springs) for lmp in self.lammps_data]

#         if min(kwargs.values()) > 0:
#             self._update_potentials(**kwargs)
#             out = np.zeros(self.expected_forces().shape)
#             structure_forces = []
#             for ld, instance in zip(self.lammps_data, instances):
#                 self._set_potentials(instance)
#                 instance.command('fix 1 cores setforce 0.0 0.0 0.0')
#                 instance.command('minimize 1e-25 1e-3 5000 10000')
#                 instance.command('unfix 1')
#                 instance.run(0)
#                 structure_forces.append(instance.system.forces[ld.core_mask()])
#             out = np.concatenate(structure_forces, axis=0)
#         else: out = np.ones(self.expected_forces().shape)*999999999 # ThisAlgorithmBecomingSkynetCost
#         return out

                    
#     def run_fit(self, excude_from_fit=None, epsilon=0.1, draws=1000, dist_func='absolute_error'):
#         """
#         Runs the PyMC3 fitting process. Initiating a pm.Model, applying the potentials and distributions, and running the simulator that calls Lammps.
#         Args:
#             excude_from_fit (list(str)): Label of the parameters not wishing to fit to. Example 'Li_O_c'. Default=None.
#             epsilion (float): Designates the value of epsilon. Default=0.1.
#             draws (int): Designates the number of draws per step for the fitting. Default=1000.
#             dist_func (optional:str): Name of distance calculation to perform in pm.SMC. Default='absolute_error'.
#         Returns:
#             trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3. This can be read by arviz to be plotted.
#         """
#         with pm.Model() as model:
#             my_dict = {}
#             for pot in self.potentials:
#                 name = '{}'.format(pot.a.label_string)
#                 if name not in excude_from_fit:
#                     my_dict[name] = pot.a.distribution()
#                 name = '{}'.format(pot.rho.label_string)
#                 if name not in excude_from_fit:
#                     my_dict[name] = pot.rho.distribution()
#                 name = '{}'.format(pot.c.label_string)
#                 if name not in excude_from_fit:
#                     my_dict[name] = pot.c.distribution()

#             simulator = pm.Simulator('simulator', self._simfunc, observed=self.expected_forces())
#             trace = pm.sample(step=pm.SMC(ABC=True, epsilon=epsilon), dis_func='absolute_error', draws=draws)
#         return trace
    






          
def get_lammps_data(params, supercell=None):
    """
    Collects the information needed for the lammps data inputs from the POSCARs and OUTCARs with additional information provided by params.
    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
        supercell (list(int)): 3 integers defining the cell increase in x, y, and z. Default=None if called directly.
    Returns:
        lammps_data (list(obj)):  LammpsData objects containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), and file_name (str).
        """
    if isinstance(supercell, list) is False :
            raise TypeError('Incorrect type for supercell. Requires integers for x,y,z expansion in a list, e.g. [1,1,1], or list of x,y,z expansions for each structure, e.g. [[1,1,1], [2,2,2], [3,3,3]]')
            
    lammps_data = []
    for i, pos in enumerate(sorted(glob.glob('poscars/POSCAR*'))):
        structure = Poscar.from_file(pos).structure
        forces = forces_from_outcar('outcars/OUTCAR{}'.format(i+1))[-1]
        structure.add_site_property('forces', forces)
        if supercell is not None:
            if isinstance(supercell[0], int) and len(supercell) == 3:
                structure = structure*supercell
            elif isinstance(supercell[i], list) and (all([len(supercell[i]) == 3 for i, x in enumerate(supercell)])):
                structure = structure*supercell[i]
            else:
                raise ValueError('Incorrect dimensions for supercell. Requires x,y,z expansion (i.e. list of 3 integers) or list of x,y,z expansions for each structure (i.e. list of list(x,y,z))')
 
        struct_data = LammpsData.from_structure(structure, params, i)
        lammps_data.append(struct_data)  
    return lammps_data
