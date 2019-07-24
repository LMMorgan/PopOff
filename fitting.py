import numpy as np
import pymc3 as pm
import glob
from vasppy.outcar import forces_from_outcar
from pymatgen.io.vasp import Poscar
import lammps
import potentials as pot
from lammps_data import LammpsData

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
    def collect_info(cls, params):
        """
        Collects information from other classes relating to the potentials and lammps_data using params input information.

        Args:
            params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
        Returns:
            (FitModel):  FitModel object containing potentials (list(obj:BuckinghamPotential)), lammps_data (obj:LammpsData), and cs_spring (dict).      
        """  
        lammps_data = get_lammps_data(params)
        parameters = pot.buckingham_parameters(params)
        potentials = pot.buckingham_potentials(params, lammps_data[0].atom_types, parameters) #REWRITE TO READ ATOM_TYPES WITHOUT [0] 
        cs_springs =  params['cs_springs']
        return cls(potentials, lammps_data, cs_springs)
    
    def write_lammps_files(self):
        """
        Writes the lammps_data to a lammps input file, which name is designated/identified by the numerical value in the related POSCAR and OUTCAR files i.e. if input files are 'POSCAR1' and 'OUTCAR1', the written lammps file will be 'coords1.lmp'.

        Args:
            None
                
        Returns:
            None
        """  
        for structure in self.lammps_data:
            lammps_file = structure.file_name
            with open( lammps_file, 'w' ) as f:
                f.write(structure.input_string())

    def initiate_lammps_instance(self):
        """
        Initialises the system from each structure (read in from a lammps input file) with non-changing parameters implemented.

        Args:
            None
                
        Returns:
            lmp_list (list(obj)): List of Lammps system objects with structure and specified commands implemented.
        """
 
        lmp_list = []
        for instance in self.lammps_data:
            lmp = lammps.Lammps(units='metal', style = 'full', args=['-log', 'none', '-screen', 'none'])
            lmp.command('read_data {}'.format(instance.file_name))

            lmp.command('group cores type {}'.format(instance.type_core()))
            lmp.command('group shells type {}'.format(instance.type_shell()))

            if self.cs_springs:
                lmp.command('pair_style buck/coul/long/cs 10.0')
                lmp.command('pair_coeff * * 0 1 0')

                lmp.command('bond_style harmonic')
                for i, spring in enumerate(self.cs_springs):
                    lmp.command('bond_coeff {} {} {}'.format(i+1,
                                                             self.cs_springs[spring][0],
                                                             self.cs_springs[spring][1]))
            else:
                lmp.command('pair_style buck/coul/long 10.0')
                lmp.command('pair_coeff * * 0 1 0')

            lmp.command('kspace_style ewald 1e-6')

            #setup for minimization
            lmp.command('min_style cg')
            lmp_list.append(lmp)
        return lmp_list       
    
    def update_potentials(self, **kwargs):
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
                    
    def set_potentials(self, lmp):
        """
        Sets the potential for the sepecified Lammps system (changes for each iteration of the potential fit).

        Args:
            lmp (obj): Lammps object with structure and specified commands implemented.

        Returns:
            None
        """
        for pot in self.potentials:
            lmp.command('{}'.format(pot.potential_string()))
            
    def simfunc(self, **kwargs):
        """
        Runs a minimization and zero step run for the instance and returns the forces.

        Args:
            **kwargs: Contain data for type of fitting and to what parameters as set with pm.Model.

        Returns:
            out (np.array): x,y,z forces on each atom.
        """
        core_mask = self.lammps_data[0].core_mask()
        instances = self.initiate_lammps_instance()
        
        if min(kwargs.values()) > 0:
            self.update_potentials(**kwargs)
            out = np.zeros([sum(core_mask), 3, len(instances)])

            for instance in instances:
                self.set_potentials(instance)
                instance.command('fix 1 cores setforce 0.0 0.0 0.0')
                instance.command('minimize 1e-25 1e-25 5000 10000')
                instance.command('unfix 1')
                instance.run(0)
                out[:,:,instances.index(instance)] = instance.system.forces[core_mask]

        else: out = np.ones([sum(core_mask),3, len(instances)])*999999999 # ThisAlgorithmBecomingSkynetCost

        return out

    def run_fit(self, excude_from_fit):
        """
        Runs the PyMC3 fitting process. Initiating a pm.Model, applying the potentials and distributions, and running the simulator that calls Lammps.

        Args:
            excude_from_fit (list(str)): Label of the parameters not wishing to fit to. Example 'Li_O_c'.

        Returns:
            trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3. This can be read by arviz to be plotted.
        """
        core_mask = self.lammps_data[0].core_mask()
        expected = np.zeros([sum(core_mask), 3, len(self.lammps_data)])
        
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

            simulator = pm.Simulator('simulator', self.simfunc, observed=expected)
            trace = pm.sample(step=pm.SMC(ABC=True, epsilon=0.1), draws=1000)
            #trace = pm.sample(step=pm.SMC(ABC=True, epsilon=1000, dist_func="sum_of_squared_distance"), draws=1000)
        return trace    
    
def get_lammps_data(params):
    """
    Collects the information needed for the lammps data inputs from the POSCARs and OUTCARs with additional information provided by params.

    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.

    Returns:
        lammps_data (list(obj)):  LammpsData objects containing atom_types (list(obj:AtomType)), bond_types (list(obj:BonType)), atoms (list(obj:Atom)), bonds (list(obj:Bond)), cell_lengths (list(float)), tilt_factor (list(float)), and file_name (str).
        """
    lammps_data = []
    for i, pos in enumerate(glob.glob('poscars/POSCAR*')):
        poscar = Poscar.from_file(pos)
        forces = forces_from_outcar('outcars/OUTCAR{}'.format(i+1))[-1]
        struct_data = LammpsData.from_structure(poscar.structure, params, forces, i)
        lammps_data.append(struct_data)  
    return lammps_data