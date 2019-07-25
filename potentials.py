from potential_parameters import buckingham_parameters

class BuckinghamPotential():
    """
    Class that contains the information for each buckingham potential, where parameters are
    BuckinghamParameter objects.
    """
    def __init__(self, labels, atom_type_index, a, rho, c):
        """
        Initialise each parameter in each buckingham potential.

        Args:
            labels (list(str)): List of the atoms in the potential i.e. ['O','O'] for O-O potential.
            atom_type_index (list(int)): List of the atom type index for the atoms in the potential.
            a (obj): BuckinghamParameter objects including parameter_type (str), label_string (str), value (float), and sd (float).
            rho (obj): BuckinghamParameter objects including parameter_type (str), label_string (str), value (float), and sd (float).
            c (obj): BuckinghamParameter objects including parameter_type (str), label_string (str), value (float), and sd (float).
                
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
            return_str (str): atype_index for atom pairs, and buckingham potential parameter values
                              formatted as a lammps command.
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
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str). Also contains bpp (list(float)) and sd (list(float)) dictionaries where the keys are atom label pairs (str), example: 'Li-O'.
        atom_types (list(obj)): AtomType objects including atom_type_index (int), label (str), mass (float), charge (float), and core_shell (str).                  
        parameters (list(obj)): BuckinghamParameter objects including parameter_type (str), label_string (str), value (float), and sd (float).                
    Returns:
        potentials (list(obj)): BuckinghamPotential objects including labels (list(str)), atom_type_index (list(int)), a (obj), rho (obj), and c (obj). Each object is a BuckinghamParameter object.
        
    """
    i = 0 #parameter_counter
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