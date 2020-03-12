import numpy as np
from pymatgen.io.vasp.outputs import Outcar

def forces_from_outcar( filename='OUTCAR' ):
    """
    Finds and returns forces from the OUTCAR file.
    
    Args:
        filename (:obj:'str', optional): the name of the ``OUTCAR`` file to be read. Default is `OUTCAR`.
        
    Returns:
        (np.array): The force as found in the ``OUTCAR`` file, as a NSTEPS x NIONS x 3 numpy array.
    """
    outcar = Outcar(filename)
    forces = outcar.read_table_pattern(
        header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
        row_pattern=r"\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s--+",
        postprocess=lambda x: float(x),
        last_one_only=False
    )
    return np.array( forces )


def stresses_from_outcar( filename='OUTCAR' ):
    """
    Finds and returns stress tensors from the OUTCAR file.
    
    Args:
        filename (:obj:'str', optional): the name of the ``OUTCAR`` file to be read. Default is `OUTCAR`.
        
    Returns:
        (np.array): The stresses as found in the ``OUTCAR`` file in kBar, as a 1D numpy array.
    """
    outcar = Outcar(filename)
    stresses = outcar.read_table_pattern(
        header_pattern=r"\s+Fock(\s+[+-]?\d+\.\d+)*\n\s+-+\n\s+Total(\s+[+-]?\d+\.\d+)*",
        row_pattern=r"\s+\D+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s+external.+",
        postprocess=lambda x: float(x),
        last_one_only=False
    )
    return np.array(stresses).flatten()