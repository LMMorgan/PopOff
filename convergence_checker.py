import numpy as np
from scipy import stats

def get_modes(trace):
    """
    Adds the data of each iteration of the fit to the summary databasae.

    Args:
        trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3. 

    Returns:
        modes (list(float)): Mode potential parameter values for current iteration.
    """
    modes = [float(stats.mode(trace.get_values(var))[0]) for var in trace.varnames]
    return modes

def converge_check(modes, distribution, prev_modes=None):
    """
    Checks the potential parameters to determine whether they are converged or not. Currently the convergence criteria is within 10% of previous. This must be true for all parameters to pass.
    Args:
        modes (list(float)): Mode potential parameter values for current iteration.
        distribution(dict(dict)): Contains buckingham potential, 'bpp':list(float), and 'sd':list(float) dictionaries where the distribution keys are atom label pairs (str), example: 'Li-O'.
        prev_modes (list(float)): Mode potential parameter values for previous iteration. If no previous, uses default. Default=None.
        
    Returns:
        (Boolean): True if all potentials are converged, otherwise False.    
    """  
    converged = []
    if prev_modes is None:
        np.set_printoptions(suppress=True)
        prev_modes = list(np.concatenate([pot['bpp'] for pot in distribution.values()]))
    for prev,cur in zip(prev_modes,modes):
        try:
            if (abs(cur - prev) / prev) * 100.0 > 3.0:
                converged.append(False)
            else:
                converged.append(True)
        except ZeroDivisionError:
            converged.append(False)
    if all(converged):
        return True
    else:
        return False