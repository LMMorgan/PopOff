from fitting import FitModel
import pymc3 as pm

def iter_fitting(params, distribution, excude_from_fit=None):
    """
    Collects information from other classes relating to the potentials and lammps_data using params input information.
    Args:
        params (dict(dict)): Contains core_shell (bool), charges (float), masses (float), and cs_springs (list(float)) dictionaries where the keys are atom label (str).
        distribution(dict(dict)): Contains buckingham potential, 'bpp':list(float), and 'sd':list(float) dictionaries where the distribution keys are atom label pairs (str), example: 'Li-O'.
        excude_from_fit (list(str)): Label of the parameters not wishing to fit to. Example 'Li_O_c'. Default=None.
        
    Returns:
        trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3.
        (FitModel):  FitModel object containing potentials (list(obj:BuckinghamPotential)), lammps_data (obj:LammpsData), and cs_spring (dict).      
    """     
    fit_data = FitModel.collect_info(params, distribution, supercell=[1,1,1])
    trace = fit_data.run_fit(excude_from_fit=excude_from_fit, epsilon=1.0, draws=100, dist_func='sum_of_squared_distance')
    return trace, fit_data


def update_potentials(trace, modes, distribution):
    """
    Updates the potentials in the distribution dictionary to the mode values calculated from the distribution and the standard deviations output by the fit in the trace, ready for the next iteration.
    Args:
        trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3.
        modes (list(float)): Mode potential parameter values for current iteration.
        distribution(dict(dict)): Contains buckingham potential, 'bpp':list(float), and 'sd':list(float) dictionaries where the distribution keys are atom label pairs (str), example: 'Li-O'.
        
    Returns:
        distribution(dict(dict)): Contains buckingham potential, 'bpp':list(float), and 'sd':list(float) dictionaries where the distribution keys are atom label pairs (str), example: 'Li-O'.      
    """   
    data = pm.summary(trace)
    i = 0
    for pot in distribution.values():
        pot['bpp'] = [round(modes[i],3), round(modes[i+1],3), round(modes[i+2],3)]
        pot['sd'] = [round(data['sd'][i],3), round(data['sd'][i+1],3), round(data['sd'][i+2],3)]
        i += 3
    return distribution