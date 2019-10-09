import arviz as az
import pandas as pd
import numpy as np
from scipy import stats

def initiate_db(summary_filename):
    """
    Initiates empty databases for collecting summart data nad output forces data.

    Args:
        summary_filename (str): Name of the summary file.

    Returns:
        None
        """
    summary_labels = ['iteration','parameter','mean', 'std', 'median', 'mode', 'data_points']
    summary_db = pd.DataFrame( columns=summary_labels ).astype('object')
    summary_db.to_csv(summary_filename, index=False)

def print_summary(i, trace, summary_filename):
    """
    Adds the data of each iteration of the fit to the summary databasae.

    Args:
        i (int): Counter for the index/iteration number.
        trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3. 
        summary_filename (str): Name of the summary file.
        
    Returns:
        None
        """
    func_dict = {"mean": np.mean,
                 "std": np.std,
                 "median": lambda x: np.percentile(x, 50),
                 "mode": lambda x : float(stats.mode(x)[0])}
    database = pd.read_csv(summary_filename)
    new_entry = az.summary(trace, stat_funcs=func_dict, extend=False)
    new_entry['iteration'] = [i for var in trace.varnames]
    new_entry['data_points'] = [trace.get_values(var) for var in trace.varnames]
    new_entry['parameter'] = [var for var in trace.varnames]
    database = database.append(new_entry).astype('object')
    database.to_csv(summary_filename, index=False)