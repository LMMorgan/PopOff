import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

def plotter(trace, i):
    """
    Plots the trace and the posterior distribution (with mode) directly from the trace using aviz.

    Args:
        trace (obj): A pymc3.backends.base.MultiTrace object containing a multitrace with information on the number of chains, iterations, and variables output from PyMC3.
        i (int): Counter for the index/iteration number.

    Returns:
        None
        """  
    az.style.use('arviz-darkgrid')
    pm.plot_trace(trace)
    plt.savefig('plots/trace_{}.png'.format(i),dpi=500, bbox_inches = "tight")
    pm.plot_posterior(trace, round_to = 3, point_estimate = 'mode')
    plt.savefig('plots/mode_{}.png'.format(i),dpi=500, bbox_inches = "tight")
    
