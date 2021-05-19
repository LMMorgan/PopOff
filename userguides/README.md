# POPOFF: POtential Parameter Optimisation for Force-Fields

## Overview
PopOff currently fits Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potenial. You can choose between fitting a formal charge, partial charge, or fitting a charge scaling factor. You can choose to fit a rigid ion model or a core-shell model. If choosing to fit a core-shell model, you are able to fix or fit the charge separation over the core and shell components and the spring constant. You can also selectively fix, fit, or set to zero the individual buckingham parameters.

# Userguides

These userguides cover building a training set and going through setting up the inputs for the fit and the different parameters which can be set. Following this, there are notebooks which show how to run a cross-validation, check the lattice parameters against a reference, and plot these for comparison.

## Notebooks

Interactive versions of each notebook can be viewed using a [Jupyter notebook](http://jupyter-notebook.readthedocs.io/en/latest/#), or viewed using [nbviewer](https://nbviewer.jupyter.org) by following each [nbviewer]() link.
- [training_set](https://github.com/LMMorgan/PopOff/blob/master/userguides/training_set.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/LMMorgan/PopOff/blob/master/userguides/training_set.ipynb))
- [fitting](https://github.com/LMMorgan/PopOff/blob/master/userguides/fitting.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/LMMorgan/PopOff/blob/master/userguides/fitting.ipynb))
- [plotting](https://github.com/LMMorgan/PopOff/blob/master/userguides/plotting.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/LMMorgan/PopOff/blob/master/userguides/plotting.ipynb))
- [lattice_parameters](https://github.com/LMMorgan/PopOff/blob/master/userguides/lattice_parameters.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/LMMorgan/PopOff/blob/master/userguides/lattice_parameters.ipynb))
- [cross-validation](https://github.com/LMMorgan/PopOff/blob/master/userguides/cross-validation.ipynb) ([nbviewer](https://nbviewer.jupyter.org/github/LMMorgan/PopOff/blob/master/userguides/cross-validation.ipynb))
