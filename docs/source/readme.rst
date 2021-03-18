
BuckFit is a Python module that fits Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potenial. You can choose between fitting a formal charge, partial charge, or fitting a charge scaling factor. You can choose to fit a rigid ion model or a core-shell model. If choosing to fit a core-shell model, you are able to fix or fit the charge separation over the core and shell components and the spring constant. You can also selectively fix, fit, or set to zero the individual buckingham parameters.

BuckFit currently fits to first principles derived forces and stress tensors obtained from a VASP training set. Details on how to build a VASP training set can be found in the `training set guide`_. This code uses the LAMMPS molecular dynamics code.

.. _training set guide: https://github.com/LMMorgan/BuckFit/blob/master/userguides/trainingsetguide.md

Installation
============

Installation of this module requires use of the LAMMPS molecular dynamics code to run and the ``lammps-cython`` interface. Instructions on how to install these can be found on the `installing dependencies <https://buckfit.readthedocs.io/en/latest/installation.html>`_ page.

The simplest way to install ``BuckFit`` is to use ``pip`` to install from `PyPI <https://pypi.org/project/BuckFit/>`_::

    pip install BuckFit

Alternatively, you can download the latest release from `GitHub <https://github.com/LMMorgan/BuckFit>`_, and install directly::

    cd buckfit
    pip install -e .

which installs an editable (-e) version of pyscses in your userspace.

Or clone the latest version from `GitHub <https://github.com/LMMorgan/BuckFit>`_ with::

    git clone git@github.com:LMMorgan/BuckFit.git

and install the same way::

    cd buckfit
    pip install -e .

Tests
=====

Tests for each module are conducted using `pytest <https://docs.pytest.org/en/stable/usage.html>`_ and can be found in::

	buckfit/tests/

The tests can be run using::

	pytest

The input for the test calculations is stored in the ``test_files`` directory.

User Guides
===========

Once the `dependencies <https://buckfit.readthedocs.io/en/latest/installation.html>`_ and ``BuckFit`` is installed ....



Citing BuckFit
==============

This code can be cited as:

Morgan, Lucy M., Clarke, Matt J., Islam, M. Saiful, & Morgan, Benjamin J. (2020). *BuckFit* Zenodo. http://doi.org/10.5281/zenodo.4311103

### BibTeX::

    @misc{wellock_georgina_l_2019_2536901,
      author       = {Morgan, Lucy M. and
							 Clarke, Matt J. and
                      Islam, M. Saiful and
                      Morgan, Benjamin J.},
      title        = {{BuckFit}},
      month        = dec,
      year         = 2020,
      doi          = {10.5281/zenodo.4311103},
      url          = {10.5281/zenodo.4311103}
    }
