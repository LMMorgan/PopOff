
PopOff is a Python module that fits Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potenial. You can choose between fitting a formal charge, partial charge, or fitting a charge scaling factor. You can choose to fit a rigid ion model or a core-shell model. If choosing to fit a core-shell model, you are able to fix or fit the charge separation over the core and shell components and the spring constant. You can also selectively fix, fit, or set to zero the individual buckingham parameters.

PopOff currently fits to first principles derived forces and stress tensors obtained from a VASP training set. Details on how to build a VASP training set can be found in the `training set guide`_. This code uses the LAMMPS molecular dynamics code.

.. _training set guide: https://github.com/LMMorgan/PopOff/blob/master/userguides/trainingsetguide.md

Installation
------------

Installation of this module requires use of the LAMMPS molecular dynamics code to run. Instructions on how to install these can be found on the `LAMMPS installation documentation <https://docs.lammps.org/Python_install.html>`_ page.

The simplest way to install ``PopOff`` is to use ``pip`` to install from `PyPI <https://pypi.org/project/PopOff/>`_::

    pip install PopOff

Alternatively, you can download the latest release from `GitHub <https://github.com/LMMorgan/PopOff>`_, and install directly::

    cd popoff
    pip install -e .

which installs an editable (-e) version of PopOff in your userspace.

Or clone the latest version from `GitHub <https://github.com/LMMorgan/PopOff>`_ with::

    git clone git@github.com:LMMorgan/PopOff.git

and install the same way::

    cd popoff
    pip install -e .

Tests
-----

Tests for each module are conducted using `pytest <https://docs.pytest.org/en/stable/usage.html>`_ and can be found in::

	popoff/tests/

The tests can be run using::

	pytest

The input for the test calculations is stored in the ``test_files`` directory. Please make sure to unzip the vasprun.xml files for the tests to run correctly. This can be done using ``gzip <file>``.


.. include:: userguide.rst

Citing PopOff
--------------

This code can be cited as:

Morgan, Lucy M., Clarke, Matt J., Islam, M. Saiful, & Morgan, Benjamin J. (2021). *PopOff* Zenodo. http://doi.org/10.5281/zenodo.4773795

### BibTeX::

    @misc{morgan_2021_popoff,
      author       = {Morgan, Lucy M. and
                      Clarke, Matt J. and
                      Islam, M. Saiful and
                      Morgan, Benjamin J.},
      title        = {{PopOff: POtential Parameter Optimisation for Force-Fields}},
      month        = may,
      year         = 2021,
      doi          = {10.5281/zenodo.4773795},
      url          = {10.5281/zenodo.4773795}
    }
