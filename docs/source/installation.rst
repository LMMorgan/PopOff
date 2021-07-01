Dependencies
============

To run the potential fitting code, you must first download LAMMPS and install at a minimum the LAMMPS python package. Installing LAMMPS itself is a relatively easy task. You just download the relevant source files, go to the src directory, and run make. Simple.

Compiling LAMMPS
----------------

Download the version of LAMMPS you want, making sure it is a release from March 23rd 2018. Before this date some of the libraries weren't properly supported. You can either ``git clone`` the latest stable version using:: 

   git clone -b stable https://github.com/lammps/lammps.git [path to build location]/lammps

(Or use ``wget`` or download the zip and move it to where you want to build LAMMPS.)

Once you have your LAMMPS folder, ``cd`` into the ``src`` directory (``lammps/src/``). Before building LAMMPS and creating the shared library, you may want to include some of the LAMMPS packages. This is fairly simple to do and you will probably want to have some of these packages. To see a list of the available packages use ``make package-status``. This will also tell you if they are active or not. To activate any of these packages you can use ``make yes-<package_name>`` or to deactivate use ``make no-<package_name>``. If you want to install a bunch of packages in one go you can use the following:: 

   export LAMMPS_PACKAGES="asphere body class2 colloid compress coreshell dipole granular kspace manybody mc misc molecule opt peri qeq replica rigid shock snap srd user-reaxc"
   for pack in $LAMMPS_PACKAGES; do make "yes-$pack"; done

These are the packages installed during development and testing of this code, though you should note that some packages are not compatible with others and can cause conflicts. So just be careful.

Right, now you are ready to install LAMMPS!

This is done using a ``make`` command. Here you can also chose which mode to use, to run as serial or mpi, and which flags to use. There are suggested flags to use to get everything working smoothly. The make command I would suggest using is::

   make mode=shlib mpi -j4 LMP_INC="-DLAMMPS_EXCEPTIONS -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64 -DBUILD_SHARED_LIBS=on"

If you only want to install the python interface package, you can do this using::

   make install-python

Boom! Now you are compiling LAMMPS. Make a cup of tea (or coffee if you prefer) while it runs. This could take about 5 minutes, or longer if you have more packages.

When LAMMPS has finished compiling you will need to make sure your libraries have installed in the correct place and that the paths to these locations are connected through a shared linker (e.g. like ``/usr/lib64/``) or part of the ``LD_LIBRARY_PATH`` environment variable, otherwise you will get an error when creating a LAMMPS object through the python module. You can check this using ``echo $LD_LIBRARY_PATH``. This should give the file path to where the ``.so`` file is saved. If this is not linked, you can set this using (with your file path)::

    # Unix/Linux
    export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

    # MacOS
    export DYLD_LIBRARY_PATH=$HOME/.local/lib:$DYLD_LIBRARY_PATH
    or
    export LD_LIBRARY_PATH=/home/[user]/.local/lib/ (depending on path)

Further details on the different types of LAMMPS installs and for debugging issue, see the `LAMMPS installation documentation <https://docs.lammps.org/Python_install.html>`_.

NOTE: This may not stay permanent on workstations or hpcs. To save having to redo this command each time you login add it to your ``.bash_profile``.

Testing
-------

If this works you should be able to run::
 
   from lammps import lammps
   lammps()
