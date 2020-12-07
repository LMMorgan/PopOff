# BuckFit

## Prelude

To run the potential fitting code, you must first install LAMMPS and the lammps-cython interface. Installing LAMMPS itself is a relatively easy task. You just download the relevant source files, go to the src directory and run make. Simple. Installing LAMMPS with lammps-cython however is a little more complex.

Here I am going to go through how to install this using *make* and *pip*, with the correct dependancies installed using *apt* (for linux based systems) and *brew* (for Macs).

[This link](https://costrouc.gitlab.io/lammps-cython/installation.html#id3) contains the instructions which most of this guide is based on.

## Installing dependencies
As with all great bits of code, you need to install some packages to get things working properly. `lammps-cython` is no exception. You will need make sure you have openblas, openmpi, and fftw. These can be installed in the follwoing ways.

For Ubuntu/linux systems use:
```
apt install build-essential libopenblas-dev libfftw3-dev libopenmpi-dev
```

For Mac/OSX systems (using [homebrew](https://brew.sh)) use:
```
brew install openblas open-mpi fftw
```
(Note: this uses `open-mpi`  in place of `mpich`, but seems to work).


**NOTE**: It should also be noted that there are certain libraries that are needed for lamps-python/lammps-cython to run properly. Lammps many still install properly but it will throw up errors later. Common ones to make sure you include are libpng and zlib. Mpich can also be used in place of open-mpi however this is not something I have tested so things may break.

## Compiling LAMMPS
Download the version of LAMMPS you want, making sure it is a release from March 23rd 2018. Before this date some of the libraries weren't properly supported. You can either `git clone` the latest stable version using:

```
git clone -b stable https://github.com/lammps/lammps.git [path to build location]/lammps
```

(Or use `wget` or download the zip and move it to where you want to build LAMMPS.)

Once you have your LAMMPS folder, `cd` into the `src` directory (`lammps/src/`). Before building LAMMPS and creating the shared library, you may want to include some of the LAMMPS packages. This is fairly simple to do and you will probably want to have some of these packages. To see a list of the available packages use `make package-status`. This will also tell you if they are active or not. To activate any of these packages you can use `make yes-<package_name>` or to deactivate use `make no-<package_name>`. If you want to install a bunch of packages in one go you can use the following:

```
export LAMMPS_PACKAGES="asphere body class2 colloid compress coreshell dipole granular kspace manybody mc misc molecule opt peri qeq replica rigid shock snap srd user-reaxc"
for pack in $LAMMPS_PACKAGES; do make "yes-$pack"; done
```

These are the packages I have installed in my version, though you should note that some packages are not compatible with others and can cause conflicts. So just be careful.

Right, now you are ready to install LAMMPS!

This is done using a `make` command. Here you can also chose which mode to use, to run as serial or mpi, and which flags to use. I won't pretend to be an expert in this (I am definitely not!), however there are suggested flags to use to get everything working smoothly. The make command I would suggest to use is:

```
make mode=shlib mpi -j4 LMP_INC="-DLAMMPS_EXCEPTIONS -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64"
```

Boom! Now you are compiling LAMMPS. Make a cup of tea (or coffee if you prefer) while it runs. This could take about 5 minutes, or longer if you have more packages.

When LAMMPS has finished compiling you will need to copy the `.so` file to a library directory and all the `.h` files to an include directory. On the workstations and hpcs, you will not have root permission (i.e. write access to `/usr/local/lib` and `/usr/local/include`) so the best place to place these files are in `/home/[user]/.local/lib/` and `/home/[user]/.local/include/`. If these locations don't exist then you can make them. The commands for copying the files over to these directories are:

```
cp liblammps_mpi.so /home/[user]/.local/lib/liblammps_mpi.so
mkdir /home/[user]/.local/include/lammps/; cp *.h /home/[user]/.local/include/lammps/
```

**NOTE**: If you are unsure on the full path of your home directory (i.e. everything before `.local`) you can type `pwd` in your home directory and it will give you your path.

Following this you may wish to include the python package to allow a lammps-python environment without installing lammps-cython. To do this, in the `src` directory, do `make install-python`. This will only take a second or so to run. If you are installing lammps-cython this step is unnecessary.

## Installing lammps-cython

Okay, so if LAMMPS compiled properly, then this should be fairly simple. In practice, this is likely the fiddly bit where everything breaks. Lammps-cython also has some prerequisites. These can easily be installed using:

```
pip install numpy mpi4py cython
```

To make sure everything is pointing to the right places, we have a few more steps to go through before installing `lammps-cython`. Firstly we need to create a `.cfg` file. This should be placed in `~/.config/lammps-site.cfg`, the content of which should be as follows:

```
# multiple values can be included seperated by commas
[lammps]
lammps_include_dir = /home/[user]/.local/include/lammps/
lammps_library_dir = /home/[user]/.local/lib/
# true library filename is liblammps_mpi.so notice lib and .so are removed
lammps_library = lammps_mpi

# use mpic++ -showme to list libraries and includes
[mpi]
mpi_include_dir = /usr/lib/x86_64-linux-gnu/openmpi/include
mpi_library_dir = /usr/lib/x86_64-linux-gnu/openmpi/lib
# no necissarily needed (default are mpi, mpi_cxx)
mpi_library     = mpi, mpi_cxx
```

**NOTE**: If a `.config` directory doesn't exist (i.e. not a writable location), you need to create it. Also if you are installing on a Mac, it doesn't seem to like `mpi_cxx`, so just use `mpi`. I.e. `mpi_library = mpi`.

This file now needs editing to make sure the file paths are correct to where you have placed your `.so` and `.h` files. On a similar note, check that the mpi file paths are also correct. These are dependant on your system, or the hpc you are using. Use `mpic++ -showme` to show a list of the libraries and includes which contain mpi. This will spit out something similar to this:

```
g++ -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi
```

You will want the file paths which contain `openmpi` and end in `/include` and `/lib`. Ignore the `-I/` at the start. This is not needed.

With the config file all setup and pointing to the right locations it should now just be a case of:

```
pip install lammps-cython
```

Now is a good time to check that your `LD_LIBRARY_PATH` is pointing to the correct place. You can check this using `echo $LD_LIBRARY_PATH`. This should give the file path to where you have saved your `.so` file, i.e. your lammps_library_dir path in the config file we made earlier. If not you will need to set this using (with your file path):

```
export LD_LIBRARY_PATH=/home/[user]/.local/lib/
```

NOTE: This may not stay permanent on workstations or hpcs. To save having to redo this command each time you login add it to your `.bash_profile`.

## Testing

**If this works you should be able to run:***
```
from lammps import Lammps
Lammps()
```

There are example/test notebooks (basic.ipynb and benchmark.ipynb) located [here](https://github.com/costrouc/lammps-cython) under `Binder Notebooks`, which you can use to test everything is running correctly and also give you an idea of how to use lammps-cython.

## Debugging a failed compile

LAMMPS itself should compile fine. If this isn't the case, there are suggestions in the LAMMPS manual you can read through. The most likely part of the process that will fail is the installation of lammps-cython. There are a few possible reasons for this, the most likely being clashes in python environments and MPI clashes.

Python environments - If, like myself, you weren't entirely sure what you were doing when you set up your computer/HPC or workstation environment then this could be an issue. The same applies if you have tried to install python through different routes i.e. direct downloads, brew install, anaconda, etc. This can lead to your python paths pointing in different places or being inconsistent. The best way to solve this is to carefully remove the different environments and stick to just one. Personally I would suggest removing all environments and use `pyenv`. This can be brew/apt installed and allows you to keep all your different python versions and environments in a controlled way. After cleaning up your environments, make sure all your paths are pointing in the right locations and that your `.bash_rc`, `.bash_profile`, etc., don't have explicit paths relating to an old environment.

MPI clashes - This one can be a bit of a challenge to fix. Firstly, check the error message you get and see where the paths are pointed. A likely issue is the paths pointing towards `mpich` when you are using a different compiler, such as `openmpi`. This is more of an issue on personal computers. To fix this you need to uninstall `mpich` and remove the files in the path. It would also be wise to remove the cache directories for `pip` and `brew` (or equivalent for `apt` and `conda`) as the paths can be held in memory. `mpi4py` should also be removed (including the cache) and reinstalled to make sure the paths are updated. This should hopefully fix the issue. If not, make sure your `lammps-site.cfg` file contains the right paths.

Also make sure Xcode is up to date. This can also cause issues if not.
