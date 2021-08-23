# PopOff: POtential Parameter Optimisation for Force-Fields
[![PyPI version](https://badge.fury.io/py/PopOff.svg)](https://badge.fury.io/py/PopOff)
[![DOI](https://zenodo.org/badge/189393218.svg)](https://zenodo.org/badge/latestdoi/189393218)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Documentation Status](https://readthedocs.org/projects/popoff/badge/?version=latest)](https://popoff.readthedocs.io/en/latest//?badge=latest)

## Overview
PopOff currently fits Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potenial. You can choose between fitting a formal charge, partial charge, or fitting a charge scaling factor. You can choose to fit a rigid ion model or a core-shell model. If choosing to fit a core-shell model, you are able to fix or fit the charge separation over the core and shell components and the spring constant. You can also selectively fix, fit, or set to zero the individual buckingham parameters.

## Prerequisites
To run the potential fitting code, you need LAMMPS with the LAMMPS-python interface package. LAMMPS is relatively easy to install and so is the python interface if you are working locally. The python interface can be challenging to compile on HPC or larger servers and you may need assistance from the dedicated HPC support team. Here, I provide the basic instructions for compiling LAMMPS and the potential fitting code locally. I also provide instructions on how to download and install the code in a conda enviroment to run on Archer2 (which may work for other HPC systems). 

## Compiling LAMMPS and the code locally
Download the version of LAMMPS you want, making sure it is a release from March 23rd 2018 onwards. Before this date some of the libraries weren't properly supported. You can either `git clone` the latest stable version using:

```
git clone -b stable https://github.com/lammps/lammps.git [path to build location]/lammps
```

(Or use `wget` or download the zip and move it to where you want to build LAMMPS.)

Once you have your LAMMPS folder, `cd` into the `src` directory (`lammps/src/`). Before building LAMMPS and creating the shared library, you may want to include some of the LAMMPS packages. This is fairly simple to do and you will probably want to have some of these packages. To see a list of the available packages use `make package-status`. This will also tell you if they are active or not. To activate any of these packages you can use `make yes-<package_name>` or to deactivate use `make no-<package_name>`. If you want to install a bunch of packages in one go you can use the following:

```
export LAMMPS_PACKAGES="asphere body class2 colloid compress coreshell dipole granular kspace manybody mc misc molecule opt peri qeq replica rigid shock snap srd user-reaxc"
for pack in $LAMMPS_PACKAGES; do make "yes-$pack"; done
```

These are the packages installed during development and testing of this code, though you should note that some packages are not compatible with others and can cause conflicts. So, just be careful.

Right, now you are ready to install LAMMPS!

This is done using a `make` command (you can also use cmake if preferred). Here you can also chose which mode to use, to run as serial or mpi, and which flags to use. There are suggested flags to use to get everything working smoothly. The make command I would suggest using is:

```
make mode=shlib mpi -j4 LMP_INC="-DLAMMPS_EXCEPTIONS -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64 -DBUILD_SHARED_LIBS=on"
```

Boom! Now you are compiling LAMMPS. Make a cup of tea (or coffee if you prefer) while it runs. This could take about 5 minutes, or longer if you have more packages.

To install the python interface package, you can do this using:

```
make install-python
```

When LAMMPS has finished compiling you will need to make sure your libraries have installed in the correct place and that the paths to these locations are connected through a shared linker (e.g. like `/usr/lib64/`) or part of the `LD_LIBRARY_PATH` environment variable, otherwise you will get an error when creating a LAMMPS object through the python module. You can check this using `echo $LD_LIBRARY_PATH`. This should give the file path to where the `.so` file is saved. If this is not linked, you can set this using (with your file path):

```
# Unix/Linux
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# MacOS
export DYLD_LIBRARY_PATH=$HOME/.local/lib:$DYLD_LIBRARY_PATH
or
export LD_LIBRARY_PATH=/home/[user]/.local/lib/ (depending on path)
```

Further details on the different types of LAMMPS installs and for debugging issue, see the [LAMMPS installation documentation](https://docs.lammps.org/Python_install.html).

NOTE: This may not stay permanent on workstations or hpcs. To save having to redo this command each time you login add it to your `.bash_profile`.



## Compiling LAMMPS and the code in a conda environment (for Archer2)
Download the relivant version of Anaconda from [here](https://docs.anaconda.com/anaconda/install/index.html). For Archer2, this will be the [Linux installer](https://docs.anaconda.com/anaconda/install/linux/). Upload the installer to the HPC to a working directory that has access to the compute nodes. For Archer2 this will be in the `/work/<project-code>/<project-code>/<username/` directory. Once in the correct place run:

```
bash <installer.sh>
```

Follow the instructions given. Review and accept the license agreement, select your install location (This will default in Archer2 to `/home` so make sure you change this to `/work`, ideally create a `.local` directory in your work user space.), then accept creating a `conda init`. If you do not create the conda init, you will need to run:

```
source <conda location>/bin/activate
conda init
```

Restart your terminal session or run `source ~/.bashrc`

Next, [create a conda virtual enviroment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) using:

```
conda create -n <yourenvname> python=x.x anaconda
```

where `x` is the version of python you want in your environment. Then activate your environment using:

```
source activate <yourenvname>
```

Now, you will need to conda install (rather than pip install) the prerequisite python packages. This is done using:

```
conda install -n <yourenvname> -c numpy scipy matplotlib
conda install -n <yourenvname> -c conda-forge pymatgen
```

LAMMPS with the python interface can then be [installed through conda](https://anaconda.org/conda-forge/lammps) using:

```
conda install -n <yourenvname> -c conda-forge/label/gcc7 lammps
```




NOTE: On the current Archer2 system, the conda init may not pass into the submission script correctly. If when submitting you get a location error, then add the conda init to your submission script.

For Archer2, you will need your usual SBATCH commands, followed by the conda init (if not read from `.bashrc`), followed by `source activate <yourenvname>`, and finally `python <input_file.py> > <output file>`, where the input file is currently named `fitting.py`.


## Testing LAMMPS-python is working

**If this works you should be able to run:**

```
from lammps import lammps
lammps()
```
