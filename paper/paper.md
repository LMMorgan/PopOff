---
title: 'popoff: POtential Parameter Optimisation for Force-Fields'
tags:
 - Python
 - Classical MD
 - potentials
 - molecular dynamics
 - force-fields
 - force field potentials
authors:
 - name: Lucy M. Morgan
   email: l.m.morgan@bath.ac.uk
   orcid: 0000-0002-6432-3760
   affiliation: "1, 2"
 - name: Matt J. Clarke
   email: mjc62@bath.ac.uk
   orcid: 0000-0003-1371-4160
   affiliation: "1"
 - name: M. Saiful Islam
   email: msi20@bath.ac.uk
   orcid: 0000-0002-8077-6241
   affiliation: "1, 2"
 - name: Benjamin J. Morgan
   email: b.j.morgan@bath.ac.uk
   orcid: 0000-0002-3056-8233
   affiliation: "1, 2"
affiliations:
 - name: Department of Chemistry, University of Bath, Claverton Down, UK, BA2 7AY
   index: 1
 - name: The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, UK
   index: 2
date: 18 June 2021
bibliography: paper.bib
---

# Summary
`PopOff` is a Python package for fitting Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potential. You can choose between fitting a formal charge, partial charge, or fitting a charge scaling factor. You can choose to fit a rigid ion model, a core-shell model, or mixture of both. If choosing to fit a core-shell model, you can fix or fit the charge separation over the core and shell components and the spring constant. You can also selectively fix, fit, or set to zero the individual buckingham parameters. `PopOff` currently fits to first principles derived forces and stress tensors obtained from a VASP training set through the `vasprun.xml` output files. The LAMMPS MD code is used as a backend to perform the MD during the fitting process.

The development of sufficiently accurate interatomic potentials, which requires model parameterization with respect to a set of target observables, for a specific chemistry is quite challenging, especially for complex systems. For example, for layered structures, such as the widely used LiNiO$_2$ based cathode materials, the short-range interactions are overwhelmed by the longer-range Coulombic term. In these cases, the system charges need to be scaled down to increase the influence of the short-range interactions, termed as partial charges, and the influence of polarizability needs to be included, usually through using a core-shell model. Variations of the Buckingham potential has been developed for these systems, some using rigid ion models,[@Lewis_1985; @Ledwaba2020; @Sayle2005; @Dawson2014] and others using core-shell models, [@Hart1998; @Fisher2010; @Lewis_1985; @Ammundsen1999; @Kerisit2014; @He2019; @lee2012atomistic] with a mixture of formal and partial charges being used. Interatomic potentials are traditionally based on mathematical functions which has been parameterised using experimental and/or First Principles derived data. [@jones_1924; @buckingham_classical_1938] There are a limited number of codes available with the explicit purpose or functionality for fitting potentials. Fitting routines from established codes including the General Utility Lattice Program (GULP), [@gale_gulp_1997] Atomicrex, [@Stukowski_2017] dftfit, [@dftfit] and potfit [@wen_kim-compliant_2017] each poses unique functionality, however, none are fully able to produce robust potentials for NMC or LiNiO$_2$ due to different aspects not being considered within the code, such as charge scaling or core-shell models. `PopOff` has been specifically developed for fitting different permutations of the Buckingham potential. Its modular design allows flexible fitting of both rigid ion and core-shell models, and formal and partial charges. 

We are currently using `PopOff` in our own research to develop potentials for cathode and solid-electrolyte materials for classical MD analysis of the Li$^+$/Na$^+$ migration properties. `PopOff` is also being used by a group at Newcastle University to develop potentials for several halide based solid-electrolyte materials in collaboration with experimental partners. We hope that this open-source resource will support development of interatomic potentials tailored to individual systems, and thus results in better approximations of the atomic interactions.

# Statement of need
Interatomic potentials give an approximate mathematical description of the interactions between atoms in a given system. These interactions are unique to the species and the surrounding environment, yet, in the wider literature interatomic potentials between specific atom types, for example Li-O, are reused in vastly different systems. This is particularly familiar for the O-O potential derived by Jackson and Catlow in 1985 for UO$_2$ [@jackson1985trapping], which has since been used in the study of many materials, in particular ionic solids. Although in some cases the species interactions may not vary substantial between different materials, in other cases it does, and to varying degrees. For example, a particular metal-oxygen interaction in a spinel might be considerably different to that in a layered oxide material, due in part to the change in charge distribution increased influence of polarizability. It is therefore crucial to develop interatomic potentials for individual systems to get the best approximation of the interactions within that environment.

There are a limited number of codes available with the explicit purpose or functionality for fitting potentials. Most of which require some pre-requisite knowledge of coding or specialism, creating a prohibitive barrier for utilising these tools in the wider atomistic modelling research community. These codes are also not able to account for all aspects which may need to be considered for particular materials, such as polarizability. Creating a resource, with minimal pre-requisite requirements to use, and able to account for a larger range of material considerations needed for accurate potentials is therefore key to producing more robust classical MD investigations.

# Optimiser
`PopOff` uses `scipy.optimize.differentail_evolution` as its global optimiser to optimise the function that calculates the mean squared error between the forces and stress tensors within the training set and those calculated with possible classical MD potentials, via:

$$
\chi^{2} = \frac{1}{n} \sum_{i=1}^{n}(Y_{i} - \hat{Y}_{i})^{2}
$$

where $n$ is the number of data points, $Y_i$ is the observed values, and $\hat{Y}_i$ is the predicted values.

Differential evolution is a stochastic population-based method, where at each pass through the population the algorithm mutates each candidate solution by mixing with other candidate solutions to create a trial candidate. There are several strategy methods which can be used to create the trial candidate, with `latinhypercube` used a default.

`PopOff`'s modular design allows the code to fit the following parameters:

- A rigid ion model, a core-shell model, or mixture of both.
- For a core-shell model, you can fix or fit the charge separation over the core and shell components and the spring constant.
- Selectively fix, fit, or set to zero the individual buckingham parameters.

## Approximations and Limitations
- Due to the different orders of magnitude between the forces acting on the atoms and the stress tensors of the system, the stress tensors are scaled by $*0.001$ to give higher weighting to the forces. This can be changed in `fitting_code.py`, however, this is not given as a suggested change in the manual.
- `PopOff` has been tested on cubic and orthorhombic structures and the compatibility to other structural types cannot be guaranteed.

# Acknowledgements
L. M. M. acknowledges support from the Faraday Institution (faraday.ac.uk; EP/S003053/1: Grant No. FIRG003). Additional support was received from the Royal Society (UF130329, URF``\``R``\``191006). 
# References
