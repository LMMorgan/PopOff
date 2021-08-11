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
 - name: The Faraday Institution, Quad One, Harwell Campus, Didcot, OX11 0RA, UK 
   index: 2
date: 2 July 2021
bibliography: paper.bib
---

# Statement of need
Interatomic potentials give an approximate mathematical description of the interactions between atoms in a given system. These interactions are unique to the species and the surrounding environment, yet in the wider literature interatomic potentials between specific atom types, for example Li-O, are reused in vastly different systems. This is particularly familiar for the O-O potential derived by Jackson and Catlow in 1985 for UO$_2$, [@jackson1985trapping] which has since been used in the study of many materials, in particular ionic solids. Although in some cases the species interactions may not vary substantially between different materials, in other cases it does, and to varying degrees. For example, a particular metal-oxygen interaction in a spinel might be considerably different to that in a layered oxide material, due in part to the charge distribution increasing the influence of polarizability. It is therefore crucial to develop interatomic potentials for individual systems to get the best approximation of the interactions within that environment.

There are a limited number of codes available with the explicit purpose or functionality for fitting potentials. Most of which require some pre-requisite knowledge of coding or specialism, creating a prohibitive barrier for utilising these tools in the wider atomistic modelling research community. These codes are also not able to account for all aspects which may need to be considered for particular materials, such as polarizability. Creating a resource, with minimal pre-requisite requirements to use, and able to account for a larger range of material considerations needed for accurate potentials is therefore key to producing more robust classical MD investigations.

# Summary
`PopOff` is a Python package for fitting Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potential. Formal charge, partial charge, or a charge scaling factor can be fitted. A rigid ion model, a core-shell model, or mixture of both can be chosen. If choosing to fit a core-shell model, the charge separation over the core and shell components can be fixed or fitted, as can the spring constant. Individual Buckingham parameters can also be selectively fixed, fittted, or set to zero. `PopOff` currently fits to first principles derived forces and stress tensors obtained from a VASP [@kresse1993ab; @kresse1994ab; @kresse1996efficiency; @kresse1996efficient] training set through the `vasprun.xml` output files. The LAMMPS MD code [@plimpton1995fast] is used as a backend to perform the MD during the fitting process and outputs the forces and stress tensors, produced using a range of potentials from the search space. The output atom forces and system stress tensors are compared and minimised.

The development of sufficiently accurate interatomic potentials, which requires model parameterization with respect to a set of target observables, for a specific chemistry is quite challenging, especially for complex systems. For example, in layered structures, such as the widely used LiNiO$_2$ based cathode materials, the short-range interactions are overwhelmed by the longer-range Coulombic term. In these cases, the system charges need to be scaled down to increase the influence of the short-range interactions, termed as partial charges, and the influence of polarizability needs to be included, usually through using a core-shell model. Variations of the Buckingham potential have been developed for these systems, some using rigid ion models,[@Lewis_1985; @Ledwaba2020; @Sayle2005; @Dawson2014] and others using core-shell models, [@Hart1998; @Fisher2010; @Lewis_1985; @Ammundsen1999; @Kerisit2014; @He2019; @lee2012atomistic] with a mixture of formal and partial charges being used. Interatomic potentials are traditionally based on mathematical functions which have been parameterized using experimental and/or First Principles derived data [@jones_1924; @buckingham_classical_1938]. There are a limited number of codes available with the explicit purpose or functionality for fitting potentials. Fitting routines from established codes including the General Utility Lattice Program (GULP), [@gale_gulp_1997] Atomicrex, [@Stukowski_2017] dftfit, [@dftfit] and potfit [@wen_kim-compliant_2017] each possess unique functionality, however, none are fully able to produce robust potentials for NMC or LiNiO$_2$ due to different aspects not being considered within the code, such as charge scaling or core-shell models. `PopOff` has been specifically developed for fitting different permutations of the Buckingham potential. Its modular design allows flexible fitting of both rigid ion and core-shell models, and formal and partial charges. 

We are currently using `PopOff` in our own research to develop potentials for cathode and solid-electrolyte materials for classical MD analysis of Li$^+$/Na$^+$ migration properties. `PopOff` is also being used by a group at Newcastle University to develop potentials for several halide based solid-electrolyte materials in collaboration with experimental partners. We hope that this open-source resource will support development of interatomic potentials tailored to individual systems, and thus results in better approximations of the atomic interactions.

# Optimiser
`PopOff` uses `scipy.optimize.differential_evolution` as its global optimiser to optimise the function that calculates the mean squared error between the forces and stress tensors within the training set and those calculated with possible classical MD potentials, via:

$$
\chi^{2} = \frac{1}{n} \sum_{i=1}^{n}(Y_{i} - \hat{Y}_{i})^{2}
$$

where $n$ is the number of data points, $Y_i$ is the observed values, and $\hat{Y}_i$ is the predicted values.

Differential evolution is a stochastic population-based method, where at each pass through the population the algorithm mutates each candidate solution by mixing with other candidate solutions to create a trial candidate. There are several strategy methods which can be used to create the trial candidate, with `latinhypercube` used as default.

`PopOff`'s modular design allows the code to fit the following parameters:

- A rigid ion model, a core-shell model, or mixture of both.
- For a core-shell model, the charge separation over the core and shell components can be fixed or fittted as can the spring constant.
- Selectively fix, fit, or set to zero the individual Buckingham parameters.

## Approximations and Limitations
- Due to the different orders of magnitude between the forces acting on the atoms and the stress tensors of the system, the stress tensors are scaled by $*0.001$ to give higher weighting to the forces. This can be changed in `fitting_code.py`, however, this is not given as a suggested change in the manual.
- Care needs to be taken in choosing the cutoff value, which must be less than half minimum cell width to prevent self interactions. This value is set as 10 Angstroms as default and can be changed in the `fitting_code.py`, however, this is not given as a suggested change in the manual.

# Acknowledgements
L. M. M. acknowledges support from the Faraday Institution (faraday.ac.uk; EP/S003053/1: Grant No. FIRG003). Additional support was received from the Royal Society (UF130329, URF``\``R``\``191006). 

# References
