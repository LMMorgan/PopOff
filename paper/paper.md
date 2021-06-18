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
`PopOff` is a Python package for fitting Coulomb-Buckingham type interatomic potentials for classical potential-based molecular dynamics (MD). This is a modular fitting code, allowing increased control over several important aspects of the potenial. You can choose between fitting a formal charge, partial charge, or fitting a charge scaling factor. You can choose to fit a rigid ion model, a core-shell model, or mixture of both. If choosing to fit a core-shell model, you are able to fix or fit the charge separation over the core and shell components and the spring constant. You can also selectively fix, fit, or set to zero the individual buckingham parameters. `PopOff` currently fits to first principles derived forces and stress tensors obtained from a VASP training set through the `vasprun.xml` output files. The LAMMPS MD code is used as a backend to perform the MD during the fitting process.

The development of sufficiently accurate interatomic potentials, which requires model paramaterization with respect to a set of target observables, for a specific chemistry is quite challenging, especially for complex systems. For example, for layered structures, such as the widely used LiNiO$_2$ based cathode materials, the short-range interactions are overwhelmed by the longer-range Coulombic term. In these cases, the system charges need to be scaled down to increase the influence of the short-range interactions, termed as partial charges, and the influence of polariability needs to be included, usually through using a core-shell model. Variations of the Buckingham potential haa been developed for these systems, some using rigid ion models,[@Lewis_1985; @Ledwaba2020; @Sayle2005; @Dawson2014] and others using core-shell models, [@Hart1998; @Fisher2010; @Lewis_1985; @Ammundsen1999; @Kerisit2014; @He2019; @lee2012atomistic] with a mixture of formal and partial charges being used. Interatomic potentials are traditionally based on mathematical functions which has been parameterised using experimental and/or First Principles derived data. [@jones_1924; @buckingham_classical_1938] There are a limited number of codes available with the explicit purpose or functionality for fitting potentials. Fitting routines from established codes including the General Utility Lattice Program (GULP), [@gale_gulp_1997] Atomicrex, [@Stukowski_2017] dftfit, [@dftfit] and potfit [@wen_kim-compliant_2017] each poses unique functionality, however, none are fully able to produce robust potentials for NMC or LiNiO$_2$ due to different aspects not being considered within the code, such as charge scaling or core-shell models. `PopOff` has been specifically developed for fitting different permutations of the Buckingham potential. Its modular design allows flexible fitting of both rigid ion and core-shell models, and formal and partial charges. 

We are currently using `PopOff` in our own research to develop potentials for cathode and solid-electrolyte materials for classical MD analysis of the Li$^+$/Na$^+$ migration properties. We hope that this open-source resource will support development of interatmic potentials taylored to individual systems, and thus results in better approximations of the atomic interactions.


# Statement of need
Interatomic potentials give an approximate mathmatical descriptions of the interactions between atoms in a given system. These interactions are unique to the species and the surrounding environment, yet, in the wider literature interatomic potentials between specific atom types, for example Li-O, are reused in vastly different systems. This is particularly familiar for the O-O potential derived by Jackson and Catlow in 1985 for UO$_2$ [@.........], which has since been used in the study of many materials, in particular ionic solids. [PROBABLY NEEDS CITATIONS] Although in some cases the species interactions may not vary substantial between different materials, in other cases it does, and to varying degrees. For example, a particular metal-oxygen interaction in a spinel might be considerably different to that in a layered oxdie material, due in part to the change in charge distribution increased influence of polarisability. It is therefore crucial to develop interatomic potentials for individual systems to get the best approximation of the interactions within that environment.

There are a limited number of codes available with the explicit purpose or functionality for fitting potentials. Most of which require some pre-requisit knowledge of coding or specialism, creating a prehibitive barrier for utilising these tools in the wider atomisic modelling research community. These codes are also not able to account for all aspects which may need to be considered for particular materials, such as polarisability. Creating a resource, with minimal pre-requsite requirements to use, and able to account for a larger range of material considerations needed for accurate potentials is therefore key to producing more robust classical MD investigations. 


# Mathematics
``pyscses`` considers simple one-dimensional models of crystallographic interfaces, and calculates equilibrium defect distributions by solving a modified Poisson-Boltzmann equation [@Maier_ProgSolStatChem1995; @DeSouzaEtAl_SolidStateIonics2011], which can be derived by considering the condition that at equilibrium the electrochemical potential for a given defect species is constant [@Maier_IonicsTextbook2005]:
$$
\mu^o_{i,x} + RT\ln \left( \frac{c_{i,x}}{1-c_{i,x}} \right) + z_i F \Phi_x = \mu^o_{i,\infty} + RT\ln \left(\frac{c_{i,\infty}}{1- c_{i,\infty}}\right) + z_i F \Phi_{\infty}.
$$
The thermodynamic driving force for point-defect segregation to or from the interface is described by defect segregation energies, $\left\{\Delta E_\mathrm{seg}^{i,x}\right\}$
$$
\Delta E_\mathrm{seg}^{i,x} = E_\mathrm{f}^{i,x} - E_\mathrm{f}^{i, \infty}
$$
i.e., $\Delta E_\mathrm{seg}^{i,x}$ is the difference in defect formation energy for defect species $i$ at site $x$ compared to a reference site in the &ldquo;bulk&rdquo; of the crystal.

Within the general framework of solving this modified 1D Poisson-Boltzmann equation, ``pyscses`` implements a range of numerical models:

- Continuum (regular grid) and site-explicit (irregular grid) models.
- Periodic and Dirichlet boundary conditions.
- &ldquo;Mott-Schottky&rdquo; and &ldquo;Gouy-Chapman&rdquo; conditions. These are implemented by setting the mobilities of different defect species. In the case of Mott-Schottky conditions, all but one defect species have mobilities of zero.
- Inclusion of &ldquo;lattice-site&rdquo; charges to account for non-defective species in the crystal structure.

Properties that can be calculated include:

- Defect mole fractions.
- Charge density.
- Electrostatic potential.
- Parallel and perpendicular grain boundary resistivities [@HwangEtAl_JElectroceram1999].
- Grain boundary activation energies [@Kim_PhysChemChemPhys2016].

<!---# Typical workflow

The necessary input to model space-charge formation at a grain boundary is a set of defect site positions and segregation energies, projected onto a one-dimensional grid (see Figure 1). For calculations using a &ldquo;continuum&rdquo; (regular) grid, the defect segregation energies and atomic positions are interpolated onto a regular grid.

![(Top) An example crystal structure for a grain boundary in CeO<sub>2</sub>. The $x$ coordinate of each potential defect site (orange spheres) is used to construct a one-dimensional &ldquo;site-explicit&rdquo; grid. Defect segregation energies calculated using e.g. atomistic modelling methods are used to assign segregation energies to every grid point (bottom).](Figures/seg_energies_joss.pdf)

`pyscses` uses these input data to solve the self-consistent modified Poisson-Boltzmann equation. The calculated outputs include the equilibrium electrostatic potential, charge density, and defect mole fractions (site occupancies) across the space charge region (Figure 2).

![Example outputs (electrostatic potentials, charge densities, and site occupancies) for a grain boundary in Gd-doped CeO<sub>2</sub>. The left and right pairs of panels show equivalent results calculated using continuum and site-explicit models.](Figures/continuum_vs_se_joss_MS.pdf)
-->
## Approximations and Limitations
- The modified Poisson-Boltzmann model implemented in ``pyscses`` assumes that defects only interact via point-charge electrostatics and volume exclusion. 
- The resistivitity and activation energy calculations implemented in ``pyscses`` assume that defect mobilities are independent of the local crystal structure.

# Acknowledgements

B. J. M. acknowledges support from the Royal Society (Grant No. UF130329), and from the Faraday Institution (FIRG003).

# References
