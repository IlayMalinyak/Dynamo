---
title: 'Dynamo: A simulation for realistic multimodal stellar observables'
tags:
  - Python
  - astronomy
  - multimodality
  - lightcruve
  - spectra
  - stellar physics
authors:
  - name: Ilay Kamai
    orcid: 0009-0008-5080-496X
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 
  - name: Hagai B. Perets
    orcid: 0000-0002-5004-199X
    affiliation: "1, 3"
  - name: Alex M. Bronstein
    orcid: 0000-0001-9699-8730
    affiliation: "2, 4"
affiliations:
 - name: Physics Department, Technion -- Israel Institute of Technology, Haifa 32000, Israel
   index: 1
 - name: Computer Science Department, Technion -- Israel Institute of Technology, Haifa 32000, Israel
   index: 2
 - name: ACRO, Open University of Israel, R'anana, Israel
   index: 3
 - name: Institute of Science and Technology Austria, Klosterneuburg 3400, Austria
   index: 4
date: 10 February 2026
bibliography: paper.bib

---

# Summary
Stellar surface is affected by various physical processes in a complex and stochastic manner that is still not well understood. Obrsevations of stellar surfaces results in photometric (lightcurve) and spectroscopic measurements that are crucial for the understading   
of the evolution of stars and the complex correlations between physical parameters. With surveys as Kepler, TESS, APOGEE, and LAMOST, which produces large data volumes, we can apply complex statistical and machine learning model to analyze them. However, while we have seperate stellar evolution and stellar surface simulations, we still lack complete simulation that would be able to connect basic physical parameters and observed signals in a modular and simple way and would fit for multimodality machine learning models. 

# Statement of need
`Dynamo` was designed to close the gap between stellar simulations that capture partiall aspects of stellar physics. While 1d stellar evolution codes are very successful (e.g., MESA), they do not account for phenomenon that changes on small timescales and are poorly understood from first principles, like stellar spots. Therefore, their applicability in lightcurve analysis is limited. On the other hand, several implementations exists for stellar surface variablity, but they are usually negelct the broader context of stellar evolution (e.g., temperature, mass and age of the star). In addition, stellar surface simulations are usually focused on single modality - photometry (lightcurves) or spectroscopy, and do not produce combined observables which are crucial for multimodal learning. In the era of big astrophysical datasets, there is a clear need for simulations that connect those seperate efforts into a unified and modular pipeline.  

# State of the field                                                                                         

There are many well known stellar evolution codes - `MESA` [@Paxton2011], `PARSEC` [@Nguyen2022_parsec], and `YREC` [@Demarque2008_yrec] are representative examples. Those codes simulate the evolution of a given star, and can produce evolutionary tracks and physical properties at different times. Since they simulate processes with time scales of the order of $10^6$ years, it is impossible to incorporate phenomenon with time scales of days, months, or several years in those simulations. For the shorter time scales, there are codes that simulate the variability on the surface of a star, mainly due to the existence of spots, in different ways. `starry-processes` [@Luger2021c] uses Gaussian Processes, `Butterpy` [@Claytor2023_butterpy] simulate the evolution of spots and use simple geometrical considerations, and `Starsim` [@Herrero2016_starsim] is a grid-based simulation that uses `PHOENIX` [@Hauschildt1999_Phoenix], a stellar atmosphere code. While `Starsim` is physically solid, it still have some major limitations - first, spots are assumed to be calcuated independently and the simulation does not have an integrated spot evolution module. In addition, it produce only single photometry, rather than multi-instrument photometry and spectroscopy. Lastly, while it accounts for $logg$ and $T_{eff}, other stellar parameters are not taken into account.    


# Software design

`Dynamo` was designed to integrate and improve existing codes, rather than creating new simulation from scratch. As such, we prioritize flexibility and modularity. Another goal of this project is to be able to create large multimodal dataset of stellar observables. This implies efficeint generation of multiple observables, which is not implemented in exisiting works. As such, we were not able to simply extend current simulations and needed to create a new framework. Nevertheless, we tried to use backbone code from exisitg works, when possible. For example, `nbspectra.py` was largely adopted from `Starsim`, and `spots.py` was largely adopted from `Butterpy`, with some modifications. The general architecture follows a modular pipeline where each component can be configured independently through a configuration file (`star.conf`). The central `Star` class combines all the parts to a unified system. The package is built around three core sub-modules:

## Spots Generation

This module, implemented in `spots.py`, is motivated by `Butterpy` [@Claytor2023_butterpy]. The `SpotsGenerator` class creates realistic spot maps by simulating the emergence and evolution of active regions on stellar surfaces. Key features include:

- **Temporal distribution**: Spots emerge uniformly in time over the simulation duration, with configurable activity levels that control the total number of spots.
- **Butterfly diagram**: Spot emergence latitudes can follow a solar-like butterfly pattern, with latitude ranges evolving through activity cycles. Cycle period, cycle overlap, and minimum/maximum emergence latitudes are configurable.
- **Spot evolution**: Each spot has a configurable emergence time scale ($\tau_1$) and decay time scale ($\tau_2$), allowing for realistic temporal evolution of spot coverage.
- **Spatial correlation**: Spots can have correlated emergence positions, controlled by a probability correlation parameter.

## Stellar Atmosphere Interpolation

This module extends the approach of `Starsim` [@Herrero2016_starsim] by enabling full three-dimensional interpolation of `PHOENIX` stellar atmosphere models [@Hauschildt1999_Phoenix] in $T_{\mathrm{eff}}$, $\log g$, and [Fe/H] space. 

## Observables Generation

The observables generation pipeline, primarily in `star.py` and `spectra.py`, transforms the spot maps and stellar atmosphere models into synthetic photometric light curves and spectra. The pipeline consists of several stages:

1. **Grid-based disk integration**: The visible stellar disk is divided into concentric rings of equal-$\mu$ grid elements using the algorithm in `nbspectra.py`. Each grid element receives a flux contribution based on its local $\mu$ value (limb darkening) and surface type (photosphere or spot).

2. **Rotating photosphere**: The `Rotator` class in `rotator.py` computes the time-dependent visibility of spots as the star rotates. Differential rotation is supported, where the rotation rate varies with latitude. At each time step, the fractional coverage of each grid element is calculated.

3. **Multi-instrument photometry**: Multiple photometric instruments (e.g., Kepler, TESS) can be simulated simultaneously with different filter bandpasses and cadences. Filter response curves are interpolated and convolved with the emergent flux.

4. **Spectroscopic observables**: For spectrographs (e.g., LAMOST, APOGEE), the code generates disk-integrated spectra with:
   - Rotational broadening applied via convolution with a rotation kernel
   - Instrumental broadening based on the spectrograph resolution
   - Resampling to the instrument's wavelength grid
   - Optional flux scaling based on stellar radius and distance 

5. **Planetary transits**: If a planet is configured, its orbital motion is computed and the planet disk blocks a fraction of the stellar surface at each time step, creating transit signatures in both photometry and spectroscopy (Rossiter-McLaughlin effect).

Additionally, the `stellar_interpolator.py` module was designed to interpolate stellar parameters (radius, luminosity, rotation period) from evolutionary grids such as YREC or MIST, given mass, metallicity, alpha-enhancement, and age. This module uses the `kiauhoku` [@CClaytor2020_kiauhoku] interfaces and create the initial conditions for the main simulation, enables important links between stellar evolution and stellar surface and ensures that the observables are physically consistent.


# Research impact statement

The impact of `Dynamo` is twofold. First, it provides a powerful tool for simulating realistic stellar observables, which can be used for a wide range of scientific applications, such as testing machine learning models for stellar parameters predictions, solving inverse problems, and multimodal learning. Second, it provides a platform for researchers to explore the relationship between stellar evolution and stellar surface, creating hypotheses that can be tested observationally.

# Figures

![Exaple observables - Kepler and TESS lightcurves and LAMOST and APOGEE spectra](assets/example_plot_1.png)

![Effect of multiple planets on lightcurve and RV](assets\multi_planet_test.png)

# AI usage disclosure
f
Generative AI was used as assistance in this project. The main task used by AI was refactoring, documentations, debuging and tests. First, the core components were implemented manualy and AI was integerated in a later step. Every task done by AI was accompanied with tests to verify the expected result.

# Acknowledgements

We acknowledge the usage of exisiting codes from `Stasim` [@Herrero2016_starsim], `kiauhoku` [@CClaytor2020_kiauhoku], and `Butterpy` [@Claytor2023_butterpy] projects.

# References