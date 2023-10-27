# XLuminA

 **XLuminA, a highly-efficient, auto-differentiating discovery framework for super-resolution microscopy**

[**XLuminA: An Auto-differentiating Discovery Framework for Super-Resolution Microscopy**](https://arxiv.org/abs/2310.08408#)\
*Carla Rodríguez, Sören Arlt, Leonhard Möckl and Mario Krenn*

# Features:

XLuminA is equipped with the necessary tools to simulate, in a fast and efficient way, light propagation and interaction in optical setups and running optimizations on optical designs. 

The simulator contains many features:

* Light sources (of any wavelength and power) using both scalar (`LightSource` from `ScalarLight`) or vectorial (`PolarizedLightSource` from `VectorizedLight`) optical fields.
* Phase masks (e.g., spatial light modulators (SLMs), polarizers and general variable retarders (LCDs)).
* Amplitude masks (e.g., circles, triangles and squares)
* Beam splitters
* The light propagation methods available in XLuminA are
  * Fast-Fourier-transform (FFT) based numerical integration of the Rayleigh-Sommerfeld diffraction integral: `RS_propagation()` for `ScalarLight` and `VRS_propagation()` for `VectorizedLight`. 
  * Chirped z-transform: `CZT()` for `ScalarLight` and `VCZT` for `VectorizedLight`. This algorithm is an accelerated version of the Rayleigh-Sommerfeld method, which allows for arbitrary selection and sampling of the region of interest. 
  * Propagation through high NA objective lenses is availale to replicate strong focusing conditions in polarized light: `high_NA_objective_lens` for `VectorizedLight`.

# Overview:

In this section we list the available functions and a brief description:

| `waveoptics.py` |          |         |      
|-----------------|----------|         |
| `ScalarLight`   | `class`  | defines |




Examples of some experiments that can be reproduced are:

* Optical telescope (or 4f-correlator),
* Polarization-based beam shaping as used in [STED (stimulated emission depletion) microscopy](https://opg.optica.org/ol/fulltext.cfm?uri=ol-19-11-780&id=12352), 
* The [sharp focus of a radially polarized light beam](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.233901).


# Basic considerations when using XLuminA:
 
 1. By default, JAX uses `float32` precision. Enable `jax.config.update("jax_enable_x64", True)` at the beginning of the file.

 2. Basic units are microns (um) and radians.
 
 3. **IMPORTANT** - RAYLEIGHT-SOMMERFELD PROPAGATION:
    FFT-based diffraction calculation algorithms can be innacurate depending on the computational window size (sampling).\
    Before propagating light, one should check which is the minimum distance available for the simulation to be accurate.\
    You can use the following functions:

      `get_RS_minimim_z()`, in `ScalarLight` class, and `get_VRS_minimim_z()`, in `VectorizedLight` class.
        
# Discovery of new optical setups 

[...]

# Development

Some functionalities of XLuminA’s optics simulator (e.g., optical propagation algorithms, planar lens or amplitude masks) are inspired in an open-source NumPy-based Python module for diffraction and interferometry simulation, [Diffractio](https://pypi.org/project/diffractio/), although we have rewritten and modified these approaches to combine them with JAX just-in-time (jit) functionality. On top of that, we developed completely new functions (e.g., beam splitters, LCDs or propagation through high NA objective lens with CZT methods, to name a few) which significantly expand the software capabilities. 

### Prerequisites 

To run XLuminA you first need to install [**JAX**](https://jax.readthedocs.io/en/latest/index.html).

### Clone repository

```
git clone https://github.com/artificial-scientist-lab/XLuminA.git
```
