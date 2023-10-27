# ✨ XLuminA ✨

 **XLuminA, a highly-efficient, auto-differentiating discovery framework for super-resolution microscopy**

[**XLuminA: An Auto-differentiating Discovery Framework for Super-Resolution Microscopy**](https://arxiv.org/abs/2310.08408#)\
*Carla Rodríguez, Sören Arlt, Leonhard Möckl and Mario Krenn*

# 👾 Features:

XLuminA allows for the simulation, in a (*very*) fast and efficient way, of classical light propagation through optics hardware configurations,and enables the optimization and automated discovery of new setup designs.

The simulator contains many features:

✦ Light sources (of any wavelength and power) using both scalar or vectorial optical fields.

✦ Phase masks (e.g., spatial light modulators (SLMs), polarizers and general variable retarders (LCDs)).

✦ Amplitude masks (e.g., circles, triangles and squares).

✦ Beam splitters.

✦ The light propagation methods available in XLuminA are:

  - [Fast-Fourier-transform (FFT) based numerical integration of the Rayleigh-Sommerfeld diffraction integral](https://doi.org/10.1364/AO.45.001102).
     
  - [Chirped z-transform](https://doi.org/10.1038/s41377-020-00362-z). This algorithm is an accelerated version of the Rayleigh-Sommerfeld method, which allows for arbitrary selection and sampling of the region of interest.
    
  - Propagation through [high NA objective lenses](https://doi.org/10.1016/j.optcom.2010.07.030) is availale to replicate strong focusing conditions in polarized light.
        
# 👀 Overview:

In this section we list the available functions in different files and a brief description:

1. In `waveoptics.py`: module for scalar optical fields.
   
   |*Class*|*Functions*|*Description*|
   |---------------|----|-----------|   
   | `ScalarLight`   | | Class for scalar optical fields defined in the XY plane: complex amplitude $U(r) = A(r)*e^{-ikz}$. | 
   |  | `.RS_propagation` | [Rayleigh-Sommerfeld](https://doi.org/10.1364/AO.45.001102) diffraction integral in z-direction (z>0 and z<0). |
   |  | `.get_RS_minimum_z()` | Given a quality factor, determines the minimum (trustworthy) distance for `RS_propagation`.|
   |  | `.CZT` | [Chirped z-transform](https://doi.org/10.1038/s41377-020-00362-z) - efficient diffraction using the Bluestein method.|
   |  | `.draw`  | Plots intensity and phase. | 
   | `LightSource`   | | Class for scalar optical fields defined in the XY plane - defines light source beams. | |
   |  | `.gaussian_beam` | Gaussian beam. |
   |  | `.plane_wave` | Plane wave. |

     
2. In `vectorizedoptics.py`: module for vectorized optical fields.

   |*Class*| *Functions* |*Description*|  
   |---------------|----|-----------|
   | `VectorizedLight`   | | Class for vectorized optical fields defined in the XY plane: $\vec{E} = (E_x, E_y, E_z)$| 
   |  | `.VRS_propagation` | [Vectorial Rayleigh-Sommerfeld](https://iopscience.iop.org/article/10.1088/1612-2011/10/6/065004) diffraction integral in z-direction (z>0 and z<0). |
   |  | `.get_VRS_minimum_z()` | Given a quality factor, determines the minimum (trustworthy) distance for `VRS_propagation`.|
   |  | `.VCZT` | [Vectorized Chirped z-transform](https://doi.org/10.1038/s41377-020-00362-z) - efficient diffraction using the Bluestein method.|
   |  | `.draw`  | Plots intensity, phase and amplitude. | 
   | `PolarizedLightSource`   | | Class for polarized optical fields defined in the XY plane - defines light source beams. | |
   |  | `.gaussian_beam` | Gaussian beam. |
   |  | `.plane_wave` | Plane wave. |


 3. In `opticalelements.py`: shelf with all the optical elements available.
   
    | *Function* |*Description*|  
    |---------------|----|
    | ***Jones matrices*** | - | 
    | `jones_LP` | Jones matrix of a [linear polarizer](https://doi.org/10.1201/b19711)| 
    | `jones_general_retarder` | Jones matrix of a [general retarder](https://www.researchgate.net/publication/235963739_Obtainment_of_the_polarizing_and_retardation_parameters_of_a_non-depolarizing_optical_system_from_the_polar_decomposition_of_its_Mueller_matrix). |
    | `jones_sSLM` | Jones matrix of the *superSLM*. |
    | `jones_LCD` | Jones matrix of liquid crystal display (LCD).|
    | ***Polarization-based devices*** | - | 
    |`sSLM` | *super*-Spatial Light Modulator: adds phase mask (pixel-wise) to $E_x$ and $E_y$ independently. |
    | `LCD` | Liquid crystal device: builds any linear wave-plate. | 
    | `linear_polarizer` | Linear polarizer.|
    | `BS` | Single-side coated dielectric beam splitter.|
    | `uncoated_BS` | Uncoated beam splitter. |
    | `VCZT_objective_lens` | High NA objective lens focusing (only for `VectorizedLight`).|
    | ***Scalar light devices*** | - | 
    | `phase_scalar_SLM` | Phase mask for the spatial light modulator available for scalar fields. |
    | `SLM` | Spatial light modulator: applies a phase mask to incident scalar field. |
    | ***General elements*** | - | 
    | `lens` | Transparent lens of variable size and focal length.|
    | `circular_mask` | Circular mask of variable size. |
    | `triangular_mask` | Triangular mask of variable size and orientation.|
    | `rectangular_mask` | Rectangular mask of variable size and orientation.|
    | `annular_aperture` | Annular aperture of variable size.|
    | `forked_grating` | Forked grating of variable size, orientation, and topological charge. |
    | ***Pre-built optical setups*** | - | 
    | `building_block` | Basic building unit. Consists of a `sSLM`, and `LCD` linked via `VRS_propagation`. |
    | `large_scale_discovery` | Optical table with the general set-up in *Fig.7a* of [our paper](https://arxiv.org/abs/2310.08408#).|
    | `vSTED` | Optical table with the vectorial-based STED setup in *Fig.4a* of [our paper](https://arxiv.org/abs/2310.08408#) .|


4. In `toolbox.py`: file with useful functions. 

   | *Function* |*Description*|  
   |---------------|----|
   | ***Basic operations*** | - | 
   | `space` | Builds the space where light is placed. |
   | `wrap_phase` | Wraps any phase mask into $[-\pi, \pi]$ range.|
   | `is_conserving_energy` | Computes the total intensity from the light source and compares is with the propagated light - [Ref](https://doi.org/10.1117/12.482883).|
   | `delta_kronecker` | Kronecker delta.|
   | `apply_low_pass_filter` | Given a phase mask, applies low pass filter to smooth.|
   | `zernike_basis` | Computes the [Zernike polynomials](https://doi.org/10.1364/OE.17.024269).|
   | `R_zernike` | Computes the radial part R(n,m) of the Zernike polynomial.|
   | `noll_to_nm` | Computes the (n,m) coefficients given the Noll index for Zernike coefficients.|
   | `decompose_zernike`| Decompose any input into Zernike basis up to a given order. |
   | `synthetic_wavefront` | Creates a synthetic wavefront built from the combination of Zernike polynomials.|
   | `draw_sSLM` | Plots the two phase masks of `sSLM`.|
   | `moving_avg` | Compute the moving average of a dataset.|
   | `rotate_mask` | Rotates the (X, Y) frame w.r.t. given point. |
   | `profile` | Determines the profile of a given input without using interpolation.|
   | `spot_size` | Computes the spot size as  $\pi (FWHM_x \cdot FWHM_y) /\lambda^2$. |
   | `compute_fwhm` | Computes FHWM in 2D. |
   
5. In `lossfunctions.py`: file with loss functions.

   | *Function* |*Description*|  
   |---------------|----|
   | `vMSE_Intensity` | Parallel computation of Mean Squared Error (Intensity) for a given electric field component $E_x$, $E_y$ or $E_z$. |
   | `MSE_Intensity` | Mean Squared Error (Intensity) for a given electric field component $E_x$, $E_y$ or $E_z$. |
   | `vMSE_Phase` | Parallel computation of Mean Squared Error (Phase) for a given electric field component $E_x$, $E_y$ or $E_z$. |
   | `MSE_Phase` | Mean Squared Error (Phase) for a given electric field component $E_x$, $E_y$ or $E_z$. |
   | `vMSE_Amplitude` | Parallel computation of Mean Squared Error (Amplitude) for a given electric field component $E_x$, $E_y$ or $E_z$. |
   | `MSE_Amplitude` | Mean Squared Error (Amplitude) for a given electric field component $E_x$, $E_y$ or $E_z$. |
   | `mean_batch_MSE_Intensity` | Batch-based `MSE_Intensity`.|
   | `small_area` | Fraction of intensity comprised inside the area of a mask.|
   | `small_area_STED` | Fraction of intensity comprised inside the area of a mask - STED version.|

# ⚠️ Considerations when using XLuminA:
 
 1. By default, JAX uses `float32` precision. If necessary, enable `jax.config.update("jax_enable_x64", True)` at the beginning of the file.

 2. Basic units are microns (um) and radians. Other units (centimeters, millimeters, nanometers, and degrees) are available at `__init.py__`.
 
 3. **IMPORTANT** - RAYLEIGH-SOMMERFELD PROPAGATION:
    [FFT-based diffraction calculation algorithms](https://doi.org/10.1117/12.482883) can be innacurate depending on the computational window size (sampling).\
    Before propagating light, one should check which is the minimum distance available for the simulation to be accurate.\
    You can use the following functions:

    `get_RS_minimum_z()`, in `ScalarLight` class, and `get_VRS_minimim_z()`, in `VectorizedLight` class.
        
# 📝 Example of usage:

Examples of some experiments that can be reproduced with XLuminA are:

* Optical telescope (or 4f-correlator),
* Polarization-based beam shaping as used in [STED (stimulated emission depletion) microscopy](https://opg.optica.org/ol/fulltext.cfm?uri=ol-19-11-780&id=12352), 
* The [sharp focus of a radially polarized light beam](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.233901).

The code for each of these optical setups is provided in the Jupyter notebook of `examples.ipynb`.

# 🚀 Testing XLuminA's efficiency:

We evaluated our framework by conducting several tests: 

 (1) we compare the running times of the different propagation methods with [Diffractio](https://pypi.org/project/diffractio/) - see [Table 1](https://arxiv.org/abs/2310.08408#).
 ![alt text](miscellaneous/propagation_comparison.pdf)

 (2) we compare the convergence times of SciPy's [BFGS optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs) *vs* JAX's ADAM optimizer when optimizing using XLuminA's optical simulator.
 ![alt text](miscellaneous/convergence_times_comparison.pdf)

The Jupyter notebook used for running these simulations is provided as `test_diffractio_vs_xlumina.ipynb`. 

# 🤖🔎 Discovery of new optical setups: 

With XLuminA we were able to re-discover three foundational optics experiments: 

➤ Optical telescope (or 4f-correlator),

➤ Polarization-based beam shaping as used in [STED (stimulated emission depletion) microscopy](https://opg.optica.org/ol/fulltext.cfm?uri=ol-19-11-780&id=12352), 

➤ The [sharp focus of a radially polarized light beam](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.233901).

The Python files used for the discovery of these optical setups, as detailed in [our paper](https://arxiv.org/abs/2310.08408#), are organized in pairs of `optical_table` and `optimizer` as follows:

| **Experiment name** | 🔬 Optical table | 🤖 Optimizer | 📄 File for data |
|----------------|---------------|-----------|----------|
| ***Optical telescope*** | `four_f_optical_table.py` | `four_f_optimizer.py`| `Generate_synthetic_data.py` |
| ***Polarization-based STED*** | `vsted_optical_table.py` | `vsted_optimizer.py`| N/A |
| ***Sharp focus*** | `sharp_focus_optical_table.py` | `sharp_focus_optimizer.py`| N/A |

★ The large-scale setup functions are defined in `XL_optical_table.py` and `XL_optimizer.py`. 

# 💻 Development

Some functionalities of XLuminA’s optics simulator (e.g., optical propagation algorithms, planar lens or amplitude masks) are inspired in an open-source NumPy-based Python module for diffraction and interferometry simulation, [Diffractio](https://pypi.org/project/diffractio/), although we have rewritten and modified these approaches to combine them with JAX just-in-time (jit) functionality. On top of that, we developed completely new functions (e.g., beam splitters, LCDs or propagation through high NA objective lens with CZT methods, to name a few) which significantly expand the software capabilities. 

### Prerequisites 

To run XLuminA you first need to install [**JAX**](https://jax.readthedocs.io/en/latest/index.html).

*For running the comparison test of the optimizers, you need to install [**SciPy**](https://scipy.org).*

### Clone repository

```
git clone https://github.com/artificial-scientist-lab/XLuminA.git
```
