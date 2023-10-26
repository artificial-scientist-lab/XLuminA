# XLuminA

 **XLuminA, a highly-efficient, auto-differentiating discovery framework for super-resolution microscopy**

[**XLuminA: An Auto-differentiating Discovery Framework for Super-Resolution Microscopy**](https://arxiv.org/abs/2310.08408#)\
*Carla Rodríguez, Sören Arlt, Leonhard Möckl and Mario Krenn*



# Discovery of new optical setups 

[...]

# Basic considerations when using XLuminA:
 
 1. By default, JAX uses `float32` precision. Enable `jax.config.update("jax_enable_x64", True)` at the beginning of the file.
 2. Basic units are microns (um) and radians.
 
 3. **IMPORTANT** - RAYLEIGHT-SOMMERFELD PROPAGATION:
    FFT-based diffraction calculation algorithms can be innacurate depending on the computational window size (sampling).\
    Before propagating light, one should check which is the minimum distance available for the simulation to be accurate.\
    You can use the following functions:

      `get_RS_minimim_z()`, in `ScalarLight` class, and `get_VRS_minimim_z()`, in `VectorizedrLight` class.

    For further check on your optical simulation, you can read if the energy is conserved w.r.t. the light source via `is_conserving_energy()` in `Toolbox.py` 
        


# Development

### Prerequisites 

To run XLuminA you first need to install [**JAX**](https://jax.readthedocs.io/en/latest/index.html).

### Clone repository

```
git clone https://github.com/artificial-scientist-lab/XLuminA.git
```
