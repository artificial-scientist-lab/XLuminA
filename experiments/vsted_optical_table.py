# Setting the path for XLuminA modules:
import os
import sys
current_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(current_path)
module_path = os.path.join(dir_path)
if module_path not in sys.path:
    sys.path.append(module_path)

# Import modules
from xlumina.__init__ import um, nm, cm, mm
from xlumina.vectorized_optics import *
from xlumina.optical_elements import vSTED
from xlumina.loss_functions import small_area_STED  
from xlumina.toolbox import space
from jax import vmap
import jax.numpy as jnp

"""
OPTICAL TABLE WITH POLARIZATION-BASED STED.
"""

# 1. System specs:
sensor_lateral_size = 2048  # Resolution
sted_wavelength = 650*nm
ex_wavelength = 532*nm
x_total = 2000*um 
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]

# 2. Define the optical functions: two orthogonally polarized beams:
w0 = (1200*um, 1200*um)  
sted_lp = PolarizedLightSource(x, y, sted_wavelength)
sted_lp.gaussian_beam(w0=w0, jones_vector=(1, 0))
ex_lp = PolarizedLightSource(x, y, ex_wavelength)
ex_lp.gaussian_beam(w0=w0, jones_vector=(0, 1))

# 3. Define the output (High Resolution) detection:
x_out, y_out = jnp.array(space(10*um, 400)) # Pixel size detector: 50 nm = 0.05 um -> 20 um / 400 pix 

# 4. High NA objective lens specs:
NA = 0.9 
radius_lens = 3.6*mm/2 
f_lens = radius_lens / NA

# 5. Static parameters - don't change during optimization:
fixed_params = [radius_lens, f_lens, x_out, y_out]

# 6. Define the loss function:
def loss_sted(parameters):
    i_eff, _, _, _, _ = vSTED(ex_lp, sted_lp, parameters, fixed_params)
    loss_val = small_area_STED(i_eff)
    return loss_val