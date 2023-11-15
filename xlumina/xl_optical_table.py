from __init__ import um, nm, cm
from vectorized_optics import *
from optical_elements import xl_setup
from loss_functions import small_area_STED  
from toolbox import space
from jax import vmap
import jax.numpy as jnp

"""
OPTICAL TABLE FOR LARGE-SCALE DISCOVERY.
"""

# 1. System specs:
sensor_lateral_size = 1024 # Resolution
wavelength_ls1 = .650*um
wavelength_ls2 = .530*um
x_total = 2500*um
x, y = space(x_total, sensor_lateral_size)

# 2. Define the optical functions: two linearly polarized gaussian beams.
w0 = (1200*um, 1200*um) 
ls1 = PolarizedLightSource(x, y, wavelength_ls1)
ls1.gaussian_beam(w0=w0, jones_vector=(1, 1))
ls2 = PolarizedLightSource(x, y, wavelength_ls2)
ls2.gaussian_beam(w0=w0, jones_vector=(1, 1))

# 3. Define the output (High Resolution) detection:
x_out, y_out = jnp.array(space(10*um, 400)) # Pixel size detector: 50 nm = 0.05 um -> 20 um / 400 pix 

# 4. High NA objective lens specs:
NA = 0.9 
radius_lens = 3.6*mm/2 
f_lens = radius_lens / NA

# 5. Static parameters - don't change during optimization:
fixed_params = [radius_lens, f_lens, x_out, y_out]

# 6. Define the loss function:
def loss_large_scale_discovery(parameters):
    i_eff, _, _ = xl_setup(ls1, ls2, parameters, fixed_params)
    loss_val = small_area_STED(i_eff)
    return loss_val