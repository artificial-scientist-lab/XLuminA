from __init__ import um, nm
from vectorized_optics import *
from optical_elements import sharp_focus
from loss_functions import small_area
from toolbox import space
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

# 2. Define the optical functions: linearly polarized gaussian beam at 45 deg.
w0 = (1200*um, 1200*um)
gb_lp = PolarizedLightSource(x, y, wavelength)
gb_lp.gaussian_beam(w0=w0, jones_vector=(1, 1))

# 3. Define the output (High Resolution) detection:
x_out, y_out = jnp.array(space(10*um, 400)) # Pixel size detector: 50 nm = 0.05 um

# 4. High NA objective lens specs:
NA = 0.9 
radius_lens = 3.6*mm/2 
f_lens = radius_lens / NA

# 5. Static parameters - don't change during optimization:
fixed_params = [radius_lens, f_lens, x_out, y_out]

# 6. Define the loss function:
def loss_sharp_focus(parameters):
    focused_light=sharp_focus(gb_lp, parameters, fixed_params)
    loss_val = small_area(focused_light)
    return loss_val