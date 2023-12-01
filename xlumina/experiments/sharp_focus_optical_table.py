# Setting the path for XLuminA modules:
import os
import sys
current_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(current_path)
module_path = os.path.join(dir_path)
if module_path not in sys.path:
    sys.path.append(module_path)

# Import modules
from xlumina.__init__ import um, nm, mm
from xlumina.vectorized_optics import *
from xlumina.optical_elements import sharp_focus
from xlumina.loss_functions import small_area
from xlumina.toolbox import space
from jax import vmap
import jax.numpy as jnp

"""
OPTICAL TABLE FOR SHARP FOCUS.
"""

# 1. System specs:
sensor_lateral_size = 2048  # Resolution
wavelength = .6328 * um
x_total = 2500 * um
x, y = space(x_total, sensor_lateral_size)

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