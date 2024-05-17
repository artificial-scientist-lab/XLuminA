# Setting the path for XLuminA modules:
import os
import sys
current_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(current_path)
module_path = os.path.join(dir_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from xlumina.__init__ import um, nm, cm, mm
from xlumina.vectorized_optics import *
from xlumina.optical_elements import hybrid_setup_sharp_focus
from xlumina.loss_functions import vectorized_loss_hybrid
from xlumina.toolbox import space, softmin
import jax.numpy as jnp

"""
Large-scale setup for Dorn, Quabis and Leuchs (2004) benchmark rediscovery:

3x3 initial setup - light gets detected across 6 detectors. 
"""

# 1. System specs:
sensor_lateral_size = 1024  # Resolution
wavelength = 635*nm
x_total = 2500*um 
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]

# 2. Define the optical functions: two orthogonally polarized beams:
w0 = (1200*um, 1200*um)  
ls1 = PolarizedLightSource(x, y, wavelength)
ls1.gaussian_beam(w0=w0, jones_vector=(1, 1))

# 3. Define the output (High Resolution) detection:
x_out, y_out = jnp.array(space(10*um, 400))

# 4. High NA objective lens specs:
NA = 0.9 
radius_lens = 3.6*mm/2 
f_lens = radius_lens / NA

# 5. Static parameters - don't change during optimization:
fixed_params = [radius_lens, f_lens, x_out, y_out]

# 6. Define the loss function:
@jit
def loss_hybrid_sharp_focus(parameters):
    # Output from hybrid_setup is jnp.array(6, N, N): for 6 detectors
    detected_z_intensities, _ = hybrid_setup_sharp_focus(ls1, ls1, ls1, ls1, ls1, ls1, parameters, fixed_params) 
    
    # Get the minimum value within loss value array of shape (6, 1, 1) 
    loss_val = softmin(vectorized_loss_hybrid(detected_z_intensities))
    
    return loss_val