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
from xlumina.optical_elements import hybrid_setup_fixed_slms_fluorophores, hybrid_setup_fixed_slms
from xlumina.loss_functions import vectorized_loss_hybrid
from xlumina.toolbox import space, softmin
import jax.numpy as jnp

"""
Large-scale setup using fixed phase masks in random positions:

3x3 initial setup - light gets detected across 6 detectors. 

This script is valid for rediscovering 

(1) Dorn, Quabis and Leuchs (2004) - use hybrid_setup_fixed_slms() in the loss function,

(2) STED microscopy - use hybrid_setup_fixed_slms_fluorophores() in the loss function.
"""

# 1. System specs:
sensor_lateral_size = 824  # Resolution
wavelength_1 = 632.8*nm
wavelength_2 = 530*nm
x_total = 2500*um
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]

# 2. Define the optical functions: two orthogonally polarized beams:
w0 = (1200*um, 1200*um)  
ls1 = PolarizedLightSource(x, y, wavelength_1)
ls1.gaussian_beam(w0=w0, jones_vector=(1, 1))
ls2 = PolarizedLightSource(x, y, wavelength_2)
ls2.gaussian_beam(w0=w0, jones_vector=(1, 1))

# 3. Define the output (High Resolution) detection:
x_out, y_out = jnp.array(space(10*um, 400)) 
X, Y = jnp.meshgrid(x,y)

# 4. High NA objective lens specs:
NA = 0.9 
radius_lens = 3.6*mm/2 
f_lens = radius_lens / NA

# 4.1 Fixed phase masks:
# Polarization converter in Dorn, Quabis, Leuchs (2004):
pi_half = (jnp.pi - jnp.pi/2) * jnp.ones(shape=(sensor_lateral_size // 2, sensor_lateral_size // 2))
minus_pi_half = - jnp.pi/2 * jnp.ones(shape=(sensor_lateral_size // 2, sensor_lateral_size // 2))
PM1_1 = jnp.concatenate((jnp.concatenate((minus_pi_half, pi_half), axis=1), jnp.concatenate((minus_pi_half, pi_half), axis=1)), axis=0)
PM1_2 = jnp.concatenate((jnp.concatenate((minus_pi_half, minus_pi_half), axis=1), jnp.concatenate((pi_half, pi_half), axis=1)), axis=0)
# Spiral phase (STED microscopy)
PM2 = jnp.arctan2(Y,X)
# Forked grating
PM3 = jnp.cos(2 * PM2 - 2 * jnp.pi * X/1000) * jnp.pi
# Linear grating
PM4_1 = jnp.sin(2*jnp.pi * Y/1000) * jnp.pi
PM4_2 = jnp.sin(2*jnp.pi * X/1000) * jnp.pi

# 5. Static parameters - don't change during optimization:
fixed_params = [radius_lens, f_lens, x_out, y_out, PM1_1, PM1_2, PM2, PM3, PM4_1, PM4_2]

# 6. Define the loss function:
def loss_hybrid_fixed_PM(parameters):
    # Output from hybrid_setup is jnp.array(6, N, N): for 6 detectors
    
    # Use (1) for Dorn, Quabis and Leuchs benchmark / Use (2) for STED microscopy benchmark
    
    # (1):
    # detected_z_intensities, _ = hybrid_setup_fixed_slms(ls1, ls1, ls1, ls1, ls1, ls1, parameters, fixed_params) 
    
    # (2):
    i_effective = hybrid_setup_fixed_slms_fluorophores(ls1, ls2, ls1, ls2, ls1, ls2, parameters, fixed_params)
    
    # Get the minimum value within loss value array of shape (6, 1, 1) 
    loss_val = softmin(vectorized_loss_hybrid(i_effective)) 
    
    return loss_val