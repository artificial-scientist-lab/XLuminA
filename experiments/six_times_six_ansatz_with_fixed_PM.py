# Setting the path for XLuminA modules:
import os
import sys

# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(current_path)

if module_path not in sys.path:
    sys.path.append(module_path)
    
from __init__ import um, nm, cm, mm
from xlumina.vectorized_optics import *
from xlumina.optical_elements import six_times_six_ansatz
from xlumina.loss_functions import vectorized_loss_hybrid
from xlumina.toolbox import space, softmin
import jax.numpy as jnp

"""
Pure topological discovery within 6x6 ansatz for Dorn, Quabis and Leuchs (2004)
"""

# 1. System specs:
sensor_lateral_size = 824  # Resolution
wavelength_1 = 635.0*nm
x_total = 2500*um
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]

# 2. Define the optical functions: two orthogonally polarized beams:
w0 = (1200*um, 1200*um)  
ls1 = PolarizedLightSource(x, y, wavelength_1)
ls1.gaussian_beam(w0=w0, jones_vector=(1, -1))

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

# Linear grating
PM2_1 = jnp.sin(2*jnp.pi * Y/1000) * jnp.pi
PM2_2 = jnp.sin(2*jnp.pi * X/1000) * jnp.pi

# 5. Static parameters - don't change during optimization:
fixed_params = [radius_lens, f_lens, x_out, y_out, PM1_1, PM1_2, PM2_1, PM2_2]

# 6. Define the loss function:
def loss_hybrid_fixed_PM(parameters):
    # Output from hybrid_setup is jnp.array(12, N, N): for 12 detectors
    i_effective = six_times_six_ansatz(ls1, ls1, ls1, ls1, ls1, ls1, ls1, ls1, ls1, ls1, ls1, ls1, parameters, fixed_params, distance_offset = 9.5)
    # Get the minimum value within loss value array of shape (12, 1, 1) 
    loss_val = softmin(vectorized_loss_hybrid(i_effective)) 
    return loss_val