# Setting the path for XLuminA modules:
import os
import sys
current_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(current_path)
module_path = os.path.join(dir_path)
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import jax.numpy as jnp
from __init__ import um, nm, mm, degrees, radians
from xlumina.wave_optics import *
from xlumina.toolbox import space

"""
Synthetic data batches generation: 4f system with magnification 2x.
    - input_fields = jnp.array(in1, in2, ...)
    - target_fields = jnp.array(out1, out2, ...)
"""

# System characteristics:
sensor_lateral_size = 1024  # Pixel resolution
wavelength = 632.8*nm
x_total = 1500*um 
x, y = space(x_total, sensor_lateral_size)

# Define the light source:
w0 = (1200*um , 1200*um)
gb = LightSource(x, y, wavelength)
gb.gaussian_beam(w0=w0, E0=1)

# Data generation functions:
def generate_synthetic_circles(gb, num_samples):
    in_circles = []
    out_circles = []
    for i in range(num_samples):
        r1 = jnp.array(np.random.uniform(100, 1000))
        r2 = jnp.array(np.random.uniform(100, 1000))
        
        in_circle = gb.apply_circular_mask(r=(r1, r2))
        in_circles.append(in_circle.field)
        # Magnification is 2x
        out_circle = gb.apply_circular_mask(r=(2*r1, 2*r2))
        out_circles.append(out_circle.field)
    return jnp.array(in_circles), jnp.array(out_circles)

def generate_synthetic_squares(gb, num_samples):
    in_squares = []
    out_squares = []
    for i in range(num_samples):
        width = jnp.array(np.random.uniform(100, 1000))
        height = jnp.array(np.random.uniform(100, 1000))
        angle = jnp.array(np.random.uniform(0, 2*jnp.pi))
        
        in_square = gb.apply_rectangular_mask(center=(0,0), width=width, height=height, angle=angle)
        in_squares.append(in_square.field)
        # Magnification is 2x
        out_square = gb.apply_rectangular_mask(center=(0,0), width=2*width, height=2*height, angle=-angle)
        out_squares.append(out_square.field)
    return jnp.array(in_squares), jnp.array(out_squares)

def generate_synthetic_annular(gb, num_samples):
    in_annulars = []
    out_annulars = []
    for i in range(num_samples):
        di = jnp.array(np.random.uniform(100, 500))
        do = jnp.array(np.random.uniform(550, 1000))
        
        in_annular = gb.apply_annular_aperture(di=di, do=do)
        in_annulars.append(in_annular.field)
        # Magnification is 2x
        out_annular = gb.apply_annular_aperture(di=2*di, do=2*do)
        out_annulars.append(out_annular.field)
    return jnp.array(in_annulars), jnp.array(out_annulars)

# Data generation loop:
num_samples = 30

for s in range(50):
    input_circles, target_circles = generate_synthetic_circles(gb, num_samples)
    input_squares, target_squares = generate_synthetic_squares(gb, num_samples)
    input_annular, target_annular = generate_synthetic_annular(gb, num_samples)

    input_fields = jnp.vstack([input_circles, input_squares, input_annular]) 
    target_fields = jnp.vstack([target_circles, target_squares, target_annular]) 

    filename = f"training_data_4f/synthetic_data_{s}.npy"
    np.save(filename, {"Input fields": input_fields, "Target fields": target_fields})