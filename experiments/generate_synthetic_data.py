# Setting the path for XLuminA modules:
import os
import sys

# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(current_path)

if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import jax.numpy as jnp
from __init__ import um, nm
from xlumina.wave_optics import *
from xlumina.optical_elements import *
from xlumina.toolbox import space
import h5py

"""
Synthetic data batches generation: 4f system with magnification 2x.
    - input_masks = jnp.array(in1, in2, ...)
    - target_intensity = jnp.array(out1, out2, ...)
"""

# System characteristics:
sensor_lateral_size = 1024  # Pixel resolution
wavelength = 632.8*nm
x_total = 1500*um 
x, y = space(x_total, sensor_lateral_size)
X, Y = jnp.meshgrid(x,y) 

# Define the light source:
w0 = (1200*um , 1200*um)
gb = LightSource(x, y, wavelength)
gb.plane_wave(A=0.5)

def generate_synthetic_circles(gb, num_samples):
    in_circles = []
    out_circles = []
    for i in range(num_samples):
        r1 = jnp.array(np.random.uniform(100, 1000))
        r2 = jnp.array(np.random.uniform(100, 1000))
        
        # Store only the mask (binary)
        in_circle = circular_mask(X, Y, r=(r1, r2))
        in_circles.append(in_circle)
        
        # Magnification is 2, store only the itensity
        out_circle = gb.apply_circular_mask(r=(2*r1, 2*r2))
        out_circles.append(jnp.abs(out_circle.field)**2)
    return jnp.array(in_circles), jnp.array(out_circles)

def generate_synthetic_squares(gb, num_samples):
    in_squares = []
    out_squares = []
    for i in range(num_samples):
        width = jnp.array(np.random.uniform(100, 1000))
        height = jnp.array(np.random.uniform(100, 1000))
        angle = jnp.array(np.random.uniform(0, 2*jnp.pi))
        
        # Binary mask only
        in_square = rectangular_mask(X, Y, center=(0,0), width=width, height=height, angle=angle)
        in_squares.append(in_square)
        
        # Magnification is 2 - we only need intensity
        out_square = gb.apply_rectangular_mask(center=(0,0), width=2*width, height=2*height, angle=-angle)
        out_squares.append(jnp.abs(out_square.field)**2)
        
    return jnp.array(in_squares), jnp.array(out_squares)

def generate_synthetic_annular(gb, num_samples):
    in_annulars = []
    out_annulars = []
    for i in range(num_samples):
        di = jnp.array(np.random.uniform(100, 500))
        do = jnp.array(np.random.uniform(550, 1000))
        
        # Binary mask only: 
        in_annular = annular_aperture(di, do, X, Y)
        in_annulars.append(in_annular)
        
        # Magnification is 2 - we only need intensity
        out_annular = gb.apply_annular_aperture(di=2*di, do=2*do)
        out_annulars.append(jnp.abs(out_annular.field)**2)
        
    return jnp.array(in_annulars), jnp.array(out_annulars)

# Data generation loop:
num_samples = 30

for s in range(50):
    input_circles, target_circles = generate_synthetic_circles(gb, num_samples)
    input_squares, target_squares = generate_synthetic_squares(gb, num_samples)
    input_annular, target_annular = generate_synthetic_annular(gb, num_samples)

    input_fields = jnp.vstack([input_circles, input_squares, input_annular]) 
    target_fields = jnp.vstack([target_circles, target_squares, target_annular]) 

    filename = f"new_training_data_4f/new_synthetic_data_{s+150}.hdf5"
    
    with h5py.File(filename, 'w') as hdf:
        # Create datasets for your data
        hdf.create_dataset("Input fields", data=input_fields)
        hdf.create_dataset("Target fields", data=target_fields)