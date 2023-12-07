import numpy as np
from xlumina.__init__ import um, nm, mm, degrees, radians
from xlumina.wave_optics import *
from xlumina.vectorized_optics import *
from xlumina.optical_elements import *
from xlumina.loss_functions import *
from xlumina.toolbox import space
import jax.numpy as jnp

""" Evaluates the convergence and gradient evaluation times for XLuminA's + numerical optimization """

# System specs: 
sensor_lateral_size = 250 # Resolution
wavelength = 650*nm
x_total = 1000*um 
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]

# Light source specs:
w0 = (1200*um , 1200*um)
gb = LightSource(x, y, wavelength)
gb.gaussian_beam(w0=w0, E0=1)
gb_gt = ScalarLight(x, y, wavelength)
# Set spiral phase mask for ground truth
gb_gt.set_spiral()

# Optical setup
def setup(gb, parameters):
    gb_modulated, _ = npSLM(gb.field, parameters, gb.x.shape[0])
    return gb_modulated

def mse_phase(input_light, target_light):
    return np.sum((np.angle(input_light) - np.angle(target_light.field)) ** 2) / sensor_lateral_size**2

def loss(parameters_flat):
    parameters = parameters_flat.reshape(shape, shape)
    out = setup(gb, parameters)
    loss_val = mse_phase(out, gb_gt)
    return loss_val

def phase(phase):
    return np.exp(1j * phase)

def npSLM(input_field, phase_array, shape):
    slm = np.fromfunction(lambda i, j: phase(phase_array[i, j]),
                           (shape, shape), dtype=int)
    light_out = input_field * slm  # Multiplies element-wise
    return light_out, slm

from scipy.optimize import minimize 
import time

results = []
times = []

parameters = np.random.uniform(-np.pi, np.pi, (shape, shape)).flatten()
tic = time.perf_counter()
    
res = minimize(loss, parameters, method='BFGS', options={'disp': True})
 
time_to_conv = time.perf_counter() - tic
times.append(time_to_conv)
results.append(res)

# We save the output (res) to divide the total time by njev from BFGS.