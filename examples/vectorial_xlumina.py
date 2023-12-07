# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(current_path)

if module_path not in sys.path:
    sys.path.append(module_path)

from xlumina.__init__ import mm
from xlumina.wave_optics import *
from xlumina.vectorized_optics import *
from xlumina.optical_elements import *
from xlumina.loss_functions import *
from xlumina.toolbox import space
import jax.numpy as jnp

import time

""" Computes the running times for vectorial version of Rayleigh-Sommerfeld (VRS) and Chirped z-transform (VCZT) algorithms using XLuminA """

# Light source settings
resolution = 2048
wavelength = .6328 * um
w0 = (1200*um , 1200*um)
x, y = space(15*mm, resolution)
x_out, y_out = jnp.array(space(15*mm, resolution))

gb_lp = PolarizedLightSource(x, y, wavelength)
gb_lp.gaussian_beam(w0=w0, jones_vector=(1, 0))

# Rayleigh-Sommerfeld:
tic = time.perf_counter() 
gb_propagated = gb_lp.VRS_propagation(z=5*mm)
print("Time taken for 1st VRS propagation - in seconds", time.perf_counter() - tic)

time_VRS_xlumina = []
time_VCZT_xlumina = []

for i in range(101):
    tic = time.perf_counter()
    gb_propagated = gb_lp.VRS_propagation(z=5*mm)
    time_VRS_xlumina.append(time.perf_counter() - tic)
    
# Chirped z-transform:
tic = time.perf_counter() 
gb_propagated = gb_lp.VCZT(z=5*mm, xout=x_out, yout=y_out)
print("Time taken for 1st VCZT propagation - in seconds", time.perf_counter() - tic)

for i in range(101):
    tic = time.perf_counter()
    gb_propagated = gb_lp.VCZT(z=5*mm, xout=x_out, yout=y_out)
    time_VCZT_xlumina.append(time.perf_counter() - tic)
    
filename = f"vectorial_propagation_xlumina_GPU.npy" 
np.save(filename, {"VRS_xlumina": time_VRS_xlumina, "VCZT_xlumina": time_VCZT_xlumina})