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

""" Computes the running times for scalar version of Rayleigh-Sommerfeld (RS) and Chirped z-transform (CZT) algorithms using XLuminA """

# Light source settings
resolution = 2048
wavelength = .6328 * um
w0 = (1200*um , 1200*um)
x, y = space(15*mm, resolution)
x_out, y_out = jnp.array(space(15*mm, resolution))

gb = LightSource(x, y, wavelength)
gb.gaussian_beam(w0=w0, E0=1)

# Rayleigh-Sommerfeld:
tic = time.perf_counter() 
ls_propagated = gb.RS_propagation(z=5*mm)
print("Time taken for 1st RS propagation - in seconds", time.perf_counter() - tic)

time_RS_xlumina = []
time_CZT_xlumina = []

for i in range(101):
    tic = time.perf_counter()
    ls_propagated = gb.RS_propagation(z=5*mm)
    t = time.perf_counter() - tic
    time_RS_xlumina.append(t)

    
# Chirped z-transform:
tic = time.perf_counter() 
ls_propagated = gb.CZT(z=5*mm, xout=x_out, yout=y_out)
print("Time taken for 1st CZT propagation - in seconds", time.perf_counter() - tic)

for i in range(101):
    tic = time.perf_counter()
    ls_propagated = gb.CZT(z=5*mm, xout=x_out, yout=y_out)
    t = time.perf_counter() - tic
    time_CZT_xlumina.append(t)
    
filename = f"Scalar_propagation_xlumina_GPU.npy" 
np.save(filename, {"RS_xlumina": time_RS_xlumina, "CZT_xlumina": time_CZT_xlumina})