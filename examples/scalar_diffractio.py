from diffractio import np, degrees, um, mm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_fields_XY import Scalar_field_XY
import time

""" Computes the running times for scalar version of Rayleigh-Sommerfeld (RS) and Chirped z-transform (CZT) algorithms using Diffractio """

# Light source settings
wavelength = .6328 * um
w0 = (1200*um , 1200*um)
x = np.linspace(-15 * mm, 15 * um, 2048)
y = np.linspace(-15 * um, 15 * um, 2048)
x_out = np.linspace(-15 * um, 15 * um, 2048)
y_out = np.linspace(-15 * um, 15 * um, 2048)

ls = Scalar_source_XY(x, y, wavelength, info='Light source')
ls.gauss_beam(r0=(0 * um, 0 * um), w0=w0, z0=(0,0), A=1, theta=0 * degrees, phi=0 * degrees)


time_RS_diffractio = []
time_CZT_diffractio = []

for i in range(101):
    tic = time.perf_counter()
    ls_propagated = ls.RS(z=5*mm)
    time_RS_diffractio.append(time.perf_counter() - tic)
    
for i in range(101):
    tic = time.perf_counter()
    ls_propagated = ls.CZT(z=5*mm, xout=x_out, yout=y_out, verbose=False)
    time_CZT_diffractio.append(time.perf_counter() - tic)
    
filename = f"Scalar_propagation_Diffractio.npy" 
np.save(filename, {"RS_Diffractio": time_RS_diffractio, "CZT_Diffractio": time_CZT_diffractio})