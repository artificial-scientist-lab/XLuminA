from diffractio import np, degrees, um, mm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.vector_sources_XY import Vector_source_XY
from diffractio.vector_fields_XY import Vector_field_XY
import time

""" Computes the running times for vectorial version of Rayleigh-Sommerfeld (VRS) and Chirped z-transform (VCZT) algorithms using Diffractio """

# Light source settings
wavelength = .6328 * um
w0 = (1200*um , 1200*um)
x = np.linspace(-15 * mm, 15 * um, 2048)
y = np.linspace(-15 * um, 15 * um, 2048)
x_out = np.linspace(-15 * um, 15 * um, 2048)
y_out = np.linspace(-15 * um, 15 * um, 2048)


ls = Scalar_source_XY(x, y, wavelength, info='Light source')
ls.gauss_beam(r0=(0 * um, 0 * um), w0=w0, z0=(0,0), A=1, theta=0 * degrees, phi=0 * degrees)

vls = Vector_source_XY(x, y, wavelength=wavelength, info='Light source polarization')
vls.constant_polarization(u=ls, v=(1, 0), has_normalization=False, radius=(15*mm, 15*mm)) 

time_VRS_diffractio = []
time_VCZT_diffractio = []

for i in range(101):
    tic = time.perf_counter()
    vls.VRS(z=5*mm, n=1, new_field=False, verbose=False, amplification=(1, 1))
    time_VRS_diffractio.append(time.perf_counter() - tic)
    
for i in range(101):
    tic = time.perf_counter()
    vls.CZT(z=5*mm, xout=x_out, yout=y_out, verbose=False)
    time_VCZT_diffractio.append(time.perf_counter() - tic)
    
filename = f"vectorial_propagation_Diffractio.npy" 
np.save(filename, {"VRS_Diffractio": time_VRS_diffractio, "VCZT_Diffractio": time_VCZT_diffractio})