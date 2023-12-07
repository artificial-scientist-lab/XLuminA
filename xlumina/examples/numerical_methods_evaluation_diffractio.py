from diffractio import np, degrees, um, mm, nm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_fields_XY import Scalar_field_XY
import time

""" Evaluates the convergence and gradient evaluation times for Diffractio + numerical optimization """

# Light source settings
wavelength = .6328 * um
w0 = (1200*um , 1200*um)
sensor_lateral_size = 10 # Resolution
wavelength = 650*nm
x_total = 1000*um 
x = np.linspace(-x_total , x_total , sensor_lateral_size)
y = np.linspace(-x_total , x_total , sensor_lateral_size)
gb = Scalar_source_XY(x, y, wavelength, info='Light source')
gb.gauss_beam(r0=(0 * um, 0 * um), w0=w0, z0=(0,0), A=1, theta=0 * degrees, phi=0 * degrees)

# Spiral phase for ground truth
gb_gt = Scalar_field_XY(x, y, wavelength)
phase_mask = np.arctan2(gb.Y,gb.X)
gb_gt.u = gb.u * np.exp(1j * phase_mask)

# Optical setup
def setup(gb, parameters):
    gb_modulated, _ = npSLM(gb.u, parameters, gb.x.shape[0])
    return gb_modulated

def mse_phase(input_light, target_light):
    return np.sum((np.angle(input_light) - np.angle(target_light.u)) ** 2) / sensor_lateral_size**2

def loss(parameters_flat):
    parameters = parameters_flat.reshape(sensor_lateral_size,sensor_lateral_size)
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

parameters = np.random.uniform(-np.pi, np.pi, (sensor_lateral_size, sensor_lateral_size)).flatten()
tic = time.perf_counter()
    
res = minimize(loss, parameters, method='BFGS', options={'disp': True})
 
time_to_conv = time.perf_counter() - tic
times.append(time_to_conv)
results.append(res)

# We save the output (res) to divide the total time by 'njev' from BFGS.