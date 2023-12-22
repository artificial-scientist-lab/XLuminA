import numpy as np

# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(current_path)

if module_path not in sys.path:
    sys.path.append(module_path)
    
from xlumina.__init__ import um, nm, mm, degrees, radians
from xlumina.wave_optics import *
from xlumina.vectorized_optics import *
from xlumina.optical_elements import *
from xlumina.loss_functions import *
from xlumina.toolbox import space
import jax.numpy as jnp

""" Evaluates the convergence and gradient evaluation times for XLuminA's autodiff optimization """

# System specs: 
sensor_lateral_size =500 # Resolution
wavelength = 650*nm
x_total = 1000*um 
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]

# Light source specs
w0 = (1200*um , 1200*um)
gb = LightSource(x, y, wavelength)
gb.gaussian_beam(w0=w0, E0=1)
gb_gt = ScalarLight(x, y, wavelength)
# Set spiral phase for the ground truth
gb_gt.set_spiral()

# Define the setup
def setup(gb, parameters):
    gb_propagated = gb.RS_propagation(z=25000*mm)
    gb_modulated, _ = SLM(gb, parameters, gb.x.shape[0])
    return gb_modulated

def mse_phase(input_light, target_light):
    return jnp.sum((jnp.angle(input_light.field) - jnp.angle(target_light.field)) ** 2) / sensor_lateral_size**2
def loss(parameters):
    out = setup(gb, parameters)
    loss_val = mse_phase(out, gb_gt)
    return loss_val

# Optimizer for phase mask
import time
import jax
from jax import grad, jit
from jax.example_libraries import optimizers

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Define the update:
@jit
def update(step_index, optimizer_state):
    # define single update step
    parameters = get_params(optimizer_state)
    # Call the loss function and compute the gradients
    computed_loss = loss_value(parameters)
    computed_gradients = grad(loss_value, allow_int=True)(parameters)

    return opt_update(step_index, computed_gradients, optimizer_state), computed_loss, computed_gradients

# Define the loss function and compute its gradients:
loss_value = jit(loss)

# Optimizer settings
STEP_SIZE = 0.1
num_iterations = 50000
n_best = 50
best_loss = 1e10
best_params = None
best_step = 0
num_samples = 5

steps = []
times = []
ratio = []

for i in range(num_samples):
    # Parameters for STED
    parameters = jnp.array([np.random.uniform(-jnp.pi, jnp.pi, (shape, shape))], dtype=jnp.float64)
    
    # Define the optimizer and initialize it
    opt_init, opt_update, get_params = optimizers.adam(STEP_SIZE)
    init_params = parameters
    opt_state = opt_init(init_params)

    print('Starting Optimization', flush=True)
    
    tic = time.perf_counter()
        
    # Optimize in a loop
    for step in range(num_iterations):
        
        opt_state, loss_value, gradients = update(step, opt_state)
        
        if loss_value < best_loss:
            # Best loss value
            best_loss = loss_value
            # Best optimized parameters
            best_params = get_params(opt_state)
            best_step = step
            # print('Best loss value is updated')

        if step % 100 == 0:
            # Stopping criteria: if best loss has not changed every 500 steps
            if step - best_step > n_best:
                print(f'Stopping criterion: no improvement in loss value for {n_best} steps')
                break
                
    steps.append(step)
    times.append(time.perf_counter() - tic)
    ratio.append((time.perf_counter() - tic)/step)

filename = f"xlumina_cpu_eval_{sensor_lateral_size}.npy" 
np.save(filename, {"Time": times, "Step": steps, "t/step": ratio})