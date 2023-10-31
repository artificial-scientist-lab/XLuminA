from vsted_optical_table import *
import time
import jax
from jax import grad, jit
from jax.example_libraries import optimizers
import numpy as np
import jax.numpy as jnp

"""
OPTIMIZER FOR POLARIZATION-BASED STED.
"""

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Global variable
shape = jnp.array([sensor_lateral_size, sensor_lateral_size])

# Define the update:
@jit
def update(step_index, optimizer_state):
    # define single update step
    parameters = get_params(optimizer_state)
    # Call the loss function and compute the gradients
    computed_loss = loss_value(parameters)
    computed_gradients = grad(loss_value, allow_int=True)(parameters)

    return opt_update(step_index, computed_gradients, optimizer_state), computed_loss, computed_gradients

# JIT the loss function:
loss_value = jit(loss_sted)

# Optimizer settings
STEP_SIZE = 0.01
num_iterations = 50000
n_best = 500
best_loss = 1e2
best_params = None
best_step = 0

# Init random parameters:
phase_1 = jnp.array([np.random.uniform(-jnp.pi, jnp.pi, shape)], dtype=jnp.float64)[0]
init_params = [phase_1]

# Define the optimizer and initialize it
opt_init, opt_update, get_params = optimizers.adam(STEP_SIZE)
opt_state = opt_init(init_params)
           
# Optimize in a loop:
print('Starting Optimization', flush=True)    
tic = time.perf_counter()

for step in range(num_iterations):
    
    # Perform an update step:
    opt_state, loss_value, gradients = update(step, opt_state)
    
    # Update the `best_loss` value:
    if loss_value < best_loss:
        # Best loss value
        best_loss = loss_value
        # Best optimized parameters
        best_params = get_params(opt_state)
        best_step = step
        print('Best loss value is updated')

    if step % 500 == 0:
        # Stopping criteria: if best_loss has not changed every 500 steps, stop.
        if step - best_step > n_best:
            print(f'Stopping criterion: no improvement in loss value for {n_best} steps')
            break

print(f'Best loss: {best_loss} at step {best_step}')
print(f'Best parameters: {best_params}')
print("Time taken to optimize - in seconds", time.perf_counter() - tic)