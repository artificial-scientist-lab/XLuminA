from sharp_focus_optical_table import *
import time
import jax
from jax import grad, jit
import optax
import numpy as np
import jax.numpy as jnp

"""
OPTIMIZER FOR SHARP FOCUS.
"""

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Global variable
shape = jnp.array([sensor_lateral_size, sensor_lateral_size])

# Define the loss function and compute its gradients:
loss_function = jit(loss_sharp_focus)

# ----------------------------------------------------

def fit(params: optax.Params, optimizer: optax.GradientTransformation, num_iterations) -> optax.Params:
    
    opt_state = optimizer.init(params)
    
    @jit
    def update(params, opt_state):
        # Define single update step:
        # JIT the loss and compute 
        loss_value, grads = jax.value_and_grad(loss_function)(params)
        # Update the state of the optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # Initialize some parameters
    iteration_steps=[]
    loss_list=[]
    
    n_best = 500
    best_loss = 1e3
    best_params = None
    best_step = 0
    
    print('Starting Optimization', flush=True)
    
    for step in range(num_iterations):
        params, opt_state, loss_value = update(params, opt_state)
        print(f"Step {step}")
        print(f"Loss {loss_value}")
        iteration_steps.append(step)
        loss_list.append(loss_value)
        
        # Update the `best_loss` value:
        if loss_value < best_loss:
            # Best loss value
            best_loss = loss_value
            # Best optimized parameters
            best_params = params
            best_step = step
            print('Best loss value is updated')

        if step % 100 == 0:
            # Stopping criteria: if best_loss has not changed every 500 steps, stop.
            if step - best_step > n_best:
                print(f'Stopping criterion: no improvement in loss value for {n_best} steps')
                break
    
    print(f'Best loss: {best_loss} at step {best_step}')
    print(f'Best parameters: {best_params}')  
    return best_params, best_loss, iteration_steps, loss_list

# ----------------------------------------------------

# Optimizer settings
num_iterations = 1e6
n_best = 500
best_loss = 1e2
best_params = None
best_step = 0

STEP_SIZE = 0.05
num_samples = 50

tic = time.perf_counter()
    
# Parameters for sharp focus:
alpha_array = jnp.array([np.random.uniform(0,1, shape)], dtype=jnp.float64)[0]
phi_array = jnp.array([np.random.uniform(0,1, shape)], dtype=jnp.float64)[0]
eta = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
theta = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
z1 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0] 
z2 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0] 
    
# Ensure that distances are larger than zmin (get_VRS_minimum_z).
init_params = [alpha_array, phi_array, eta, theta, z1, z2]

# Init optimizer:
optimizer = optax.adam(STEP_SIZE)
    
# Apply fit function:
best_params, best_loss, iteration_steps, loss_list = fit(init_params, optimizer, num_iterations)

print("Time taken to optimize one sample - in seconds", time.perf_counter() - tic)