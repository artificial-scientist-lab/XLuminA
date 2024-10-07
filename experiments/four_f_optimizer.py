import os
import sys

# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(current_path)

if module_path not in sys.path:
    sys.path.append(module_path)

from four_f_optical_table import *
from xlumina.toolbox import MultiHDF5DataLoader
import time
import jax
import optax
from jax import jit
import numpy as np
import jax.numpy as jnp

"""
OPTIMIZER FOR THE OPTICAL TELESCOPE (4F-SYSTEM).
"""

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Call the data loader and set batchsize
dataloader = MultiHDF5DataLoader("training_data_4f", batch_size=10)

# Define the loss function and compute its gradients:
loss_function = jit(loss_dualSLM)

# ----------------------------------------------------

def fit(params: optax.Params, optimizer: optax.GradientTransformation, num_iterations) -> optax.Params:
    
    opt_state = optimizer.init(params)
    
    @jit
    def update(params, opt_state, input_fields, target_fields):
        # Define single update step:
        # JIT the loss and compute 
        loss_value, grads = jax.value_and_grad(loss_function, allow_int=True)(params, input_fields, target_fields)
        # Update the state of the optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # Initialize some parameters
    iteration_steps=[]
    loss_list=[]

    # Optimizer settings
    n_best = 500
    best_loss = 1e2
    best_params = None
    best_step = 0
    
    print('Starting Optimization', flush=True)
    
    for step in range(num_iterations):        
        # Load data:
        input_fields, target_fields = next(dataloader)
        params, opt_state, loss_value = update(params, opt_state, input_fields, target_fields)
        
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
num_iterations = 50000
num_samples = 1
# Step size engineering:
STEP_SIZE = 0.01
WEIGHT_DECAY = 0.0001

for i in range(num_samples):
    tic = time.perf_counter()
    
    # Init random parameters
    phase_mask_slm1 = jnp.array([np.random.uniform(0, 1, (shape, shape))], dtype=jnp.float64)[0]
    phase_mask_slm2 = jnp.array([np.random.uniform(0, 1, (shape, shape))], dtype=jnp.float64)[0]
    distance_0 = jnp.array([np.random.uniform(0.027, 1)], dtype=jnp.float64)
    distance_1 = jnp.array([np.random.uniform(0.027, 1)], dtype=jnp.float64) 
    distance_2 = jnp.array([np.random.uniform(0.027, 1)], dtype=jnp.float64)
    init_params = [distance_0, distance_1, distance_2, phase_mask_slm1, phase_mask_slm2]
    
    # Init optimizer:
    optimizer = optax.adamw(STEP_SIZE, weight_decay=WEIGHT_DECAY)
    
    # Apply fit function:
    best_params, best_loss, iteration_steps, loss_list = fit(init_params, optimizer, num_iterations)
    
    print("Time taken to optimize one sample - in seconds", time.perf_counter() - tic)