from vsted_optical_table import *
import time
import jax
from jax import grad, jit
import optax
import numpy as np
import jax.numpy as jnp

"""
OPTIMIZER FOR POLARIZATION-BASED STED.
"""

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Global variable
shape = jnp.array([sensor_lateral_size, sensor_lateral_size])

# Define the loss function and compute its gradients:
loss_function = jit(loss_sted)

# ----------------------------------------------------

def fit(params: optax.Params, optimizer: optax.GradientTransformation, num_iterations):
    """
    Optimizer function: defines the update and runs the optimizer loop. 
    
    Parameters:
        params (optax.Params): initial set of parameters.
        optimizer (optax.GradientTransformation): type of optimizer to use.
        num_iterations (int): total number of iterations to perform.
        
    Returns:
        best_params (optax.Params): set of best parameters.
        best_loss (float): best loss achieved.
        iteration_steps (list): list of performed iterations.
        loss_list (list): list of loss values.
    """
    # Init the optimizer
    opt_state = optimizer.init(params)
    
    @jit
    def update(params, opt_state):
        """
        Define single update step
        """
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
    best_loss = 1e2
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
num_iterations = 10000
n_best = 500
best_loss = 1e2
best_params = None
best_step = 0

STEP_SIZE = 0.01
  
# Parameters for STED
init_params = [jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]]

# Define the optimizer:
optimizer = optax.adam(STEP_SIZE)

# Apply fit function:
best_params, best_loss, iteration_steps, loss_list = fit(init_params, optimizer, num_iterations)