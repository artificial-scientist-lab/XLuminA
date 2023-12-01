from xl_optical_table import *
import time
import jax
from jax import grad, jit
import optax
import numpy as np
import jax.numpy as jnp

"""
OPTIMIZER FOR LARGE-SCALE DISCOVERY
"""

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Global variable
shape = jnp.array([sensor_lateral_size, sensor_lateral_size])

# Define the loss function and compute its gradients:
loss_function = jit(loss_large_scale_discovery)

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
num_iterations = 1000000
n_best = 500
best_loss = 1e7
best_params = None
best_step = 0

STEP_SIZE = 0.03
num_samples = 50

# Init parameters:   
alpha_a = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
alpha_b = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
alpha_c = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
alpha_d = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    
eta_a = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
eta_b = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
eta_c = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
eta_d = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    
theta_a = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
theta_b = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
theta_c = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
theta_d = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    
z_a = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0] 
z_b = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0]
z_c = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0] 
z_d = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0]
z_ab = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0] 
z_ba = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0] 
z_cd = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0] 
z_dc = jnp.array([np.random.uniform(0.009, 1, 1)], dtype=jnp.float64)[0]
    
phi_a = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
phi_b = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
phi_c = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
phi_d = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]

init_params = [alpha_a, phi_a, z_a, eta_a, theta_a, alpha_b, phi_b, z_b, eta_b, theta_b, alpha_c, phi_c, z_c, eta_c, theta_c, alpha_d, phi_d, z_d, eta_d, theta_d, z_ab, z_ba, z_cd, z_dc]

# Init optimizer:
optimizer = optax.adam(STEP_SIZE)
    
# Apply fit function:
best_params, best_loss, iteration_steps, loss_list = fit(init_params, optimizer, num_iterations)