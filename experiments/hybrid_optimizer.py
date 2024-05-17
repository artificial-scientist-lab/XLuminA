import os
import sys
current_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(current_path)
module_path = os.path.join(dir_path)
if module_path not in sys.path:
    sys.path.append(module_path)
    
import time
import jax
from jax import grad, jit
import optax
import numpy as np
import jax.numpy as jnp
import gc # Garbage collector

# Use this for pure topological discovery:
from xlumina.six_times_six_ansatz_with_fixed_PM import * #<--- use this for 6x6 ansatz
# from xlumina.hybrid_with_fixed_PM import * # <--- use this for 3x3 with fixed masks

# Use this for hybrid optimization:
# from xlumina.hybrid_sharp_optical_table import *  # <--- use this for sharp focus 
# from xlumina.hybrid_sted_optical_table import * # <--- use this for sted

"""
OPTIMIZER - LARGE-SCALE SETUPS
"""

# Print device info (GPU or CPU)
print(jax.devices(), flush=True)

# Global variable
shape = jnp.array([sensor_lateral_size, sensor_lateral_size])

# Define the loss function and compute its gradients:
# loss_function = jit(loss_hybrid_sharp_focus) # <--- use this for sharp focus 
# loss_function = jit(loss_hybrid_sted) # <--- use this for sted
loss_function = jit(loss_hybrid_fixed_PM) # <--- use this for sharp focus with fixed phase masks

# ----------------------------------------------------

def clip_adamw(learning_rate, weight_decay) -> optax.GradientTransformation:
    """
    Custom optimizer - adamw: applies several transformations in sequence
    1) Apply ADAM
    2) Apply weight decay
    """
    return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

def fit(params: optax.Params, optimizer: optax.GradientTransformation, num_iterations) -> optax.Params:
    
    # Init the optimizer with initial parameters
    opt_state = optimizer.init(params)
    
    @jit
    def update(parameters, opt_state):
        # Define single update step:
        loss_value, grads = jax.value_and_grad(loss_function)(parameters)
        
        # Update the state of the optimizer
        updates, state = optimizer.update(grads, opt_state, parameters)

        # Update the parameters
        new_params = optax.apply_updates(parameters, updates)
        
        return new_params, parameters, state, loss_value, updates

    
    # Initialize some parameters
    iteration_steps=[]
    loss_list=[]
    
    n_best = 500
    best_loss = 3*1e2
    best_params = None
    best_step = 0
    
    print('Starting Optimization', flush=True)
    
    for step in range(num_iterations):
        
        params, old_params, opt_state, loss_value, grads = update(params, opt_state)
            
        print(f"Step {step}")
        print(f"Loss {loss_value}")
        iteration_steps.append(step)
        loss_list.append(loss_value)
        
        # Update the `best_loss` value:
        if loss_value < best_loss:
            # Best loss value
            best_loss = loss_value
            # Best optimized parameters
            best_params = old_params
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
num_iterations = 100000
num_samples = 20

for i in range(num_samples):
    
    STEP_SIZE = 0.05  
    WEIGHT_DECAY = 0.0001
    
    gc.collect()
    tic = time.perf_counter()
    
    # Parameters -- know which ones to comment based on the setup you want to optimize:
    # super-SLM phase masks:
    phase1_1 = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    phase1_2 = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    phase2_1 = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    phase2_2 = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    phase3_1 = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    phase3_2 = jnp.array([np.random.uniform(0, 1, shape)], dtype=jnp.float64)[0]
    
    # Wave plate variables:
    eta1 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    theta1 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    eta2 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    theta2 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    eta3 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    theta3 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    eta4 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    theta4 = jnp.array([np.random.uniform(0, 1, 1)], dtype=jnp.float64)[0]
    
    # Distances:
    z1_1 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z1_2 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z2_1 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z2_2 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z3_1 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z3_2 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z4_1 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z4_2 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z4 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z5 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z1 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z2 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z3 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z4 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z5 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z6 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z7 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z8 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z9 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z10 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z11 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    z12 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    
    # Beam splitter ratios
    bs1 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs2 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs3 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs4 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs5 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs6 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs7 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs8 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs9 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs10 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs11 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs12 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs13 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs14 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs15 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs16 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs17 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs18 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs19 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs20 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs21 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs22 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs23 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs24 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs25 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs26 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs27 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs28 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs29 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs30 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs31 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs32 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs33 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs34 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs35 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    bs36 = jnp.array([np.random.uniform(0, 1)], dtype=jnp.float64)
    
    # Set which set of init parameters to use: 
    # REMEMBER TO COMMENT (#) THE VARIABLES YOU DON'T USE! 
    
    # 1. For 3x3 hybrid optimization (topology + optical parameters):
    # init_params = [phase1_1, phase1_2, eta1, theta1, z1_1, z1_2, phase2_1, phase2_2, eta2, theta2, z2_1, z2_2, phase3_1, phase3_2, eta3, theta3, z3_1, z3_2, bs1, bs2, bs3, bs4, bs5, bs6, bs7, bs8, bs9, z4, z5]
    
    # 2. Parameters for pure topological optimization on 3x3 systems with fixed phase masks at random positions:
    # init_params = [z1_1, z1_2, z2_1, z2_2, z3_1, z3_2, z4_1, z4_2, bs1, bs2, bs3, bs4, bs5, bs6, bs7, bs8, bs9, eta1, theta1, eta2, theta2, eta3, theta3, eta4, theta4]

    # 3. Parameters for pure topological optimization on the 6x6 system with fixed phase masks:
    init_params = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12,
                   bs1, bs2, bs3, bs4, bs5, bs6, 
                   bs7, bs8, bs9, bs10, bs11, bs12,
                   bs13, bs14, bs15, bs16, bs17, bs18, 
                   bs19, bs20, bs21, bs22, bs23, bs24,
                   bs25, bs26, bs27, bs28, bs29, bs30,
                   bs31, bs32, bs33, bs34, bs35, bs36, 
                   eta1, theta1, eta2, theta2]
                   
    # Init optimizer:
    optimizer = clip_adamw(STEP_SIZE, WEIGHT_DECAY)

    # Apply fit function:
    best_params, best_loss, iteration_steps, loss_list = fit(init_params, optimizer, num_iterations)