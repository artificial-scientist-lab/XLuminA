import jax.numpy as jnp
from jax import jit, vmap, config
from __init__ import um

# Comment this line if float32 is enough precision for you. 
config.update("jax_enable_x64", True)

""" Loss functions:

    - small_area_STED
    - small_area
    - mean_batch_MSE_Intensity
    - vMSE_Amplitude
    - vMSE_Phase
    - vMSE_Intensity
    - MSE_Amplitude
    - MSE_Phase
    - MSE_Intensity
    
"""

def small_area_STED(sted_i_effective):
    """
    Computes the fraction of intensity comprised inside the area of a mask for STED-like output.

    Parameters:
        sted_i_effective (jnp.array): Effective intensity in the focal plane of the objective lens for STED.
        + epsilon (float): fraction of minimum intensity comprised inside the area.
        
    Return loss function (jnp.array).
    """
    epsilon = 0.5
    I = sted_i_effective / jnp.sum(sted_i_effective)
    mask = jnp.where(I > epsilon*jnp.max(I), 1, 0)
    return jnp.sum(mask) / (jnp.sum(mask * I))

def small_area(focused_field):
    """
    Computes the fraction of intensity comprised inside the area of a mask.

    Parameters:
    focused_field (object): VectorizedLight in the focal plane of an objective lens.
    + epsilon (float): fraction of minimum intensity comprised inside the area.
        
    Return type jnp.array.
    """
    epsilon = 0.5
    I = jnp.abs(focused_field.Ez)**2 / jnp.sum(jnp.abs(focused_field.Ez)**2)
    mask = jnp.where(I > epsilon*jnp.max(I), 1, 0)
    return jnp.sum(mask) / jnp.sum(mask * I)

def mean_batch_MSE_Intensity(optimized, target):
    """
    [Computed for batch optimization in 4f system]. Vectorized version of MSE_Intensity.
    
    Returns the mean value of all the MSE for each (optimized, target) pairs and a jnp.array with MSE values from each pair.
    """
    MSE = vmap(MSE_Intensity, in_axes=(0, 0))(optimized, target)
    return jnp.mean(MSE), MSE

def vMSE_Amplitude(input_light, target_light):
    """
    Computes the Mean Squared Error (in Amplitude) for each electric field component (computed in parallel) Ei (i = x, y, z).

    Parameters:
        input_field (object): VectorizedLight in the focal plane of an objective lens.
        target_field (object): VectorizedLight in the focal plane of an objective lens.

    Returns the MSE in jnp.array [MSEx, MSEy, MSEz]. 
    """
    E_in = jnp.stack([input_light.Ex, input_light.Ey, input_light.Ez], axis=-1)
    E_target = jnp.stack([target_light.Ex, target_light.Ey, target_light.Ez], axis=-1)
    vectorized_MSE = vmap(MSE_Amplitude, in_axes=(2, 2))
    MSE_out = vectorized_MSE(E_in, E_target)
    return MSE_out

def vMSE_Phase(input_light, target_light):
    """
    Computes the Mean Squared Error (in Phase) for each electric field component (computed in parallel) Ei (i = x, y, z).

    Parameters:
        input_field (object): VectorizedLight in the focal plane of an objective lens.
        target_field (object): VectorizedLight in the focal plane of an objective lens.
        
    Returns the MSE in jnp.array [MSEx, MSEy, MSEz]. 
    """
    E_in = jnp.stack([input_light.Ex, input_light.Ey, input_light.Ez], axis=-1)
    E_target = jnp.stack([target_light.Ex, target_light.Ey, target_light.Ez], axis=-1)
    vectorized_MSE = vmap(MSE_Phase, in_axes=(2, 2))
    MSE_out = vectorized_MSE(E_in, E_target)
    return MSE_out

def vMSE_Intensity(input_light, target_light):
    """
    Computes the Mean Squared Error (in Intensity) for each electric field component (computed in parallel) Ei (i = x, y, z).

    Parameters:
        input_field (object): VectorizedLight in the focal plane of an objective lens.
        target_field (object): VectorizedLight in the focal plane of an objective lens.
        
    Returns the MSE in jnp.array [MSEx, MSEy, MSEz]. 
    """
    E_in = jnp.stack([input_light.Ex, input_light.Ey, input_light.Ez], axis=-1)
    E_target = jnp.stack([target_light.Ex, target_light.Ey, target_light.Ez], axis=-1)
    vectorized_MSE = vmap(MSE_Intensity, in_axes=(2, 2))
    MSE_out = vectorized_MSE(E_in, E_target)
    return MSE_out

@jit
def MSE_Amplitude(input_light, target_light):
    """
    Computes the Mean Squared Error (in Amplitude) for a given electric field component Ex, Ey or Ez.

    Parameters:
        input_light (array): If origin light is VectorizedLight, field Ex, Ey or Ez in the detector. For ScalarLight, it corresponds to .field.
        target_light (array): Ground truth - field Ex, Ey or Ez in the detector.
        
    Returns the MSE (jnp.array). 
    """
    num_pix = input_light.shape[0] * input_light.shape[1]
    return jnp.sum((jnp.abs(input_light) - jnp.abs(target_light)) ** 2) / num_pix

@jit
def MSE_Phase(input_light, target_light):
    """
    Computes the Mean Squared Error (in Phase) for a given electric field component Ex, Ey or Ez.

    Parameters:
        input_light (array): If origin light is VectorizedLight, field Ex, Ey or Ez in the detector. For ScalarLight, it corresponds to .field.
        target_light (array): Ground truth - field Ex, Ey or Ez in the detector.
        
    Returns the MSE (jnp.array). 
    """
    num_pix = input_light.shape[0] * input_light.shape[1]
    return jnp.sum((jnp.angle(input_light) - jnp.angle(target_light)) ** 2) / num_pix

@jit
def MSE_Intensity(input_light, target_light):
    """
    Computes the Mean Squared Error (in Intensity) for a given electric field component Ex, Ey or Ez.

    Parameters:
        input_light (array): If origin light is VectorizedLight, field Ex, Ey or Ez in the detector. For ScalarLight, it corresponds to .field.
        target_light (array): Ground truth - field Ex, Ey or Ez in the detector.

    Returns the MSE (jnp.array). 
    """
    num_pix = jnp.shape(input_light)[0] * jnp.shape(input_light)[1]
    return jnp.sum((jnp.abs(input_light)**2 - jnp.abs(target_light)**2) ** 2) / num_pix