# Setting the path for XLuminA modules:
import os
import sys
current_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(current_path)
module_path = os.path.join(dir_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from xlumina.__init__ import um, nm, cm
from xlumina.wave_optics import *
from xlumina.optical_elements import SLM
from xlumina.toolbox import space
from jax import vmap
import jax.numpy as jnp

"""
OPTICAL TABLE WITH AN OPTICAL TELESCOPE (4F-SYSTEM).
"""

# 1. System specs:
sensor_lateral_size = 1024  # Resolution
wavelength = 632.8*nm
x_total = 1500*um 
x, y = space(x_total, sensor_lateral_size)
shape = jnp.shape(x)[0]


# 2. Define the optical functions:
def batch_dualSLM_4f(input_mask, x, y, wavelength, parameters):
    """
    [4f system coded exclusively for batch optimization purposes].

    Define an optical table with a 4f system composed by 2 SLMs (to be used with ScalarLight).
    
    Illustrative scheme:
    U(x,y) --> input_mask --> SLM(phase1) --> Propagate: RS(z1) --> SLM(phase2) --> Propagate: RS(z2) --> Detect

    Parameters:
        input_mask (jnp.array): Input mask, comes in the form of an array
        parameters (list): Parameters to pass to the optimizer [z1, z2, z3, phase1 and phase2] for RS propagation and the two SLMs. 
        
    Returns the intensity (jnp.array) after second propagation, and phase masks slm1 and slm2.
    
    + Parameters in the optimizer are (0,1). We need to convert them back [Offset is determined by .get_RS_minimum_z() for the corresponding pixel resolution].
        Convert (0,1) to distance in cm. Conversion factor (offset, 100) -> (offset/100, 1).
        Convert (0,1) to phase (in radians). Conversion factor (0, 2pi) -> (0, 1) 
    """
    global shape
    
    # From get_RS_minimum_z()
    offset = 1.2 
    
    # Apply input mask (comes from vmap)
    input_light.field = input_light.field * input_mask
    
    """ Stage 0: Propagation """
    # Propagate light from mask
    light_stage0, _ = input_light.RS_propagation(z=(jnp.abs(parameters[0]) * 100 + offset)*cm)
    
    """ Stage 0: Modulation """
    # Feed SLM_1 with parameters[2] and apply the mask to the forward beam
    modulated_slm1, slm_1 = SLM(light_stage0, parameters[3] * (2*jnp.pi) - jnp.pi, shape)

    """ Stage 1: Propagation """
    # Propagate the SLM_1 output beam to another distance z
    light_stage1, _ = modulated_slm1.RS_propagation(z=(jnp.abs(parameters[1]) * 100 + offset)*cm)

    """ Stage 1: Modulation """
    # Apply the SLM_2 to the forward beam
    modulated_slm2, slm_2 = SLM(light_stage1, parameters[4] * (2*jnp.pi) - jnp.pi, shape)

    """ Stage 2: Propagation """
    # Propagate the SLM_2 output beam to another distance z
    fw_to_detector, _ = modulated_slm2.RS_propagation(z=(jnp.abs(parameters[2]) * 100 + offset)*cm)

    return jnp.abs(fw_to_detector.field)**2, slm_1, slm_2

def vector_dualSLM_4f_system(input_masks, x, y, wavelength, parameters):
    """
    [Coded exclusively for the batch optimization].
    
    Vectorized (efficient) version of 4f system for batch optimization. 
    
    Parameters:
        input_masks (jnp.array): Array with input masks
        x, y, wavelength (jnp.arrays and float): Light specs to pass to batch_dualSLM_4f.
        parameters (list): Parameters to pass to the optimizer [z1, z2, z3, phase1 and phase2] for RS propagation and the two SLMs. 
    
    Returns vectorized version of detected light (intensity).
    """
    detected_intensity, _, _ = vmap(batch_dualSLM_4f, in_axes=(0, None, None, None, None))(input_masks, x, y, wavelength, parameters)
    return detected_intensity


# 3. Define the loss function for batch optimization.
def loss_dualSLM(parameters, input_masks, target_intensities):
    """
    Loss function for 4f system batch optimization. It computes the MSE between the optimized light and the target field.
    
    Parameters:
        parameters (list): Optimized parameters.
        input_masks (jnp.array): Array with input masks.
        target_intensities (jnp.array): Array with target intensities.
    
    Returns the mean value of the loss computed for all the inputs. 
    """
    global x, y, wavelength
    # Input fields and target fields are arrays with synthetic data. Global variables defined in the optical table script.
    optimized_intensities = vector_dualSLM_4f_system(input_masks, x, y, wavelength, parameters)
    mean_loss, loss_array = mean_batch_MSE_Intensity(optimized_intensities, target_intensities)
    return mean_loss

def mean_batch_MSE_Intensity(optimized, target):
    """
    [Computed for batch optimization in 4f system]. Vectorized version of MSE_Intensity.
    
    Returns the mean value of all the MSE for each (optimized, target) pairs and a jnp.array with MSE values from each pair.
    """
    MSE = vmap(MSE_Intensity, in_axes=(0, 0))(optimized, target)
    return jnp.mean(MSE), MSE

@jit
def MSE_Intensity(input_light, target_light):
    """
    Computes the Mean Squared Error (in Intensity) for a given electric field component Ex, Ey or Ez.

    Parameters:
        input_light (array): intensity: input_light = jnp.abs(input_light.field)**2
        target_light (array): Ground truth - intensity in the detector: target_light = jnp.abs(target_light.field)**2
        
    Returns the MSE (jnp.array). 
    """
    num_pix = jnp.shape(input_light)[0] * jnp.shape(input_light)[1]
    return jnp.sum((input_light - target_light)** 2) / num_pix