from __init__ import um, nm, cm
from wave_optics import *
from optical_elements import SLM
from loss_functions import mean_batch_MSE_Intensity    
from toolbox import space
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
def batch_dualSLM_4f(input_field, x, y, wavelength, parameters):
    """
    [4f system coded exclusively for batch optimization purposes].

    Define an optical table with a 4f system composed by 2 SLMs (to be used with ScalarLight).
    
    Illustrative scheme:
    U(x,y) --> SLM(phase1) --> Propagate: RS(z1) --> SLM(phase2) --> Propagate: RS(z2) --> Detect

    Parameters:
        input_field (jnp.array): Light to be modulated. Comes in the form of an array. Not ScalarLight. 
        parameters (list): Parameters to pass to the optimizer [z1, z2, z3, phase1 and phase2] for RS propagation and the two SLMs. 
        
    Returns the field (jnp.array) after second propagation, and phase masks slm1 and slm2.
    
    + Parameters in the optimizer are (0,1). We need to convert them back [Offset is determined by .get_RS_minimum_z() for the corresponding pixel resolution].
        Convert (0,1) to distance in cm. Conversion factor (offset, 100) -> (offset/100, 1).
        Convert (0,1) to phase (in radians). Conversion factor (0, 2pi) -> (0, 1) 
    """
    global shape
    
    # From get_RS_minimum_z()
    offset = 1.2 
    
    # Restore input_field as ScalarLight object - comes from vmap (AbstractTracer).
    input_light = ScalarLight(x, y, wavelength)
    input_light.field = jnp.array(input_field)
    
    """ Stage 0: Propagation """
    # Propagate light from mask
    light_stage0, quality_0 = input_light.RS_propagation(z=(jnp.abs(parameters[0]) * 100 + offset)*cm)
    
    """ Stage 0: Modulation """
    # Feed SLM_1 with parameters[2] and apply the mask to the forward beam
    modulated_slm1, slm_1 = SLM(light_stage0, parameters[3] * (2*jnp.pi) - jnp.pi, shape)

     """ Stage 1: Propagation """
    # Propagate the SLM_1 output beam to another distance z
    light_stage1, quality_1 = modulated_slm1.RS_propagation(z=(jnp.abs(parameters[1]) * 100 + offset)*cm)

    """ Stage 1: Modulation """
    # Apply the SLM_2 to the forward beam
    modulated_slm2, slm_2 = SLM(light_stage1, parameters[4] * (2*jnp.pi) - jnp.pi, shape)

    """ Stage 2: Propagation """
    # Propagate the SLM_2 output beam to another distance z
    fw_to_detector, quality_2 = modulated_slm2.RS_propagation(z=(jnp.abs(parameters[2]) * 100 + offset)*cm)

    return fw_to_detector.field, slm_1, slm_2

def vector_dualSLM_4f_system(input_fields, x, y, wavelength, parameters):
    """
    [Coded exclusively for the batch optimization].
    
    Vectorized (efficient) version of 4f system for batch optimization. 
    
    Parameters:
        input_fields (jnp.array): Array with input fields.
        x, y, wavelength (jnp.arrays and float): Light specs to pass to batch_dualSLM_4f.
        parameters (list): Parameters to pass to the optimizer [z1, z2, z3, phase1 and phase2] for RS propagation and the two SLMs. 
    
    Returns vectorized version of detected light.
    """
    detected_light, _, _ = vmap(batch_dualSLM_4f, in_axes=(0, None, None, None, None))(input_fields, x, y, wavelength, parameters)
    return detected_light


# 3. Define the loss function for batch optimization.
def loss_dualSLM(parameters, input_fields, target_fields):
    """
    Loss function for 4f system batch optimization. It computes the MSE between the optimized light and the target field.
    
    Parameters:
        parameters (list): Optimized parameters.
        input_fields (jnp.array): Array with input light fields.
        target_fields (jnp.array): Array with target light fields.
    
    Returns the mean value of the loss computed for all the input fields. 
    """
    global x, y, wavelength
    # Input fields and target fields are arrays with synthetic data. Global variables defined in the optical table script.
    optimized_fields = vector_dualSLM_4f_system(input_fields, x, y, wavelength, parameters)
    mean_loss, loss_array = mean_batch_MSE_Intensity(optimized_fields, target_fields)
    return mean_loss