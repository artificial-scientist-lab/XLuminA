import jax.numpy as jnp

"""
Define basic units:
    
    um = microns -> we set um = 1.
    mm = nanometers 
    mm = milimeters
    cm = centimeters
    
    radians -> we set radians = 1.
    degrees = 180 / jnp.pi -> When *degrees, the units are transformed to degrees. 
"""

um = 1
nm = 1e-3
mm = 1e3
cm = 1e4

radians = 1
degrees = 180/jnp.pi