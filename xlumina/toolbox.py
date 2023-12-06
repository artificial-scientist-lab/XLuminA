import numpy as np
import jax.numpy as jnp
from math import factorial
from jax import config
import matplotlib.pyplot as plt

# Comment this line if float32 is enough precision for you. 
config.update("jax_enable_x64", True)

""" 
Contains useful functions:

    - space
    - wrap_phase
    - is_conserving_energy
    - delta_kronecker
    - build_LCD_cell
    - draw_sSLM
    - moving_avg
    - rotate_mask
    - nearest
    - extract_profile
    - profile
    - spot size
    - compute_fwhm
    - find_max_min
    - fwhm_2d
    - fwhm_1d
    
"""

def space(x_total, num_pix):
    """
    Define the space of the simulation.

    Parameters:
        x_total (float): Length of half the array (in microns).
        num_pix (float): 1D resolution.
        
    Returns x and y (jnp.array).
    """
    x = np.linspace(-x_total, x_total, num_pix, dtype=np.float64)
    y = np.linspace(-x_total, x_total, num_pix, dtype=np.float64)
    return x, y

def wrap_phase(phase):
    """
    Wraps the input phase into [-pi, pi] range. 

    Parameters:
        phase (jnp.array): Phase to wrap.
        
    Returns the wrapped-phase (jnp.array). 
    """
    return jnp.arctan2(jnp.sin(phase), jnp.cos(phase))

def is_conserving_energy(light_source, propagated_light):
    """
    Computes the total intensity from the light source and compares the propagated light.
    [Ref: J. Li, Z. Fan, Y. Fu, Proc. SPIE 4915, (2002)].
    
    Parameters:
        light_source (object): VectorizedLight light source.
        propagated_light (object): Propagated VectorizedLight. 
    
    Ideally, Itotal_propagated / I_source = 1. 
    Values <1 can happen when light is lost (i.e., light gets outside the computational window).
    
    Returns Itotal_propagated / I_source (jnp.array). 
    """
    if light_source.info == 'Wave optics light' or light_source.info == 'Wave optics light source':
        I_source = jnp.sum(jnp.abs(light_source.field**2))
        I_propagated = jnp.sum(jnp.abs(propagated_light.field**2)) 
    
    else:
        I_source = jnp.sum(jnp.abs(light_source.Ex**2)) + jnp.sum(jnp.abs(light_source.Ey**2)) + jnp.sum(jnp.abs(light_source.Ez**2)) 
        I_propagated = jnp.sum(jnp.abs(propagated_light.Ex**2)) + jnp.sum(jnp.abs(propagated_light.Ey**2)) + jnp.sum(jnp.abs(propagated_light.Ez**2))
    
    return I_propagated / I_source

def delta_kronecker(a, b):
    """
    Computes the Kronecker delta.

    Parameters:
        a (float): Number
        b (float): Number

    Returns (int).
    """
    if a == b:
        return 1
    else:
        return 0
    
def build_LCD_cell(eta, theta, shape):
    """
    Builds the LCD cell: eta and theta are constant across the cell [not pixel-wise modulation!!].

    Parameters:
        eta (float): Phase difference between Ex and Ey (in radians).
        theta (float): Tilt of the fast axis w.r.t. horizontal (in radians).
        shape (float): 1D resolution.
        
    Returns the phase and tilt (jnp.array). 
    """
    # Builds constant eta and theta LCD cell
    eta_array = eta * jnp.ones(shape=(shape, shape))
    theta_array = theta * jnp.ones(shape=(shape, shape))
    return eta_array, theta_array

def draw_sSLM(alpha, phi, extent, extra_title=None, save_file=False, filename=''):
    """
    Plots the phase masks of the sSLM (for VectorizedLight).

    Parameters:
        alpha (jnp.array): Phase mask to be applied to Ex (in radians).
        phi (jnp.array): Phase mask to be applied to Ey (in radians).
        extent (jnp.array): Limits for x and y for plotting purposes.
        extra_title (str): Adds extra info to the plot title.
        save_file (bool): If True, saves the figure.
        filename (str): Name of the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    cmap = 'twilight'

    ax = axes[0]
    im = ax.imshow(alpha, cmap=cmap, extent=extent, origin='lower')
    ax.set_title(f"SLM #1. {extra_title}")
    ax.set_xlabel('$x (\mu m)$')
    ax.set_ylabel('$y (\mu m)$')
    fig.colorbar(im, ax=ax)
    im.set_clim(vmin=-jnp.pi, vmax=jnp.pi)

    ax = axes[1]
    im = ax.imshow(phi, cmap=cmap, extent=extent, origin='lower')
    ax.set_title(f"SLM #2. {extra_title}")
    ax.set_xlabel('$x (\mu m)$')
    ax.set_ylabel('$y (\mu m)$')
    fig.colorbar(im, ax=ax)
    im.set_clim(vmin=-jnp.pi, vmax=jnp.pi)
    
    plt.tight_layout()

    if save_file is True:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        
    plt.show()
    
def moving_avg(window_size, data):
    """
    Compute the moving average of a dataset.
    
    Parameters:
        window_size (int): Number of datapoints to compute the avg.
        data (jnp.array): Data.
    
    Returns moving average (jnp.array).
    """ 
    return jnp.convolve(jnp.array(read_loss_list), jnp.ones(window_size)/window_size, mode='valid')

def rotate_mask(X, Y, angle, origin=None):
    """
    Rotates the (X, Y) frame of a mask w.r.t. origin.

    Parameters:
        origin (float, float): Coordinates w.r.t. which perform the rotation (in microns).
        angle (float): Rotation angle (in radians).

    Returns the rotated meshgrid X, Y (jnp.array).

    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) <<
    """
    if origin is None:
        x0 = (x[-1] + x[0]) / 2
        y0 = (y[-1] + y[0]) / 2
    else:
        x0, y0 = origin

    Xrot = (X - x0) * jnp.cos(angle) + (Y - y0) * jnp.sin(angle)
    Yrot = -(X - x0) * jnp.sin(angle) + (Y - y0) * jnp.cos(angle)
    return Xrot, Yrot

def nearest(array, value):
    """
    Finds the nearest value and its index in an array.
    
    Parameters:
        array (jnp.array): Array to analyze.
        value (float): Number to which determine the position.
    
    Returns index (idx), the value of the array at idx position and the distance. 
    
    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
    """
    idx = (jnp.abs(array - value)).argmin()
    return idx, array[idx], abs(array[idx] - value)

def extract_profile(data_2d, x_points, y_points):
    """
    [From profile] Extract the values along a line defined by x_points and y_points.
    
    Parameters: 
        data_2d (jnp.array): Input data from which extract the profile.
        x_points, y_points (jnp.array): X and Y arrays.
    
    Returns the profile (jnp.array).
    """
    x_indices = jnp.round(x_points).astype(int)
    y_indices = jnp.round(y_points).astype(int)

    x_indices = jnp.clip(x_indices, 0, data_2d.shape[1] - 1)
    y_indices = jnp.clip(y_indices, 0, data_2d.shape[0] - 1)

    profile = [data_2d[y, x] for x, y in zip(x_indices, y_indices)]
    return jnp.array(profile)

def profile(data_2d, x, y, point1='', point2=''):
    """
    Determine profile for a given input without using interpolation.
    
    Parameters:
        data_2d (jnp.array): Input 2D array from which extract the profile.
        point1 (float, float): Initial point.
        point2 (float, float): Final point.
        
    Returns the profile (h and z) of the input (jnp.array).
    """
    x1, y1 = point1
    x2, y2 = point2

    ix1, value, distance = nearest(x, x1)
    ix2, value, distance = nearest(x, x2)
    iy1, value, distance = nearest(y, y1)
    iy2, value, distance = nearest(y, y2)

    # Create a set of x and y points along the line between point1 and point2
    x_points = jnp.linspace(ix1, ix2, int(jnp.hypot(ix2-ix1, iy2-iy1)))
    y_points = jnp.linspace(iy1, iy2, int(jnp.hypot(ix2-ix1, iy2-iy1)))

    h = jnp.linspace(0, jnp.sqrt((y2 - y1)**2 + (x2 - x1)**2), len(x_points))
    h = h - h[-1] / 2

    z_profile = extract_profile(data_2d, x_points, y_points)

    return h, z_profile

def spot_size(fwhm_x, fwhm_y, wavelength):
    """
    Computes the spot size in wavelength**2 units.
    
    Parameters:
        fwhm_x (float): FWHM in x (in microns).
        fwhm_y (float): FWHM in y (in microns).
        wavelength (float): Wavelength (in microns).
    
    Returns the spot size (jnp.array).
    """
    return jnp.pi * (fwhm_x/2) * (fwhm_y/2) / wavelength**2

def compute_fwhm(light, light_specs, field=''):
    """
    Computes FWHM2D (in um).
    
    Parameters:
        light (object): can be a jnp.array if field='' is set to 'Intensity'.
        light_specs (list): list with light specs - measurement plane [x, y].
        field (str): component for which compute FWHM. Can be 'Ex', 'Ey', 'Ez', 'r' (radial), 'rz' (total field) and 'Intensity' if the input is not a field, but an intensity array.
    Returns:
        FWHMx and FWHMy
    
    > Warning: It is not accurate when assymetric beam shapes appear.
    """
    if field == 'Ex':
        intensity = (jnp.abs(light.Ex)) ** 2
    elif field == 'Ey':
        intensity = (jnp.abs(light.Ey)) ** 2
    elif field == 'Ez':
        intensity = (jnp.abs(light.Ez)) ** 2
    elif field == 'r':
        intensity = jnp.abs(light.Ex) ** 2 + (jnp.abs(light.Ey)) ** 2
    elif field == 'rz':
        intensity = (jnp.abs(light.Ex)) ** 2 + (jnp.abs(light.Ey)) ** 2 + (jnp.abs(light.Ez)) ** 2
    elif field == 'Intensity':
        intensity = jnp.abs(light)

    FWHMx, FWHMy = fwhm_2d(light_specs[0], light_specs[1], intensity)

    return FWHMx, FWHMy

def find_max_min(value, x, y, kind =''):
    """
    Find the position of maximum and minimum values within a 2D array.
    
    Parameters:
        value (jnp.array): 2-dimensional array with values. 
        x (jnp.array): x-position array
        y (jnp.array): y-position array
        kind (str): choose whether to detect minimum 'min' or maximum 'max'. 
    
    Returns:
        idx (int, int): indexes of the position of max/min.
        xy (float, float): space position of max/min.
        ext_value (float): max/min value.
        
    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) <<  
    """
    if kind =='min':
        val = jnp.where(value==jnp.min(value))
    if kind =='max':
        val = jnp.where(value==jnp.max(value))
    
    # Extract coordinates into separate arrays:
    coordinates = jnp.array(list(zip(val[1], val[0])))
    coords_0 = coordinates[:, 0]
    coords_1 = coordinates[:, 1]

    # Index array:
    idx = coordinates.astype(int)

    # Array with space positions:
    xy = jnp.stack([x[coords_0], y[coords_1]], axis=1)

    # Array with extreme values:
    ext_value = value[coords_1, coords_0]
    
    return idx, xy, ext_value

def fwhm_2d(x, y, intensity):
    """
    Computes FWHM of an 2-dimensional intensity array.
    
    Parameters:
        x (jnp.array): x-position array.
        y (jnp.array): y-position array.
        intensity (jnp.array): 2-dimensional intensity array.
        
    Returns: 
        fwhm_x and fwhm_y
        
    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
    """
    i_position, _, _ = find_max_min(jnp.transpose(intensity), x, y, kind='max')

    Ix = intensity[:, i_position[0, 1]]
    Iy = intensity[i_position[0, 0], :]

    fwhm_x = fwhm_1d(x, Ix)
    fwhm_y = fwhm_1d(y, Iy)
    
    return fwhm_x, fwhm_y

def fwhm_1d(x, intensity):
    """
    Computes FWHM of 1-dimensional intensity array.
    
    Parameters:
        x (jnp.array): 1D-position array.
        intensity (jnp.array): 1-dimensional intensity array.
    Returns: 
        fwhm in 1 dimension
        
    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
    """
    # Setting-up:
    dx = x[1] - x[0]
    I_max = jnp.max(intensity)
    I_half = I_max * 0.5
    
    # Pixels with I max:
    i_max = jnp.where(intensity == I_max)
    # Compute the pixel location:
    i_max = int(i_max[0][0])

    # Compute the slopes:
    i_left, _, distance_left = nearest(intensity[0:i_max], I_half)
    slope_left = (intensity[i_left + 1] - intensity[i_left]) / dx

    i_right, _, distance_right = nearest(intensity[i_max::], I_half)
    slope_right = (intensity[i_max + i_right] - intensity[i_max + i_right - 1]) / dx

    x_right = (i_right + i_max) * dx - distance_right / slope_right
    x_left = i_left * dx - distance_left / slope_left

    # Compute fwhm:
    fwhm = x_right - x_left
    
    return fwhm