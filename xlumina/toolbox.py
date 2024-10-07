import numpy as np
import jax.numpy as jnp
import h5py
import random
from PIL import Image
from jax import config, jit, nn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Set this to False if f64 is enough precision for you.
enable_float64 = True
if enable_float64:
    config.update("jax_enable_x64", True)
    d_type = np.float64

""" 
Contains useful functions:

    - space
    - wrap_phase
    - is_conserving_energy
    - softmin
    - delta_kronecker
    - build_LCD_cell
    - draw_sSLM
NEW:- draw_sSLM_amplitude
    - moving_avg
NEW:- image_to_binary_mask
    - rotate_mask
    - nearest
    - extract_profile
    - profile
NEW:- gaussian
NEW:- gaussian_2d
NEW:- lorentzian
NEW:- lorentzian_2d
NEW:- fwhm_1d_fit
NEW:- fwhm_2d_fit
    - spot size
NEW:- compute_fwhm 
    - find_max_min 
    - fwhm_1d (no fitting)
    - fwhm_2d (no fitting)
NEW:> Functions to process datasets (hdf5 files):
        CLASS MultiHDF5DataLoader

"""

def space(x_total, num_pix):
    """
    Define the space of the simulation.

    Parameters:
        x_total (float): Length of half the array (in microns).
        num_pix (float): 1D resolution.
        
    Returns x and y (jnp.array).
    """
    x = np.linspace(-x_total, x_total, num_pix, dtype=d_type)
    y = np.linspace(-x_total, x_total, num_pix, dtype=d_type)
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

@jit
def softmin(args, beta=90):
    """
    Differentiable version for min() function.
    """
    return - nn.logsumexp(-beta * args) / beta


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

def draw_sSLM_amplitude(A1, A2, extent, extra_title=None, save_file=False, filename=''):
    """
    Plots the amplitude masks of the sSLM (for VectorizedLight).

    Parameters:
        A1 (jnp.array): Amp mask to be applied to Ex (AU).
        A2 (jnp.array): Amp mask to be applied to Ey (AU).
        extent (jnp.array): Limits for x and y for plotting purposes.
        extra_title (str): Adds extra info to the plot title.
        save_file (bool): If True, saves the figure.
        filename (str): Name of the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    cmap = 'Greys_r'

    ax = axes[0]
    im = ax.imshow(A1, cmap=cmap, extent=extent, origin='lower')
    ax.set_title(f"SLM #1: amplitude mask {extra_title}")
    ax.set_xlabel('$x (\mu m)$')
    ax.set_ylabel('$y (\mu m)$')
    fig.colorbar(im, ax=ax)
    im.set_clim(vmin=0, vmax=1)

    ax = axes[1]
    im = ax.imshow(A2, cmap=cmap, extent=extent, origin='lower')
    ax.set_title(f"SLM #2: amplitude mask {extra_title}")
    ax.set_xlabel('$x (\mu m)$')
    ax.set_ylabel('$y (\mu m)$')
    fig.colorbar(im, ax=ax)
    im.set_clim(vmin=0, vmax=1)
    
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
    return jnp.convolve(jnp.array(data), jnp.ones(window_size)/window_size, mode='valid')

def image_to_binary_mask(filename, x, y, mirror = 'vertical', normalize=True, invert=False, threshold=0.5):
    """
    Converts image > binary mask (given a threshold)

    Parameters:
        filename (str): os path to the image file
        x, y (jnp.array): corresponds to the space dimensions where the mask is placed
        mirror (str): direction to mirror the image. Can be 'horizontal', 'vertical' or 'both'. Default is 'vertical'
        normalize (bool): if True, normalizes the image
        invert (bool): if True, inverts the binarization
        threshold (float): pix value threshold for binarization (0 to 1)

    Returns:
        Binary mask (jnp.array)
    """
    with Image.open(filename) as img:
        img_gray = img.convert('L')
    
    if mirror == 'horizontal':
        img_gray = img_gray.transpose(Image.FLIP_LEFT_RIGHT)
    if mirror == 'vertical':
        img_gray = img_gray.transpose(Image.FLIP_TOP_BOTTOM)
    if mirror == 'both':
        img_gray = img_gray.transpose(Image.FLIP_LEFT_RIGHT)
        img_gray = img_gray.transpose(Image.FLIP_TOP_BOTTOM)

    size_x, size_y = jnp.size(x), jnp.size(y)
    img_gray = img_gray.resize((size_y, size_x))

    img_array = jnp.array(img_gray) # convert into jax array

    if normalize:
        img_array = (img_array - jnp.min(img_array)) / (jnp.max(img_array) - jnp.min(img_array))

    if invert:
        img_array = jnp.max(img_array) - img_array

    binary_mask = (img_array > threshold).astype(jnp.uint8)

    return binary_mask

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
        x0 = (X[-1] + X[0]) / 2
        y0 = (Y[-1] + Y[0]) / 2
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

def gaussian(x, amplitude, mean, std_dev):
    """ 
    [In fwhm_1d_fit] 
    Returns 1D Gaussian (Normal distribution) for FWHM calculation

    Parameters:
        x (jnp.array): 1D-position array
        mean_x (float): location of the peak of the distribution in X or Y
        stdev_x (float): standard deviation in X or Y
    """
    return amplitude * jnp.exp(-((x - mean) / std_dev)**2 / 2)

def gaussian_2d(xy, amplitude, mean_x, mean_y, stdev_x, stdev_y):
    """ 
    [In fwhm_2d_fit] 
    Returns 2D Gaussian (Normal distribution) for FWHM 2D calculation

    Parameters:
        xy (tuple): contains two 1D arrays, X and Y, which are the meshgrid of x and y coordinates
        mean_x (float): location of the peak of the distribution in X
        mean_y (float): location of the peak of the distribution in Y
        stdev_x (float): standard deviation in X
        stdev_y (float): standard deviation in Y. 
    """
    X, Y = xy
    return amplitude * jnp.exp(-((X - mean_x)**2 / (2 * stdev_x**2) + (Y - mean_y)**2 / (2 * stdev_y**2)))

def lorentzian(x, x0, gamma):
    """
    [In fwhm_1d_fit]
    Returns Lorentzian -- pathological distribution (expected value and variance are undefined)
    Parameters: 
        x (jnp.array): 1D-position array
        x0 (float): location of the peak of the distribution
        gamma (float): scale parameter. Specifies FWHM = 2 * gamma. 
    """
    return (1/jnp.pi) * (gamma / (x-x0)**2 + gamma**2)

def lorentzian_2d(xy, amplitude, x0, y0, gamma_x, gamma_y):
    """ 
    [In fwhm_2d_fit] 
    Returns 2D Lorentzian -- pathological distribution (expected value and variance are undefined)
    
    Parameters: 
        xy (tuple): contains two 1D arrays, X and Y, which are the meshgrid of x and y coordinates
        amplitude (float): amplitude of the peak
        x0 (float): location of the peak of the distribution in X
        y0 (float): location of the peak of the distribution in Y
        gamma_x (float): scale parameter in X.
        gamma_y (float): scale parameter in Y. 
    """
    X, Y = xy
    return amplitude / (1 + ((X - x0) / gamma_x)**2 + ((Y - y0) / gamma_y)**2)

def fwhm_1d_fit(x, intensity, fit = 'gaussian'):
    """
    Compute FWHM of a 1D-array using fit function (gaussian or lorentzian) 
    
    Parameters:
        x (jnp.array): 1D-position array.
        intensity (jnp.array): 1-dimensional intensity array.
        fit (str): can be 'gaussian' or 'lorentzian'
    
    Returns:
        popt (amplitude_fit, mean_fit, stddev_fit), 
        fwhm (float, in um) 
        and r_squared (float, r-squared metric of the fit).
    """
    if fit == 'lorentzian':
        # initial guess (p0) for curve_fit
        x0_guess = jnp.max(intensity)
        gamma_guess = 0.5
        # lorentizan fit -- call scipy.curve_fit
        popt, _ = curve_fit(lorentzian, np.array(x), np.array(intensity), p0=[x0_guess, gamma_guess], maxfev=100000)
        _, gamma_fit = popt
        
        fwhm = 2 * gamma_fit 

        # from https://en.wikipedia.org/wiki/Coefficient_of_determination 
        # r^2 = 1 - SSres / SStot
        # here we compute residual sum of squares (SSres) and total sum of squares (SStot)
        ss_res = jnp.sum((intensity - lorentzian(x, *popt))**2)

    if fit == 'gaussian':
        # initial guess (p0) for curve_fit
        a_guess = jnp.max(intensity)
        mean_guess = x[jnp.argmax(intensity)]
        std_dev_guess = jnp.sqrt(jnp.sum((x - mean_guess)**2 * intensity) / jnp.sum(intensity))

        # gaussian fit -- call scipy.curve_fit
        popt, _ = curve_fit(gaussian, np.array(x), np.array(intensity), p0=[a_guess, mean_guess, std_dev_guess], maxfev=100000)
        _, _, stddev_fit = popt

        # FWHM normal distribution = 2*sqrt(2ln2)*sigma
        fwhm = 2 * jnp.sqrt(2 * jnp.log(2)) * stddev_fit
        
        # from https://en.wikipedia.org/wiki/Coefficient_of_determination 
        # r^2 = 1 - SSres / SStot
        # here we compute residual sum of squares (SSres) and total sum of squares (SStot)
        ss_res = jnp.sum((intensity - gaussian(x, *popt))**2)
    
    else:
        raise ValueError("fit must be either 'gaussian' or 'lorentzian'")
    
    ss_tot = jnp.sum((intensity - jnp.mean(intensity))**2)
    r_squared = 1 - (ss_res / ss_tot)
     
    return popt, fwhm, r_squared

def fwhm_2d_fit(x, y, intensity, fit = 'gaussian'):
    """
    Computes FWHM of an 2-dimensional using fit function (gaussian or lorentzian) 
    
    Parameters:
        x (jnp.array): 1D-position array.
        y (jnp.array): 1D-position array.
        intensity (jnp.array): 2-dimensional intensity array.
        fit (str): can be 'gaussian' or 'lorentzian'
    
    Returns:
        popt (amplitude_fit, mean_fit, stddev_fit), 
        FWHM_2D = (fwhm_x, fwhm_y) (tuple, in um) 
        and r_squared (float, r-squared metric of the fit).
    """
    X, Y = jnp.meshgrid(x, y)
    xy = jnp.vstack((X.ravel(), Y.ravel())) # vertical stack of ravel arrays for scipy's curve_fit (accepts indep. var. as a single arg)

    if fit == 'lorentzian':
        amplitude_guess = jnp.max(intensity)
        # get x[idxx, idxy] and y[idxx', idxy'] where max intensity 
        mean_x_guess, mean_y_guess = (
            x[jnp.unravel_index(jnp.argmax(intensity), intensity.shape)[1]], 
            y[jnp.unravel_index(jnp.argmax(intensity), intensity.shape)[0]]
        )
        gamma_x_guess, gamma_y_guess = 0.5, 0.5

        popt, _ = curve_fit(lorentzian_2d, xy, intensity.ravel(), 
                            p0=[amplitude_guess, mean_x_guess, mean_y_guess, gamma_x_guess, gamma_y_guess], 
                            maxfev=100000)
        
        _, _, _, gamma_x_fit, gamma_y_fit = popt
        FWHM2D = 2 * gamma_x_fit, 2 * gamma_y_fit

        # from https://en.wikipedia.org/wiki/Coefficient_of_determination 
        # r^2 = 1 - SSres / SStot
        # here we compute residual sum of squares (SSres) and total sum of squares (SStot)
        ss_res = jnp.sum((intensity - lorentzian_2d(xy, *popt).reshape(intensity.shape))**2)

    if fit == 'gaussian':
        # initial guess for curve_fit
        amplitude_guess = jnp.max(intensity)
        # get x[idxx, idxy] and y[idxx', idxy'] where max intensity 
        x0_guess, y0_guess = (
            x[jnp.unravel_index(jnp.argmax(intensity), intensity.shape)[1]], 
            y[jnp.unravel_index(jnp.argmax(intensity), intensity.shape)[0]]
        )
        sigma_x_guess = jnp.sqrt(jnp.sum((X - x0_guess)**2 * intensity) / jnp.sum(intensity))
        sigma_y_guess = jnp.sqrt(jnp.sum((Y - y0_guess)**2 * intensity) / jnp.sum(intensity))

        popt, _ = curve_fit(gaussian_2d, xy, intensity.ravel(), 
                            p0=[amplitude_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess], 
                            maxfev=100000)
        
        _, _, _, sigma_x_fit, sigma_y_fit = popt
        FWHM2D = 2 * jnp.sqrt(2 * jnp.log(2)) * sigma_x_fit, 2 * jnp.sqrt(2 * jnp.log(2)) * sigma_y_fit

        # compute r-squared
        # from https://en.wikipedia.org/wiki/Coefficient_of_determination 
        # r^2 = 1 - SSres / SStot
        # here we compute residual sum of squares (SSres) and total sum of squares (SStot)
        ss_res = jnp.sum((intensity - gaussian_2d(xy, *popt).reshape(intensity.shape))**2)
    
    else:
        raise ValueError("fit must be either 'gaussian' or 'lorentzian'")

    ss_tot = jnp.sum((intensity - jnp.mean(intensity))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return popt, FWHM2D, r_squared

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

def compute_fwhm(light, light_specs, field='', fit = 'gaussian', dimension = '2D', pix_slice = None):
    """
    Computes FWHM in 1D or 2D (in um).
    
    Parameters:
        light (object): can be a jnp.array if field='' is set to 'Intensity'.
        light_specs (list): list with light specs - measurement plane [x, y].
        field (str): component for which compute FWHM. Can be 'Ex', 'Ey', 'Ez', 'r' (radial), 'rz' (total field) and 'Intensity' if the input is not a field, but an intensity array.
        dimension (str): can be '1D' or '2D'.
        pix_slice (int): pix number in which to perform a slice for 1D calculation. E.g., in the center: pix_slice = resolution // 2 
    
    Returns:
        popt, fwhm, r_squared;  where fwhm = FWHM_1D or FWHM_2D (tuple)
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

    if '1D' in dimension:
        if dimension == '1D_x':
            intensity = intensity[:, pix_slice] # Slice intensity array in pix_slice
            axis = light_specs[0]
        elif dimension == '1D_y':
            intensity = intensity[pix_slice, :] # Slice intensity array in pix_slice
            axis = light_specs[1]
        else:
            axis = light_specs[0]
        popt, fwhm, r_squared = fwhm_1d_fit(axis, intensity, fit)

    if dimension == '2D':
        popt, fwhm, r_squared = fwhm_2d_fit(light_specs[0], light_specs[1], intensity, fit)
        # FWHM (tuple) = FWHM_x, FWHM_y

    return popt, fwhm, r_squared

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

    fwhm_x, _, _, _ = fwhm_1d(x, Ix)
    fwhm_y, _, _, _ = fwhm_1d(y, Iy)
    
    return fwhm_x, fwhm_y

def fwhm_1d(x, intensity, I_max = None):
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
    if I_max is None:
        I_max = jnp.max(intensity)
    I_half = I_max * 0.5
    
    # Pixels with I max:
    i_max = jnp.argmax(intensity)
    # i_max = jnp.where(intensity == I_max)
    # # Compute the pixel location:
    # i_max = int(i_max[0][0])

    # Compute the slopes:
    i_left, _, distance_left = nearest(intensity[0:i_max], I_half)
    slope_left = (intensity[i_left + 1] - intensity[i_left]) / dx

    i_right, _, distance_right = nearest(intensity[i_max::], I_half)
    i_right += i_max
    slope_right = (intensity[i_right] - intensity[i_right - 1]) / dx

    x_right = x[i_right] - distance_right / slope_right
    x_left = x[i_left] - distance_left / slope_left

    # Compute fwhm:
    fwhm = x_right - x_left
    
    return fwhm, x_left, x_right, I_half

# -----------------------------------------------------------

""" Functions to process datasets (hdf5 files) """

class MultiHDF5DataLoader:
    """
    Class for JAX-DataLoader
    """   
    def __init__(self, directory, batch_size):
        self.directory = directory
        self.batch_size = batch_size

        # Get a list of all HDF5 files in the directory
        self.files = [f for f in os.listdir(self.directory) if f.endswith('.hdf5')]

        if not self.files:
            raise ValueError(f"No HDF5 files found in directory: {self.directory}")

    def __iter__(self):
        return self

    def __next__(self):
        """
        Randomly selects one file and randomly picks batch_size number of files from it. Returns the jnp.array version. 
        """
        # Randomly select one of the HDF5 files
        selected_file = random.choice(self.files)
        filepath = os.path.join(self.directory, selected_file)

        # Open the selected HDF5 file to get the total number of samples; 
        # This is where we open the selected HDF5 file in read mode. However, we're not reading the entire dataset into memory. 
        # Instead, we're just accessing its shape to get the total number of samples. The data remains on disk.
        with h5py.File(filepath, 'r') as hf:
            total_samples = hf["Input fields"].shape[0]
            
        # Randomly select indices for the batch
        batch_indices = sorted(random.sample(range(total_samples), self.batch_size))

        # Load the batch from the selected HDF5 file;
        # We open the HDF5 file in read mode again, but this time, we're using the randomly selected indices (batch_indices) to fetch only a specific subset of the data. 
        # Only the data corresponding to these indices is loaded into memory, and not the entire dataset. 
        with h5py.File(filepath, 'r') as hf:
            input_batch = hf["Input fields"][batch_indices]
            target_batch = hf["Target fields"][batch_indices]

        return jnp.array(input_batch), jnp.array(target_batch)