import jax.numpy as jnp
import numpy as np
from jax import jit, config
from functools import partial
import matplotlib.pyplot as plt
import time

from toolbox import rotate_mask

# Comment this line if float32 is enough precision for you. 
config.update("jax_enable_x64", True)

""" 
Module for scalar optical fields.

    - ScalarLight:
        - draw
        - apply_circular_mask
        - apply_triangular_mask
        - apply_rectangular_mask
        - apply_annular_aperture
        - RS_propagation
        - get_RS_minimum_z
        - CZT
    - build_grid
    - RS_propagation_jit
    - transfer_function_RS
    - build_CZT_grid
    - CZT_jit
    - CZT_for_high_NA_jit
    - compute_np2
    - compute_fft
    - Bluestein_method
    
    - LightSource:
        - gaussian_beam
        - plane_wave
"""


class ScalarLight:
    """ Class for Scalar fields - complex amplitude U(r) = A(r)*exp(-ikz). """
    def __init__(self, x, y, wavelength):
        self.x = x
        self.y = y
        self.X, self.Y = jnp.meshgrid(x, y)
        self.wavelength = wavelength
        self.k = 2 * jnp.pi / wavelength
        self.n = 1
        self.field = jnp.zeros((jnp.shape(x)[0], jnp.shape(y)[0]))
        self.info = 'Wave optics light'
        
    def draw(self, xlim='', ylim='', kind='', extra_title='', save_file=False, filename=''):
        """
        Plots ScalarLight.

        Parameters:
            xlim (float, float): x-axis limits.
            ylim (float, float): y-axis limits. 
            kind (str): Feature to plot: 'Intensity' or 'Phase'. 
            extra_title (str): Adds extra info to the plot title.
            save_file (bool): If True, saves the figure.
            filename (str): Name of the figure.
        """
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        if kind == 'Intensity':
            field_to_plot = jnp.abs(self.field) ** 2  # Compute intensity (magnitude squared)
            title = f"Detected intensity. {extra_title}"
            cmap = 'gist_heat'
            plt.imshow(field_to_plot, cmap=cmap, extent=extent, origin='lower')
            plt.colorbar(orientation='vertical')

        elif kind == 'Phase':
            field_to_plot = jnp.angle(self.field)  # Calculate phase (in radians)
            title = f"Phase in radians. {extra_title}"
            cmap = 'twilight'
            plt.imshow(field_to_plot, cmap=cmap, extent=extent, origin='lower')
            plt.colorbar(orientation='vertical')
            plt.clim(vmin=-jnp.pi, vmax=jnp.pi)

        else:
            raise ValueError(f"Invalid kind option: {kind}. Please choose 'Intensity' or 'Phase'.")

        plt.title(title)
        plt.xlabel('$x(\mu m)$')
        plt.ylabel('$y(\mu m)$')

        if save_file is True:
            plt.savefig(filename)
            print(f"Plot saved as {filename}")
            
        plt.show()
        
    def apply_circular_mask(self, r):
        """
        Apply a circular mask of variable radius.

        Parameters:
            r (float, float): Radius of the circle (in microns).
        
        Returns ScalarLight object after applying the pupil mask.
        """
        rx, ry = r
        pupil = jnp.where((self.X**2 / rx**2 + self.Y**2 / ry**2) < 1, 1, 0)
        output = ScalarLight(self.x, self.y, self.wavelength)
        output.field = self.field * pupil
        return output
    
    def apply_triangular_mask(self, r, angle, m, height):
        """
        Apply a triangular mask of variable size. 
        
        Equation to generate the triangle: y = -m (x - x0) + y0.

        Parameters:
            r (float, float): Coordinates of the top corner of the triangle (in microns).
            angle (float): Rotation of the triangle (in radians).
            m (float): Slope of the edges.
            height (float): Distance between the top corner and the basis (in microns).

        Returns ScalarLight object after applying the triangular mask.

        >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
        """
        x0, y0 = r
        Xrot, Yrot = rotate_mask(self.X, self.Y, angle, origin=(x0, y0))
        Y = -m * jnp.abs(Xrot - x0) + y0
        mask = jnp.where((Yrot < Y) & (Yrot > (y0 - height)), 1, 0)
        output = ScalarLight(self.x, self.y, self.wavelength)
        output.field = self.field * mask
        return output
    
    def apply_rectangular_mask(self, center, width, height, angle):
        """
        Apply a square mask of variable size. Can generate rectangles, squares and rotate them to create diamond shapes.

        Parameters:
            center (float, float): Coordinates of the center (in microns).
            width (float): Width of the rectangle (in microns).
            height (float): Height of the rectangle (in microns).
            angle (float): Angle of rotation of the rectangle (in degrees).

        Returns ScalarLight object after applying the rectangular mask.
        """
        x0, y0 = center
        angle = angle * (jnp.pi/180)
        Xrot, Yrot = rotate_mask(self.X, self.Y, angle, center)
        mask = jnp.where((Xrot < (width/2)) & (Xrot > (-width/2)) & (Yrot < (height/2)) & (Yrot > (-height/2)), 1, 0)
        output = ScalarLight(self.x, self.y, self.wavelength)
        output.field = self.field * mask
        return output
    
    def apply_annular_aperture(self, di, do):
        """
        Apply an annular aperture of variable size.

        Parameters:
            di (float): Radius of the inner circle (in microns).
            do (float): Radius of the outer circle (in microns).

        Returns ScalarLight object after applying the annular mask.
        """
        di = di/2
        do = do/2
        stop = jnp.where(((self.X**2 + self.Y**2) / di**2) < 1, 0, 1)
        ring = jnp.where(((self.X**2 + self.Y**2) / do**2) < 1, 1, 0)
        output = ScalarLight(self.x, self.y, self.wavelength)
        output.field = self.field * stop*ring
        return output
    
    def RS_propagation(self, z):
        """
        Rayleigh-Sommerfeld diffraction integral in both, z>0 and z<0, for ScalarLight. 
        [Ref 1: F. Shen and A. Wang, Appl. Opt. 45, 1102-1110 (2006)].
        [Ref 2: J. Li, Z. Fan, Y. Fu, Proc. SPIE 4915, (2002)].
        
        Parameters:
            z (float): Distance to propagate (in microns). 
        
        Returns ScalarLight object after propagation and the quality factor of the algorithm.
        """
        tic = time.perf_counter()
        nx, ny, dx, dy, Xext, Yext = build_grid(self.x, self.y)
        
        # Quality factor for accurate simulation [Eq. 17 in Ref 2]:
        dr_real = jnp.sqrt(dx**2 + dy**2)
        rmax = jnp.sqrt((self.x**2).max() + (self.y**2).max())
        dr_ideal = jnp.sqrt((self.wavelength / 1)**2 + rmax**2 + 2 * (self.wavelength / 1) * jnp.sqrt(rmax**2 + z**2)) - rmax
        quality_factor = dr_ideal / dr_real
        
        propagated_light = ScalarLight(self.x, self.y, self.wavelength)
        propagated_light.field = RS_propagation_jit(self.field, z, nx, ny, dx, dy, Xext, Yext, self.k)
        print("Time taken to perform one RS propagation (in seconds):", time.perf_counter() - tic)
        return propagated_light, quality_factor
    
    def get_RS_minimum_z(self, n=1, quality_factor=1):
        """
        Given a quality factor, determines the minimum available (trustworthy) distance for RS_propagation(). 
        [Ref 1: Laser Phys. Lett., 10(6), 065004 (2013)].

        Parameters:
            n (float): refraction index of the medium.
            quality_factor (int): Defaults to 1.

        Returns the minimum distance z (in microns) necessary to achieve qualities larger than quality_factor.
        
        >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
        """
        # Check sampling 
        range_x = self.x[-1] - self.x[0]
        range_y = self.y[-1] - self.y[0]
        num_x = jnp.size(self.x)
        num_y = jnp.size(self.y)

        dx = range_x / num_x
        dy = range_y / num_y
        # Delta rho 
        dr_real = jnp.sqrt(dx**2 + dy**2)
        # Rho
        rmax = jnp.sqrt(range_x**2 + range_y**2)

        factor = (((quality_factor * dr_real + rmax)**2 - (self.wavelength / n)**2 - rmax**2) / (2 * self.wavelength / n))**2 - rmax**2

        if factor > 0:
            z_min = jnp.sqrt(factor)
        else:
            z_min = 0

        return print("Minimum distance to propagate (in microns):", z_min)
    
    def CZT(self, z, xout=None, yout=None):
        """
        Chirped z-transform propagation - efficient diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).

        Parameters:
            z (float): Propagation distance (in microns).
            xout (jnp.array): Array with the x-positions for the output plane.
            yout (jnp.array): Array with the y-positions for the output plane.

        Returns ScalarLight object after propagation.
        """
        tic = time.perf_counter()
        if xout is None:
            xout = self.x

        if yout is None:
            yout = self.y

        # Define main set of parameters
        nx, ny, dx, dy, Xout, Yout, Dm, fy_1, fy_2, fx_1, fx_2 = build_CZT_grid(z, self.wavelength, self.x, self.y, xout, yout)
        
        # Compute the diffraction integral using Bluestein method
        field_at_z = CZT_jit(self.field, z, self.wavelength, self.k, nx, ny, dx, dy, Xout, Yout, self.X, self.Y, Dm, fy_1, fy_2, fx_1, fx_2)

        # Build ScalarLight object with output field.
        field_out = ScalarLight(xout, yout, self.wavelength)
        field_out.field = field_at_z
        print("Time taken to perform one CZT propagation (in seconds):", time.perf_counter() - tic)
        return field_out

def build_grid(x, y):
    """[From RS_propagation]: Returns the grid where the transfer function is defined."""
    nx = len(x)
    ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # Build 2N-1 x 2N-1 (X, Y) space:
    x_padded = jnp.pad((x[0] - x[::-1]), (0, jnp.size(x) - 1), 'reflect')
    y_padded = jnp.pad((y[0] - y[::-1]), (0, jnp.size(y) - 1), 'reflect')
    # Convert the right half into positive values:
    I = jnp.ones((1, int(len(x_padded) / 2) + 1))
    II = -jnp.ones((1, int(len(x_padded) / 2)))
    III = jnp.ravel(jnp.concatenate((I, II), 1))
    Xext, Yext = jnp.meshgrid(x_padded * III, y_padded * III)
    return nx, ny, dx, dy, Xext, Yext

@partial(jit, static_argnums=(2, 3, 4, 5, 8))
def RS_propagation_jit(input_field, z, nx, ny, dx, dy, Xext, Yext, k):
    """[From RS_propagation]: JIT function for Equation (10) in [Ref 1]."""
    # input_field is jnp.array of (N, N)
    H = transfer_function_RS(z, Xext, Yext, k)
    U = jnp.zeros((2 * ny - 1, 2 * nx - 1), dtype=complex)
    U = U.at[0:ny, 0:nx].set(input_field)
    output_field = (jnp.fft.ifft2(jnp.fft.fft2(U) * jnp.fft.fft2(H)) * dx * dy)[ny - 1:, nx - 1:]
    return output_field

@partial(jit, static_argnums=(3,))
def transfer_function_RS(z, Xext, Yext, k):
    """[From RS_propagation]: JIT function for optical transfer function."""
    r = jnp.sqrt(Xext ** 2 + Yext ** 2 + z ** 2)
    factor = 1 / (2 * jnp.pi) * z / r ** 2 * (1 / r - 1j * k)
    result = jnp.where(z > 0, jnp.exp(1j * k * r) * factor, jnp.exp(-1j * k * r) * factor)
    return result

def build_CZT_grid(z, wavelength, xin, yin, xout, yout):
    """
    [From CZT]: Defines the resolution / sampling of initial and output planes.
    
    Parameters:
        xin (jnp.array): Array with the x-positions of the input plane.
        yin (jnp.array): Array with the y-positions of the input plane.
        xout (jnp.array): Array with the x-positions of the output plane.
        yout (jnp.array): Array with the y-positions of the output plane.
    
    Returns the set of parameters: nx, ny, Xout, Yout, dx, dy, delta_out, Dm, fy_1, fy_2, fx_1 and fx_2.
    """
    # Resolution of the output plane:
    nx = len(xout)
    ny = len(yout)
    Xout, Yout = jnp.meshgrid(xout, yout)
    
    # Sampling of initial plane:
    dx = xin[1] - xin[0]
    dy = yin[1] - yin[0]
    
    # For Bluestein method implementation: 
    # Dimension of the output field - Eq. (11) in [Ref].
    Dm = wavelength * z / dx
    
    # (1) for FFT in Y-dimension:
    fy_1 = yout[0] + Dm / 2
    fy_2 = yout[-1] + Dm / 2
    # (1) for FFT in X-dimension:
    fx_1 = xout[0] + Dm / 2
    fx_2 = xout[-1] + Dm / 2
    
    return nx, ny, dx, dy, Xout, Yout, Dm, fy_1, fy_2, fx_1, fx_2

def CZT_jit(field, z, wavelength, k, nx, ny, dx, dy, Xout, Yout, X, Y, Dm, fy_1, fy_2, fx_1, fx_2):
    """
    [From CZT]: Diffraction integral implementation using Bluestein method.
    [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
    """  
    # Compute the scalar diffraction integral using RS transfer function:
    # See Eq.(3) in [Ref].
    F0 = transfer_function_RS(z, Xout, Yout, k)
    F = transfer_function_RS(z, X, Y, k)
    
    # Compute (E0 x F) in Eq.(6) in [Ref].
    field = field * F
    
    # Bluestein method implementation:
    
    # (1) FFT in Y-dimension:
    U = Bluestein_method(field, fy_1, fy_2, Dm, ny)

    # (2) FFT in X-dimension using output from (1):
    U = Bluestein_method(U, fx_1, fx_2, Dm, nx)
    
    # Compute Eq.(6) in [Ref].
    field_at_z = F0 * U * z * dx * dy * wavelength
    
    return field_at_z

def CZT_for_high_NA_jit(field, nx, ny, Dm, fy_1, fy_2, fx_1, fx_2):
    """
    [From VCZT_objective_lens - in optical_elements.py]: Function for Debye integral implementation using Bluestein method.
    [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
    """
    # Bluestein method implementation:
    # (1) FFT in Y-dimension:
    U = Bluestein_method(field, fy_1, fy_2, Dm, ny)

    # (2) FFT in X-dimension using output from (1):
    U = Bluestein_method(U, fx_1, fx_2, Dm, nx)
    
    return U

def compute_np2(x):
    """
    [For Bluestein method]: Exponent of next higher power of 2. 

    Parameters:
        x (float): value

    Returns the exponent for the smallest powers of two that satisfy 2**p >= X for each element in X.
    """
    np2 = 2**(np.ceil(np.log2(x))).astype(int)
    return np2
    
@partial(jit, static_argnums=(4, 5, 6, 7, 8)) 
def compute_fft(x, D1, D2, Dm, m, n, mp, M_out, np2):
    """
    [From Bluestein_method]: JIT-computes the FFT part of the algorithm. 
    """
    # A-Complex exponential term
    A = jnp.exp(1j * 2 * jnp.pi * D1 / Dm)
    # W-Complex exponential term
    W = jnp.exp(-1j * 2 * jnp.pi * (D2 - D1) / (M_out * Dm)) 
    
    # Window function
    h = jnp.arange(-m + 1, max(M_out -1, m -1) +1)
    h = W**(h**2/ 2)
    h_sliced = h[:mp + 1]
    
    # Compute the 1D Fourier Transform of 1/h up to length 2**nextpow2(mp)
    ft = jnp.fft.fft(1 / h_sliced, np2)
    # Compute intermediate result for Bluestein's algorithm
    b = A**(-(jnp.arange(m))) * h[jnp.arange(m - 1, 2 * m -1)] 
    tmp = jnp.tile(b, (n, 1)).T
    # Compute the 1D Fourier Transform of input data * intermediate result
    b = jnp.fft.fft(x * tmp, np2, axis=0)
    # Compute the Inverse Fourier Transform
    b = jnp.fft.ifft(b * jnp.tile(ft, (n, 1)).T, axis=0)
    
    return b, h

def Bluestein_method(x, f1, f2, Dm, M_out):
    """
    [From CZT]: Performs the DFT using Bluestein method. 
    [Ref1]: Hu, Y., et al. Light Sci Appl 9, 119 (2020).
    [Ref2]: L. Bluestein, IEEE Trans. Au. and Electro., 18(4), 451-455 (1970).
    [Ref3]: L. Rabiner, et. al., IEEE Trans. Au. and Electro., 17(2), 86-92 (1969).
    
    Parameters:
        x (jnp.array): Input sequence, x[n] in Eq.(12) in [Ref 1].
        f1 (float): Starting point in frequency range.
        f2 (float): End point in frequency range. 
        Dm (float): Dimension of the imaging plane.
        M_out (float): Length of the transform (resolution of the output plane).
    
    Returns the output X[m] (jnp.array).
    
    >> Adapted from MATLAB code provided by https://github.com/yanleihu/Bluestein-Method<<
    """
    
    # Correspond to the length of the input sequence.  
    m, n = x.shape
    
    # Intermediate frequency
    D1 = f1 + (M_out * Dm + f2 - f1) / (2 * M_out) 
    # Upper frequency limit
    D2 = f2 + (M_out * Dm + f2 - f1) / (2 * M_out) 

    # Length of the output sequence
    mp = m + M_out - 1
    np2 = compute_np2(mp)
    b, h = compute_fft(x, D1, D2, Dm, m, n, mp, M_out, np2)
    
    # Extract the relevant portion and multiply by the window function, h
    if M_out > 1:
        b = b[m:mp +1, 0:n].T * jnp.tile(h[m - 1:mp], (n, 1))
    else:
        b = b[0] * h[0]

    # Create a linearly spaced array from 0 to M_out-1
    l = jnp.linspace(0, M_out - 1, M_out)
    # Scale the array to the frequency range [D1, D2]
    l = l / M_out * (D2 - D1) + D1
    
    # Eq. S14 in Supplementaty Information Section 3 in [Ref1]. Frequency shift to center the spectrum.
    M_shift = -m / 2
    M_shift = jnp.tile(jnp.exp(-1j * 2 * jnp.pi * l * (M_shift + 1 / 2) / Dm), (n, 1))
    # Apply the frequency shift to the final output
    b = b * M_shift
    return b

class LightSource(ScalarLight):
    """ Class for generating 2D wave optics light source beams. """
    def __init__(self, x, y, wavelength):
        super().__init__(x, y, wavelength)
        self.info = 'Wave optics light source'

    def gaussian_beam(self, w0, E0, center=(0, 0), z_w0=(0, 0), alpha=0):
        """
        Defines a gaussian beam.

        Parameters:
            w0 (float, float): Waist radius (in microns).
            E0 (float): Electric field amplitude at the origin (r=0, z=0).
            center (float, float): Position of the center of the beam (in microns).
            z_w0 (float, float): Position of the waist for (x, y) (in microns).
            alpha (float, float): Amplitude rotation (in radians).

        Returns LightSource object.
        """
        # Waist radius
        w0_x, w0_y = w0

        # (x, y) center position
        x0, y0 = center

        # z-position of the beam waist
        z_w0x, z_w0y = z_w0

        # Rayleigh range
        Rayleigh_x = self.k * w0_x ** 2 * self.n / 2
        Rayleigh_y = self.k * w0_y ** 2 * self.n / 2

        # Gouy phase
        Gouy_phase_x = jnp.arctan2(z_w0x, Rayleigh_x)
        Gouy_phase_y = jnp.arctan2(z_w0y, Rayleigh_y)

        # Spot size (radius of the beam at position z)
        w_x = w0_x * jnp.sqrt(1 + (z_w0x / Rayleigh_x) ** 2)
        w_y = w0_y * jnp.sqrt(1 + (z_w0y / Rayleigh_y) ** 2)

        # Radius of curvature
        if z_w0x == 0:
            R_x = 1e12
        else:
            R_x = z_w0x * (1 + (Rayleigh_x / z_w0x) ** 2)
        if z_w0x == 0:
            R_y = 1e12
        else:
            R_y = z_w0y * (1 + (Rayleigh_y / z_w0y) ** 2)

        # Gaussian beam coordinates
        X, Y = jnp.meshgrid(self.x, self.y)
        # Accounting the rotation of the coordinates by alpha:
        x_rot = X * jnp.cos(alpha) + Y * jnp.sin(alpha)
        y_rot = -X * jnp.sin(alpha) + Y * jnp.cos(alpha)

        # Define the phase and amplitude of the field:
        phase = jnp.exp(-1j * ((self.k * z_w0x + self.k * X ** 2 / (2 * R_x) - Gouy_phase_x) + (
                self.k * z_w0y + self.k * Y ** 2 / (2 * R_y) - Gouy_phase_y)))

        self.field = (E0 * (w0_x / w_x) * (w0_y / w_y) * jnp.exp(
            -(x_rot - x0) ** 2 / (w_x ** 2) - (y_rot - y0) ** 2 / (w_y ** 2))) * phase
        
    def plane_wave(self, A=1, theta=0, phi=0, z0=0):
        """
        Defines a plane wave. 
        
        Parameters:
            A (float): Maximum amplitude.
            theta (float): Angle (in radians).
            phi (float): Angle (in radians).
            z0 (float): Constant value for phase shift.
        
        Equation:
        self.field = A * exp(1.j * k * (self.X * sin(theta) * cos(phi) + self.Y * sin(theta) * sin(phi) + z0 * cos(theta)))
        
        Returns a LightSource object.
        
        >> Diffractio-adapted function (https://pypi.org/project/diffractio/) <<<
        """
        self.field = A * jnp.exp(1j * self.k * (self.X * jnp.sin(theta) * jnp.cos(phi) + self.Y * jnp.sin(theta) * jnp.sin(phi) + z0 * jnp.cos(theta)))