import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, config, 
from functools import partial
import matplotlib.pyplot as plt
import time

from toolbox import rotate_frame, profile
from wave_optics import build_grid, RS_propagation_jit, build_CZT_grid, CZT_jit, CZT_for_high_NA_jit

# Comment this line if float32 is enough precision for you. 
config.update("jax_enable_x64", True)
 
""" 
Module for vectorized optical fields:

    - VectorizedLight:
        - draw
        - draw_intensity_profile
        - VRS_propagation
        - get_VRS_minimum_z
        - VCZT
    - VRS_propagation_jit
    - VCZT_jit
    - vectorized_CZT_for_high_NA
    
    - PolarizedLightSource:
        - gaussian_beam
        - plane_wave
"""

class VectorizedLight:
    """ Class for Vectorial EM fields - (Ex, Ey, Ez) """
    def __init__(self, x=None, y=None, wavelength=None):
        self.x = x
        self.y = y
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        self.wavelength = wavelength
        self.k = 2 * jnp.pi / wavelength
        self.n = 1
        shape = (jnp.shape(x)[0], jnp.shape(y)[0])
        self.Ex = jnp.zeros(shape, dtype=jnp.complex128)
        self.Ey = jnp.zeros(shape, dtype=jnp.complex128)
        self.Ez = jnp.zeros(shape, dtype=jnp.complex128)
        self.info = 'Vectorized light'
        
    def draw(self, xlim='', ylim='', kind='', extra_title='', save_file=False, filename=''):
        """
        Plots VectorizedLight.

        Parameters:
            xlim (float, float): x-axis limit for plot purpose.
            ylim (float, float): y-axis limit for plot purpose. 
            kind (str): Feature to plot: 'Intensity', 'Phase' or 'Field'. 
            extra_title (str): Adds extra info to the plot title.
            save_file (bool): If True, saves the figure.
            filename (str): Name of the figure.
        """        
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        if kind == 'Intensity':
            # Compute intensity
            Ix = jnp.abs(self.Ex) ** 2  # Ex
            Iy = jnp.abs(self.Ey) ** 2  # Ey
            Iz = jnp.abs(self.Ez) ** 2  # Ez
            Ir = Ix + Iy  # Er
            
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            cmap = 'gist_heat'
            
            ax = axes[0,0]
            im = ax.imshow(Ix, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Intensity x. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Ix), vmax=jnp.max(Ix))

            ax = axes[0,1]
            im = ax.imshow(Iy, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Intensity y. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Iy), vmax=jnp.max(Iy))

            ax = axes[1,0]
            im = ax.imshow(Iz, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Intensity z. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Iz), vmax=jnp.max(Iz))

            ax = axes[0,2]
            im = ax.imshow(Ir, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Intensity r. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Ir), vmax=jnp.max(Ir))
            
            axes[1,1].axis('off')
            axes[1,2].axis('off')
            plt.subplots_adjust(wspace=0.6, hspace=0.6)
            

        elif kind == 'Phase':
            # Compute phase
            phi_x = jnp.angle(self.Ex)  # Ex
            phi_y = jnp.angle(self.Ey)  # Ey
            phi_z = jnp.angle(self.Ez)  # Ez

            fig, axes = plt.subplots(1, 3, figsize=(14, 3))
            cmap = 'twilight'

            ax = axes[0]
            im = ax.imshow(phi_x, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Phase x (in radians). {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=-jnp.pi, vmax=jnp.pi)

            ax = axes[1]
            im = ax.imshow(phi_y, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Phase y (in radians). {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=-jnp.pi, vmax=jnp.pi)

            ax = axes[2]
            im = ax.imshow(phi_z, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Phase z (in radians). {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=-jnp.pi, vmax=jnp.pi)

        elif kind == 'Field':
            # Compute field amplitudes
            Ax = jnp.abs(self.Ex)  # Ex
            Ay = jnp.abs(self.Ey)  # Ey
            Az = jnp.abs(self.Ez)  # Ez
            
            fig, axes = plt.subplots(1, 3, figsize=(14, 3))
            cmap = 'viridis'

            ax = axes[0]
            im = ax.imshow(Ax, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Amplitude x. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Ax), vmax=jnp.max(Ax))

            ax = axes[1]
            im = ax.imshow(Ay, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Amplitude y. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Ay), vmax=jnp.max(Ay))

            ax = axes[2]
            im = ax.imshow(Az, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"Amplitude z. {extra_title}")
            ax.set_xlabel('$x (\mu m)$')
            ax.set_ylabel('$y (\mu m)$')
            fig.colorbar(im, ax=ax)
            im.set_clim(vmin=jnp.min(Az), vmax=jnp.max(Az))

        else:
            raise ValueError(f"Invalid kind option: {kind}. Please choose 'Intensity', 'Phase' or 'Field'.")

        plt.tight_layout()

        if save_file is True:
            plt.savefig(filename)
            print(f"Plot saved as {filename}")
            
        plt.show()
    
    def draw_intensity_profile(self, p1='', p2=''):
        """
        Draws the intensity profile of VectorizedLight.

        Parameters: 
            p1 (float, float): Initial point.
            p2 (float, float): Final point.
        """
        h, z_profile_x = profile(jnp.abs(self.Ex)**2, self.x, self.y, point1=p1, point2=p2)
        _, z_profile_y = profile(jnp.abs(self.Ey)**2, self.x, self.y, point1=p1, point2=p2)
        _, z_profile_z = profile(jnp.abs(self.Ez)**2, self.x, self.y, point1=p1, point2=p2)
        _, z_profile_r = profile(jnp.abs(self.Ex)**2 + jnp.abs(self.Ey)**2, self.x, self.y, point1=p1, point2=p2)
        _, z_profile_total = profile(jnp.abs(self.Ex)**2 + jnp.abs(self.Ey)**2 + jnp.abs(self.Ez)**2, self.x, self.y, point1=p1, point2=p2)

        fig, axes = plt.subplots(3, 2, figsize=(14, 14))

        ax = axes[0, 0]
        im = ax.plot(h, z_profile_x, 'k', lw=2)
        ax.set_title(f"Ix profile")
        ax.set_xlabel('$\mu m$')
        ax.set_ylabel('$Ix$')
        ax.set(xlim=(h.min(), h.max()), ylim=(z_profile_x.min(), z_profile_x.max()))

        ax = axes[0, 1]
        im = ax.plot(h, z_profile_y, 'k', lw=2)
        ax.set_title(f"Iy profile")
        ax.set_xlabel('$\mu m$')
        ax.set_ylabel('$Iy$')
        ax.set(xlim=(h.min(), h.max()), ylim=(z_profile_y.min(), z_profile_y.max()))

        ax = axes[1, 0]
        im = ax.plot(h, z_profile_z, 'k', lw=2)
        ax.set_title(f"Iz profile")
        ax.set_xlabel('$\mu m$')
        ax.set_ylabel('$Iz$')
        ax.set(xlim=(h.min(), h.max()), ylim=(z_profile_z.min(), z_profile_z.max()))

        ax = axes[1, 1]
        im = ax.plot(h, z_profile_r, 'k', lw=2)
        ax.set_title(f"Ir profile")
        ax.set_xlabel('$\mu m$')
        ax.set_ylabel('$Ir$')
        ax.set(xlim=(h.min(), h.max()), ylim=(z_profile_r.min(), z_profile_r.max()))

        ax = axes[2, 0]
        im = ax.plot(h, z_profile_total, 'k', lw=2)
        ax.set_title(f"Itotal profile")
        ax.set_xlabel('$\mu m$')
        ax.set_ylabel('$Itotal$')
        ax.set(xlim=(h.min(), h.max()), ylim=(z_profile_total.min(), z_profile_total.max()))
    
        axes[2, 1].axis('off')
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
        plt.show()
    
    def VRS_propagation(self, z):
        """
        Rayleigh-Sommerfeld diffraction integral in both, z>0 and z<0, for VectorizedLight. 
        [Ref 1: Laser Phys. Lett., 10(6), 065004 (2013)].
        [Ref 2: Optics and laser tech., 39(4), 10.1016/j.optlastec.2006.03.006].
        [Ref 3: J. Li, Z. Fan, Y. Fu, Proc. SPIE 4915, (2002)].
        
        Parameters:
            z (float): Distance to propagate. 
        
        Returns VectorizedLight object after propagation and the quality factor of the algorithm.
        """ 
        tic = time.perf_counter()
        # Define r [From Ref 1, eq. 1a-1c]:
        r = jnp.sqrt(self.X ** 2 + self.Y ** 2 + z ** 2)

        # Set the value of Ez:
        Ez = jnp.array(self.Ex * self.X / r + self.Ey * self.Y / r)
        nx, ny, dx, dy, Xext, Yext = build_grid(self.x, self.y)
        
        # Quality factor for accurate simulation [Eq. 22 in Ref1]:
        dr_real = jnp.sqrt(dx**2 + dy**2)
        # Rho
        rmax = jnp.sqrt(jnp.max(self.x**2) + jnp.max(self.y**2))
        # Delta rho ideal
        dr_ideal = jnp.sqrt((self.wavelength)**2 + rmax**2 + 2 * (self.wavelength) * jnp.sqrt(rmax**2 + z**2)) - rmax
        quality_factor = dr_ideal / dr_real
        
        # Stack the input field in a (3, N, N) shape and pass to jit.
        E_in = jnp.stack([self.Ex, self.Ey, Ez], axis=0)
        E_out = VRS_propagation_jit(E_in, z, nx, ny, dx, dy, Xext, Yext, self.k)
        E_out = jnp.moveaxis(E_out, [0, 1, 2], [2, 0, 1])
        
        # Define the output light:
        light_out = VectorizedLight(self.x, self.y, self.wavelength)
        light_out.Ex = E_out[:, :, 0]
        light_out.Ey = E_out[:, :, 1]
        light_out.Ez = E_out[:, :, 2]

        print("Time taken to perform one VRS propagation (in seconds):", time.perf_counter() - tic)
        return light_out, quality_factor
    
    def get_VRS_minimum_z(self, n=1, quality_factor=1):
        """
        Given a quality factor, determines the minimum available (trustworthy) distance for VRS_propagation(). 
        [Ref 1: Laser Phys. Lett., 10(6), 065004 (2013)].

        Parameters:
            n (float): refraction index of the surrounding medium.
            quality_factor (int): Defaults to 1.

        Returns the minimum distance z (in microns) necessary to achieve qualities larger than quality_factor.
        
        >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
        """
        # Check sampling 
        range_x = self.x[-1] - self.x[0]
        range_y = self.y[-1] - self.y[0]
        num_x = len(self.x)
        num_y = len(self.y)

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

        return print("Minimum distance to propagate (in um):", z_min)
    
    def VCZT(self, z, xout, yout):
        """
        Vectorial version of the Chirped z-transform propagation - efficient RS diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).

        Parameters:
            z (float): Propagation distance.
            xout (jnp.array): Array with the x-positions for the output plane.

        Returns VectorizedLight object after propagation.
        """
        tic = time.perf_counter()
        if xout is None:
            xout = self.x

        if yout is None:
            yout = self.y

        # Define r:
        r = jnp.sqrt(self.X ** 2 + self.Y ** 2 + z ** 2)

        # Set the value of Ez:
        Ez = jnp.array((self.Ex * self.X / r + self.Ey * self.Y / r) * z / r)

        # Define main set of parameters
        nx, ny, dx, dy, Xout, Yout, Dm, fy_1, fy_2, fx_1, fx_2 = build_CZT_grid(z, self.wavelength, self.x, self.y, xout, yout)

        # Stack the input field in a (3, N, N) shape and pass to jit.
        E_in = jnp.stack([self.Ex, self.Ey, Ez], axis=0)
        E_out = VCZT_jit(E_in, z, self.wavelength, self.k, nx, ny, dx, dy, Xout, Yout, self.X, self.Y, Dm, fy_1, fy_2, fx_1, fx_2)
        E_out = jnp.moveaxis(E_out, [0, 1, 2], [2, 0, 1])

        # Define the output light:
        light_out = VectorizedLight(xout, yout, self.wavelength)
        light_out.Ex = E_out[:, :, 0]
        light_out.Ey = E_out[:, :, 1]
        light_out.Ez = E_out[:, :, 2]
        
        print("Time taken to perform one VCZT propagation (in seconds):", time.perf_counter() - tic)
        return light_out
    
    
@partial(jit, static_argnums=(2, 3, 8))
def VRS_propagation_jit(input_field, z, nx, ny, dx, dy, Xext, Yext, k):
    """[From VRS_propagation]: JIT function that vectorizes the propagation and calls RS_propagation_jit from wave_optics.py."""
    
    # Input field has (3, N, N) shape
    vectorized_RS_propagation = vmap(RS_propagation_jit,
                                     in_axes=(0, None, None, None, None, None, None, None, None))
    # Call the vectorized function
    E_out = vectorized_RS_propagation(input_field, z, nx, ny, dx, dy, Xext, Yext, k)
    return E_out # (3, N, N) -> ([Ex, Ey, Ez], N, N)

def VCZT_jit(field, z, wavelength, k, nx, ny, dx, dy, Xout, Yout, X, Y, Dm, fy_1, fy_2, fx_1, fx_2):
    """[From CZT]: JIT function that vectorizes the propagation and calls CZT_jit from wave_optics.py."""
    
    # Input field has (3, N, N) shape
    vectorized_CZT = vmap(CZT_jit, 
                          in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None))
    
    # Call the vectorized function
    E_out = vectorized_CZT(field, z, wavelength, k, nx, ny, dx, dy, Xout, Yout, X, Y, Dm, fy_1, fy_2, fx_1, fx_2)
    return E_out # (3, N, N) -> ([Ex, Ey, Ez], N, N)

def vectorized_CZT_for_high_NA(field, nx, ny, Dm, fy_1, fy_2, fx_1, fx_2):
    """[From VCZT_objective_lens - in optical_elements.py]: JIT function that vectorizes the propagation and calls CZT_for_high_NA_jit from wave_optics.py."""
    
    # Input field has (3, N, N) shape
    vectorized = vmap(CZT_for_high_NA_jit, in_axes=(0, None, None, None, None, None, None, None))
    
    # Call the vectorized function
    E_out = vectorized(field, nx, ny, Dm, fy_1, fy_2, fx_1, fx_2)
    return E_out # (3, N, N) -> ([Ex, Ey, Ez], N, N)

class PolarizedLightSource(VectorizedLight):
    """ Class for generating polarized light source beams."""
    def __init__(self, x, y, wavelength):
        super().__init__(x, y, wavelength)
        self.info = 'Vectorized light source'

    def gaussian_beam(self, w0, jones_vector, center=(0, 0), z_w0=(0, 0), alpha=0):
        """
        Defines a gaussian beam.

        Parameters:
            w0 (float, float): Waist radius (in microns).
            jones_vector (float, float): (Ex, Ey) at the origin (r=0, z=0). Doesn't need to be normalized. 
            center (float, float): Position of the center of the beam (in microns).
            z_w0 (float, float): Position of the waist for (x, y) (in microns).
            alpha (float, float): Amplitude rotation (in radians).

        Returns PolarizedLightSource object.
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
        # Accounting the rotation of the coordinates by alpha:
        x_rot = self.X * jnp.cos(alpha) + self.Y * jnp.sin(alpha)
        y_rot = -self.X * jnp.sin(alpha) + self.Y * jnp.cos(alpha)

        # Define the phase and amplitude of the field:
        phase = jnp.exp(-1j * ((self.k * z_w0x + self.k * self.X ** 2 / (2 * R_x) - Gouy_phase_x) + (
                    self.k * z_w0y + self.k * self.Y ** 2 / (2 * R_y) - Gouy_phase_y)))

        # Normalize Jones vector:
        normalized_jones = np.array(jones_vector) / jnp.linalg.norm(np.array(jones_vector))
        
        Ex = normalized_jones[0] * (w0_x / w_x) * (w0_y / w_y) * jnp.exp(
            -(x_rot - x0) ** 2 / (w_x ** 2) - (y_rot - y0) ** 2 / (w_y ** 2))
        Ey = normalized_jones[1] * (w0_x / w_x) * (w0_y / w_y) * jnp.exp(
            -(x_rot - x0) ** 2 / (w_x ** 2) - (y_rot - y0) ** 2 / (w_y ** 2))
        
        self.Ex = Ex*phase
        self.Ey = Ey*phase
        self.Ez = jnp.zeros((jnp.shape(self.x)[0], jnp.shape(self.x)[0]))
        
    def plane_wave(self, jones_vector, theta=0, phi=0, z0=0):
        """
        Defines a plane wave. 
        
        Parameters:
            jones_vector (float, float): (Ex, Ey) at the origin (r=0, z=0). Doesn't need to be normalized. 
            theta (float): Angle (in radians).
            phi (float): Angle (in radians).
            z0 (float): Constant value for phase shift.
        
        Equation:
        self.field = A * exp(1.j * k * (self.X * sin(theta) * cos(phi) + self.Y * sin(theta) * sin(phi) + z0 * cos(theta)))
        
        Returns PolarizedLightSource object.
        >> Diffractio-adapted function (https://pypi.org/project/diffractio/) <<<
        """
        # Normalize Jones vector:
        normalized_jones = np.array(jones_vector) / jnp.linalg.norm(np.array(jones_vector))
       
        pw = jnp.exp(1j * self.k * (self.X * jnp.sin(theta) * jnp.cos(phi) + self.Y * jnp.sin(theta) * jnp.sin(phi) + z0 * jnp.cos(theta)))

        self.Ex = normalized_jones[0] * pw
        self.Ey = normalized_jones[1] * pw
        self.Ez = jnp.zeros((jnp.shape(self.x)[0], jnp.shape(self.x)[0]))


            
        