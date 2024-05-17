import jax.numpy as jnp
from jax import config, jit, vmap
from functools import partial
import time
from .__init__ import um, cm
from .wave_optics import ScalarLight
from .vectorized_optics import VectorizedLight, vectorized_CZT_for_high_NA
from .toolbox import build_LCD_cell, rotate_mask

# Comment this line if float32 is enough precision for you. 
config.update("jax_enable_x64", True)

""" Contains optical elements:

    (1) Scalar light devices:
        - phase_scalar_SLM
        - SLM
    
    (2) Jones matrices:
        - jones_LP
        - jones_general_retarder
        - jones_sSLM
        - jones_LCD
    
    (3) Polarization-based devices:
        - sSLM
        - LCD
        - linear_polarizer
NEW:    - BS_symmetric 
NEW:    - bs_port0
NEW:    - bs_port1
NEW:    - BS_symmetric_SI
        - BS
        - high_NA_objective_lens
            + _high_NA_objective_lens_
    (3.1) Propagation methods through objective lenses:
        - VCZT_objective_lens
            + build_high_NA_VCZT_grid
    
    (4) General elements:
        - lens
        - circular_mask
        - triangular_mask
        - rectangular_mask
        - annular_aperture
        - forked_grating
    
    (5) Pre-built optical set-ups:
        - building_block
NEW:    - fluorescence
NEW:    - vectorized_fluorophores
NEW:    - hybrid_setup_fixed_slms_fluorophores
NEW:    - hybrid_setup_fixed_slms
NEW:    - hybrid_setup_sharp_focus
NEW:    - hybrid_setup_fluorophores
NEW:    - six_times_six_ansatz
        - xl_setup
        - vSTED
        - sharp_focus
        - general_setup
"""

# ------------------------------------------------------------------------------------------------

""" (1) Scalar light devices: """

def phase_scalar_SLM(phase):
    """
    Phase for ScalarLight SLM. 
    
    Parameters:
        phase (float): Global phase (in radians).
   
    Returns phase (jnp.array).
    """
    return jnp.exp(1j * phase)

def SLM(input_field, phase_array, shape):
    """
    SLM (spatial light modulator) for ScalarLight: applies a phase mask [pixel-wise].
    
    Parameters:
        input_field (ScalarLight): Light to be modulated.
        phase_array (jnp.array): Phase to be applied (in radians). 
    
    Returns ScalarLight after applying the transformation.
    """
    slm = jnp.fromfunction(lambda i, j: phase_scalar_SLM(phase_array[i, j]),
                           (shape, shape), dtype=int)
    light_out = ScalarLight(input_field.x, input_field.y, input_field.wavelength)
    light_out.field = input_field.field * slm  # Multiplies element-wise
    
    return light_out, slm

# ------------------------------------------------------------------------------------------------

""" (2) Jones matrices: """

@jit
def jones_LP(alpha):
    """
    Define the Jones matrix of a Linear polarizer.

    Parameters:
        alpha (float): Transmission angle w.r.t. horizontal (in radians).
        
    Returns the Jones matrix (jnp.array).
    """
    return jnp.array([[jnp.cos(alpha) ** 2, jnp.cos(alpha) * jnp.sin(alpha)],
                      [jnp.cos(alpha) * jnp.sin(alpha), jnp.sin(alpha) ** 2]])

@jit
def jones_general_retarder(eta, theta, delta):
    """
    Define the Jones matrix of a general retarder.

    Parameters:
        eta (float): Phase difference between Ex and Ey (in radians).
        theta (float): Angle of the fast axis w.r.t. horizontal (in radians).
        delta (float): Ellipticity of the eigenvalues of the retarder.
        
    Returns the Jones matrix (jnp.array).
    """
    return jnp.array([[jnp.exp(-(eta / 2) * 1j) * jnp.cos(theta) ** 2 + jnp.exp((eta / 2) * 1j) * jnp.sin(theta) ** 2,
                      (jnp.exp(-(eta / 2) * 1j) - jnp.exp((eta / 2) * 1j)) * jnp.exp(-delta * 1j) * jnp.sin(
                          theta) * jnp.cos(theta)],
                     [(jnp.exp(-(eta / 2) * 1j) - jnp.exp((eta / 2) * 1j)) * jnp.exp(delta * 1j) * jnp.sin(
                         theta) * jnp.cos(theta),
                      jnp.exp(-(eta / 2) * 1j) * jnp.sin(theta) ** 2 + jnp.exp((eta / 2) * 1j) * jnp.cos(theta) ** 2]])

@jit
def jones_sSLM(alpha, phi):
    """
    Define the Jones matrix of the sSLM.

    Parameters:
        alpha (float): Phase mask for Ex (in radians).
        phi (float): Phase mask for Ey (in radians).
        
    Returns the Jones matrix (jnp.array).
    """
    return jnp.array([[jnp.exp(1j * alpha), 0], [0, jnp.exp(1j * phi)]])

@jit
def jones_LCD(eta, theta):
    """
    Define the Jones matrix of LCD (liquid crystal display).

    Parameters:
        eta (float): Phase difference between Ex and Ey (in radians).
        theta (float): Angle of the fast axis w.r.t. horizontal (in radians).

    Returns the Jones matrix (jnp.array).
    """
    return jones_general_retarder(eta, theta, delta=0)

# ------------------------------------------------------------------------------------------------

""" (3) Polarization-based devices: """

def sSLM(input_field, alpha_array=None, phi_array=None):
    """
    Define super-Spatial Light Modulator (sSLM): adds phase mask [pixel-wise] to Ex and Ey independently. 
    
    Illustrative scheme:
    (Ex, Ey) --> PBS --> Ex --> SLM(alpha) --> Ex' --> PBS --> (Ex', Ey')
                  |                                     ^ 
                  v                                     |
                  Ey ---------> SLM(phi) ----> Ey' -----/

    Parameters:
        input_field (VectorizedLight): Light to be modulated.
        alpha_array (jnp.array): Phase mask to be applied to Ex (in radians).
        phi_array (jnp.array): Phase mask to be applied to Ey (in radians).
        
    Returns VectorizedLight after applying the transformation.
    """
    # Consider Ex and Ey:
    input_field_xy = jnp.moveaxis(jnp.stack([input_field.Ex, input_field.Ey]), [0, 1, 2], [2, 0, 1])
    shape = jnp.shape(input_field_xy)[1]

    # Compute phase for each 
    sslm = jnp.fromfunction(lambda i, j: jones_sSLM(alpha_array[i, j], phi_array[i, j]),
                            (shape, shape), dtype=int)
    
    sslm = jnp.reshape(sslm, (shape ** 2, 2, 2))
    field = jnp.reshape(input_field_xy, (shape ** 2, 2, 1))
    field_out = sslm @ field
    field_out = field_out.reshape(shape, shape, 2)
    
    light_out = VectorizedLight(input_field.x, input_field.y, input_field.wavelength)
    light_out.Ex = field_out[:, :, 0]
    light_out.Ey = field_out[:, :, 1]
    # Maintain the input Ez.
    light_out.Ez = input_field.Ez
    
    return light_out

def LCD(input_field, eta, theta):
    """
    Liquid Crystal Device for VectorizedLight: builds any linear wave-plate.
    
    Parameters:
        input_field (VectorizedLight): Light to be modulated.
        eta (float): Retardance between Ex and Ey (in radians).
        theta (float): = Tilt of the fast axis w.r.t. the horizontal (in radians).
    
    Examples: tuning "eta" and "theta" one can achieve 
        HWP at 0º: eta = pi, theta = 0,
        HWP at 90º: eta = pi, theta = pi/2,
        QWP at 0º: eta = pi/2, theta = 0,
        QWP at 90º: eta = pi/2, theta = pi/2, etc. 
    
    Returns VectorizedLight after applying the transformation.
    """
    # Consider Ex and Ey:
    input_field_xy = jnp.moveaxis(jnp.stack([input_field.Ex, input_field.Ey]), [0, 1, 2], [2, 0, 1])
    shape = jnp.shape(input_field_xy)[1]
    
    # Define the constant eta and theta cell
    eta_array, theta_array = build_LCD_cell(eta, theta, shape)
    
    # Compute phase for each 
    lcd = jnp.fromfunction(lambda i, j: jones_LCD(eta_array[i, j], theta_array[i, j]),
                           (shape, shape), dtype=int)

    lcd = jnp.reshape(lcd, (shape ** 2, 2, 2))
    field = jnp.reshape(input_field_xy, (shape ** 2, 2, 1))
    field_out = lcd @ field
    field_out = field_out.reshape(shape, shape, 2)
    
    light_out = VectorizedLight(input_field.x, input_field.y, input_field.wavelength)
    light_out.Ex = field_out[:, :, 0]
    light_out.Ey = field_out[:, :, 1]
    # Maintain the input Ez.
    light_out.Ez = input_field.Ez    
    
    return light_out

def linear_polarizer(input_field, alpha):
    """
    Linear polarizer VectorizedLight.
    
    Parameters:
        input_field (VectorizedLight): Light to be modulated.
        alpha (jnp.array): Transmission angle w.r.t. horizontal (in radians).
    
    Returns VectorizedLight after applying the transformation.
    """
    # General function for linear polarizer.
    # Transmission angle alpha[i,j] from the horizontal.
    input_field_xy = jnp.moveaxis(jnp.stack([input_field.Ex, input_field.Ey]), [0, 1, 2], [2, 0, 1])
    shape = jnp.shape(input_field_xy)[1]

    E_reshape = jnp.reshape(input_field_xy, (shape ** 2, 2, 1))
    LP = jnp.fromfunction(lambda i, j: jones_LP(alpha[i, j]), (shape, shape), dtype=jnp.float64)
    LP_reshape = jnp.reshape(LP, (shape ** 2, 2, 2))
    E_out = LP_reshape @ E_reshape
    E_out = E_out.reshape(shape, shape, 2)
    
    light_out = VectorizedLight(input_field.x, input_field.y, input_field.wavelength)
    light_out.Ex = E_out[:, :, 0]
    light_out.Ey = E_out[:, :, 1]
    
    return light_out

def BS_symmetric(a, b, theta):
    """
    Classical lossless two-mode beam splitter of reflectance R = |sin(theta)|**2, and transmittance T = |cos(theta)|**2. 
    
    Scheme:
                 a
                 |
                 v        
          b --> [\] --> c = (r_ac) * a + (t_bc) * b
                 | 
                 v
                 d = (t_ad) * a + (r_bd) * b 
   
    ------------------------------------------------------------ 
    
    BS = [[ R     i*T],
          [i*T     R ]]
    
    c = [[R * Ex_a + T * i * Ex_b],
         [R * Ey_a + T * i * Ey_b]] 
               
    d = [[T * i * Ex_a + R * Ex_b],
         [T * i * Ey_a + R * Ey_b]]    
         
    Reflectance = R**2
    Transmissivity = T**2
    
    ------------------------------------------------------------          
    
    Parameters: 
        a (VectorizedLight): electric field in port a1
        b (VectorizedLight): electric field in port a2
        theta (float): reflectance (or transmittance); theta = arcsin(R) = arccos(T)
    
    Returns c and d (VectorizedLight). 
    """
    # Define light at ports b1 and b2.
    c = VectorizedLight(a.x, a.y, a.wavelength)
    d = VectorizedLight(a.x, a.y, a.wavelength)
    
    # Define reflectance and transmittance
    T = jnp.abs(jnp.cos(theta))
    R = jnp.abs(jnp.sin(theta))
    
    c.Ex = (R * a.Ex) + (1j * T * b.Ex) 
    c.Ey = (R * a.Ey) + (1j * T * b.Ey)

    d.Ex = (1j * T * a.Ex) + (R * b.Ex)
    d.Ey = (1j * T * a.Ey) + (R * b.Ey)  
    
    return c, d

@jit
def bs_port0(a_Ex, a_Ey, c_Ex, c_Ey, d_Ex, d_Ey, T, R):
    """ [For BS_symmetric_SI]: BS single input - port 0 - """
    c_Ex = (R * a_Ex) 
    c_Ey = (R * a_Ey)
    d_Ex = (1j * T * a_Ex)
    d_Ey = (1j * T * a_Ey)
    return c_Ex, c_Ey, d_Ex, d_Ey

@jit
def bs_port1(a_Ex, a_Ey, c_Ex, c_Ey, d_Ex, d_Ey, T, R):
    """ [For BS_symmetric_SI]: BS single input - port 1 - """
    d_Ex = (R * a_Ex) 
    d_Ey = (R * a_Ey)
    c_Ex = (1j * T * a_Ex)
    c_Ey = (1j * T * a_Ey)
    return c_Ex, c_Ey, d_Ex, d_Ey

def BS_symmetric_SI(a, theta, port=0):
    """
    Classical lossless single input beam splitter of reflectance R = |sin(theta)|**2, and transmittance T = |cos(theta)|**2. 
    
    Scheme:
        a (port 0)
        |
        v        
       [\] --> c = (r_ac) * a 
        | 
        v
        d = (t_ad) * a 
   
    ------------------------------------------------------------ 
    
    c = [[R * Ex_a],
         [R * Ey_a]] 
               
    d = [[T * i * Ex_a],
         [T * i * Ey_a]]    
         
    Reflectance = R**2
    Transmissivity = T**2
    
    ------------------------------------------------------------          
    
    Parameters: 
        a (VectorizedLight): electric field in port a1
        theta (float): reflectance (or transmittance); theta = arcsin(R) = arccos(T)
        port (int): if 0, light incoming through port a; if 1, port b.
    
    Returns c and d (VectorizedLight). 
    """
    # Define light at ports b1 and b2.
    c = VectorizedLight(a.x, a.y, a.wavelength)
    d = VectorizedLight(a.x, a.y, a.wavelength)
    
    # Define reflectance and transmittance
    T = jnp.abs(jnp.cos(theta))
    R = jnp.abs(jnp.sin(theta))
    
    # Apply BS single input in a differentiable way
    c.Ex, c.Ey, d.Ex, d.Ey = lax.cond(port == 0, bs_port0, bs_port1, a.Ex, a.Ey, c.Ex, c.Ey, d.Ex, d.Ey, T, R)
    
    return c, d

def BS(a1, a2, R, T, phase):
    """
    Lossless two-mode beam splitter of reflectance R, and transmittance T. 
    
    If: 
        phase = 0 -> light in port b1
        phase = pi -> light in port b2
        phase = pi/2 -> light in ports b1, b2
    
    Scheme:
                 a1
                 |
                 v        
         a2 --> [\] --> b2 = a2_t + a1_r
                 | 
                 v
                 b1 = a1_t + a2_r
   
    ------------------------------------------------------------ 
    
    BS = [[     √T          e^(i*phase)*√R],
          [-e^(-i*phase)√R        √T      ]]
    
    b1 = [[√T * Ex_a1 + √R * e^(i*phase) * Ex_a2],
          [√T * Ey_a1 + √R * e^(i*phase) * Ey_a2]] 
               
    b2 = [[- √R * e^(-i*phase) * Ex_a1 + √T * Ex_a2],
          [- √R * e^(-i*phase) * Ey_a1 + √T * Ey_a2]]          
    
    ------------------------------------------------------------          
    
    Parameters: 
        a1 (VectorizedLight): electric field in port a1
        a2 (VectorizedLight): electric field in port a2
        R (float): Reflectance (between 0 and 1)
        T (float): Transmittance (between 0 and 1)
        phase (float): phase shift to apply.
    
    Returns b1 and b2 (VectorizedLight). 
    """
    # Define light at ports b1 and b2.
    b1 = VectorizedLight(a1.x, a1.y, a1.wavelength)
    b2 = VectorizedLight(a1.x, a1.y, a1.wavelength)
    
    b1.Ex = (jnp.sqrt(T) * a1.Ex) + (jnp.sqrt(R) * jnp.exp(1j*phase) * a2.Ex) 
    b1.Ey = (jnp.sqrt(T) * a1.Ey) + (jnp.sqrt(R) * jnp.exp(1j*phase) * a2.Ey)
    
    b2.Ex = (- jnp.sqrt(R) * jnp.exp(- 1j*phase) * a1.Ex) + (jnp.sqrt(T) * a2.Ex)
    b2.Ey = (- jnp.sqrt(R) * jnp.exp(- 1j*phase) * a1.Ey) + (jnp.sqrt(T) * a2.Ey) 
    
    return b1, b2


def high_NA_objective_lens(input_field, radius, f):
    """
    High NA objective lens for VectorizedLight - to be used with [VCZT_objective_lens].
    [Ref1: Opt. Comm. 283 (2010), 4859 - 4865].
    [Ref2: Hu, Y., et al. Light Sci Appl 9, 119 (2020)].
        
    Parameters:
        input_field (VectorizedLight): Light to be focused.
        radius (float): Radius of the objective lens (in microns).
        f (float): Focal length of the objective lens (in microns).
        
    Returns the field directly after applying the lens.
    """ 
    sin_theta_max = radius / jnp.sqrt(radius ** 2 + f ** 2)
    
    # Coordinates:
    X, Y = input_field.X, input_field.Y
    r = jnp.sqrt(X ** 2 + Y ** 2)
    phi = jnp.arctan2(Y, X)
    theta = r / f
    
    # Set the value of Ez:
    # r_field = jnp.sqrt(X** 2 + Y** 2 + z ** 2)
    Ez = jnp.array(input_field.Ex * X/ r + input_field.Ey * Y / r) 
    
    # Spatial frequencies
    # Eq. (6) - Opt. Comm. 283 (2010), 4859 - 4865.
    u = X / radius
    v = Y / radius

    # G(u,v) - Eq. (9) - Opt. Comm. 283 (2010), 4859 - 4865.
    pupil_mask = jnp.where((X ** 2 + Y ** 2) / radius ** 2 < 1, 1, 0)
    G = jnp.real(pupil_mask * (1 / jnp.sqrt(jnp.abs(1 - (u ** 2 + v ** 2) * sin_theta_max ** 2))))
    incoming_light = jnp.stack([input_field.Ex, input_field.Ey, Ez], axis=-1)
    
    # jit-compute Equation (7) from [Ref 2].
    out_field = _high_NA_objective_lens_(incoming_light, theta, phi, G)
    
    return out_field, sin_theta_max

@jit
def _high_NA_objective_lens_(incoming_light, theta, phi, G):
    """[From high_NA_objective_lens]: JIT function that applies the lens."""
    Ex = incoming_light[:, :, 0]
    Ey = incoming_light[:, :, 1]
    Ez = incoming_light[:, :, 2]
    
    # For matrix elements:
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)
    
    # Apodization factor
    apod = jnp.sqrt(jnp.abs(c_theta))
    
    # Eq. (1) - Opt. Comm. 283 (2010), 4859 - 4865.
    # E0 = apod * RL * Ei 
    RL00 = c_theta * c_phi**2 + s_phi**2
    RL01 = c_theta * c_phi * s_phi - s_phi * c_phi
    RL02 = - c_phi * s_theta
    RL10 = s_phi * c_theta * c_phi - c_phi * s_phi
    RL11 = c_theta * s_phi**2 + c_phi**2
    RL12 = - s_phi * s_theta
    RL20 = s_theta * c_phi 
    RL21 = s_theta * s_phi
    RL22 = c_theta
    
    # Eq. (1) - Opt. Comm. 283 (2010), 4859 - 4865.
    E0_x = RL00 * Ex + RL01 * Ey + RL02 * Ez
    E0_y = RL10 * Ex + RL11 * Ey + RL12 * Ez
    E0_z = RL20 * Ex + RL21 * Ey + RL22 * Ez

    Ef_x = apod * G * E0_x
    Ef_y = apod * G * E0_y
    Ef_z = apod * G * E0_z

    # Stack the field in a (3, N, N).
    
    return jnp.stack([Ef_x, Ef_y, Ef_z], axis=0)

# ------------------------------------------------------------------------------------------------

""" (3.1) Propagation methods through objective lenses """

def VCZT_objective_lens(input_field, r, f, xout, yout):
    """
    Vectorial Chirp z-transform algorithm for high NA objective lens.
    [Ref 1] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
    [Ref 2] Opt. Comm. 283 (2010), 4859 - 4865.
    
    Parameters:
        input_field (VectorizedLight): Light to be focused.
        r (float): Radius of the objective lens (in microns).
        f (float): Focal length of the objective lens (in microns).
        xout, yout (jnp.arrays): Desired output (high resolution) arrays in the focal plane.

    Returns the VectorizedLight in the focal plane sampled in the new arrays (xout, yout).
    """
    tic = time.perf_counter()
    
    # Apply high NA objective lens - returns (3, N, N) electric field. 
    field_in_lens, sin_theta_max = high_NA_objective_lens(input_field, r, f)

    # Define main set of parameters
    nx, ny, Dm, fy_1, fy_2, fx_1, fx_2 = build_high_NA_VCZT_grid(f, r, input_field.wavelength, input_field.x, xout, yout)

    # Apply VCZT [Ref 1] to propagate through the focus.
    # Pass to jit the input field in shape (3, N, N).
    U = vectorized_CZT_for_high_NA(field_in_lens, nx, ny, Dm, fy_1, fy_2, fx_1, fx_2)

    # Eq. (8) in [Ref 2]
    cte = -(1j * sin_theta_max**2 / (f * input_field.wavelength))
    
    field_at_z = cte * U 
    
    # Define the output light:
    light_out = VectorizedLight(xout, yout, input_field.wavelength)
    light_out.Ex = field_at_z[0, :, :]
    light_out.Ey = field_at_z[1, :, :]
    light_out.Ez = field_at_z[2, :, :]
    print("Time taken to perform one VCZT propagation through objective lens (in seconds):", time.perf_counter() - tic)
    
    return light_out

def build_high_NA_VCZT_grid(f, r, wavelength, xin, xout, yout):
    """
    [For VCZT_objective_lens]: Defines the resolution / sampling of initial and output planes.
    
    Parameters:
        f (float): Focal length of the objective lens (in microns).
        r (float): Radius of the objective lens (in microns).
        wavelength (float): Wavelength of the light beam (in microns).
        xin (jnp.array): Array with the x-positions of the input plane.
        xout (jnp.array): Array with the x-positions of the output plane.
        yout (jnp.array): Array with the y-positions of the output plane.
    
    Returns the set of parameters: nx, ny, Xout, Yout, dx, dy, delta_out, Dm, fy_1, fy_2, fx_1 and fx_2.
    """
    # Resolution of the output plane:
    nx = len(xout)
    ny = len(yout)
    
    # Resolution of the input plane:
    Din = len(xin)
    
    # For Bluestein method implementation: 
    # Dimension of the imaging plane
    Dm = f * wavelength * (Din - 1)/(2 * r)
    
    # (1) for FFT in Y-dimension:
    fy_1 = yout[0] + Dm / 2
    fy_2 = yout[-1] + Dm / 2
    # (1) for FFT in X-dimension:
    fx_1 = xout[0] + Dm / 2
    fx_2 = xout[-1] + Dm / 2
    
    return nx, ny, Dm, fy_1, fy_2, fx_1, fx_2

# ------------------------------------------------------------------------------------------------

""" (4) General elements: """

def lens(input_field, radius, focal):
    """
    Define a transparent lens of variable size (in microns) for ScalarLight / VectorizedLight.

    Parameters:
        radius (float, focal): Radius of the lens (in microns).
        focal (float, float): Focal length of the lens (in microns).

    Returns ScalarLight (or VectorizedLight) after applying the lens and the lens mask.
    """
    fx, fy = focal
    pupil = circular_mask(input_field.X, input_field.Y, radius)
    lens_ = pupil * jnp.exp(-1j * input_field.k * (input_field.X**2 / (2*fx) + input_field.Y**2 / (2*fy)))
    
    if input_field.info == 'Wave optics light' or input_field.info =='Wave optics light source':
        output = ScalarLight(input_field.x, input_field.y, input_field.wavelength)
        output.field = input_field.field * lens_
        
    elif input_field.info == 'Vectorized light' or input_field.info =='Vectorized light source':
        output = VectorizedLight(input_field.x, input_field.y, input_field.wavelength)
        output.Ex = input_field.Ex * lens_
        output.Ey = input_field.Ey * lens_
    else:
        raise ValueError(f"Invalid input. Please use ScalarLight or VectorizedLight object.")
   
    return output, lens_

def circular_mask(X, Y, r):
    """
    Define a circular mask of variable size (in microns).
    
    Parameters:
        X (float, float): X array.
        Y (float, float): Y array.
        r (float, float): Radius of the circle (in microns).
    
    Returns the circular mask (jnp.array)
    """
    rx, ry = r
    pupil = jnp.where((X**2 / rx**2 + Y**2 / ry**2) < 1, 1, 0)
    
    return pupil

def triangular_mask(X, Y, r, angle, m, height):
    """
    Define a triangular mask of variable size (in microns); equation to generate the triangle: y = -m (x - x0) + y0.
    
    Parameters:
        X (float, float): x array.
        Y (float, float): y array.
        center (float, float): Coordinates of the top corner of the triangle (in microns).
        angle (float): Rotation of the triangle (in degrees).
        m (float): Slope of the edges.
        height (float): Distance between the top corner and the basis (in microns).
    
    Returns the triangular mask (jnp.array).
    
    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) << 
    """
    x0, y0 = r
    angle = angle * (jnp.pi/180)
    Xrot, Yrot = rotate_mask(X, Y, angle, r)
    Y = -m * jnp.abs(Xrot - x0) + y0
    return jnp.where((Yrot < Y) & (Yrot > y0 - height), 1, 0)

def rectangular_mask(X, Y, center, width, height, angle):
    """
    Apply a square mask of variable size. Can generate rectangles, squares and rotate them to create diamond shapes.
    
    Parameters:
        X (float, float): X array.
        Y (float, float): Y array.
        center (float, float): Coordinates of the center (in microns).
        width (float): Width of the rectangle (in microns).
        height (float): Height of the rectangle (in microns).
        angle (float): Angle of rotation of the rectangle (in degrees).
    
    Returns the rectangular mask (jnp.array). 
    """
    x0, y0 = center
    angle = angle * (jnp.pi/180)
    Xrot, Yrot = rotate_mask(X, Y, angle, center)
    return jnp.where((Xrot < (width/2)) & (Xrot > (-width/2)) & (Yrot < (height/2)) & (Yrot > (-height/2)), 1, 0)

def annular_aperture(di, do, X, Y):
    """
    Define annular aperture of variable size (in microns).
    
    Parameters:
        di (float): Radius of the inner circle (in microns).
        do (float): Radius of the outer circle (in microns).
        X (float, float): X array.
        Y (float, float): Y array.
    
    Returns the circular mask (jnp.array).
    """
    di = di/2
    do = do/2
    stop = jnp.where(((X**2 + Y**2) / di**2) < 1, 0, 1)
    ring = jnp.where(((X**2 + Y**2) / do**2) < 1, 1, 0)
    return stop*ring

def forked_grating(X, Y, center, angle, period, l, kind=''):
    """
    Defines a forked grating mask of variable size (in microns).
    
    Parameters:
        X (float, float): X array.
        Y (float, float): Y array.
        center (float, float): Coordinates of the center (in microns).
        angle (float): Angle of rotation of the grating (in degrees).
        period (float): Period of the grating.
        l (int): Number of lines inside the wrap.
        alpha (int): 
        kind (str): Set to 'Amplitude' or 'Phase'
    
    Returns the forked grating mask (jnp.array). 
    
    >> Diffractio-adapted function (https://pypi.org/project/diffractio/) <<
    """
    x0, y0 = center
    angle = angle * (jnp.pi/180)
    Xrot, Yrot = rotate_mask(X, Y, angle, center)

    theta = jnp.arctan2(Xrot, Yrot)
    alpha = 1 # Scaling factor 
    
    forked_grating = jnp.angle(jnp.exp(1j * alpha * jnp.cos(l * theta - 2 * jnp.pi / period * (Xrot))))

    forked_grating_phase = jnp.where(forked_grating < 0, 0, 1) 

    if kind == 'Amplitude':
        return forked_grating_phase 
    elif kind == 'Phase':
        return jnp.exp(1j * jnp.pi * forked_grating_phase)

# ------------------------------------------------------------------------------------------------

""" (5) Pre-built optical set-ups: """

def building_block(input_light, alpha, phi, z, eta, theta):
    """
    Basic building block for general setup construction. 
    
    Scheme:
    Light in --> sSLM (alpha, phi) -- VRS(z) -- LCD (eta, theta) --> Light out
    
    Parameters:
        input_light (VectorizedLight): Input light to the block (can be light source or light inside the system).
        alpha, phi (jnp.array): sSLM phase masks.
        z (jnp.array): Distance to propagate between sSLM and LCD. 
        eta, theta (jnp.array): Global retardance and tilt of LCD.
    
    Returns output light (VectorizedLight) from the block.
    """
    # Apply sSLM (alpha - Ex-, phi - Ey-)
    l_modulated = sSLM(input_light, alpha, phi)
    # Propagate (z)
    l_propagated, _ = l_modulated.VRS_propagation(z)
    # Apply LCD:
    return LCD(l_propagated, eta, theta)

def fluorescence(i_ex, i_dep, beta=1):
    """
    Fluorescence function - allows STED.
    
    Parameters: 
        i_ex (jnp.array): Excitation intensity
        i_dep (jnp.array): Depletion intensity
        beta (float, int): Strength parameter
        
    Returns effective intensity from stimulated emission-depletion.
    """
    # Epsilon - small numerical constant to prevent division by 0
    eps = 1e-8
    return i_ex * (1 - beta * (1- jnp.exp(-(i_dep/(i_ex + eps))))) 

@jit
def vectorized_fluorophores(i_ex, i_dep):
    """
    Vectorized version of [fluorescence]: Allows to compute effective intensity across an array of detectors.
    """
    vfluo = vmap(fluorescence, in_axes=(0, 0))
    return vfluo(i_ex, i_dep)


def hybrid_setup_fixed_slms_fluorophores(ls1, ls2, ls3, ls4, ls5, ls6, parameters, fixed_params, distance_offset = 8.9):
    """
    Hybrid steup with fixed phase masks for STED pure topology discovery: ls1 = red wavelength and ls2 = green wavelength
    """
    # Get fixed params:
    r=fixed_params[0]
    f=fixed_params[1]
    xout=fixed_params[2]
    yout=fixed_params[3]
    
    # Dorn, Quabis, Leuchs PM:
    phase1_1 = fixed_params[4]
    phase1_2 = fixed_params[5]
    # STED spiral PM
    phase2_1 = fixed_params[6]
    phase2_2 = fixed_params[6]
    # Forked grating PM
    phase3_1 = fixed_params[7]
    phase3_2 = fixed_params[7]
    # Ladder grating PM
    phase4_1 = fixed_params[8]
    phase4_2 = fixed_params[9]
    
    # Distances
    z1_1 = (jnp.abs(parameters[0]) * 100 + distance_offset)*cm
    z1_2 = (jnp.abs(parameters[1]) * 100 + distance_offset)*cm
    z2_1 = (jnp.abs(parameters[2]) * 100 + distance_offset)*cm
    z2_2 = (jnp.abs(parameters[3]) * 100 + distance_offset)*cm
    z3_1 = (jnp.abs(parameters[4]) * 100 + distance_offset)*cm
    z3_2 = (jnp.abs(parameters[5]) * 100 + distance_offset)*cm
    z4_1 = (jnp.abs(parameters[6]) * 100 + distance_offset)*cm
    z4_2 = (jnp.abs(parameters[7]) * 100 + distance_offset)*cm
    
    # Beam splitter ratios (theta: raw input from (0,1) to (-pi, pi)) 
    bs1 = parameters[8]* 2*jnp.pi - jnp.pi
    bs2 = parameters[9]* 2*jnp.pi - jnp.pi
    bs3 = parameters[10]* 2*jnp.pi - jnp.pi
    bs4 = parameters[11]* 2*jnp.pi - jnp.pi
    bs5 = parameters[12]* 2*jnp.pi - jnp.pi
    bs6 = parameters[13]* 2*jnp.pi - jnp.pi
    bs7 = parameters[14]* 2*jnp.pi - jnp.pi
    bs8 = parameters[15]* 2*jnp.pi - jnp.pi
    bs9 = parameters[16]* 2*jnp.pi - jnp.pi

    # LCDs:
    eta1   = parameters[17]* 2*jnp.pi - jnp.pi
    theta1 = parameters[18]* 2*jnp.pi - jnp.pi
    eta2   = parameters[19]* 2*jnp.pi - jnp.pi
    theta2 = parameters[20]* 2*jnp.pi - jnp.pi
    eta3   = parameters[21]* 2*jnp.pi - jnp.pi
    theta3 = parameters[22]* 2*jnp.pi - jnp.pi
    eta4   = parameters[23]* 2*jnp.pi - jnp.pi
    theta4 = parameters[24]* 2*jnp.pi - jnp.pi
    
    # 1st row:
    # BS port 0 for red
    c1_r, d1_r = BS_symmetric_SI(ls1, bs1, port=0)
    # BS port 1 for green
    c1_g, d1_g = BS_symmetric_SI(ls4, bs1, port=1)
    
    # Common PM
    b2_r, _ = (building_block(c1_r, phase1_1, phase1_2, z1_1, eta1, theta1)).VRS_propagation(z1_2)
    b2_g, _ = (building_block(c1_g, phase1_1, phase1_2, z1_1, eta1, theta1)).VRS_propagation(z1_2)
    
    # BS port 1 for red
    c2_r, d2_r = BS_symmetric_SI(b2_r, bs2, port=1)
    # BS symm for green
    c2_g, d2_g = BS_symmetric(ls2, b2_g, bs2)
    
    # Common prop
    b3_r, _ = c2_r.VRS_propagation(z2_1 + z2_2)
    b3_g, _ = c2_g.VRS_propagation(z2_1 + z2_2)
    
    # Symm for red
    c3_r, d3_r = BS_symmetric(ls3, b3_r, bs3)
    # SI port 1 green
    c3_g, d3_g = BS_symmetric_SI(b3_g, bs3, port=1)
    
    # Common objective lens
    det_1_r = VCZT_objective_lens(c3_r, r, f, xout, yout)
    det_1_g = VCZT_objective_lens(c3_g, r, f, xout, yout)
    
    
    # Mid space:
    a5_r, _ = d2_r.VRS_propagation(z3_1 + z3_2)
    a5_g, _ = d2_g.VRS_propagation(z3_1 + z3_2)
    a6_r, _ = (building_block(d3_r, phase3_1, phase3_2, z3_1, eta3, theta3)).VRS_propagation(z3_2)
    a6_g, _ = (building_block(d3_g, phase3_1, phase3_2, z3_1, eta3, theta3)).VRS_propagation(z3_2)
    
    # 2nd row:
    # Symm for red
    c4_r, d4_r = BS_symmetric(d1_r, ls5, bs4)
    # SI port 0 for green
    c4_g, d4_g = BS_symmetric_SI(d1_g, bs4, port=0)
    
    # Common prop
    b5_r, _ = c4_r.VRS_propagation(z1_1 + z1_2)
    b5_g, _ = c4_g.VRS_propagation(z1_1 + z1_2)
    
    # BS symm for both
    c5_r, d5_r = BS_symmetric(a5_r, b5_r, bs5)
    c5_g, d5_g = BS_symmetric(a5_g, b5_g, bs5)
    
    # Common PM
    b6_r, _ = (building_block(c5_r, phase2_1, phase2_2, z2_1, eta2, theta2)).VRS_propagation(z2_2)
    b6_g, _ = (building_block(c5_g, phase2_1, phase2_2, z2_1, eta2, theta2)).VRS_propagation(z2_2)
    
    # BS symm for both
    c6_r, d6_r = BS_symmetric(a6_r, b6_r, bs6)
    c6_g, d6_g = BS_symmetric(a6_g, b6_g, bs6)
    
    # Common objective lens
    det_2_r = VCZT_objective_lens(c6_r, r, f, xout, yout)
    det_2_g = VCZT_objective_lens(c6_g, r, f, xout, yout)
    
    # Mid space:
    # Common PM
    a7_r, _ = (building_block(d4_r, phase4_1, phase4_2, z4_1, eta4, theta4)).VRS_propagation(z4_2)
    a7_g, _ = (building_block(d4_g, phase4_1, phase4_2, z4_1, eta4, theta4)).VRS_propagation(z4_2)
    a8_r, _ = d5_r.VRS_propagation(z4_1 + z4_2)
    a8_g, _ = d5_g.VRS_propagation(z4_1 + z4_2)
    a9_r, _ = d6_r.VRS_propagation(z4_1 + z4_2)
    a9_g, _ = d6_g.VRS_propagation(z4_1 + z4_2)
    
    # 3rd row:
    # SI port 0 for red
    c7_r, d7_r = BS_symmetric_SI(a7_r, bs7, port=0)
    # BS sym for green
    c7_g, d7_g = BS_symmetric(a7_g, ls6, bs7)
    
    # Common prop
    b8_r, _ = c7_r.VRS_propagation(z1_1 + z1_2)
    b8_g, _ = c7_g.VRS_propagation(z1_1 + z1_2)
    
    # BS sym for both
    c8_r, d8_r = BS_symmetric(a8_r, b8_r, bs8)
    c8_g, d8_g = BS_symmetric(a8_g, b8_g, bs8)
    
    # Common prop
    b9_r, _ = c8_r.VRS_propagation(z2_1 + z2_2)
    b9_g, _ = c8_g.VRS_propagation(z2_1 + z2_2)
    
    # BS sym for both
    c9_r, d9_r = BS_symmetric(a9_r, b9_r, bs9)
    c9_g, d9_g = BS_symmetric(a9_g, b9_g, bs9)
    
    # Common objective lens
    det_3_r = VCZT_objective_lens(c9_r, r, f, xout, yout)
    det_3_g = VCZT_objective_lens(c9_g, r, f, xout, yout)
    
    # Detector row:
    det_4_r = VCZT_objective_lens(d7_r, r, f, xout, yout)
    det_4_g = VCZT_objective_lens(d7_g, r, f, xout, yout)
    
    det_5_r = VCZT_objective_lens(d8_r, r, f, xout, yout)
    det_5_g = VCZT_objective_lens(d8_g, r, f, xout, yout)
    
    det_6_r = VCZT_objective_lens(d9_r, r, f, xout, yout)
    det_6_g = VCZT_objective_lens(d9_g, r, f, xout, yout)    
    
    # Array of detector information
    i1_ex = jnp.abs(det_1_g.Ex)**2 + jnp.abs(det_1_g.Ey)**2
    i2_ex = jnp.abs(det_2_g.Ex)**2 + jnp.abs(det_2_g.Ey)**2
    i3_ex = jnp.abs(det_3_g.Ex)**2 + jnp.abs(det_3_g.Ey)**2
    i4_ex = jnp.abs(det_4_g.Ex)**2 + jnp.abs(det_4_g.Ey)**2
    i5_ex = jnp.abs(det_5_g.Ex)**2 + jnp.abs(det_5_g.Ey)**2
    i6_ex = jnp.abs(det_6_g.Ex)**2 + jnp.abs(det_6_g.Ey)**2
    
    i1_dep = jnp.abs(det_1_r.Ex)**2 + jnp.abs(det_1_r.Ey)**2
    i2_dep = jnp.abs(det_2_r.Ex)**2 + jnp.abs(det_2_r.Ey)**2
    i3_dep = jnp.abs(det_3_r.Ex)**2 + jnp.abs(det_3_r.Ey)**2
    i4_dep = jnp.abs(det_4_r.Ex)**2 + jnp.abs(det_4_r.Ey)**2
    i5_dep = jnp.abs(det_5_r.Ex)**2 + jnp.abs(det_5_r.Ey)**2
    i6_dep = jnp.abs(det_6_r.Ex)**2 + jnp.abs(det_6_r.Ey)**2
    
    # Array with intensities
    iex = jnp.stack([i1_ex/jnp.sum(i1_ex), i2_ex/jnp.sum(i2_ex), i3_ex/jnp.sum(i3_ex), i4_ex/jnp.sum(i4_ex), i5_ex/jnp.sum(i5_ex), i6_ex/jnp.sum(i6_ex)])
    idep = jnp.stack([i1_dep/jnp.sum(i1_dep), i2_dep/jnp.sum(i2_dep), i3_dep/jnp.sum(i3_dep), i4_dep/jnp.sum(i4_dep), i5_dep/jnp.sum(i5_dep), i6_dep/jnp.sum(i6_dep)])                        
    
    # Resulting STED-like beam
    i_eff = vectorized_fluorophores(iex, idep)
    
    return i_eff

def hybrid_setup_fixed_slms(ls1, ls2, ls3, ls4, ls5, ls6, parameters, fixed_params, distance_offset = 8.9):
    """
    ++++++++
    Large-scale setup with fixed phase masks at random positions for pure tpology discovery. 
    
    
    For Dorn, Quabis, Leuchs: ls1, ls2, ls3, ls4, ls5, ls6 (650 nm)
    ++++++++
    
    Scheme:
            ls1                       ls2                      ls3
             |                         |                        |
             |                         |                        |
             v                         v                        v    
    ls4 --> [BS] --> [PM #1] - z1 -> [BS] -------- z2 -------> [BS] ----> OL --> Detector 1
             |                         |                        |
             |                         |                     [PM #3]
             |                         |                        |
             z3                        z3                       z3
             |                         |                        |
             v                         v                        v    
    ls5 --> [BS] -------- z1 -------> [BS] --> [PM #2] - z2 -> [BS] ----> OL --> Detector 2
             |                         |                        |
             |                         |                        |
          [PM #4]                      |                        |
             |                         |                        |
             z4                        z4                       z4     
             |                         |                        |
             v                         v                        v      
    ls6 --> [BS] -------- z1 -------> [BS] ------- z2 -------> [BS] ----> OL --> Detector 3
             |                         |                        |        
             |                         |                        |
             v                         v                        v    
             OL                        OL                       OL
             |                         |                        |
             v                         v                        v    
           Detector 4                Detector 5              Detector 6
    
    
    Parameters: 
        ls1, ls2, ..., ls6 (PolarizedLightSource)
        parameters (jnp.array): parameters to pass to the optimizer
            BS ratios: [bs1, bs2, bs3, bs4, bs5, bs6, bs7, bs8, bs9]
            Distances: [z1, z2, z3, z4]
        fixed_params (jnp.array): parameters to maintain fixed during optimization [r, f, xout, yout, PM1, PM2, PM3 and PM4]; 
        that is radius and focal length of the objective lens, XY out and phase masks 1-4.
        distance_offset (float): From get_VRS_minimum() estimate the 'offset'.
    
    Returns:
        i1, i2, i3, i4, i5 (jnp.array): Detected intensities
    """
    # Get fixed params:
    r=fixed_params[0]
    f=fixed_params[1]
    xout=fixed_params[2]
    yout=fixed_params[3]
    
    # Dorn, Quabis, Leuchs PM:
    phase1_1 = fixed_params[4]
    phase1_2 = fixed_params[5]
    # STED spiral PM
    phase2_1 = fixed_params[6]
    phase2_2 = fixed_params[6]
    # Forked grating PM
    phase3_1 = fixed_params[7]
    phase3_2 = fixed_params[7]
    # Ladder grating PM
    phase4_1 = fixed_params[8]
    phase4_2 = fixed_params[9]
    
    # Distances
    z1_1 = (jnp.abs(parameters[0]) * 100 + distance_offset)*cm
    z1_2 = (jnp.abs(parameters[1]) * 100 + distance_offset)*cm
    z2_1 = (jnp.abs(parameters[2]) * 100 + distance_offset)*cm
    z2_2 = (jnp.abs(parameters[3]) * 100 + distance_offset)*cm
    z3_1 = (jnp.abs(parameters[4]) * 100 + distance_offset)*cm
    z3_2 = (jnp.abs(parameters[5]) * 100 + distance_offset)*cm
    z4_1 = (jnp.abs(parameters[6]) * 100 + distance_offset)*cm
    z4_2 = (jnp.abs(parameters[7]) * 100 + distance_offset)*cm
    
    # Beam splitter ratios (theta: raw input from (0,1) to (-pi, pi)) 
    bs1 = parameters[8]* 2*jnp.pi - jnp.pi
    bs2 = parameters[9]* 2*jnp.pi - jnp.pi
    bs3 = parameters[10]* 2*jnp.pi - jnp.pi
    bs4 = parameters[11]* 2*jnp.pi - jnp.pi
    bs5 = parameters[12]* 2*jnp.pi - jnp.pi
    bs6 = parameters[13]* 2*jnp.pi - jnp.pi
    bs7 = parameters[14]* 2*jnp.pi - jnp.pi
    bs8 = parameters[15]* 2*jnp.pi - jnp.pi
    bs9 = parameters[16]* 2*jnp.pi - jnp.pi

    # LCDs:
    eta1   = parameters[17]* 2*jnp.pi - jnp.pi
    theta1 = parameters[18]* 2*jnp.pi - jnp.pi
    eta2   = parameters[19]* 2*jnp.pi - jnp.pi
    theta2 = parameters[20]* 2*jnp.pi - jnp.pi
    eta3   = parameters[21]* 2*jnp.pi - jnp.pi
    theta3 = parameters[22]* 2*jnp.pi - jnp.pi
    eta4   = parameters[23]* 2*jnp.pi - jnp.pi
    theta4 = parameters[24]* 2*jnp.pi - jnp.pi
    
    # 1st row
    c1, d1 = BS_symmetric(ls1, ls4, bs1)
    b2, _ = (building_block(c1, phase1_1, phase1_2, z1_1, eta1, theta1)).VRS_propagation(z1_2)
    c2, d2 = BS_symmetric(ls2, b2, bs2)
    b3, _ = c2.VRS_propagation(z2_1 + z2_2)
    c3, d3 = BS_symmetric(ls3, b3, bs3)
    det_1 = VCZT_objective_lens(c3, r, f, xout, yout)
    
    # Mid space:
    a5, _ = d2.VRS_propagation(z3_1 + z3_2)
    a6, _ = (building_block(d3, phase3_1, phase3_2, z3_1, eta3, theta3)).VRS_propagation(z3_2)
    
    # 2nd row
    c4, d4 = BS_symmetric(d1, ls5, bs4)
    b5, _ = c4.VRS_propagation(z1_1 + z1_2)
    c5, d5 = BS_symmetric(a5, b5, bs5)
    b6, _ = (building_block(c5, phase2_1, phase2_2, z2_1, eta2, theta2)).VRS_propagation(z2_2)
    c6, d6 = BS_symmetric(a6, b6, bs6)
    det_2 = VCZT_objective_lens(c6, r, f, xout, yout)
    
    # Mid space:
    a7, _ = (building_block(d4, phase4_1, phase4_2, z4_1, eta4, theta4)).VRS_propagation(z4_2)
    a8, _ = d5.VRS_propagation(z4_1 + z4_2)
    a9, _ = d6.VRS_propagation(z4_1 + z4_2)
    
    # 3rd row
    c7, d7 = BS_symmetric(a7, ls6, bs7)
    b8, _ = c7.VRS_propagation(z1_1 + z1_2)
    c8, d8 = BS_symmetric(a8, b8, bs8)
    b9, _ = c8.VRS_propagation(z2_1 + z2_2)
    c9, d9 = BS_symmetric(a9, b9, bs9)
    det_3 = VCZT_objective_lens(c9, r, f, xout, yout)
    
    # Detector row:
    det_4 = VCZT_objective_lens(d7, r, f, xout, yout)
    
    det_5 = VCZT_objective_lens(d8, r, f, xout, yout)
    
    det_6 = VCZT_objective_lens(d9, r, f, xout, yout)
    
    # Array of detector information
    detector_array = [det_1, det_2, det_3, det_4, det_5, det_6]
    
    i1 = jnp.abs(det_1.Ez)**2
    i2 = jnp.abs(det_2.Ez)**2
    i3 = jnp.abs(det_3.Ez)**2
    i4 = jnp.abs(det_4.Ez)**2
    i5 = jnp.abs(det_5.Ez)**2
    i6 = jnp.abs(det_6.Ez)**2
    
    # Array with specific z-intensities
    intensities = jnp.stack([i1, i2, i3, i4, i5, i6])
    
    return intensities, detector_array

def hybrid_setup_sharp_focus(ls1, ls2, ls3, ls4, ls5, ls6, parameters, fixed_params, distance_offset = 8.9):
    """
    ++++++++
    For Dorn, Quabis, Leuchs benchmark: ls1, ls2, ls3, ls4, ls5, ls6 (650 nm)
    ++++++++
    
    Scheme:
            ls1                       ls2                      ls3
             |                         |                        |
             |                         |                        |
             v                         v                        v    
    ls4 --> [BS] --> [BB 1] -- z1 --> [BS] --- z2_1 + z2_2 --> [BS] --- z3_1 + z3_2 ---> OL --> Detector 1
             |                         |                        |
             |                         z4                       z4             
             |                         |                        |
             v                         v                        v    
    ls5 --> [BS] --> z1_1 + z1_2 ---> [BS] --> [BB 2]-- z2 --> [BS] --- z3_1 + z3_2 ---> OL --> Detector 2
             |                         |                        |
             |                         z5                       z5             
             |                         |                        |
             v                         v                        v      
    ls6 --> [BS] --- z1_1 + z1_2 ---> [BS] --- z2_1 + z2_2 --> [BS] --> [BB 3] -- z3 --> OL --> Detector 3
             |                         |                        |        
             |                         |                        |
             v                         v                        v    
             OL                        OL                       OL
             |                         |                        |
             v                         v                        v    
           Detector 4                Detector 5              Detector 6
    
    
    Parameters: 
        ls1, ls2, ..., ls6 (PolarizedLightSource)
        parameters (jnp.array): parameters to pass to the optimizer
            BB 1: [phase1_1, phase1_2, eta1, theta1, z1_1, z1_2]
            BB 2: [phase2_1, phase2_2, eta2, theta2, z2_1, z2_2] 
            BB 3: [phase3_1, phase3_2, eta3, theta3, z3_1, z3_2]
            BS ratios: [bs1, bs2, bs3, bs4, bs5, bs6, bs7, bs8, bs9]
            Extra distances: [z4, z5]
        fixed_params (jnp.array): parameters to maintain fixed during optimization [r, f, xout and yout]; that is radius and focal length of the objective lens.
        distance_offset (float): From get_VRS_minimum() estimate the 'offset'.
    
    Returns:
        i1, i2, i3, i4, i5 (jnp.array): Detected intensities
    """
    # Get fixed params:
    r=fixed_params[0]
    f=fixed_params[1]
    xout=fixed_params[2]
    yout=fixed_params[3]
    
    # Extract params:
    # 1st bulding block:
    phase1_1 = parameters[0]* 2*jnp.pi - jnp.pi
    phase1_2 = parameters[1]* 2*jnp.pi - jnp.pi
    eta1 = parameters[2]* 2*jnp.pi - jnp.pi
    theta1 = parameters[3]* 2*jnp.pi - jnp.pi
    z1_1 = (jnp.abs(parameters[4]) * 100 + distance_offset)*cm
    z1_2 = (jnp.abs(parameters[5]) * 100 + distance_offset)*cm
    
    # 2nd building block:
    phase2_1 = parameters[6]* 2*jnp.pi - jnp.pi
    phase2_2 = parameters[7]* 2*jnp.pi - jnp.pi
    eta2 = parameters[8]* 2*jnp.pi - jnp.pi
    theta2 = parameters[9]* 2*jnp.pi - jnp.pi
    z2_1 = (jnp.abs(parameters[10]) * 100 + distance_offset)*cm
    z2_2 = (jnp.abs(parameters[11]) * 100 + distance_offset)*cm
    
    # 3rd building block:
    phase3_1 = parameters[12]* 2*jnp.pi - jnp.pi
    phase3_2 = parameters[13]* 2*jnp.pi - jnp.pi
    eta3 = parameters[14]* 2*jnp.pi - jnp.pi
    theta3 = parameters[15]* 2*jnp.pi - jnp.pi
    z3_1 = (jnp.abs(parameters[16]) * 100 + distance_offset)*cm
    z3_2 = (jnp.abs(parameters[17]) * 100 + distance_offset)*cm
    
    # Beam splitter ratios (theta: raw input from (0,1) to (-pi, pi)) 
    bs1 = parameters[18]* 2*jnp.pi - jnp.pi
    bs2 = parameters[19]* 2*jnp.pi - jnp.pi
    bs3 = parameters[20]* 2*jnp.pi - jnp.pi
    bs4 = parameters[21]* 2*jnp.pi - jnp.pi
    bs5 = parameters[22]* 2*jnp.pi - jnp.pi
    bs6 = parameters[23]* 2*jnp.pi - jnp.pi
    bs7 = parameters[24]* 2*jnp.pi - jnp.pi
    bs8 = parameters[25]* 2*jnp.pi - jnp.pi
    bs9 = parameters[26]* 2*jnp.pi - jnp.pi
    
    # Extra distances:
    z4 = (jnp.abs(parameters[27]) * 100 + distance_offset)*cm
    z5 = (jnp.abs(parameters[28]) * 100 + distance_offset)*cm
    
    # 1st row
    c1, d1 = BS_symmetric(ls1, ls4, bs1)
    b2, _ = (building_block(c1, phase1_1, phase1_2, z1_1, eta1, theta1)).VRS_propagation(z1_2)
    c2, d2 = BS_symmetric(ls2, b2, bs2)
    b3, _ = c2.VRS_propagation(z2_1+z2_2)
    c3, d3 = BS_symmetric(ls3, b3, bs3)
    b_det1, _ = c3.VRS_propagation(z3_1+z3_2)
    det_1 = VCZT_objective_lens(b_det1, r, f, xout, yout)
    
    # Mid space:
    a5, _ = d2.VRS_propagation(z4)
    a6, _ = d3.VRS_propagation(z4)
    
    # 2nd row
    c4, d4 = BS_symmetric(d1, ls5, bs4)
    b5, _ = c4.VRS_propagation(z1_1 + z1_2)
    c5, d5 = BS_symmetric(a5, b5, bs5)
    b6, _ = (building_block(c5, phase2_1, phase2_2, z2_1, eta2, theta2)).VRS_propagation(z2_2)
    c6, d6 = BS_symmetric(a6, b6, bs6)
    b_det2, _ = c6.VRS_propagation(z3_1 + z3_2)
    det_2 = VCZT_objective_lens(b_det2, r, f, xout, yout)
    
    # Mid space:
    a8, _ = d5.VRS_propagation(z5)
    a9, _ = d6.VRS_propagation(z5)
    
    # 3rd row
    c7, d7 = BS_symmetric(d4, ls6, bs7)
    b8, _ = c7.VRS_propagation(z1_1 + z1_2)
    c8, d8 = BS_symmetric(a8, b8, bs8)
    b9, _ = c8.VRS_propagation(z2_1 + z2_2)
    c9, d9 = BS_symmetric(a9, b9, bs9)
    b_det3, _ = (building_block(c9, phase3_1, phase3_2, z3_1, eta3, theta3)).VRS_propagation(z3_2)
    det_3 = VCZT_objective_lens(b_det3, r, f, xout, yout)
    
    # Detector row:
    det_4 = VCZT_objective_lens(d7, r, f, xout, yout)
    
    det_5 = VCZT_objective_lens(d8, r, f, xout, yout)
    
    det_6 = VCZT_objective_lens(d9, r, f, xout, yout)
    
    # Array of detector information
    detector_array = [det_1, det_2, det_3, det_4, det_5, det_6]
    
    i1 = jnp.abs(det_1.Ez)**2
    i2 = jnp.abs(det_2.Ez)**2
    i3 = jnp.abs(det_3.Ez)**2
    i4 = jnp.abs(det_4.Ez)**2
    i5 = jnp.abs(det_5.Ez)**2
    i6 = jnp.abs(det_6.Ez)**2
    
    # Array with specific z-intensities
    intensities = jnp.stack([i1, i2, i3, i4, i5, i6])
    
    return intensities, detector_array

def hybrid_setup_fluorophores(ls1, ls2, ls3, ls4, ls5, ls6, parameters, fixed_params, distance_offset = 8.9):
    """
    [hybrid_setup with fluorophores] - STED-like
    
    Scheme:
            ls1                       ls2                      ls3
             |                         |                        |
             |                         |                        |
             v                         v                        v    
    ls4 --> [BS] --> [BB 1] -- z1 --> [BS] --- z2_1 + z2_2 --> [BS] --- z3_1 + z3_2 ---> OL --> Detector 1
             |                         |                        |
             |                         z4                       z4             
             |                         |                        |
             v                         v                        v    
    ls5 --> [BS] --> z1_1 + z1_2 ---> [BS] --> [BB 2]-- z2 --> [BS] --- z3_1 + z3_2 ---> OL --> Detector 2
             |                         |                        |
             |                         z5                       z5             
             |                         |                        |
             v                         v                        v      
    ls6 --> [BS] --- z1_1 + z1_2 ---> [BS] --- z2_1 + z2_2 --> [BS] --> [BB 3] -- z3 --> OL --> Detector 3
             |                         |                        |        
             |                         |                        |
             v                         v                        v    
             OL                        OL                       OL
             |                         |                        |
             v                         v                        v    
           Detector 4                Detector 5              Detector 6
    
    
    ls1, ls2, ..., ls6 (PolarizedLightSource)
    parameters (jnp.array): parameters to pass to the optimizer
    
        BB 1: [_,_, eta1, theta1, z1_1, z1_2]
        BB 2: [_,_, eta2, theta2, z2_1, z2_2] 
        BB 3: [phase3_1, phase3_2, eta3, theta3, z3_1, z3_2]
        BS ratios: [bs1, bs2, bs3, bs4, bs5, bs6, bs7, bs8, bs9]
        Extra distances: z4, z5
        
    fixed_params (jnp.array): parameters to maintain fixed during optimization [r, f, xout and yout]; that is radius and focal length of the objective lens.
    
    Returns:
        intensities (jnp.array): stack of i1, i2, i3, i4, i5 effective detected intensities
    """
    
    # Get fixed params:
    r=fixed_params[0]
    f=fixed_params[1]
    xout=fixed_params[2]
    yout=fixed_params[3]
    
    # Extract params:
    # 1st bulding block:
    phase1_1 = parameters[0]* 2*jnp.pi - jnp.pi
    phase1_2 = parameters[1]* 2*jnp.pi - jnp.pi
    eta1 = parameters[2]* 2*jnp.pi - jnp.pi
    theta1 = parameters[3]* 2*jnp.pi - jnp.pi
    z1_1 = (jnp.abs(parameters[4]) * 100 + distance_offset)*cm
    z1_2 = (jnp.abs(parameters[5]) * 100 + distance_offset)*cm
    
    # 2nd building block:
    phase2_1 = parameters[6]* 2*jnp.pi - jnp.pi
    phase2_2 = parameters[7]* 2*jnp.pi - jnp.pi
    eta2 = parameters[8]* 2*jnp.pi - jnp.pi
    theta2 = parameters[9]* 2*jnp.pi - jnp.pi
    z2_1 = (jnp.abs(parameters[10]) * 100 + distance_offset)*cm
    z2_2 = (jnp.abs(parameters[11]) * 100 + distance_offset)*cm
    
    # 3rd building block:
    phase3_1 = parameters[12]* 2*jnp.pi - jnp.pi
    phase3_2 = parameters[13]* 2*jnp.pi - jnp.pi
    eta3 = parameters[14]* 2*jnp.pi - jnp.pi
    theta3 = parameters[15]* 2*jnp.pi - jnp.pi
    z3_1 = (jnp.abs(parameters[16]) * 100 + distance_offset)*cm
    z3_2 = (jnp.abs(parameters[17]) * 100 + distance_offset)*cm
    
    # Beam splitter ratios (theta: raw input from (0,1) to (-pi, pi)) 
    bs1 = parameters[18]* 2*jnp.pi - jnp.pi
    bs2 = parameters[19]* 2*jnp.pi - jnp.pi
    bs3 = parameters[20]* 2*jnp.pi - jnp.pi
    bs4 = parameters[21]* 2*jnp.pi - jnp.pi
    bs5 = parameters[22]* 2*jnp.pi - jnp.pi
    bs6 = parameters[23]* 2*jnp.pi - jnp.pi
    bs7 = parameters[24]* 2*jnp.pi - jnp.pi
    bs8 = parameters[25]* 2*jnp.pi - jnp.pi
    bs9 = parameters[26]* 2*jnp.pi - jnp.pi
    
    # Extra distances:
    z4 = (jnp.abs(parameters[27]) * 100 + distance_offset)*cm
    z5 = (jnp.abs(parameters[28]) * 100 + distance_offset)*cm
    
    # 1st row: _r (red light), _g (green light); BS (a, b) -> c d 
    c1_r, d1_r = BS_symmetric_SI(ls1, bs1, port=0)
    c1_g, d1_g = BS_symmetric_SI(ls4, bs1, port=1)
    
    b2_r, _ = (building_block(c1_r, phase1_1, phase1_2, z1_1, eta1, theta1)).VRS_propagation(z1_2)
    b2_g, _ = (building_block(c1_g, phase1_1, phase1_2, z1_1, eta1, theta1)).VRS_propagation(z1_2)

    c2_r, d2_r = BS_symmetric_SI(b2_r, bs2, port=1)
    c2_g, d2_g = BS_symmetric(ls2, b2_g, bs2)
    
    b3_r, _ = c2_r.VRS_propagation(z2_1+z2_2)
    b3_g, _ = c2_g.VRS_propagation(z2_1+z2_2)
    
    c3_r, d3_r = BS_symmetric(ls3, b3_r, bs3)
    c3_g, d3_g = BS_symmetric_SI(b3_g, bs3, port=1)
    
    b_det1_r, _ = c3_r.VRS_propagation(z3_1+z3_2)
    b_det1_g, _ = c3_g.VRS_propagation(z3_1+z3_2)
    
    det_1_r = VCZT_objective_lens(b_det1_r, r, f, xout, yout)
    det_1_g = VCZT_objective_lens(b_det1_g, r, f, xout, yout)
    
    # Mid space:
    a5_r, _ = d2_r.VRS_propagation(z4)
    a5_g, _ = d2_g.VRS_propagation(z4)
    a6_r, _ = d3_r.VRS_propagation(z4)
    a6_g, _ = d3_g.VRS_propagation(z4)
    
    # 2nd row
    c4_r, d4_r = BS_symmetric(d1_r, ls5, bs4)
    c4_g, d4_g = BS_symmetric_SI(d1_g, bs4, port=0)
    
    b5_r, _ = c4_r.VRS_propagation(z1_1 + z1_2)
    b5_g, _ = c4_g.VRS_propagation(z1_1 + z1_2)
    
    c5_r, d5_r = BS_symmetric(a5_r, b5_r, bs5)
    c5_g, d5_g = BS_symmetric(a5_g, b5_g, bs5)
    
    b6_r, _ = (building_block(c5_r, phase2_1, phase2_2, z2_1, eta2, theta2)).VRS_propagation(z2_2)
    b6_g, _ = (building_block(c5_g, phase2_1, phase2_2, z2_1, eta2, theta2)).VRS_propagation(z2_2)
    
    c6_r, d6_r = BS_symmetric(a6_r, b6_r, bs6)
    c6_g, d6_g = BS_symmetric(a6_g, b6_g, bs6)
    
    b_det2_r, _ = c6_r.VRS_propagation(z3_1 + z3_2)
    b_det2_g, _ = c6_g.VRS_propagation(z3_1 + z3_2)
    
    det_2_r = VCZT_objective_lens(b_det2_r, r, f, xout, yout)
    det_2_g = VCZT_objective_lens(b_det2_g, r, f, xout, yout)
    
    # Mid space:
    a8_r, _ = d5_r.VRS_propagation(z5)
    a8_g, _ = d5_g.VRS_propagation(z5)
    
    a9_r, _ = d6_r.VRS_propagation(z5)
    a9_g, _ = d6_g.VRS_propagation(z5)
    
    # 3rd row
    c7_r, d7_r = BS_symmetric_SI(d4_r, bs7, port=0)
    c7_g, d7_g = BS_symmetric(d4_g, ls6, bs7)
    
    b8_r, _ = c7_r.VRS_propagation(z1_1 + z1_2)
    b8_g, _ = c7_g.VRS_propagation(z1_1 + z1_2)
    
    c8_r, d8_r = BS_symmetric(a8_r, b8_r, bs8)
    c8_g, d8_g = BS_symmetric(a8_g, b8_g, bs8)
    
    b9_r, _ = c8_r.VRS_propagation(z2_1 + z2_2)
    b9_g, _ = c8_g.VRS_propagation(z2_1 + z2_2)
    
    c9_r, d9_r = BS_symmetric(a9_r, b9_r, bs9)
    c9_g, d9_g = BS_symmetric(a9_g, b9_g, bs9)
    
    b_det3_r, _ = (building_block(c9_r, phase3_1, phase3_2, z3_1, eta3, theta3)).VRS_propagation(z3_2)
    b_det3_g, _ = (building_block(c9_g, phase3_1, phase3_2, z3_1, eta3, theta3)).VRS_propagation(z3_2)
    
    det_3_r = VCZT_objective_lens(b_det3_r, r, f, xout, yout)
    det_3_g = VCZT_objective_lens(b_det3_g, r, f, xout, yout)
    
    # Detector row:
    det_4_r = VCZT_objective_lens(d7_r, r, f, xout, yout)
    det_4_g = VCZT_objective_lens(d7_g, r, f, xout, yout)
    
    det_5_r = VCZT_objective_lens(d8_r, r, f, xout, yout)
    det_5_g = VCZT_objective_lens(d8_g, r, f, xout, yout)
    
    det_6_r = VCZT_objective_lens(d9_r, r, f, xout, yout)
    det_6_g = VCZT_objective_lens(d9_g, r, f, xout, yout)
    
    # Array of detector information
    # detector_array = [det_1, det_2, det_3, det_4, det_5, det_6]
    i1_ex = jnp.abs(det_1_g.Ex)**2 + jnp.abs(det_1_g.Ey)**2# + jnp.abs(det_1_g.Ez)**2
    i2_ex = jnp.abs(det_2_g.Ex)**2 + jnp.abs(det_2_g.Ey)**2# + jnp.abs(det_2_g.Ez)**2
    i3_ex = jnp.abs(det_3_g.Ex)**2 + jnp.abs(det_3_g.Ey)**2# + jnp.abs(det_3_g.Ez)**2
    i4_ex = jnp.abs(det_4_g.Ex)**2 + jnp.abs(det_4_g.Ey)**2# + jnp.abs(det_4_g.Ez)**2
    i5_ex = jnp.abs(det_5_g.Ex)**2 + jnp.abs(det_5_g.Ey)**2# + jnp.abs(det_5_g.Ez)**2
    i6_ex = jnp.abs(det_6_g.Ex)**2 + jnp.abs(det_6_g.Ey)**2# + jnp.abs(det_6_g.Ez)**2
    
    i1_dep = jnp.abs(det_1_r.Ex)**2 + jnp.abs(det_1_r.Ey)**2# + jnp.abs(det_1_r.Ez)**2
    i2_dep = jnp.abs(det_2_r.Ex)**2 + jnp.abs(det_2_r.Ey)**2# + jnp.abs(det_2_r.Ez)**2
    i3_dep = jnp.abs(det_3_r.Ex)**2 + jnp.abs(det_3_r.Ey)**2# + jnp.abs(det_3_r.Ez)**2
    i4_dep = jnp.abs(det_4_r.Ex)**2 + jnp.abs(det_4_r.Ey)**2# + jnp.abs(det_4_r.Ez)**2
    i5_dep = jnp.abs(det_5_r.Ex)**2 + jnp.abs(det_5_r.Ey)**2# + jnp.abs(det_5_r.Ez)**2
    i6_dep = jnp.abs(det_6_r.Ex)**2 + jnp.abs(det_6_r.Ey)**2# + jnp.abs(det_6_r.Ez)**2
    
    # Array with intensities
    iex = jnp.stack([i1_ex/jnp.sum(i1_ex), i2_ex/jnp.sum(i2_ex), i3_ex/jnp.sum(i3_ex), i4_ex/jnp.sum(i4_ex), i5_ex/jnp.sum(i5_ex), i6_ex/jnp.sum(i6_ex)])
    idep = jnp.stack([i1_dep/jnp.sum(i1_dep), i2_dep/jnp.sum(i2_dep), i3_dep/jnp.sum(i3_dep), i4_dep/jnp.sum(i4_dep), i5_dep/jnp.sum(i5_dep), i6_dep/jnp.sum(i6_dep)])                       
    # Resulting STED-like beam
    i_eff = vectorized_fluorophores(iex, idep)
    
    return i_eff


def six_times_six_ansatz(ls1, ls2, ls3, ls4, ls5, ls6, ls7, ls8, ls9, ls10, ls11, ls12, parameters, fixed_params, distance_offset = 7.6):
    """
    ++++++++
    6x6 grid:
        12 light sources,
        36 BS,
        2 super slms (i.e., 4 slms with fixed PM),
        2 waveplates
        
    Look for Dorn, Quabis, Leuchs benchmark: ls1-ls12 (635 nm)
    ++++++++
   
    Parameters: 
        ls1, ls2, ..., ls12 (PolarizedLightSource)
        parameters (jnp.array): parameters to pass to the optimizer
            BS ratios: [bs1 - bs36]
            Distances: [z1 - z12]
            Waveplate angles: [eta1, theta1, eta2, theta2]
        fixed_params (jnp.array): parameters to maintain fixed during optimization [r, f, xout, yout, PM1, PM2, PM3 and PM4]; 
        that is radius and focal length of the objective lens, XY out and phase masks 1-4.
        distance_offset (float): From get_VRS_minimum() estimate the 'offset'.
    
    Returns:
        i1, i2, i3, i4, i5, i6 (jnp.array): Detected intensities
    """
    # Get fixed params:
    r=fixed_params[0]
    f=fixed_params[1]
    xout=fixed_params[2]
    yout=fixed_params[3]
    
    # Dorn, Quabis, Leuchs PM:
    phase1_1 = fixed_params[4]
    phase1_2 = fixed_params[5]
    # Ladder grating
    phase2_1 = fixed_params[6]
    phase2_2 = fixed_params[7]
    
    # Distances
    z1 = (jnp.abs(parameters[0]) * 100 + distance_offset)*cm
    z2 = (jnp.abs(parameters[1]) * 100 + distance_offset)*cm
    z3 = (jnp.abs(parameters[2]) * 100 + distance_offset)*cm
    z4 = (jnp.abs(parameters[3]) * 100 + distance_offset)*cm
    z5 = (jnp.abs(parameters[4]) * 100 + distance_offset)*cm
    z6 = (jnp.abs(parameters[5]) * 100 + distance_offset)*cm
    z7 = (jnp.abs(parameters[6]) * 100 + distance_offset)*cm
    z8 = (jnp.abs(parameters[7]) * 100 + distance_offset)*cm
    z9 = (jnp.abs(parameters[8]) * 100 + distance_offset)*cm
    z10 = (jnp.abs(parameters[9]) * 100 + distance_offset)*cm
    z11 = (jnp.abs(parameters[10]) * 100 + distance_offset)*cm
    z12 = (jnp.abs(parameters[11]) * 100 + distance_offset)*cm
    
    # Beam splitter ratios (theta: raw input from (0,1) to (-pi, pi)) 
    bs1 = parameters[12]* 2*jnp.pi - jnp.pi
    bs2 = parameters[13]* 2*jnp.pi - jnp.pi
    bs3 = parameters[14]* 2*jnp.pi - jnp.pi
    bs4 = parameters[15]* 2*jnp.pi - jnp.pi
    bs5 = parameters[16]* 2*jnp.pi - jnp.pi
    bs6 = parameters[17]* 2*jnp.pi - jnp.pi
    bs7 = parameters[18]* 2*jnp.pi - jnp.pi
    bs8 = parameters[19]* 2*jnp.pi - jnp.pi
    bs9 = parameters[20]* 2*jnp.pi - jnp.pi
    bs10 = parameters[21]* 2*jnp.pi - jnp.pi
    bs11 = parameters[22]* 2*jnp.pi - jnp.pi
    bs12 = parameters[23]* 2*jnp.pi - jnp.pi
    bs13 = parameters[24]* 2*jnp.pi - jnp.pi
    bs14 = parameters[25]* 2*jnp.pi - jnp.pi
    bs15 = parameters[26]* 2*jnp.pi - jnp.pi
    bs16 = parameters[27]* 2*jnp.pi - jnp.pi
    bs17 = parameters[28]* 2*jnp.pi - jnp.pi
    bs18 = parameters[29]* 2*jnp.pi - jnp.pi
    bs19 = parameters[30]* 2*jnp.pi - jnp.pi
    bs20 = parameters[31]* 2*jnp.pi - jnp.pi
    bs21 = parameters[32]* 2*jnp.pi - jnp.pi
    bs22 = parameters[33]* 2*jnp.pi - jnp.pi
    bs23 = parameters[34]* 2*jnp.pi - jnp.pi
    bs24 = parameters[35]* 2*jnp.pi - jnp.pi
    bs25 = parameters[36]* 2*jnp.pi - jnp.pi
    bs26 = parameters[37]* 2*jnp.pi - jnp.pi
    bs27 = parameters[38]* 2*jnp.pi - jnp.pi
    bs28 = parameters[39]* 2*jnp.pi - jnp.pi
    bs29 = parameters[40]* 2*jnp.pi - jnp.pi
    bs30 = parameters[41]* 2*jnp.pi - jnp.pi
    bs31 = parameters[42]* 2*jnp.pi - jnp.pi
    bs32 = parameters[43]* 2*jnp.pi - jnp.pi
    bs33 = parameters[44]* 2*jnp.pi - jnp.pi
    bs34 = parameters[45]* 2*jnp.pi - jnp.pi
    bs35 = parameters[46]* 2*jnp.pi - jnp.pi
    bs36 = parameters[47]* 2*jnp.pi - jnp.pi

    # LCDs:
    eta1   = parameters[48]* 2*jnp.pi - jnp.pi
    theta1 = parameters[49]* 2*jnp.pi - jnp.pi
    eta2   = parameters[50]* 2*jnp.pi - jnp.pi
    theta2 = parameters[51]* 2*jnp.pi - jnp.pi
    
    # 1st row and mid space
    c1, d1 = BS_symmetric(ls1, ls7, bs1)
    b2, _ = c1.VRS_propagation(z1)
    a7, _ = d1.VRS_propagation(z8)
    
    c2, d2 = BS_symmetric(ls2, b2, bs2)
    b3, _ = c2.VRS_propagation(z2+z3)
    a8, _ = d2.VRS_propagation(z8)
    
    c3, d3 = BS_symmetric(ls3, b3, bs3)
    b4, _ = c3.VRS_propagation(z4)
    a9, _ = d3.VRS_propagation(z8)
    
    c4, d4 = BS_symmetric(ls4, b4, bs4)
    b5, _ = c4.VRS_propagation(z5+z6)
    a10, _ = d4.VRS_propagation(z8)
    
    c5, d5 = BS_symmetric(ls5, b5, bs5)
    b6, _ = c5.VRS_propagation(z7)
    a11, _ = d5.VRS_propagation(z8)
    
    c6, d6 = BS_symmetric(ls6, b6, bs6)
    a12, _ = d6.VRS_propagation(z8)
    det_1 = VCZT_objective_lens(c6, r, f, xout, yout)
    
    # 2nd row and mid space
    c7, d7 = BS_symmetric(a7, ls8, bs7)
    b8, _ = c7.VRS_propagation(z1)
    a13, _ = d7.VRS_propagation(z9)
    
    c8, d8 = BS_symmetric(a8, b8, bs8)
    b9, _ = (building_block(c8, phase1_1, phase1_2, z2, eta1, theta1)).VRS_propagation(z3)    
    a14, _ = d8.VRS_propagation(z9)
    
    c9, d9 = BS_symmetric(a9, b9, bs9)
    b10, _ = c9.VRS_propagation(z4)
    a15, _ = d9.VRS_propagation(z9)
    
    c10, d10 = BS_symmetric(a10, b10, bs10)
    b11, _ = c10.VRS_propagation(z5+z6)
    a16, _ = d10.VRS_propagation(z9)
    
    c11, d11 = BS_symmetric(a11, b11, bs11)
    b12, _ = c11.VRS_propagation(z7)
    a17, _ = d11.VRS_propagation(z9)
    
    c12, d12 = BS_symmetric(a12, b12, bs12)
    a18, _ = d12.VRS_propagation(z9)
    det_2 = VCZT_objective_lens(c12, r, f, xout, yout)

    # 3rd row and mid space
    c13, d13 = BS_symmetric(a13, ls9, bs13)
    b14, _ = c13.VRS_propagation(z1)
    a19, _ = d13.VRS_propagation(z10)
    
    c14, d14 = BS_symmetric(a14, b14, bs14)
    b15, _ = c14.VRS_propagation(z2+z3)
    a20, _ = d14.VRS_propagation(z10)
    
    c15, d15 = BS_symmetric(a15, b15, bs15)
    b16, _ = c15.VRS_propagation(z4)
    a21, _ = d15.VRS_propagation(z10)
    
    c16, d16 = BS_symmetric(a16, b16, bs16)
    b17, _ = c16.VRS_propagation(z5+z6)
    a22, _ = d16.VRS_propagation(z10)
    
    c17, d17 = BS_symmetric(a17, b17, bs17)
    b18, _ = c17.VRS_propagation(z7)
    a23, _ = d17.VRS_propagation(z10)
    
    c18, d18 = BS_symmetric(a18, b18, bs18)
    a24, _ = d18.VRS_propagation(z10)
    det_3 = VCZT_objective_lens(c18, r, f, xout, yout)
    
    # 4th row and mid space
    c19, d19 = BS_symmetric(a19, ls10, bs19)
    b20, _ = c19.VRS_propagation(z1)
    a25, _ = d19.VRS_propagation(z11)
    
    c20, d20 = BS_symmetric(a20, b20, bs20)
    b21, _ = c20.VRS_propagation(z2+z3)
    a26, _ = d20.VRS_propagation(z11)
    
    c21, d21 = BS_symmetric(a21, b21, bs21)
    b22, _ = c21.VRS_propagation(z4)
    a27, _ = d21.VRS_propagation(z11)
    
    c22, d22 = BS_symmetric(a22, b22, bs22)
    b23, _ = (building_block(c22, phase2_1, phase2_2, z5, eta2, theta2)).VRS_propagation(z6)    
    a28, _ = d22.VRS_propagation(z11)
    
    c23, d23 = BS_symmetric(a23, b23, bs23)
    b24, _ = c23.VRS_propagation(z7)
    a29, _ = d23.VRS_propagation(z11)
    
    c24, d24 = BS_symmetric(a24, b24, bs24)
    a30, _ = d24.VRS_propagation(z11)
    det_4 = VCZT_objective_lens(c24, r, f, xout, yout)
    
    # 5th row and mid space
    c25, d25 = BS_symmetric(a25, ls11, bs25)
    b26, _ = c25.VRS_propagation(z1)
    a31, _ = d25.VRS_propagation(z12)
    
    c26, d26 = BS_symmetric(a26, b26, bs26)
    b27, _ = c26.VRS_propagation(z2+z3)
    a32, _ = d26.VRS_propagation(z12)
    
    c27, d27 = BS_symmetric(a27, b27, bs27)
    b28, _ = c27.VRS_propagation(z4)
    a33, _ = d27.VRS_propagation(z12)
    
    c28, d28 = BS_symmetric(a28, b28, bs28)
    b29, _ = c28.VRS_propagation(z5+z6)
    a34, _ = d28.VRS_propagation(z12)
    
    c29, d29 = BS_symmetric(a29, b29, bs29)
    b30, _ = c29.VRS_propagation(z7)
    a35, _ = d29.VRS_propagation(z12)
    
    c30, d30 = BS_symmetric(a30, b30, bs30)
    a36, _ = d30.VRS_propagation(z12)
    det_5 = VCZT_objective_lens(c30, r, f, xout, yout)

    # 6th row and mid space
    c31, d31 = BS_symmetric(a31, ls12, bs31)
    b32, _ = c31.VRS_propagation(z1)
    det_7 = VCZT_objective_lens(c31, r, f, xout, yout)
    
    c32, d32 = BS_symmetric(a32, b32, bs32)
    b33, _ = c32.VRS_propagation(z2+z3)
    det_8 = VCZT_objective_lens(c32, r, f, xout, yout)
    
    c33, d33 = BS_symmetric(a33, b33, bs33)
    b34, _ = c33.VRS_propagation(z4)
    det_9 = VCZT_objective_lens(c33, r, f, xout, yout)
    
    c34, d34 = BS_symmetric(a34, b34, bs34)
    b35, _ = c34.VRS_propagation(z5+z6)
    det_10 = VCZT_objective_lens(c34, r, f, xout, yout)
    
    c35, d35 = BS_symmetric(a35, b35, bs35)
    b36, _ = c35.VRS_propagation(z7)
    det_11 = VCZT_objective_lens(c35, r, f, xout, yout)

    c36, d36 = BS_symmetric(a36, b36, bs36)
    det_6 = VCZT_objective_lens(c36, r, f, xout, yout)
    det_12 = VCZT_objective_lens(d36, r, f, xout, yout)
    
    # Array of detector information
    # detector_array = [det_1, det_2, det_3, det_4, det_5, det_6, det_7, det_8, det_9, det_10, det_11, det_12]
    
    i1 = jnp.abs(det_1.Ez)**2
    i2 = jnp.abs(det_2.Ez)**2
    i3 = jnp.abs(det_3.Ez)**2
    i4 = jnp.abs(det_4.Ez)**2
    i5 = jnp.abs(det_5.Ez)**2
    i6 = jnp.abs(det_6.Ez)**2
    i7 = jnp.abs(det_7.Ez)**2
    i8 = jnp.abs(det_8.Ez)**2
    i9 = jnp.abs(det_9.Ez)**2
    i10 = jnp.abs(det_10.Ez)**2
    i11 = jnp.abs(det_11.Ez)**2
    i12 = jnp.abs(det_12.Ez)**2
    
    # Array with specific z-intensities
    intensities = jnp.stack([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12])
    
    return intensities #, detector_array





def xl_setup(ls1, ls2, parameters, fixed_params):
    """
    Optical table with a more general set-up. Building blocks are [ls, sSLM, LCD], joint by BS. 
    
    Scheme:
    [See Fig. 7a in our paper: https://doi.org/10.48550/arXiv.2310.08408].
    
    Parameters:
        ls1, ls2, (VectorizedLight): Light sources of the same type (need for interference).
        parameters (list): Parameters to pass to the optimizer for sSLM, LCD and VRS.
        parameters[0 -> 4] = alpha, phi, z, eta, theta, for 1st block.
        parameters[5 -> 9] = alpha, phi, z, eta, theta, for 2nd block.
        parameters[10 -> 14] = alpha, phi, z, eta, theta, for 3rd block.
        parameters[15 -> 19] = alpha, phi, z, eta, theta, for 4th block.
        parameters[20 -> 22] = Distance to beam splitters.
        fixed_params (jnp.array): Parameters to maintain fixed during optimization [r, f]; that is radius and focal of the objective lens.
  
    Parameters in the optimizer are from (0,1). Conversion factor here are:
    
        1. Convert (0,1) to distance in cm -> Conversion factor for (offset, 100)*cm = (offset/100, 1).
        2. Convert (0,1) to phase (in radians) -> Conversion factor (-pi, pi) + pi = (0, 2pi) = (0, 1). 
    
    + From get_VRS_minimum_z(), estimate the 'offset'. 
  
    Returns intensity (jnp.array) in the focal plane and fields at stop ends.
    """
    offset = 8.9 
    
    # Apply building blocks:
    
    # Wavelength 1:
    light_path_a, _ = (building_block(ls1, parameters[0]* 2*jnp.pi - jnp.pi, parameters[1]* 2*jnp.pi - jnp.pi, (jnp.abs(parameters[2]) * 100 + offset)*cm, parameters[3]* 2*jnp.pi - jnp.pi, parameters[4]* 2*jnp.pi - jnp.pi)).VRS_propagation((jnp.abs(parameters[20]) * 100 + offset)*cm)
    light_path_b, _ = (building_block(ls1, parameters[5]* 2*jnp.pi - jnp.pi, parameters[6]* 2*jnp.pi - jnp.pi, (jnp.abs(parameters[7]) * 100 + offset)*cm, parameters[8]* 2*jnp.pi - jnp.pi, parameters[9]* 2*jnp.pi - jnp.pi)).VRS_propagation((jnp.abs(parameters[21]) * 100 + offset)*cm)
        # Join the building blocks of equal wavelength with BS: 
    ab_reflected, ab_transmitted = BS(light_path_b, light_path_a, 0.5, 0.5, jnp.pi)
    
    # Wavelength 2
    light_path_c, _ = (building_block(ls2, parameters[10]* 2*jnp.pi - jnp.pi, parameters[11]* 2*jnp.pi - jnp.pi, (jnp.abs(parameters[12]) * 100 + offset)*cm, parameters[13]* 2*jnp.pi - jnp.pi, parameters[14]* 2*jnp.pi - jnp.pi)).VRS_propagation((jnp.abs(parameters[22]) * 100 + offset)*cm)
    light_path_d, _ = (building_block(ls2, parameters[15]* 2*jnp.pi - jnp.pi, parameters[16]* 2*jnp.pi - jnp.pi, (jnp.abs(parameters[17]) * 100 + offset)*cm, parameters[18]* 2*jnp.pi - jnp.pi, parameters[19]* 2*jnp.pi - jnp.pi)).VRS_propagation((jnp.abs(parameters[23]) * 100 + offset)*cm)
        # Join the building blocks of equal wavelength with BS: 
    cd_reflected, cd_transmitted = BS(light_path_d, light_path_c, 0.5, 0.5, jnp.pi)
    
    # Propagate to focal plane and extract the intensity
    ls1_f = VCZT_objective_lens(ab_reflected, r=fixed_params[0], f=fixed_params[1], xout=fixed_params[2], yout=fixed_params[3])
    ls2_f = VCZT_objective_lens(cd_transmitted, r=fixed_params[0], f=fixed_params[1], xout=fixed_params[2], yout=fixed_params[3])
    
    # TOTAL (3D) intensity
    i_ls1 = jnp.abs(ls1_f.Ex)**2 + jnp.abs(ls1_f.Ey)**2 + jnp.abs(ls1_f.Ez)**2
    i_ls2 = jnp.abs(ls2_f.Ex)**2 + jnp.abs(ls2_f.Ey)**2 + jnp.abs(ls2_f.Ez)**2
    
    # Resulting STED function computed for 3D:
    beta = 1 # Efficiency in the depletion
    
    i_eff = i_ls2 * (1 - beta * (1- jnp.exp(-(i_ls1/i_ls2))))

    return i_eff, ls1_f, ls2_f
    
def vSTED(excitation_beam, depletion_beam, parameters, fixed_params):
    """
    Vectorial-based STED. 
    [Ref] D. Wildanger, E. Rittweger, L. Kastrup, and S. W. Hell, Opt. Express 16, 9614-9621 (2008).
    
    Scheme: 
    STED beam ---> Modulate: sSLM (phase mask) --> VRS(z) --> High NA lens 
    Excitation beam ----------------------------------------> High NA lens 
              
    Parameters:
        excitation_beam (object): input field for the excitation.
        depletion_beam (object): input field for the depletion.
        parameters (jnp.array): parameters to pass to the optimizer [phase 1] for sSLM.
        fixed_params (jnp.array): parameters to maintain fixed during optimization [r, f, xout and yout]; that is radius and focal length of the objective lens.
           
    Returns:
        i_eff (jnp.array) effective PSF of the system, 
        i_ex (jnp.array) excitation intensity in the focal plane,
        i_dep (jnp.array) depletion intensity in the focal plane,
        ex_f (object) excitation beam in the focal plane,
        dep_f (object) depletion beam in the focal plane.
        
    Parameters in the optimizer are from (0,1). Conversion factor here are: 
        
        Convert (0,1) to phase (in radians) -> Conversion factor (-pi, pi) + pi = (0, 2pi) = (0, 1). 
    """
    # Estimate the offset via get_VRS_minimum().
    offset = 24000 #microns
    
    # Apply phase mask to depletion beam. We use only the SLM in sSLM corresponding to the input polarization state. The other is set to zero.
    dep_modulated = sSLM(depletion_beam, parameters[0]* 2 * jnp.pi - jnp.pi, jnp.zeros((2048, 2048)))
    
    # Propagate:
    dep_propagated, _ = dep_modulated.VRS_propagation(z=offset)
    
    # Propagate to focal plane and extract the intensity
    ex_f = VCZT_objective_lens(excitation_beam, r=fixed_params[0], f=fixed_params[1], xout=fixed_params[2], yout=fixed_params[3])
    dep_f = VCZT_objective_lens(dep_propagated, r=fixed_params[0], f=fixed_params[1], xout=fixed_params[2], yout=fixed_params[3])
    
    # Ir intensity
    i_ex = jnp.abs(ex_f.Ex)**2 + jnp.abs(ex_f.Ey)**2
    i_dep = jnp.abs(dep_f.Ex)**2 + jnp.abs(dep_f.Ey)**2
    
    # Resulting STED-like beam
    beta = 1 # Efficiency in the depletion
    
    i_eff = i_ex * (1 - beta * (1- jnp.exp(-(i_dep/i_ex)))) 

    return i_eff, i_ex, i_dep, ex_f, dep_f


def sharp_focus(input_field, parameters, fixed_params):
    """
    Define an optical table for sharp focus. 
    
    Illustrative scheme:

    (Ex, Ey) --> PBS --> Ex --> Modulate: SLM(alpha) --> Ex' --> PBS --> (Ex', Ey') --> Modulate: LCD(eta, theta) 
                  |                                               |                               |
                  Ey ---------> Modulate: SLM(phi) ----> Ey' -----/                          (Ex'', Ey'') --> Propagate: VRS(z) --> objective_lens(r,f)

    Parameters:
        input_field (VectorizedLight): Light to be modulated.
        parameters (list): Parameters to pass to the optimizer [alpha, phi, eta, theta, z1 and z2] for sSLM, LCD and VRS.
        fixed_params (jnp.array): Parameters to maintain fixed during optimization [r, f] that is radius and focal of the high NA objective lens.
        
    Returns VectorizedLight in the focal plane.
    
    Parameters in the optimizer are from (0,1). Conversion factor here are:
    
        1. Convert (0,1) to distance in cm -> Conversion factor for (offset, 100)*cm = (offset/100, 1).
        2. Convert (0,1) to phase (in radians) -> Conversion factor (-pi, pi) + pi = (0, 2pi) = (0, 1). 
    """
    offset = 3.8 # cm 

    # 1. Apply super-SLM:
    modulated_light = sSLM(input_field, parameters[0]* 2 * jnp.pi - jnp.pi, parameters[1]* 2 * jnp.pi - jnp.pi)
    
    # 2. Propagate:
    propagated_1, _ = modulated_light.VRS_propagation(z=(jnp.abs(parameters[4])*100+offset)*cm)
    
    # 3. Apply LCD: 
    modulated_light_2 = LCD(propagated_1, parameters[2]* 2 * jnp.pi - jnp.pi, parameters[3]* 2 * jnp.pi - jnp.pi)
    
    # 4. Propagate:
    propagated_2, _ = modulated_light_2.VRS_propagation(z=(jnp.abs(parameters[5])*100+offset)*cm)
    
    # 5. Strong focus using high NA objective:
    focused_light = VCZT_objective_lens(propagated_2, r=fixed_params[0], f=fixed_params[1], xout=fixed_params[2], yout=fixed_params[3])
    
    return focused_light

def XL_Setup(ls1, ls2, ls3, z, phase, angle, r, f, xout, yout):
    """
    Optical table with the general set-up in Fig. 6a (https://arxiv.org/abs/2310.08408#):  
    Building blocks consist of [sSLM -- z --> LCD], joint by z and beam splitters (BS). 
    
    Parameters:
    ls1, ls2, ls3 (VectorizedLight objects): Light sources.
    z (float): Distance to propagate.
    phase (jnp.array): Array with phase masks for sSLM.
    angle (float): Angle for LCDs.
    r (float): Radius of the objective lens.
    f (float): Focal length of the objective lens.
    xout, yout (jnp.arrays): Size of the detection window. 
    
    Returns VectorizedLight objects at 6 detectors. 
    
    -------------------------------------------------------------------
    
    * Scheme of the setup (distance z, phase masks, angles, and objective lens specs are common - this setup is for testing, not optimizing):
    

                             ls1                     ls2                     ls3
                              |                       |                       |
                            [BB 2]                  [BB 4]                 [BB 6]
                              |                       |                       |
                              z                       z                       z             
                              |                       |                       |
                              v                       v                       v    
    ls1 --> [BB 1] -- z --> [BS] --> [BB 7] -- z --> [BS] --> [BB 8] -- z -> [BS] --> OL --> Detector
                              |                       |                       |
                              z                       z                       z             
                              |                       |                       |
                              v                       v                       v    
                           [BB 13]                 [BB 15]                 [BB 17]
                              |                       |                       |
                              z                       z                       z             
                              |                       |                       |
                              v                       v                       v    
    ls2 --> [BB 3] -- z --> [BS] --> [BB 9] -- z --> [BS] --> [BB 10] - z -> [BS] --> OL --> Detector
                              |                       |                       |
                              z                       z                       z             
                              |                       |                       |
                              v                       v                       v    
                           [BB 14]                 [BB 16]                 [BB 18]
                              |                       |                       |
                              z                       z                       z             
                              |                       |                       |
                              v                       v                       v  
    ls3 --> [BB 5] -- z --> [BS] --> [BB 11] -- z -> [BS] -> [BB 12] -- z -> [BS] --> OL --> Detector
                              |                       |                       |
                              z                       z                       z             
                              |                       |                       |
                              v                       v                       v    
                              OL                      OL                      OL
                              |                       |                       |
                              v                       v                       v    
                           Detector                Detector                Detector
    """
    tic = time.perf_counter()
    # Define empty object for single-input BS.
    empty = VectorizedLight(ls1.x, ls1.y, ls1.wavelength)
    
    # Apply initial building blocks:
    path_ls1_1, _ = (building_block(ls1, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_2, _ = (building_block(ls1, phase, phase, z, angle, angle)).VRS_propagation(z)

    path_ls2_1, _ = (building_block(ls2, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_2, _ = (building_block(ls2, phase, phase, z, angle, angle)).VRS_propagation(z)

    path_ls3_1, _ = (building_block(ls3, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls3_2, _ = (building_block(ls3, phase, phase, z, angle, angle)).VRS_propagation(z)

    # Compute the first row
    ls1_ref, ls1_tra = BS(path_ls1_2, path_ls1_1, 0.5, 0.5, jnp.pi)
    path_ls1_3, _ = (building_block(ls1_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_3_ref, path_ls1_3_tra = BS(empty, path_ls1_3, 0.5,0.5, jnp.pi)
    path_ls1_4, _ = (building_block(path_ls1_3_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_4_ref, path_ls1_4_tra = BS(empty, path_ls1_4, 0.5,0.5, jnp.pi)                 
    ls1_f1 = VCZT_objective_lens(path_ls1_4_tra, r=r, f=f, xout=xout, yout=yout)

    path_ls2_2_ref, path_ls2_2_tra = BS(path_ls2_2, empty, 0.5, 0.5, jnp.pi)                 
    path_ls2_3, _ = (building_block(path_ls2_2_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_3_ref, path_ls2_3_tra = BS(empty, path_ls2_3, 0.5,0.5, jnp.pi)
    ls2_f1 = VCZT_objective_lens(path_ls2_3_tra, r=r, f=f, xout=xout, yout=yout)  

    path_ls3_2_ref, path_ls3_2_tra = BS(path_ls3_2, empty, 0.5, 0.5, jnp.pi)                     
    ls3_f1 = VCZT_objective_lens(path_ls3_2_ref, r=r, f=f, xout=xout, yout=yout)  

    # Compute 2nd row
    path_ls1_5, _ = (building_block(ls1_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_6, _ = (building_block(path_ls1_3_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_7, _ = (building_block(path_ls1_4_ref, phase, phase, z, angle, angle)).VRS_propagation(z)

    path_ls2_4, _ = (building_block(path_ls2_2_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_5, _ = (building_block(path_ls2_3_ref, phase, phase, z, angle, angle)).VRS_propagation(z)  
    path_ls3_3, _ = (building_block(path_ls3_2_tra, phase, phase, z, angle, angle)).VRS_propagation(z)   

    # Compute 3rd row
    path_ls1_3_ref, path_ls1_3_tra = BS(path_ls1_3, empty, 0.5,0.5, jnp.pi)
    path_ls1_8, _ = (building_block(path_ls1_3_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_8_ref, path_ls1_8_tra = BS(path_ls1_6, path_ls1_8, 0.5,0.5, jnp.pi)
    path_ls1_9, _ = (building_block(path_ls1_8_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_9_ref, path_ls1_9_tra = BS(path_ls1_7, path_ls1_9, 0.5,0.5, jnp.pi)                 
    ls1_f2 = VCZT_objective_lens(path_ls1_9_tra, r=r, f=f, xout=xout, yout=yout)  

    path_ls2_1_ref, path_ls2_1_tra = BS(path_ls2_1, empty, 0.5,0.5, jnp.pi)
    path_ls2_6, _ = (building_block(path_ls2_1_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_6_ref, path_ls2_6_tra = BS(path_ls2_4, path_ls2_6, 0.5,0.5, jnp.pi)
    path_ls2_7, _ = (building_block(path_ls2_6_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_7_ref, path_ls2_7_tra = BS(path_ls2_5, path_ls2_7,0.5,0.5, jnp.pi)
    ls2_f2 = VCZT_objective_lens(path_ls2_7_tra, r=r, f=f, xout=xout, yout=yout)  

    path_ls3_3_ref, path_ls3_3_tra = BS(path_ls3_2_tra, empty, 0.5,0.5, jnp.pi)                 
    ls3_f2 = VCZT_objective_lens(path_ls3_3_ref, r=r, f=f, xout=xout, yout=yout)  

    # Compute 4th row
    path_ls1_10, _ = (building_block(path_ls1_3_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_8, _ = (building_block(path_ls2_1_ref, phase, phase, z, angle, angle)).VRS_propagation(z)

    path_ls1_11, _ = (building_block(path_ls1_8_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_9, _ = (building_block(path_ls2_6_ref, phase, phase, z, angle, angle)).VRS_propagation(z)

    path_ls1_12, _ = (building_block(path_ls1_9_ref, phase, phase, z, angle, angle)).VRS_propagation(z)      
    path_ls2_10, _ = (building_block(path_ls2_7_ref, phase, phase, z, angle, angle)).VRS_propagation(z)  
    path_ls3_4, _ = (building_block(path_ls3_3_tra, phase, phase, z, angle, angle)).VRS_propagation(z)   

    # Compute 5th row       
    path_ls1_10_ref, path_ls1_10_tra = BS(path_ls1_10, empty, 0.5,0.5, jnp.pi)  
    path_ls1_13, _ = (building_block(path_ls1_10_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_13_ref, path_ls1_13_tra = BS(path_ls1_11, path_ls1_13,0.5,0.5, jnp.pi)                 
    path_ls1_14, _ = (building_block(path_ls1_13_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls1_14_ref, path_ls1_14_tra = BS(path_ls1_12, path_ls1_14,0.5,0.5, jnp.pi)                    
    ls1_f3 = VCZT_objective_lens(path_ls1_14_tra, r=r, f=f, xout=xout, yout=yout)  

    path_ls2_8_ref, path_ls2_8_tra = BS(path_ls2_8, empty, 0.5,0.5, jnp.pi)  
    path_ls2_11, _ = (building_block(path_ls2_8_ref, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_11_ref, path_ls2_11_tra = BS(path_ls2_9, path_ls2_11,0.5,0.5, jnp.pi)                  
    path_ls2_12, _ = (building_block(path_ls2_11_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls2_12_ref, path_ls2_12_tra = BS(path_ls2_10, path_ls2_12,0.5,0.5, jnp.pi)     
    ls2_f3 = VCZT_objective_lens(path_ls2_12_tra, r=r, f=f, xout=xout, yout=yout)  

    path_ls3_1_ref, path_ls3_1_tra = BS(empty, path_ls3_1, 0.5,0.5, jnp.pi)  
    path_ls3_5, _ = (building_block(path_ls3_1_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls3_5_ref, path_ls3_5_tra = BS(empty, path_ls3_5, 0.5,0.5, jnp.pi)                  
    path_ls3_6, _ = (building_block(path_ls3_5_tra, phase, phase, z, angle, angle)).VRS_propagation(z)
    path_ls3_6_ref, path_ls3_6_tra = BS(path_ls3_4, path_ls3_6,0.5,0.5, jnp.pi)  
    ls3_f3 = VCZT_objective_lens(path_ls3_6_tra, r=r, f=f, xout=xout, yout=yout)              

    # Compute 6th row
    ls1_f4 = VCZT_objective_lens(path_ls1_10_tra, r=r, f=f, xout=xout, yout=yout)  
    ls2_f4 = VCZT_objective_lens(path_ls2_8_tra, r=r, f=f, xout=xout, yout=yout)  
    ls3_f4 = VCZT_objective_lens(path_ls3_1_ref, r=r, f=f, xout=xout, yout=yout)
    
    ls1_f5 = VCZT_objective_lens(path_ls1_13_ref, r=r, f=f, xout=xout, yout=yout)  
    ls2_f5 = VCZT_objective_lens(path_ls2_11_ref, r=r, f=f, xout=xout, yout=yout)  
    ls3_f5 = VCZT_objective_lens(path_ls3_5_ref, r=r, f=f, xout=xout, yout=yout)
    
    ls1_f6 = VCZT_objective_lens(path_ls1_14_ref, r=r, f=f, xout=xout, yout=yout)  
    ls2_f6 = VCZT_objective_lens(path_ls2_12_ref, r=r, f=f, xout=xout, yout=yout)  
    ls3_f6 = VCZT_objective_lens(path_ls3_6_ref, r=r, f=f, xout=xout, yout=yout) 
    print("Time taken for generate XL experiment - in seconds", time.perf_counter() - tic)
    
    return ls1_f1, ls1_f2, ls1_f3, ls1_f4, ls1_f5, ls1_f6, ls2_f1, ls2_f2, ls2_f3, ls2_f4, ls2_f5, ls2_f6, ls3_f1, ls3_f2, ls3_f3, ls3_f4, ls3_f5, ls3_f6