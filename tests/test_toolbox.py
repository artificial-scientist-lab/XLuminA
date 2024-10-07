# Test for optical elements
import os
import sys

# Setting the path for XLuminA modules:
current_path = os.path.abspath(os.path.join('..'))
module_path = os.path.join(current_path)

if module_path not in sys.path:
    sys.path.append(module_path)

import unittest
import jax.numpy as jnp
import numpy as np
from jax import random
from xlumina.toolbox import (
    space, wrap_phase, is_conserving_energy, softmin, delta_kronecker,
    build_LCD_cell, rotate_mask, nearest,
    extract_profile, gaussian, lorentzian, fwhm_1d_fit, spot_size,
    compute_fwhm, find_max_min, gaussian_2d
)
from xlumina.vectorized_optics import VectorizedLight, PolarizedLightSource

class TestToolbox(unittest.TestCase):
    def setUp(self):
        seed = 9999
        self.key = random.PRNGKey(seed)
        self.resolution = 512
        self.x = np.linspace(-1500, 1500, self.resolution)
        self.y = np.linspace(-1500, 1500, self.resolution)
        self.wavelength = 633e-3

    def test_space(self):
        x, y = space(1500, self.resolution)
        self.assertTrue(jnp.allclose(x, self.x))
        self.assertTrue(jnp.allclose(y, self.y))

    def test_wrap_phase(self):
        phase = jnp.array([0, jnp.pi, 2*jnp.pi, 3*jnp.pi, -3*jnp.pi])
        wrapped = wrap_phase(phase)
        self.assertTrue(jnp.allclose(wrapped, jnp.array([0, jnp.pi, 0, jnp.pi, -jnp.pi])))

    def test_is_conserving_energy(self):
        light1 = VectorizedLight(self.x, self.y, self.wavelength)
        light2 = VectorizedLight(self.x, self.y, self.wavelength)
        light1.Ex = jnp.ones((self.resolution, self.resolution))
        light2.Ex = jnp.ones((self.resolution, self.resolution))
        conservation = is_conserving_energy(light1, light2)
        self.assertAlmostEqual(conservation, 1.0, places=6)
        light2.Ex = 0*light2.Ex 
        conservation = is_conserving_energy(light1, light2)
        self.assertEqual(conservation, 0)

    def test_softmin(self):
        result = softmin(jnp.array([1.0, 2.0, 3.0]))
        self.assertTrue(result == 1.0)

    def test_delta_kronecker(self):
        self.assertEqual(delta_kronecker(1, 1), 1)
        self.assertEqual(delta_kronecker(1, 2), 0)

    def test_build_LCD_cell(self):
        eta, theta = build_LCD_cell(jnp.pi/2, jnp.pi/4, self.resolution)
        self.assertTrue(jnp.allclose(eta, jnp.pi/2 * jnp.ones((self.resolution, self.resolution))))
        self.assertTrue(jnp.allclose(theta, jnp.pi/4 * jnp.ones((self.resolution, self.resolution))))

    def test_rotate_mask(self):
        X, Y = jnp.meshgrid(self.x, self.y)
        Xrot, Yrot = rotate_mask(X, Y, jnp.pi/4)
        self.assertEqual(Xrot.shape, (self.resolution, self.resolution))
        self.assertEqual(Yrot.shape, (self.resolution, self.resolution))

    def test_nearest(self):
        array = jnp.array([1, 2, 3, 4, 5])
        idx, value, distance = nearest(array, 3.7)
        self.assertEqual(idx, 3)
        self.assertEqual(value, 4)
        self.assertAlmostEqual(distance, 0.3, places=6)

    def test_extract_profile(self):
        data_2d = jnp.ones((10, 10))
        x_points = jnp.array([0, 1, 2])
        y_points = jnp.array([0, 1, 2])
        profile = extract_profile(data_2d, x_points, y_points)
        self.assertEqual(profile.shape, x_points.shape)

    def test_gaussian(self):
        y = gaussian(self.x, 1, 0, 1)
        self.assertEqual(y.shape, self.x.shape)

    def test_lorentzian(self):
        y = lorentzian(self.x, 0, 1)
        self.assertEqual(y.shape, self.x.shape)

    def test_fwhm_1d_fit(self):
        sigma = 120
        amplitude = 1000
        mean = 0
        y = gaussian(self.x, amplitude, mean, sigma)
        _, fwhm, _ = fwhm_1d_fit(self.x, y, fit='gaussian')
        fwhm_theoretical = 2*sigma*jnp.sqrt(2*jnp.log(2))  # 2*sigma*sqrt(2*ln2) is the theoretical FWHM for a gaussian
        self.assertAlmostEqual(fwhm, fwhm_theoretical, places=2) 

    def test_spot_size(self):
        size = spot_size(1, 1, self.wavelength)
        self.assertGreater(size, 0)

    def test_compute_fwhm(self):
        sigma = 120
        light_1d = gaussian(self.x, 1000, 0, sigma)
        XY = jnp.meshgrid(self.x, self.y)
        light_2d = gaussian_2d(XY, 1000, 0, 0, sigma, sigma)
    
        popt, fwhm, r_squared = compute_fwhm(light_1d, [self.x, self.y], field='Intensity', fit = 'gaussian', dimension='1D')
        fwhm_theoretical = 2*sigma*jnp.sqrt(2*jnp.log(2))
        self.assertAlmostEqual(fwhm, fwhm_theoretical, places=4) 

        popt, fwhm, r_squared = compute_fwhm(light_2d, [self.x, self.y], field='Intensity', fit = 'gaussian', dimension='2D')
        fwhm_x, fwhm_y = fwhm
        fwhm_theoretical = 2*sigma*jnp.sqrt(2*jnp.log(2))
        self.assertAlmostEqual(fwhm_x, fwhm_theoretical, places=4) 
        self.assertAlmostEqual(fwhm_y, fwhm_theoretical, places=4)

    def test_find_max_min(self):
        value = jnp.array([[1, 2], [3, 4]])
        idx, xy, ext_value = find_max_min(value, self.x[:2], self.y[:2], kind='max')
        self.assertEqual(idx.shape, (1, 2))
        self.assertEqual(xy.shape, (1, 2))
        self.assertEqual(ext_value, 4)
        idx, xy, ext_value = find_max_min(value, self.x[:2], self.y[:2], kind='min')
        self.assertEqual(ext_value, 1)

if __name__ == '__main__':
    unittest.main()