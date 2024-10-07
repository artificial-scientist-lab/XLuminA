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
from xlumina.optical_elements import (
    SLM, jones_LP, jones_general_retarder, jones_sSLM, jones_LCD,
    sSLM, LCD, linear_polarizer, BS_symmetric, high_NA_objective_lens,
    VCZT_objective_lens, lens, cylindrical_lens, axicon_lens, building_block
)

from xlumina.vectorized_optics import VectorizedLight, PolarizedLightSource
from xlumina.wave_optics import LightSource

class TestOpticalElements(unittest.TestCase):
    def setUp(self):
        self.wavelength = 633e-3 #nm
        self.resolution = 512
        self.x = np.linspace(-1500, 1500, self.resolution)
        self.y = np.linspace(-1500, 1500, self.resolution)
        self.k = 2 * jnp.pi / self.wavelength

    def test_slm(self):
        light = LightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), E0=1)
        phase = jnp.zeros((self.resolution, self.resolution))
        slm_output, _ = SLM(light, phase, self.resolution)
        self.assertEqual(slm_output.field.shape, (self.resolution, self.resolution)) # Check output shape == input shape
        self.assertTrue(jnp.allclose(slm_output.field, light.field)) # Phase added by SLM is 0, field shouldn't change.

    def test_shape_jones_matrices(self):
        lp = jones_LP(jnp.pi/4)
        self.assertEqual(lp.shape, (2, 2))

        retarder = jones_general_retarder(jnp.pi/2, jnp.pi/4, 0)
        self.assertEqual(retarder.shape, (2, 2))

        sslm = jones_sSLM(jnp.pi/2, jnp.pi/4)
        self.assertEqual(sslm.shape, (2, 2))

        lcd = jones_LCD(jnp.pi/2, jnp.pi/4)
        self.assertEqual(lcd.shape, (2, 2))

    def test_polarization_devices(self):
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 1))
        
        # super-SLM with zero phase -- input SoP is diagonal
        alpha = jnp.zeros((self.resolution, self.resolution))
        phi = jnp.zeros((self.resolution, self.resolution))
        sslm_output = sSLM(light, alpha, phi)
        self.assertEqual(sslm_output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(sslm_output.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(sslm_output.Ez.shape, (self.resolution, self.resolution))
        self.assertTrue(jnp.allclose(sslm_output.Ex, light.Ex))
        self.assertTrue(jnp.allclose(sslm_output.Ey, light.Ey))
        self.assertTrue(jnp.allclose(sslm_output.Ez, light.Ez))

        # super-SLM with pi phase in Ex and Ey -- input SoP is diagonal
        alpha = jnp.pi * jnp.ones((self.resolution, self.resolution))
        phi = jnp.pi * jnp.ones((self.resolution, self.resolution))
        sslm_output = sSLM(light, alpha, phi)
        self.assertTrue(jnp.allclose(sslm_output.Ex, light.Ex * jnp.exp(1j * jnp.pi)))
        self.assertTrue(jnp.allclose(sslm_output.Ey, light.Ey * jnp.exp(1j * jnp.pi)))
        self.assertTrue(jnp.allclose(sslm_output.Ez, light.Ez))

        # LCD 
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        lcd_output = LCD(light, 0, 0)
        self.assertEqual(lcd_output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(lcd_output.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(lcd_output.Ez.shape, (self.resolution, self.resolution))
        self.assertTrue(jnp.allclose(lcd_output.Ex, light.Ex))
        self.assertTrue(jnp.allclose(lcd_output.Ey, light.Ey))
        self.assertTrue(jnp.allclose(lcd_output.Ez, light.Ez))

        # LP aligned with incident SoP
        empty = jnp.zeros((self.resolution, self.resolution))
        lp_output = linear_polarizer(light, empty)
        self.assertEqual(lp_output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(lp_output.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(lp_output.Ez.shape, (self.resolution, self.resolution))
        self.assertTrue(jnp.allclose(lp_output.Ex, light.Ex))
        self.assertTrue(jnp.allclose(lp_output.Ey, light.Ey))
        self.assertTrue(jnp.allclose(lp_output.Ez, light.Ez))

        # LP crossed to input SoP
        pi_half = jnp.pi/2 * jnp.ones_like(empty)
        lp_output = linear_polarizer(light, pi_half)
        self.assertTrue(jnp.allclose(lp_output.Ex, empty))
        self.assertTrue(jnp.allclose(lp_output.Ey, empty))

    def test_beam_splitter(self):
        light1 = PolarizedLightSource(self.x, self.y, self.wavelength)
        light1.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        light2 = PolarizedLightSource(self.x, self.y, self.wavelength)
        light2.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        
        c, d = BS_symmetric(light1, light2, 0) # fully transmissive
        # Adds a pi phase: jnp.exp(1j * pi) = 1j
        # Noise = T*0.01 
        T = jnp.abs(jnp.cos(0))
        R = jnp.abs(jnp.sin(0))
        noise = 0.01
        self.assertTrue(jnp.allclose(c.Ex,  (T - noise) * 1j * light2.Ex + (R - noise) * light1.Ex))
        self.assertTrue(jnp.allclose(c.Ey,  (T - noise) * 1j * light2.Ey + (R - noise) * light1.Ey))
        self.assertTrue(jnp.allclose(d.Ex,  (T - noise) * 1j * light1.Ex + (R - noise) * light2.Ex))
        self.assertTrue(jnp.allclose(d.Ey,  (T - noise) * 1j * light1.Ey + (R - noise) * light2.Ey))

    def test_high_na_objective_lens(self):
        radius_lens = 3.6*1e3/2  # mm
        f_lens = radius_lens / 0.9 
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        output, _ = high_NA_objective_lens(light, radius_lens, f_lens)
        self.assertEqual(output.shape, (3, self.resolution, self.resolution))

    def test_vczt_objective_lens(self):
        radius_lens = 3.6*1e3/2  # mm
        f_lens = radius_lens / 0.9 
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        output = VCZT_objective_lens(light, radius_lens, f_lens, self.x, self.y)
        self.assertEqual(output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(output.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(output.Ez.shape, (self.resolution, self.resolution))

    def test_lenses_scalar(self):
        light = LightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), E0=1)
        lens_output, _ = lens(light, (50, 50), (1000, 1000))
        self.assertEqual(lens_output.field.shape, (self.resolution, self.resolution))
        cyl_lens_output, _ = cylindrical_lens(light, 1000)
        self.assertEqual(cyl_lens_output.field.shape, (self.resolution, self.resolution))
        axicon_output, _ = axicon_lens(light, 0.1)
        self.assertEqual(axicon_output.field.shape, (self.resolution, self.resolution))
    
    def test_lenses_vectorial(self):
        ls = PolarizedLightSource(self.x, self.y, self.wavelength)
        ls.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        light = VectorizedLight(self.x, self.y, self.wavelength)
        light.Ex = ls.Ex
        light.Ey = ls.Ey
        light.Ez = ls.Ez
        lens_output, _ = lens(light, (50, 50), (1000, 1000))
        self.assertEqual(lens_output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(lens_output.Ey.shape, (self.resolution, self.resolution))
        cyl_lens_output, _ = cylindrical_lens(light, 1000)
        self.assertEqual(cyl_lens_output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(cyl_lens_output.Ey.shape, (self.resolution, self.resolution))
        axicon_output, _ = axicon_lens(light, 0.1)
        self.assertEqual(axicon_output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(axicon_output.Ey.shape, (self.resolution, self.resolution))

    def test_building_block(self):
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        output = building_block(light, jnp.zeros((self.resolution, self.resolution)), jnp.zeros((self.resolution, self.resolution)), 1000, jnp.pi/2, jnp.pi/4)
        self.assertEqual(output.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(output.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(output.Ez.shape, (self.resolution, self.resolution))

if __name__ == '__main__':
    unittest.main()