# Test for wave optics  module
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
from xlumina.wave_optics import ScalarLight, LightSource

class TestWaveOptics(unittest.TestCase):
    def setUp(self):
        self.wavelength = 633e-3 #nm
        self.resolution = 1024 
        self.x = np.linspace(-1500, 1500, self.resolution)
        self.y = np.linspace(-1500, 1500, self.resolution)
        self.k = 2 * jnp.pi / self.wavelength

    def test_scalar_light(self):
        light = ScalarLight(self.x, self.y, self.wavelength)
        self.assertEqual(light.wavelength, self.wavelength)
        self.assertEqual(light.k, self.k)
        self.assertEqual(light.field.shape, (self.resolution, self.resolution))

    def test_light_source_gb(self):
        source = LightSource(self.x, self.y, self.wavelength)
        source.gaussian_beam(w0=(1200, 1200), E0=1)
        self.assertEqual(source.wavelength, self.wavelength)
        self.assertEqual(source.field.shape, (self.resolution, self.resolution))
        self.assertGreater(jnp.sum(jnp.abs(source.field)**2), 0)
    
    def test_light_source_pw(self):
        source = LightSource(self.x, self.y, self.wavelength)
        source.plane_wave(A=1, theta=0, phi=0, z0=0)
        self.assertGreater(jnp.sum(jnp.abs(source.field)**2), 0)

    def test_rs_propagation(self):
        light = LightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), E0=1)
        propagated, _ = light.RS_propagation(z=1000)
        self.assertEqual(propagated.field.shape, (self.resolution, self.resolution))

    def test_czt(self):
        light = LightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), E0=1)
        propagated = light.CZT(z=1000)
        self.assertEqual(propagated.field.shape, (self.resolution, self.resolution))

if __name__ == '__main__':
    unittest.main()