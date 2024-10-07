# Test for vectorized optics module
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
from xlumina.vectorized_optics import PolarizedLightSource, VectorizedLight

class TestVectorizedOptics(unittest.TestCase):
    def setUp(self):
        self.wavelength = 633e-3 #nm
        self.resolution = 1024 
        self.x = np.linspace(-1500, 1500, self.resolution)
        self.y = np.linspace(-1500, 1500, self.resolution)
        self.k = 2 * jnp.pi / self.wavelength

    def test_vectorized_light(self):
        light = VectorizedLight(self.x, self.y, self.wavelength)
        self.assertEqual(light.wavelength, self.wavelength)
        self.assertEqual(light.k, self.k)
        self.assertEqual(light.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(light.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(light.Ez.shape, (self.resolution, self.resolution))

    def test_polarized_light_source_horizontal(self):
        source = PolarizedLightSource(self.x, self.y, self.wavelength)
        source.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 0))
        # TEST POLARIZATION
        self.assertGreater(jnp.sum(jnp.abs(source.Ex)**2), 0)
        self.assertEqual(jnp.sum(jnp.abs(source.Ey)**2), 0)
    
    def test_polarized_light_source_vertical(self):
        source = PolarizedLightSource(self.x, self.y, self.wavelength)
        source.gaussian_beam(w0=(1200, 1200), jones_vector=(0, 1))
        # TEST POLARIZATION
        self.assertGreater(jnp.sum(jnp.abs(source.Ey)**2), 0)
        self.assertEqual(jnp.sum(jnp.abs(source.Ex)**2), 0)

    def test_polarized_light_source_diagonal(self):
        source = PolarizedLightSource(self.x, self.y, self.wavelength)
        source.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 1))
        # TEST POLARIZATION
        self.assertGreater(jnp.sum(jnp.abs(source.Ex)**2), 0)
        self.assertGreater(jnp.sum(jnp.abs(source.Ey)**2), 0)

    def test_vrs_propagation(self):
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 1))
        propagated, _ = light.VRS_propagation(z=1000)
        self.assertEqual(propagated.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(propagated.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(propagated.Ez.shape, (self.resolution, self.resolution))
        # Check same SoP
        self.assertGreater(jnp.sum(jnp.abs(light.Ex)**2), 0)
        self.assertGreater(jnp.sum(jnp.abs(light.Ey)**2), 0)

    def test_vczt(self):
        light = PolarizedLightSource(self.x, self.y, self.wavelength)
        light.gaussian_beam(w0=(1200, 1200), jones_vector=(1, 1))
        propagated = light.VCZT(1000, self.x, self.y)
        self.assertEqual(propagated.Ex.shape, (self.resolution, self.resolution))
        self.assertEqual(propagated.Ey.shape, (self.resolution, self.resolution))
        self.assertEqual(propagated.Ez.shape, (self.resolution, self.resolution))
        # Check same SoP
        self.assertGreater(jnp.sum(jnp.abs(light.Ex)**2), 0)
        self.assertGreater(jnp.sum(jnp.abs(light.Ey)**2), 0)

if __name__ == '__main__':
    unittest.main()
