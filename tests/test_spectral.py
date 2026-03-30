import unittest
import numpy as np

from numerics.spectral import chebyshev_matrices


class TestSpectralMatrices(unittest.TestCase):
    def test_shapes_and_boundaries(self):
        z, D, D2 = chebyshev_matrices(N=4, L=2.0)

        self.assertEqual(z.shape, (5,))
        self.assertEqual(D.shape, (5, 5))
        self.assertEqual(D2.shape, (5, 5))
        self.assertAlmostEqual(z[0], 2.0)
        self.assertAlmostEqual(z[-1], -2.0)
        self.assertTrue(np.allclose(z, -z[::-1]))

    def test_derivatives_on_simple_functions(self):
        z, D, D2 = chebyshev_matrices(N=6, L=1.5)

        ones = np.ones_like(z)
        self.assertTrue(np.allclose(D @ ones, np.zeros_like(z), atol=1e-12))

        linear = z.copy()
        d_linear = D @ linear
        self.assertTrue(np.allclose(d_linear[1:-1], np.ones_like(d_linear[1:-1]), atol=1e-9))

        quadratic = 0.5 * (z ** 2)
        d2_quadratic = D2 @ quadratic
        self.assertTrue(np.allclose(d2_quadratic[1:-1], np.ones_like(d2_quadratic[1:-1]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()