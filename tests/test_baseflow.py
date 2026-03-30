import unittest
import numpy as np

from domain.baseflow import evaluate_baseflow


class TestBaseflowEvaluation(unittest.TestCase):
    def test_parabolic_values_and_symbolics(self):
        z = np.array([-1.0, 0.0, 1.0])
        result = evaluate_baseflow("parabolic", z)

        self.assertTrue(np.allclose(result["U"].real, np.array([1.0, 0.0, 1.0])))
        self.assertTrue(np.allclose(result["Upp"].real, np.array([2.0, 2.0, 2.0])))

        sympy_meta = result["symbolic"]
        for key in ["U", "U_prime", "U_double_prime", "U_pretty", "U_double_prime_pretty"]:
            self.assertIn(key, sympy_meta)

    def test_tanh_monotonicity(self):
        z = np.linspace(-2.0, 2.0, 9)
        result = evaluate_baseflow("tanh_shear", z)
        U_real = result["U"].real
        self.assertTrue(np.all(np.diff(U_real) >= -1e-12))


if __name__ == "__main__":
    unittest.main()