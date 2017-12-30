import unittest
import numpy as np

import fdm
import fractulus as fr
import fractional_mechanics as fm


class RieszCaputoForLinearDerivative(unittest.TestCase):
    """

    """
    def setUp(self):
        values = [[-0.02625],  # *values* are chosen in the way, the first derivative is a linear function
                  [0.],
                  [0.02375],
                  [0.045],
                  [0.06375],
                  [0.08],
                  [0.09375],
                  [0.105],
                  [0.11375],
                  [0.12],
                  [0.12375],
                  [0.125],
                  [0.12375],
                  [0.12],
                  [0.11375],
                  [0.105],
                  [0.09375],
                  [0.08],
                  [0.06375],
                  [0.045],
                  [0.02375],
                  [0.],
                  [-0.02625]]

        def get_value(index):
            index += 1
            if index >= len(values):
                index = len(values) - 1
            return values[index][0]

        self._function = get_value

    def test_Calculate_Always_ClassicAndFractionalDerivativeIsEqual(self):
        alpha = 0.8
        points = 21.
        L = 1.
        h = L / (points - 1)
        lf = 4
        test_range = [h * i for i in range(21)]

        classical, fractional = zip(*self._compute(alpha, lf, h, test_range))

        def cut_boundary(item):
            return item[lf: -lf]

        np.testing.assert_almost_equal(
            cut_boundary(classical),
            cut_boundary(fractional),
            decimal=3
        )

    def _compute(self, alpha, lf, h, test_range):
        return [self._compute_for_item(i, lf, h, alpha) for i, value in enumerate(test_range)]

    def _compute_for_item(self, i, lf, h, alpha):
        fractional_stencil = self._create_fractional_derivative(alpha, lf)
        classical_stencil = self._create_classic_derivative()
        return (
            self._compute_by_stencil(classical_stencil.expand(i), self._function, h),
            self._compute_by_stencil(fractional_stencil.expand(i), self._function, h)
        )

    @staticmethod
    def _compute_by_stencil(scheme, function, h):
        coefficients = scheme.to_coefficients(h)
        return sum([function(node) * weight for node, weight in coefficients.items()])

    @staticmethod
    def _create_fractional_derivative(alpha, lf):
        settings = fr.CaputoSettings(alpha, lf, lf)
        return fm.create_caputo_operator_by_pattern(settings)

    @staticmethod
    def _create_classic_derivative():
        return fdm.Operator(
            fdm.Stencil.central(1)
        )