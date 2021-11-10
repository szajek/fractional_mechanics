import unittest
import numpy as np

import fdm
from fdm.geometry import Point
import fractulus as fr
import fractional_mechanics as fm


class RieszCaputoForLinearDerivative(unittest.TestCase):
    def setUp(self):
        def quadratic_polynomial(point):
            return point.x * (point.x - 1)  # *values* are chosen in the way, the first derivative is a linear function

        self._quadratic_polynomial = quadratic_polynomial

    def test_Calculate_Always_ClassicAndFractionalDerivativeIsEqual(self):
        length_scale = 4
        test_range = range(21)

        settings = fr.Settings(alpha=0.8, lf=length_scale, resolution=4)

        classical = self._compute_classical(test_range, self._quadratic_polynomial)
        fractional = self._compute_fractional(test_range, self._quadratic_polynomial, settings)

        def cut_boundary(item):
            return item[length_scale: -length_scale]

        np.testing.assert_almost_equal(
            cut_boundary(classical),
            cut_boundary(fractional),
            decimal=3
        )

    def _compute_classical(self, test_range, function):
        return [self._compute_by_stencil(
            self._create_classic_derivative().expand(Point(i)),
            function
        ) for i, value in enumerate(test_range)]

    def _compute_fractional(self, test_range, function, settings):
        return [self._compute_by_stencil(
            self._create_fractional_derivative(settings).expand(Point(i)),
            function
        ) for i, value in enumerate(test_range)]

    @staticmethod
    def _compute_by_stencil(scheme, function):
        return sum([function(node) * weight for node, weight in scheme.items()])

    @staticmethod
    def _create_fractional_derivative(settings):
        return fm.create_riesz_caputo_strain_operator_by_pattern('caputo', settings, "CCC", settings.lf / settings.resolution)

    @staticmethod
    def _create_classic_derivative():
        return fdm.Operator(
            fdm.Stencil.central(1)
        )

