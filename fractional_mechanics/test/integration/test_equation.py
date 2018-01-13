import unittest
import numpy as np

import fdm
from fdm.geometry import Point
import fractulus as fr
import fractional_mechanics as fm

import fractional_mechanics
import fractulus


class RieszCaputoForLinearDerivative(unittest.TestCase):
    def setUp(self):

        def quadratic_polynomial(point):
            return point.x*(point.x - 1)  # *values* are chosen in the way, the first derivative is a linear function

        self._function = quadratic_polynomial

    def test_Calculate_Always_ClassicAndFractionalDerivativeIsEqual(self):

        lenght_scale = 4
        test_range = range(21)

        settings = fr.CaputoSettings(alpha=0.8, lf=lenght_scale, resolution=4)

        classical, fractional = zip(*self._compute(settings, test_range))

        def cut_boundary(item):
            return item[lenght_scale: -lenght_scale]

        np.testing.assert_almost_equal(
            cut_boundary(classical),
            cut_boundary(fractional),
            decimal=3
        )

    def _compute(self, settings, test_range):
        return [self._compute_for_item(i, settings) for i, value in enumerate(test_range)]

    def _compute_for_item(self, i, settings):
        return (
            self._compute_by_stencil(
                self._create_classic_derivative().expand(Point(i)),
                self._function
            ),
            self._compute_by_stencil(
                self._create_fractional_derivative(settings).expand(Point(i)),
                self._function
            )
        )

    @staticmethod
    def _compute_by_stencil(scheme, function):
        return sum([function(node) * weight for node, weight in scheme.items()])

    @staticmethod
    def _create_fractional_derivative(settings):
        return fm.create_riesz_caputo_operator_by_pattern(settings, "CCC", settings.lf / settings.resolution)

    @staticmethod
    def _create_classic_derivative():
        return fdm.Operator(
            fdm.Stencil.central(1)
        )


