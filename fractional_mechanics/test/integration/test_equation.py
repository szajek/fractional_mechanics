import unittest
import numpy as np

import fractulus as fr
import fdm


class LeftCaputoForLinearFunctionStudies(unittest.TestCase):
    def setUp(self):
        self._function = lambda x: x  # f(x) = x

    def test_Calculate_AlphaAlmostOne_ConstantValue(self):

        alpha = 0.9999
        test_range = range(1, 15)

        result = self._compute(alpha, test_range)

        expected = [1. for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)

    def test_Calculate_AlphaAlmostZero_ReturnLinearFunction(self):

        alpha = 0.00001
        test_range = range(1, 2)

        result = self._compute(alpha, test_range)

        expected = test_range

        np.testing.assert_almost_equal(expected, result, decimal=3)

    def _compute(self, alpha, test_range):
        return [self._compute_for_item(i, alpha) for i in test_range]

    def _compute_for_item(self, i, alpha):
        stencil = self._create_derivative(alpha, i)
        return self._compute_by_stencil(stencil.expand(i), self._function)

    @staticmethod
    def _compute_by_stencil(scheme, function):
        coefficients = scheme.to_coefficients(1.)
        return sum([function(node) * weight for node, weight in coefficients.items()])

    @staticmethod
    def _create_derivative(alpha, lf):
        return fdm.Operator(
            fr.equation.create_left_caputo_stencil(alpha, lf),
            fdm.Operator(
                fdm.Stencil.central(.1)
            )
        )
