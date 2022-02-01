import unittest
import timeit
import textwrap

import fdm
import fractional_mechanics.beam
import fractulus

from fractional_mechanics.test.utils import Profiler
from fdm.geometry import Point


class TestSuite(unittest.TestCase):
    def assertStencilEqual(self, expected, actual, p_tol=1e-6, w_tol=1e-6):
        exp_weights, actual_weights = expected._weights, actual._weights
        equal = True

        if len(exp_weights) != len(actual_weights):
            equal = False

        for p, w in expected._weights.items():
            try:
                equal = abs(expected._weights[p] - actual._weights[p]) <= w_tol
            except KeyError:
                equal = False
            if not equal:
                break

        if not equal:
            self.assertEqual(expected, actual)

    def assertStencilRange(self, expected, actual):
        expected_min, expected_max = self._get_stencil_range(expected)
        actual_min, actual_max = self._get_stencil_range(actual)
        self.assertEqual(expected_min, actual_min)
        self.assertEqual(expected_max, actual_max)

    @staticmethod
    def _get_stencil_range(stencil):
        max_x, min_x = 0., 0.
        for p, _ in stencil._weights.items():
            max_x = max(p.x, max_x)
            min_x = min(p.x, min_x)
        return min_x, max_x

    @staticmethod
    def _get_scheme_range(scheme):
        return scheme.start.x, scheme.end.x


class AOperatorFactoryTest(TestSuite):
    def test_Create_AlphaOne_ReturnAsForClassicDerivative(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.99999, 0.1, 3))
        factory = self.create(settings_factory, span=1.)

        result = factory(Point(0.))

        actual = result.drop(1e-3)
        expected = fdm.Stencil(
            {
                Point(-0.5, 0.0, 0.0): -1.,
                Point(0.5, 0.0, 0.0): 1.,
            })

        self.assertStencilEqual(expected, actual, w_tol=1e-3)

    def test_Create_AlphaAlmostZero_AllInteriorWeightsUniform(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.00001, 0.1, 3))
        factory = self.create(settings_factory)

        result = factory(Point(0.))

        actual = result.drop(1e-3)
        expected = fdm.Scheme(
            {
                Point(-0.6, 0.0, 0.0): -0.08333,
                Point(-0.56666666, 0.0, 0.0): -0.16666,
                Point(-0.53333333, 0.0, 0.0): -0.16666,
                Point(-0.5, 0.0, 0.0): -0.16666,
                Point(-0.46666666, 0.0, 0.0): -0.16666,
                Point(-0.43333333, 0.0, 0.0): -0.16666,
                Point(-0.4, 0.0, 0.0): -0.08333,
                Point(0.4, 0.0, 0.0): 0.08333,
                Point(0.433333333, 0.0, 0.0): 0.16666,
                Point(0.466666666, 0.0, 0.0): 0.16666,
                Point(0.5, 0.0, 0.0): 0.16666,
                Point(0.533333333, 0.0, 0.0): 0.16666,
                Point(0.566666666, 0.0, 0.0): 0.16666,
                Point(0.6, 0.0, 0.0): 0.08333,
            })

        self.assertStencilEqual(expected, actual, w_tol=1e-4)

    def test_Create_AlphaNotOneAllCentral_StencilRangeUpTo(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.5, 0.1, 3))
        factory = self.create(settings_factory, span=0.1)

        result = factory(Point(0.))

        actual_min, actual_max = self._get_scheme_range(result)
        expected_min, expected_max = -0.15, 0.15

        self.assertAlmostEqual(expected_min, actual_min)
        self.assertAlmostEqual(expected_max, actual_max)

    @staticmethod
    def fake_settings_factory(settings):
        return lambda point: settings

    @staticmethod
    def create(settings_factory, integration_method='caputo', span=1.):
        return _create_operator_factory(integration_method, span, settings_factory)['A']


class BOperatorFactoryTest(TestSuite):
    def test_Create_AlphaOne_ReturnAsForClassicDerivative(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.99999, 0.1, 3))
        factory = self.create(settings_factory, span=1.)

        result = factory(Point(0.))

        actual = result.drop(1e-3)
        expected = fdm.Scheme(
            {
                Point(1., 0.0, 0.0): 1.,
                Point(0., 0.0, 0.0): -2.,
                Point(-1., 0.0, 0.0): 1.
            })

        self.assertStencilEqual(expected, actual, w_tol=1e-3)

    def test_Create_AlphaNotOneAllCentral_StencilRangeUpTo(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.5, 0.1, 3))
        factory = self.create(settings_factory, span=0.1)

        result = factory(Point(0.))

        actual_min, actual_max = self._get_scheme_range(result)
        expected_min, expected_max = -0.3, 0.3

        self.assertAlmostEqual(expected_min, actual_min)
        self.assertAlmostEqual(expected_max, actual_max)

    @staticmethod
    def fake_settings_factory(settings):
        return lambda point: settings

    @staticmethod
    def create(settings_factory, integration_method='caputo', span=0.01):
        return _create_operator_factory(integration_method, span, settings_factory)['B']


class COperatorFactoryTest(TestSuite):
    def test_Create_AlphaOne_ReturnAsForClassicDerivative(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.99999, 0.1, 3))
        factory = self.create(settings_factory, span=1.)

        result = factory(Point(0.))

        actual = result.drop(1e-3)
        expected = fdm.Stencil(
            {
                Point(1.5, 0.0, 0.0): 1.,
                Point(0.5, 0.0, 0.0): -3.,
                Point(-0.5, 0.0, 0.0): 3.,
                Point(-1.5, 0.0, 0.0): -1.
            }
        )

        self.assertStencilEqual(expected, actual, w_tol=1e-3)

    def test_Create_AlphaNotOneAllCentral_StencilRangeUpTo(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.5, 0.1, 3))
        factory = self.create(settings_factory, span=0.1)

        result = factory(Point(0.))

        actual_min, actual_max = self._get_scheme_range(result)
        expected_min, expected_max = -0.45, 0.45

        self.assertAlmostEqual(expected_min, actual_min)
        self.assertAlmostEqual(expected_max, actual_max)

    @staticmethod
    def fake_settings_factory(settings):
        return lambda point: settings

    @staticmethod
    def create(settings_factory, integration_method='caputo', span=0.01):
        return _create_operator_factory(integration_method, span, settings_factory)['C']


class DOperatorFactoryTest(TestSuite):
    def test_Create_AlphaOne_ReturnAsForClassicDerivative(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.999999, 0.1, 3))
        factory = self.create(settings_factory, span=1.)

        result = factory(Point(0.))

        actual = result.drop(1e-3)
        expected = fdm.Scheme({
            Point(2., 0.0, 0.0): 1.,
            Point(1., 0.0, 0.0): -4.,
            Point(0., 0.0, 0.0): 6.,
            Point(-1., 0.0, 0.0): -4.,
            Point(-2., 0.0, 0.0): 1.,

        })

        self.assertStencilEqual(expected, actual, w_tol=1e-3)

    def test_Create_AlphaNotOneAllCentral_StencilRangeUpTo(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.5, 0.1, 3))
        factory = self.create(settings_factory, span=0.1)

        result = factory(Point(0.))

        actual_min, actual_max = self._get_scheme_range(result)
        expected_min, expected_max = -0.6, 0.6

        self.assertAlmostEqual(expected_min, actual_min)
        self.assertAlmostEqual(expected_max, actual_max)

    def test_Create_AlphaNotOneAllCentral_StencilHasCorrectWeights(self):
        settings_factory = self.fake_settings_factory(fractulus.Settings(0.8, 0.1, 3))
        factory = self.create(settings_factory, span=0.1)

        with Profiler(False):
            result = factory(Point(0.))

        actual = result.drop(1e-3)

        expected = fdm.Scheme({
            Point(-0.6000000000000001): 0.6000000000000001,
            Point(-0.6000000000000001): 0.0011520355579769184,
            Point(-0.5666666666666668): 0.011913299705236607,
            Point(-0.5333333333333334): 0.07107923032309445,
            Point(-0.5): 0.4353072613356501,
            Point(-0.46666666666666673): 2.0267930181868614,
            Point(-0.4333333333333334): 7.457577659764114,
            Point(-0.4): 27.33612096746882,
            Point(-0.3666666666666667): 79.20845676394275,
            Point(-0.33333333333333337): 189.3411259928102,
            Point(-0.30000000000000004): 451.8839442977507,
            Point(-0.2666666666666667): 710.1534895646513,
            Point(-0.23333333333333336): 810.4861267993128,
            Point(-0.2): 702.9719303277944,
            Point(-0.16666666666666669): -2029.803359776797,
            Point(-0.13333333333333336): -4239.501580585547,
            Point(-0.1): -7327.145310283915,
            Point(-0.06666666666666667): -418.21326330311723,
            Point(-0.033333333333333326): 4888.761641336766,
            Point(0.0): 12289.033710788015,
            Point(0.033333333333333354): 4888.761641336766,
            Point(0.06666666666666668): -418.21326330311706,
            Point(0.10000000000000002): -7327.145310283915,
            Point(0.13333333333333336): -4239.501580585548,
            Point(0.16666666666666669): -2029.8033597767972,
            Point(0.20000000000000004): 702.9719303277947,
            Point(0.23333333333333336): 810.4861267993128,
            Point(0.2666666666666667): 710.1534895646513,
            Point(0.30000000000000004): 451.8839442977507,
            Point(0.33333333333333337): 189.34112599281022,
            Point(0.36666666666666675): 79.20845676394276,
            Point(0.4000000000000001): 27.33612096746882,
            Point(0.4333333333333334): 7.457577659764114,
            Point(0.46666666666666673): 2.0267930181868614,
            Point(0.5000000000000001): 0.4353072613356501,
            Point(0.5333333333333334): 0.07107923032309446,
            Point(0.5666666666666668): 0.011913299705236607,
            Point(0.6000000000000001): 0.0011520355579769184,
        })

        self.assertStencilEqual(expected, actual, w_tol=1e-5)

    def test_Create_VariedLengthScale_ConsiderActualValueOfLengthScaleForAllNestedOperators(self):
        max_lf = 0.1
        settings_factory = self.fake_reduced_lf_settings_factory(0.5, 3, Point(0.), 0.0, Point(0.1), max_lf, 0.01)
        factory = self.create(settings_factory, span=0.01)

        result = factory(Point(0.1))

        actual_min, actual_max = self._get_scheme_range(result)
        expected_min, expected_max = -0.05, 1.675

        self.assertAlmostEqual(expected_min, actual_min)
        self.assertAlmostEqual(expected_max, actual_max)

    @staticmethod
    def fake_settings_factory(settings):
        return lambda point: settings

    @staticmethod
    def fake_reduced_lf_settings_factory(alpha, resolution, point_1, value_1, point_2, value_2, min_value):
        def calc_lf(point):
            return max(value_1 + (value_2 - value_1)/(point_2.x - point_1.x)*point.x, min_value)
        return lambda point: fractulus.Settings(alpha, calc_lf(point), resolution)

    @staticmethod
    def create(settings_factory, integration_method='caputo', span=0.01):
        return _create_operator_factory(integration_method, span, settings_factory)['D']


def _create_operator_factory(integration_method, span, settings_factory):
    base_stencils = {
        'A': fdm.Stencil.central(span),
        'B': fdm.Stencil.central(span),
        'C': fdm.Stencil.central(span),
        'D': fdm.Stencil.central(span),
    }
    stencils = fractional_mechanics.beam.create_beam_stiffness_stencils_factory(
        integration_method, base_stencils, settings_factory)
    return fractional_mechanics.beam.create_beam_stiffness_operators_factory(stencils, settings_factory)
