import unittest
import math
import numpy
import numpy as np
from numpy.testing import assert_allclose

import fractional_mechanics.builder as builder
from fdm.analysis import solve, AnalysisStrategy
from fdm.analysis.analyzer import AnalysisType
from fractional_mechanics.builder import (
    FractionalVirtualBoundaryStrategy
)
from fdm.builder import (
    LoadType, Side, BoundaryType, FieldType
)
from fractional_mechanics.test.utils import Profiler


class Truss1dStatics6nodesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 6

    def test_ConstantSection_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.99999, 5)
            .set_length_scale_controller('uniform', .1)
            .set_field(FieldType.LINEAR, a=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.032],
                [0.056],
                [0.064],
                [0.048],
                [0.0],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha05_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.5, 5)
            .add_virtual_nodes(2, 2)
            .set_length_scale_controller('vanish', 0.6, min_value=.2)
            .set_field(FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.05299994],
                [0.09866683],
                [0.09866683],
                [0.05299994],
                [0.],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha03_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.3, 3)
            .add_virtual_nodes(2, 2)
            .set_length_scale_controller('vanish', 0.6, min_value=.2)
            .set_field(FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.00045699],
                [0.07412265],
                [0.07412265],
                [0.00045699],
                [0.],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.9999, 3)
            .set_length_scale_controller('vanish', 0.6, min_value=.2)
            .add_virtual_nodes(2, 2)
            .set_field(FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[8.88464753e-17],
             [8.00209890e-02],
             [1.20028113e-01],
             [1.20028113e-01],
             [8.00209890e-02],
             [0.00000000e+00], ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlpha05_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.5, 6)
            .set_length_scale_controller('vanish', 0.5, min_value=.2)
            .add_virtual_nodes(2, 2)
            .set_field(FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.05281015],
                [0.09917294],
                [0.09917294],
                [0.05281015],
                [0.],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlpha03_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.3, 6)
            .set_length_scale_controller('vanish', 0.5, min_value=.2)
            .set_field(FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.00045699],
                [0.07412265],
                [0.07412265],
                [0.00045699],
                [0.],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlphaAlmostOne_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.9999, 6)
            .set_length_scale_controller('vanish', 0.5, min_value=.2)
            .set_field(FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[7.10756467e-17],
             [8.00189611e-02],
             [1.20024393e-01],
             [1.20024393e-01],
             [8.00189611e-02],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_VariedSection_Always_ReturnCorrectDisplacement(self):
        def young_modulus(point):
            return 2. - (point.x / self._length) * 1.

        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.9999, 5)
            .set_length_scale_controller('vanish', 0.1, min_value=.2)
            .set_field(FieldType.CONSTANT, m=1.)
            .set_young_modulus_controller('user', young_modulus)
            .set_virtual_boundary_strategy(builder.FractionalVirtualBoundaryStrategy.BASED_ON_SECOND_DERIVATIVE)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.04786519],
                [0.0778324],
                [0.08512865],
                [0.06277831],
                [0.],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def _create_predefined_builder(self):
        return (
            builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_boundary(Side.RIGHT, BoundaryType.FIXED)
                .set_boundary(Side.LEFT, BoundaryType.FIXED)
                .set_load(LoadType.MASS)
                .add_virtual_nodes(2, 2)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    @staticmethod
    def _solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model).displacement


class Truss1dStatics31nodesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 31

    def test_Call_Fractional_ReturnCorrectDisplacements(self):
        span = self._length / (self._node_number - 1)
        length_scale = span * 3.
        alpha = 0.7

        model = (
            self._create_predefined_builder()
            .set_fractional_settings(alpha, 4)
            .set_length_scale_controller('step_vanish', length_scale, min_value=span, span=span)
        ).create()

        results = self.solve(model)
        actual = max(results.displacement)

        expected = 0.063734

        assert_allclose(expected, actual, atol=1e-6)

    def _create_predefined_builder(self):
        b = (
            builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_density_controller('spline_interpolated_linearly', 6)
                .add_virtual_nodes(2, 2)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(LoadType.MASS)
                .set_field(FieldType.SINUSOIDAL, n=1.)
                .set_stiffness_operator_strategy('minimize_virtual_layer')
                .set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)
                .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
                .set_virtual_boundary_strategy(builder.FractionalVirtualBoundaryStrategy.BASED_ON_SECOND_DERIVATIVE)
        )

        b.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        return b

    @staticmethod
    def solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)


class Truss1dEigenproblem11nodesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 11
        self._span = self._length / float(self._node_number - 1)

    def test_AlphaAlmostToOne_ReturnCorrectEigenValuesAndVectors(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.999999, 4)

        ).create()

        result = self.solve(model)

        expected_eigenvectors = np.array(
            [
                [0., 0.309017, 0.587785, 0.809017, 0.951057, 1., 0.951057, 0.809017, 0.587785, 0.309017, 0.],
                [0., 0.618034, 1., 1., 0.618034, 0., -0.618034, -1., -1., -0.618034, 0.],
                [0., 0.809017, 0.951057, 0.309017, -0.587785, -1., -0.587785, 0.309017, 0.951057, 0.809017, 0.]
            ]
        )
        expected_eigenvalues = [
            3.1287,
            6.1803,
            9.0798,
        ]  # rad/s

        for i, (expected_value, expected_vector) in enumerate(zip(expected_eigenvalues, expected_eigenvectors)):
            self.assertAlmostEqual(expected_value, result.eigenvalues[i], places=3)
            np.testing.assert_allclose(expected_vector, result.eigenvectors[i], atol=1e-5)

    def test_AlphaEqualsZeroFive_ReturnCorrectEigenValuesAndVectors(self):
        model = (
            self._create_predefined_builder()
            .set_fractional_settings(0.5, 4)
        ).create()

        result = self.solve(model)

        expected_eigenvectors = np.array(  # note: results obtained from simulation - not confirmed
            [
                [0., 0.30874, 0.587676, 0.80894931, 0.95104,  1., 0.95104, 0.80895, 0.58767, 0.30874, 0.],
                [0., 0.6156, 0.9994, 1., 0.61829, 0., -0.61828, -1., -0.99941, -0.6156, -0.],
                [0., 0.80197, 0.95354, 0.31233, -0.5862, -1., -0.586196, 0.31233, 0.95348, 0.801977, 0.]
            ]
        )

        expected_eigenvalues = [  # note: results obtained from simulation - not confirmed
            3.10612,
            6.00445,
            8.51189,
        ]  # rad/s

        for i, (expected_value, expected_vector) in enumerate(zip(expected_eigenvalues, expected_eigenvectors)):
            self.assertAlmostEqual(expected_value, result.eigenvalues[i], places=3)
            np.testing.assert_allclose(expected_vector, result.eigenvectors[i], atol=1e-4)

    @staticmethod
    def solve(model):
        return solve(AnalysisType.EIGENPROBLEM, model)

    def _create_predefined_builder(self):
        return (
            builder.create('truss1d', self._length, self._node_number)
            .set_analysis_type('EIGENPROBLEM')
            .set_young_modulus_controller('uniform', value=1.)
            .set_boundary(Side.LEFT, BoundaryType.FIXED)
            .set_boundary(Side.RIGHT, BoundaryType.FIXED)
            .set_load(LoadType.MASS)
            .add_virtual_nodes(1, 1)
            .set_length_scale_controller('vanish', 0.1, min_value=0.01)
            .set_stiffness_operator_strategy('minimize_virtual_layer')
            .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
            .set_virtual_boundary_strategy(builder.FractionalVirtualBoundaryStrategy.BASED_ON_SECOND_DERIVATIVE)
        )


class BeamTest(unittest.TestCase):
    @staticmethod
    def _create_builder(analysis_type, length, node_number):
        span = length/(node_number - 1.)
        return (
            builder.create('beam1d', length, node_number)
            .set_analysis_type(analysis_type)
            .set_density_controller('uniform', 1.)
            .set_young_modulus_controller('uniform', 1.)
            .set_moment_of_inertia_controller('uniform', 1.)
            .set_boundary(Side.LEFT, BoundaryType.FIXED)
            .set_boundary(Side.RIGHT, BoundaryType.FIXED)
            .add_middle_nodes()
            .set_load(LoadType.MASS)
            .set_field(FieldType.CONSTANT, value=1.)
            .add_virtual_nodes(8, 8)
            .set_length_scale_controller('vanish', 0.1, min_value=span)
            .set_stiffness_operator_strategy('minimize_virtual_layer')
            .set_virtual_boundary_strategy(builder.FractionalVirtualBoundaryStrategy.BASED_ON_FOURTH_DERIVATIVE)
        )

    @staticmethod
    def _solve(model, analysis_type):
        return solve(analysis_type, model)


class Beam1dStatics21nodesTest(BeamTest):
    def setUp(self):
        self._length = 1.
        self._node_number = 21
        self._span = self._length / float(self._node_number - 1)

    def test_UpToDown_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_fractional_settings(0.99999, 3)
        model = builder.create()

        result = self.solve(model).displacement

        E = J = q = 1.
        expected_max_theoretical = -1. / 384. * q * self._length ** 4 / (E * J)
        expected_max = -0.002808

        np.testing.assert_allclose([expected_max], min(result), atol=1e-5)

    def test_UpToDown_Alpha08_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_fractional_settings(0.8, 3)
        model = builder.create()

        result = self.solve(model).displacement

        E = J = q = 1.
        expected_max_classical = -1. / 384. * q * self._length ** 4 / (E * J)
        expected_max = -0.002577  # it agrees with down up when denser mesh

        np.testing.assert_allclose(min(result), [expected_max], atol=1e-5)

    def create_builder(self):
        return super(Beam1dStatics21nodesTest, self)._create_builder(
            'SYSTEM_OF_LINEAR_EQUATIONS', self._length, self._node_number
        )

    def solve(self, model):
        return super(Beam1dStatics21nodesTest, self)._solve(
            model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS,
        )


class Beam1dStatics101nodesTest(BeamTest):
    def setUp(self):
        self._length = 1.
        self._node_number = 101
        self._span = self._length / float(self._node_number - 1)

    def test_UpToDown_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_fractional_settings(0.999999, 2)
        model = builder.create()

        result = self.solve(model).displacement

        E = J = q = 1.
        expected_max_theoretical = -1. / 384. * q * self._length ** 4 / (E * J)
        expected_max = -0.002647

        np.testing.assert_allclose(min(result), [expected_max], atol=1e-5)

    def test_DownToUp_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_fractional_settings(0.999999, 2)
        model = builder.create()

        result = self.solve(model).displacement

        E = J = q = 1.
        expected_max_theoretical = -1. / 384. * q * self._length ** 4 / (E * J)
        expected_max = -0.002709  # it converges to analytical solution along with denser mesh

        np.testing.assert_allclose(min(result), [expected_max], atol=1e-5)

    def test_DownToUp_Alpha08_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_fractional_settings(0.8, 3)
        model = builder.create()

        with Profiler(False):
            result = self.solve(model).displacement

        expected_max = -0.002491

        np.testing.assert_allclose(min(result), [expected_max], atol=1e-5)

    def create_builder(self):
        return super(Beam1dStatics101nodesTest, self)._create_builder(
            'SYSTEM_OF_LINEAR_EQUATIONS', self._length, self._node_number
        )

    def solve(self, model):
        return super(Beam1dStatics101nodesTest, self)._solve(
            model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS,
        )


class Beam1dDynamics101nodesTest(BeamTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls._length = 1.
        cls._nodes_number = 101

    def test_DownToUp_AlphaAlmostOne_Fixed_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_fractional_settings(0.999999, 2)
        model = builder.create()

        result = self.solve(model).eigenvalues

        actual = result[:3]

        omega_rad_s = calc_first_eigenvalue(
            [BoundaryType.FIXED, BoundaryType.FIXED],
            L=self._length,
            E=1.,  # N/m^2
            I=1.,  # m^4
            A=1.,  # m^2
            rho=1.  # kg/m^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [
            21.934,  # 22.373 -> 3.561Hz
            60.450,  # 61.688 -> 9.818Hz
            118.468,  # 121.020 -> 19.261Hz
        ]

        np.testing.assert_allclose(expected, actual, atol=1e-1)

    def test_DownToUp_Alpha08_Fixed_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_fractional_settings(0.8, 3)
        model = builder.create()

        result = self.solve(model).eigenvalues

        actual = result[:3]

        expected = [  # from analysis - not verified
            22.7,
            60.1,
            111.0,
        ]

        np.testing.assert_allclose(expected, actual, atol=1e-1)

    def test_DownToUp_AlphaAlmostOne_Hinged_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.HINGE)
        builder.set_boundary(Side.RIGHT, BoundaryType.HINGE)
        builder.set_fractional_settings(0.999999, 2)
        model = builder.create()

        result = self.solve(model).eigenvalues

        actual = result[:3]

        omega_rad_s = calc_first_eigenvalue(
            [BoundaryType.FIXED, BoundaryType.FIXED],
            L=self._length,
            E=1.,  # N/m^2
            I=1.,  # m^4
            A=1.,  # m^2
            rho=1.  # kg/m^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [
            9.87,  # 9.871 -> 1.571Hz
            39.47,  # 39.484 -> 6.284Hz
            88.76,  # 88.876 -> 14.145Hz
        ]

        np.testing.assert_allclose(expected, actual, atol=1e-1)

    def test_DownToUp_Alpha08_Hinged_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.HINGE)
        builder.set_boundary(Side.RIGHT, BoundaryType.HINGE)
        builder.set_fractional_settings(0.8, 2)
        model = builder.create()

        result = self.solve(model).eigenvalues

        actual = result[:3]

        omega_rad_s = calc_first_eigenvalue(
            [BoundaryType.FIXED, BoundaryType.FIXED],
            L=self._length,
            E=1.,  # N/m^2
            I=1.,  # m^4
            A=1.,  # m^2
            rho=1.  # kg/m^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [
            9.73,
            37.79,
            81.06,
        ]

        np.testing.assert_allclose(expected, actual, atol=1e-1)

    def create_builder(self):
        return super(Beam1dDynamics101nodesTest, self)._create_builder(
            'EIGENPROBLEM', length=self._length, node_number=self._nodes_number
        )

    def solve(self, model):
        return super(Beam1dDynamics101nodesTest, self)._solve(
            model, AnalysisType.EIGENPROBLEM
        )


class Beam1dDynamics31nodesTest(BeamTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls._length = 1.
        cls._nodes_number = 31

    def test_DownToUp_AlphaAlmostOne_Fixed_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_fractional_settings(0.999999, 2)
        model = builder.create()

        result = self.solve(model).eigenvalues

        actual = result[:3]

        omega_rad_s = calc_first_eigenvalue(
            [BoundaryType.FIXED, BoundaryType.FIXED],
            L=self._length,
            E=1.,  # N/m^2
            I=1.,  # m^4
            A=1.,  # m^2
            rho=1.  # kg/m^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [
            20.96,  # 22.373 -> 3.561Hz
            57.68,  # 61.688 -> 9.818Hz
            112.68,  # 121.020 -> 19.261Hz
        ]

        np.testing.assert_allclose(expected, actual, atol=1e-1)

    def test_DownToUp_AlphaAlmostOne_Hinged_ReturnCorrectDisplacement(self):
        builder = self.create_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.HINGE)
        builder.set_boundary(Side.RIGHT, BoundaryType.HINGE)
        builder.set_fractional_settings(0.999999, 2)
        model = builder.create()

        result = self.solve(model).eigenvalues

        actual = result[:3]

        omega_rad_s = calc_first_eigenvalue(
            [BoundaryType.FIXED, BoundaryType.FIXED],
            L=self._length,
            E=1.,  # N/m^2
            I=1.,  # m^4
            A=1.,  # m^2
            rho=1.  # kg/m^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [
            9.86,  # 9.871 -> 1.571Hz
            39.33,  # 39.484 -> 6.284Hz
            88.1,  # 88.876 -> 14.145Hz
        ]

        np.testing.assert_allclose(expected, actual, atol=1e-1)

    def create_builder(self):
        return super(Beam1dDynamics31nodesTest, self)._create_builder(
            'EIGENPROBLEM', length=self._length, node_number=self._nodes_number
        )

    def solve(self, model):
        return super(Beam1dDynamics31nodesTest, self)._solve(
            model, AnalysisType.EIGENPROBLEM
        )


class BeamStaticsCaseTest(unittest.TestCase):
    """
    Results from https://doi.org/10.1016/j.ijmecsci.2020.105902
    """

    def setUp(self):
        self._length = 2.
        self._node_number = 81
        self._length_scale = 0.2  # note: in the article lf/L is provided; so lf/L = 0.1 is equivalent of 0.2 here
        self._span = self._length / float(self._node_number - 1)
        self._resolution = int(self._length_scale / self._span / 2.)

    def test_DownToUp_FixedAndAlphaAlmostOne_ReturnCorrectDisplacement(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.FIXED)
        builder.set_boundary(Side.RIGHT, BoundaryType.FIXED)
        builder.set_fractional_settings(0.999999999999, self._resolution)
        model = builder.create()

        result = self._solve(model)

        L = self._length
        L1, L2 = 0.5 * L, 0.5 * L
        E = 30e6
        I = 0.25 * 0.2 ** 3 / 12.
        P = -100.
        expected_max_theoretical = P * L1 ** 3 * L2 ** 3 / (3. * L ** 3 * E * I)
        expected_max = -0.000875

        np.testing.assert_allclose(min(result), [expected_max], atol=1e-6)

    def test_DownToUp_FixedAndAlpha06_ReturnCorrectDisplacement(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.FIXED)
        builder.set_boundary(Side.RIGHT, BoundaryType.FIXED)

        builder.set_fractional_settings(0.6, self._resolution)
        model = builder.create()

        with Profiler(False):
            result = self._solve(model)

        expected_max = -0.000887

        np.testing.assert_allclose(min(result), [expected_max], atol=1e-5)

    def test_DownToUp_SimplySupportedAndAlphaAlmostOne_ReturnCorrectDisplacement(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.HINGE)
        builder.set_boundary(Side.RIGHT, BoundaryType.HINGE)
        builder.set_fractional_settings(0.999999999999, self._resolution)

        model = builder.create()

        result = self._solve(model)

        L = self._length
        L1, L2 = 0.5 * L, 0.5 * L
        E = 30e6
        I = 0.25 * 0.2 ** 3 / 12.
        P = -100.
        expected_max_theoretical = P * L2 * (3. * L ** 2 - 4. * L2 ** 2) / (48 * E * I)
        expected_max = -0.003372
        np.testing.assert_allclose([expected_max], min(result), atol=1e-6)

    def test_DownToUp_SimplySupportedAndAlpha06_ReturnCorrectDisplacement(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(Side.LEFT, BoundaryType.HINGE)
        builder.set_boundary(Side.RIGHT, BoundaryType.HINGE)
        builder.set_fractional_settings(0.8, self._resolution)
        model = builder.create()

        with Profiler(False):
            result = self._solve(model)

        expected_max = -0.003439
        np.testing.assert_allclose([expected_max], min(result), atol=1e-6)

    def _create_predefined_builder(self):
        return (
            builder.create('beam1d', self._length, self._node_number)
            .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
            .set_load(LoadType.POINT, ordinate=0.5, magnitude=-100, )
            .add_virtual_nodes(8, 8)
            .add_middle_nodes()
            .set_length_scale_controller('vanish', self._length_scale, min_value=self._span)
            .set_young_modulus_controller('uniform', 30e6)
            .set_moment_of_inertia_controller('uniform', 0.25 * 0.2 ** 3 / 12.)
            .set_stiffness_operator_strategy('minimize_virtual_layer')
            .set_virtual_boundary_strategy(FractionalVirtualBoundaryStrategy.BASED_ON_FOURTH_DERIVATIVE)
            .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
        )

    @staticmethod
    def _solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model).displacement


class BeamStaticsCase2Test(unittest.TestCase):
    """
    Results from https://doi.org/10.3390/ma14081817
    """

    def setUp(self):
        self._length = 3054.*1e-6  # mm
        self._node_number = 101
        self._length_scale = 160.*1e-6  # mm
        self._span = self._length / float(self._node_number - 1)
        self._resolution = int(self._length_scale / self._span / 2.)

    def test_Statics_Fixed_AlphaAlmostOne_ReturnCorrectDisplacements(self):
        builder = (
            self.create_builder()
            .set_boundary(Side.LEFT, BoundaryType.FIXED)
            .set_boundary(Side.RIGHT, BoundaryType.FIXED)
            .set_fractional_settings(0.99999999, self._resolution)
        )

        model = builder.create()

        results = self.solve(model)

        actual = results.displacement

        L = self._length
        L1, L2 = 0.54 * L, 0.46 * L
        E = 295e3  # MPa
        d = 57.e-6  # mm
        I = math.pi*d**4/64.  # m^4
        P = 122.7e-9  # N
        expected_max_theoretical = P * L1 ** 3 * L2 ** 3 / (3. * L ** 3 * E * I)  # 0.1168
        expected_max = 121.1e-6  # mm

        np.testing.assert_allclose(max(actual), [expected_max], atol=1e-7)

    def test_Statics_Fixed_Alpha66_ReturnCorrectDisplacements(self):
        builder = (
            self.create_builder()
            .set_boundary(Side.LEFT, BoundaryType.FIXED)
            .set_boundary(Side.RIGHT, BoundaryType.FIXED)
            .set_fractional_settings(0.66, self._resolution)
        )

        model = builder.create()

        results = self.solve(model)

        actual = results.displacement

        expected_max = 116.4e-6

        np.testing.assert_allclose(max(actual), [expected_max], atol=1e-7)

    def create_builder(self):
        d = 57.e-6  # mm
        return (
            builder.create('beam1d', self._length, self._node_number)
            .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
            .add_virtual_nodes(8, 8)
            .add_middle_nodes()
            .set_load(LoadType.POINT, ordinate=0.54, magnitude=122.7e-9, )  # N
            .set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
            .set_length_scale_controller('vanish', self._length_scale, min_value=self._span)
            .set_young_modulus_controller('uniform', 295e3)  # N/mm^2
            .set_moment_of_inertia_controller('uniform', math.pi*d**4/64.)
            .set_stiffness_operator_strategy('minimize_virtual_layer')
            .set_virtual_boundary_strategy(FractionalVirtualBoundaryStrategy.BASED_ON_FOURTH_DERIVATIVE)
            .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
        )

    @staticmethod
    def solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)


class BeamDynamicsCase2Test(unittest.TestCase):
    """
    Results from https://doi.org/10.3390/ma14081817
    """

    def setUp(self):
        self._length = 4700.e-6  # mm
        self._node_number = 101
        self._length_scale = 160.e-6  # mm
        self._span = self._length / float(self._node_number - 1)
        self._resolution = int(self._length_scale / self._span / 2.)

    def test_Dynamics_Fixed_AlphaAlmostOne_ReturnCorrectEigenvalues(self):
        builder = (
            self.create_builder()
            .set_boundary(Side.LEFT, BoundaryType.FIXED)
            .set_boundary(Side.RIGHT, BoundaryType.FIXED)
            .set_fractional_settings(0.999999, self._resolution)
        )

        model = builder.create()

        results = self.solve(model)

        actual = results.eigenvalues[:3]

        d = 66.e-6  # mm
        h = (d ** 2 - (d / 2.) ** 2) ** 0.5  # nm
        omega_rad_s = calc_first_eigenvalue(
            bc=[BoundaryType.FIXED, BoundaryType.FIXED],
            L=self._length,
            E=295.e3,  # N/mm^2
            I=d * h ** 3 / 36.,  # mm^4
            A=0.5*d*h,  # mm^2
            rho=6.150e-9  # t/mm^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [  # rad/s
            92645726.,  # 14.74 MHz
            255337836.,
            500399080.
        ]

        np.testing.assert_allclose(expected, actual, atol=1e0)

    def test_Dynamics_Fixed_Alpha66_ReturnCorrectEigenvalues(self):
        builder = (
            self.create_builder()
            .set_boundary(Side.LEFT, BoundaryType.FIXED)
            .set_boundary(Side.RIGHT, BoundaryType.FIXED)
            .set_fractional_settings(0.66, self._resolution)
        )

        model = builder.create()

        results = self.solve(model)

        actual = results.eigenvalues[:3]

        expected = [
            94808600.,  # 15.09 MHz
            258421400.,
            496429200.
        ]

        np.testing.assert_allclose(expected, actual, atol=1e2)

    @unittest.skip('accuracy problem - todo')
    def test_Dynamics_Hinged_AlphaAlmostOne_ReturnCorrectEigenvalues(self):
        builder = (
            self.create_builder()
            .set_boundary(Side.LEFT, BoundaryType.HINGE)
            .set_boundary(Side.RIGHT, BoundaryType.HINGE)
            .set_fractional_settings(0.99999, self._resolution)
        )

        model = builder.create()

        results = self.solve(model)

        actual = results.eigenvalues[:3]

        d = 66.e-6  # mm
        h = (d ** 2 - (d / 2.) ** 2) ** 0.5  # nm
        omega_rad_s = calc_first_eigenvalue(
            bc=[BoundaryType.HINGE, BoundaryType.HINGE],
            L=self._length,
            E=295.e3,  # N/mm^2
            I=d * h ** 3 / 36.,  # mm^4
            A=0.5*d*h,  # mm^2
            rho=6.150e-9  # t/mm^3
        )
        omega_hz = omega_rad_s/(2.*math.pi)

        expected = [  # rad/s
            0.,  # 0 MHz
            0.,
            0.
        ]

        np.testing.assert_allclose(expected, actual, atol=1e0)

    def create_builder(self):
        triangle_b = 66.e-6  # mm
        triangle_h = (triangle_b**2 - (triangle_b/2.)**2)**0.5
        area = triangle_b*triangle_h*0.5

        rho = 6.150e-9  # t/mm^3
        mass_per_length = rho*area

        E = 295.e3  # N/mm^2
        I = triangle_b * triangle_h ** 3 / 36.  # mm^4

        return (
            builder.create('beam1d', self._length, self._node_number)
            .set_analysis_type('EIGENPROBLEM')
            .set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
            .set_density_controller('uniform', mass_per_length)
            .set_young_modulus_controller('uniform', E)
            .set_moment_of_inertia_controller('uniform', I)
            .add_middle_nodes()
            .add_virtual_nodes(8, 8)
            .set_length_scale_controller('vanish', self._length_scale, min_value=self._span)
            .set_stiffness_operator_strategy('minimize_virtual_layer')
            .set_virtual_boundary_strategy(FractionalVirtualBoundaryStrategy.BASED_ON_FOURTH_DERIVATIVE)
        )

    @staticmethod
    def solve(model):
        return solve(AnalysisType.EIGENPROBLEM, model)


def calc_first_eigenvalue(bc, L, E, I, A, rho):
    n = 1
    m = rho * A
    if bc == [BoundaryType.FIXED, BoundaryType.FIXED]:
        return ((n + 0.5) ** 2 * math.pi ** 2) / L ** 2 * (E * I / m) ** 0.5  # rad/s
    elif bc == [BoundaryType.HINGE, BoundaryType.HINGE]:
        return (n**2*math.pi**2)/L**2 * (E*I/m)**0.5
    else:
        raise NotImplementedError


def plot(x):
    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.show()


def plot2(xs, ys):
    import matplotlib.pyplot as plt
    for x, y in zip(xs, ys):
        plt.plot(x, y)
    plt.show()
