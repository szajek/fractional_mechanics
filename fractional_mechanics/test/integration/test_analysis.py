import unittest

import numpy as np

import fractional_mechanics.builder as builder
from fdm.analysis import solve
from fdm.analysis.analyzer import AnalysisType


class Truss1dStatics6nodesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 6

    def test_ConstantSection_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.99999, 5)
                .set_length_scale_controller('uniform', .1)
                .add_virtual_nodes(2, 2)
                .set_field(builder.FieldType.LINEAR, a=1.)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FREE)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.AS_AT_BORDER)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.08],
                [0.152],
                [0.208],
                [0.24],
                [0.24],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha05_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.5, 3)
                .set_length_scale_controller('uniform', 0.6)
                .add_virtual_nodes(4, 4)
                .set_field(builder.FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[9.41385462e-16],
             [3.47172214e-01],
             [4.99518990e-01],
             [4.99518990e-01],
             [3.47172214e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha03_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.3, 3)
                .set_length_scale_controller('uniform', 0.6)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[5.46428567e-16],
             [9.59393722e-01],
             [1.74524563e+00],
             [1.74524563e+00],
             [9.59393722e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.9999, 3)
                .set_length_scale_controller('uniform', 0.6)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
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
                .set_length_scale_controller('uniform', 0.5)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[-2.79921788e-16],
             [2.93429923e-01],
             [3.71316963e-01],
             [3.71316963e-01],
             [2.93429923e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlpha03_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.3, 6)
                .set_length_scale_controller('uniform', 0.5)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[1.42549063e-16],
             [5.85829500e-01],
             [7.06500004e-01],
             [7.06500004e-01],
             [5.85829500e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlphaAlmostOne_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.9999, 6)
                .set_length_scale_controller('uniform', 0.5)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
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
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FREE)
                .set_fractional_settings(0.9999, 5)
                .set_length_scale_controller('uniform', 0.1)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
                .set_young_modulus_controller('user', young_modulus)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.AS_AT_BORDER)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[-3.92668354e-16],
             [8.42105263e-02],
             [1.54798762e-01],
             [2.08132095e-01],
             [2.38901326e-01],
             [2.38901326e-01],
             ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def _create_predefined_builder(self):
        return (
            builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    @staticmethod
    def _solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model).displacement


class Truss1dStatics31nodesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1
        self._node_number = 31

    def test_Call_Fractional_ReturnCorrectDisplacements(self):

        span = self._length / (self._node_number - 1)
        length_scale = span * 3.
        alpha = 0.7

        model = (
            self._create_predefined_builder()
            .set_fractional_settings(alpha, None)
            .set_length_scale_controller('step_vanish', length_scale, min_value=2. * span, span=span)
            .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
        ).create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
            [0.00369007],
            [0.00801109],
            [0.0127756],
            [0.01789321],
            [0.0233234],
            [0.02897823],
            [0.03473435],
            [0.0404404],
            [0.04591583],
            [0.05095925],
            [0.05536753],
            [0.05895956],
            [0.06159406],
            [0.06317172],
            [0.0636338],
            [0.06296574],
            [0.06119491],
            [0.05837687],
            [0.0545781],
            [0.0499014],
            [0.04452662],
            [0.03870359],
            [0.03270816],
            [0.02678991],
            [0.02113941],
            [0.01588936],
            [0.01111796],
            [0.0068419],
            [0.00308798],
            [0.],
        ])

        np.testing.assert_array_almost_equal(expected, result)

    def _create_predefined_builder(self):
        b = (
            builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_density_controller('spline_interpolated_linearly', 6)
                .add_virtual_nodes(1, 1)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_field(builder.FieldType.SINUSOIDAL, n=1.)
                .set_stiffness_operator_strategy('minimize_virtual_layer')
                .set_virtual_boundary_strategy('based_on_second_derivative')
                .set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)
        )

        b.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        return b

    @staticmethod
    def _solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)


class Truss1dEigenproblemTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 11

    def test_AlphaAlmostToOne_ReturnCorrectEigenValuesAndVectors(self):

        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.999999, 4)
                .set_length_scale_controller('uniform', 0.1)
                .add_virtual_nodes(1, 1)
        ).create()

        result = self._solve(model)

        expected_eigenvectors = np.array(
            [
                [0., 0.309017, 0.587785, 0.809017, 0.951057, 1., 0.951057, 0.809017, 0.587785, 0.309017, 0.],
                [0., 0.618034, 1., 1., 0.618034, 0., -0.618034, -1., -1., -0.618034, 0.],
                [0., 0.809017, 0.951057, 0.309017, -0.587785, -1., -0.587785, 0.309017, 0.951057, 0.809017, 0.]
            ]
        )
        expected_eigenvalues = [
            9.7887,
            38.197,
            82.443,
        ]  # rad/s

        for i, (expected_value, expected_vector) in enumerate(zip(expected_eigenvalues, expected_eigenvectors)):
            self.assertAlmostEqual(expected_value, result.eigenvalues[i], places=3)
            np.testing.assert_allclose(expected_vector, result.eigenvectors[i], atol=1e-5)

    def test_AlphaEqualsZeroFive_ReturnCorrectEigenValuesAndVectors(self):

        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.5, 4)
                .set_length_scale_controller('uniform', 0.1)
                .add_virtual_nodes(1, 1)
        ).create()

        result = self._solve(model)

        expected_eigenvectors = np.array(  # note: results obtained from simulation - not confirmed
            [
                [0., 0.364569, 0.61176, 0.8227, 0.954241, 1., 0.954241, 0.8227, 0.61176, 0.364569, 0.],
                [0., 0.71529, 1., 0.973215, 0.590133, 0., -0.590133, -0.973215, -1., -0.71529, 0.],
                [0., 0.937754, 0.897694, 0.242287, -0.618689, -1., -0.618689, 0.242287, 0.897694, 0.937754, 0.]
            ]
        )
        expected_eigenvalues = [  # note: results obtained from simulation - not confirmed
            8.9793402824906465,
            33.613075188092054,
            67.755313752691023,
        ]  # rad/s

        for i, (expected_value, expected_vector) in enumerate(zip(expected_eigenvalues, expected_eigenvectors)):
            self.assertAlmostEqual(expected_value, result.eigenvalues[i], places=3)
            np.testing.assert_allclose(expected_vector, result.eigenvectors[i], atol=1e-5)

    def _create_predefined_builder(self):
        return (
            builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('EIGENPROBLEM')
                .set_young_modulus_controller('uniform', value=1.)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_stiffness_operator_strategy('standard')
                .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    def _solve(self, model):
        return solve(AnalysisType.EIGENPROBLEM, model)


class Beam1dStatics6nodesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 21

    def test_ConstantSection_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        span = self._length/float(self._node_number - 1)
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.99999, 2)
                .set_length_scale_controller('vanish', 0.1, min_value=span)
                .add_virtual_nodes(8, 8)
                .set_load(builder.LoadType.MASS)
                .set_field(builder.FieldType.CONSTANT, value=1.)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_virtual_boundary_strategy('zero_value')
        ).create()

        result = self._solve(model)

        E = J = q = 1.
        expected_max_theoretical = -1./384.*q*self._length**4/(E*J)
        expected_max = -0.00315

        np.testing.assert_allclose(min(result), [expected_max], rtol=5e-2)

    def test_ConstantSection_Alpha08_ReturnCorrectDisplacement(self):
        span = self._length/float(self._node_number - 1)
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.8, 3)
                .set_length_scale_controller('vanish', 0.1, min_value=span)
                .add_virtual_nodes(8, 8)
                .set_load(builder.LoadType.MASS)
                .set_field(builder.FieldType.CONSTANT, value=1.)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_virtual_boundary_strategy('zero_value')
        ).create()

        result = self._solve(model)

        E = J = q = 1.
        expected_max_classical = -1./384.*q*self._length**4/(E*J)
        expected_max = -0.00345

        np.testing.assert_allclose(min(result), [expected_max], rtol=5e-1)

    def _create_predefined_builder(self):
        return (
            builder.create('beam1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    @staticmethod
    def _solve(model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model).displacement
