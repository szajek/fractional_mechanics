import unittest

import numpy as np

import fractional_mechanics.builder as builder
from fdm.analysis import solve, AnalysisType


class TrussStaticEquationFractionalDifferencesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 6

    def test_ConstantSection_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.99999, .1, 5)
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
                .set_fractional_settings(0.5, 0.6, 3)
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
                .set_fractional_settings(0.3, 0.6, 3)
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
                .set_fractional_settings(0.9999, 0.6, 3)
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
                .set_fractional_settings(0.5, 0.5, 6)
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
                .set_fractional_settings(0.3, 0.5, 6)
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
                .set_fractional_settings(0.9999, 0.5, 6)
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
                .set_fractional_settings(0.9999, 0.1, 5)
                .add_virtual_nodes(3, 3)
                .set_field(builder.FieldType.CONSTANT, m=1.)
                .set_young_modulus(young_modulus)
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
            builder.create(self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    def _solve(self, model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model).displacement


class TrussDynamicEigenproblemEquationFractionalDifferencesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 11

    def test_AlphaAlmostToOne_ReturnCorrectEigenValuesAndVectors(self):

        model = (
            self._create_predefined_builder()
                .set_fractional_settings(0.999999, 0.1, 4)
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
                .set_fractional_settings(0.5, 0.1, 4)
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
            builder.create(self._length, self._node_number)
                .set_analysis_type('EIGENPROBLEM')
                .set_young_modulus(1.)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_operator_dispatcher_strategy('standard')
                .set_fractional_operator_pattern(central="FCB", backward="BBB", forward="FFF")
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    def _solve(self, model):
        return solve(AnalysisType.EIGENPROBLEM, model)

