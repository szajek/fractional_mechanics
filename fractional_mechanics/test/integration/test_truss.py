import unittest

import dicttools
import numpy as np
import itertools

from fdm.mesh import Mesh1DBuilder
from fdm.equation import Operator, Stencil, Number, LinearEquationTemplate
from fdm.geometry import Point
from fdm.model import create_bc, Model, VirtualNodeStrategy
from fdm.system import solve
from fractulus.equation import CaputoSettings
from fractional_mechanics.strains import create_riesz_caputo_operator_by_pattern


def _create_linear_function(a, b):
    def calc(point):
        return a*point.x + b
    return calc


def _create_mesh(length, node_number, virtual_nodes_number=2):
    dx = length / (node_number - 1)
    builder = Mesh1DBuilder(length)
    builder.add_uniformly_distributed_nodes(node_number)
    builder.add_virtual_nodes(*itertools.chain(
        [-(i+1)*dx for i in range(virtual_nodes_number)],
        [length + (i+1)*dx for i in range(virtual_nodes_number)]
    ))
    return builder.create()


def _create_equation(linear_operator, free_vector):
    return LinearEquationTemplate(
        linear_operator,
        free_vector
    )


def _build_fractional_operator(E, A, approx_span, settings):
    return Operator(
        Stencil.central(approx_span),
        Number(A) * Number(E) * create_riesz_caputo_operator_by_pattern(settings, "CCC", approx_span)
    )


def _create_fixed_and_free_end_bc(length, approx_span):
    return {
        Point(0): create_bc('dirichlet', value=0.),
        Point(length): create_bc('neumann', Stencil.backward(span=approx_span), value=0.)
    }


def _create_fixed_ends_bc(length, approx_span):
    return {
        Point(0): create_bc('dirichlet', value=0.),
        Point(length): create_bc('dirichlet', value=0.)
    }


_bcs = {
    'fixed_free': _create_fixed_and_free_end_bc,
    'fixed_fixed': _create_fixed_ends_bc,
}


def _create_bc(_type, length, approx_span):
    return _bcs[_type](length, approx_span)


def _create_virtual_nodes_bc(strategy, length, span, virtual_points_number):
    xs = [(i + 1)*span for i in range(virtual_points_number)]
    coords = itertools.chain(*[(-x, x) for x in xs])
    return {Point(x if x <0. else length + x): create_bc('virtual_node', x, strategy) for x in coords}


class TrussStaticEquationFractionalDifferencesTest(unittest.TestCase):
    def test_ConstantSection_ReturnCorrectDisplacement(self):
        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number)
        approx_span = length/(node_number - 1)
        settings = CaputoSettings(0.99999, .1, 5)

        results = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_free', 2, settings, load_function_coefficients=(-1., 0.))

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

        np.testing.assert_allclose(expected, results, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha05_ReturnCorrectDisplacement(self):
        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=4)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.5, 0.6, 3)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_fixed', 4, settings, load_function_coefficients=(0., -1.))

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
        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=3)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.3, 0.6, 3)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_fixed', 3, settings, load_function_coefficients=(0., -1.))

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
        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=3)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.9999, 0.6, 3)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_fixed', 3, settings, load_function_coefficients=(0., -1.))

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

        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=3)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.5, 0.5, 6)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_fixed', 3, settings, load_function_coefficients=(0., -1.))

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
        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=3)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.3, 0.5, 6)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_fixed', 3, settings, load_function_coefficients=(0., -1.))

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

        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=3)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.9999, 0.5, 6)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_fixed', 3, settings, load_function_coefficients=(0., -1.))

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

        node_number = 6
        length = 1.
        mesh = _create_mesh(length=length, node_number=node_number, virtual_nodes_number=3)
        approx_span = length / (node_number - 1)
        settings = CaputoSettings(0.9999, 0.1, 5)

        result = _solve_for_fractional('linear_system_of_equations', mesh, approx_span, 'fixed_free', 3, settings, load_function_coefficients=(0., -1.),
                                       cross_section=_create_linear_function(-1. / length, 2.)
                                       )

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


def _solve_for_fractional(analysis_type, mesh, approx_span, bc_type, virtual_points_number, settings, load_function_coefficients, cross_section=1.):
    a, b = load_function_coefficients
    length = mesh.boundary_box.dimensions[0]
    return solve(
        analysis_type,
        Model(
            _create_equation(
                _build_fractional_operator(A=cross_section, E=1., approx_span=approx_span, settings=settings),
                _create_linear_function(a=a, b=b),
            ),
            mesh,
            dicttools.merge(
                _create_bc(bc_type, length, approx_span),
                _create_virtual_nodes_bc(VirtualNodeStrategy.SYMMETRY, length, approx_span, virtual_points_number),
            )
        )
    )


class TrussDynamicEigenproblemEquationFractionalDifferencesTest(TrussStaticEquationFractionalDifferencesTest):
    @unittest.skip("No result to compare")
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):  # todo: compute result to compare
        mesh = _create_mesh(length=1., node_number=101)
        settings = CaputoSettings(0.8, 10, 10)
        ro = 2.

        result = _solve_for_fractional('eigenproblem', mesh, 'fixed_fixed', 3, settings,
                                       load_function_coefficients=(0., -ro))

        expected = np.array(
            [0.,],  # no result to compare
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)
