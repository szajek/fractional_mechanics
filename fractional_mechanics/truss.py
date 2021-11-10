from fdm import Operator, Number
from fractional_mechanics.equation import create_inwards_classic_derivative_dispatcher
from fractulus.equation import create_riesz_caputo_stencil


__all__ = ['create_riesz_caputo_strain_operator_by_pattern']


def create_riesz_caputo_strain_operator_by_pattern(method, settings, pattern, span):
    return create_riesz_caputo_strain_operator(
        method, settings,
        create_inwards_classic_derivative_dispatcher(pattern, span)
    )


def create_riesz_caputo_strain_operator(method, settings, element=None):
    alpha, lf, p = settings
    return Operator(
        Number(lf ** (alpha - 1)) * _operator_builder[method](settings),
        element
    )


def _create_operator_factory(_type):
    return lambda settings: create_riesz_caputo_stencil(_type, settings)


_operator_builder = {
    _type: _create_operator_factory(_type) for _type in ['caputo', 'rectangle', 'trapezoidal', 'simpson']
}


