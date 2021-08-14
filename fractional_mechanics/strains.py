from fdm import Operator, Number, Stencil
from fractulus.equation import create_riesz_stencil


__all__ = ['create_riesz_operator_by_pattern']


def create_riesz_operator_by_pattern(method, settings, pattern, span):
    return create_riesz_operator(
        method, settings,
        _create_classic_derivative_dispatcher(pattern, span)
    )


def create_riesz_operator(method, settings, element=None):
    alpha, lf, p = settings
    return Operator(
        Number(lf ** (alpha - 1)) *
        _operator_builder[method](settings),
        element
    )


def _create_operator_factory(_type):
    return lambda settings: create_riesz_stencil(_type, settings)


_operator_builder = {
    _type: _create_operator_factory(_type) for _type in ['caputo', 'rectangle', 'trapezoidal', 'simpson']
}


def _create_classic_derivative_dispatcher(pattern, span):
    left, center, right = pattern

    stencils = {
        "B": Stencil.backward(span),
        "C": Stencil.central(span),
        "F": Stencil.forward(span),
    }

    return lambda start, end, position: \
        Operator(
            stencils[{start: left, end: right}.get(position, center)]
        )

