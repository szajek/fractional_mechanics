from fdm import Operator, Number, Stencil
from fractulus.equation import create_riesz_caputo_stencil


__all__ = ['create_riesz_caputo_operator_by_pattern']


def create_riesz_caputo_operator(settings, element=None):
    alpha, lf, p = settings
    return Operator(
        Number(lf ** (alpha - 1)) *
        create_riesz_caputo_stencil(settings),
        element
    )


def create_riesz_caputo_operator_by_pattern(settings, pattern, span):
    return create_riesz_caputo_operator(
        settings,
        _create_classic_derivative_dispatcher(pattern, span)
    )


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

