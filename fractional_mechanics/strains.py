from fdm import Operator, Number, Stencil
from fractulus.equation import create_riesz_caputo_stencil


__all__ = ['create_caputo_operator_by_pattern']


def create_caputo_operator(settings, element=None):
    alpha, lf, p = settings
    return Operator(
        Number(p ** (alpha - 1)) *  # l_ef**(alpha-1) = p**(alpha-1) * 1./h**(1-alpha) <- lef describes by grid span
        create_riesz_caputo_stencil(settings, increase_order_by=1 - alpha),
        element
    )


def create_caputo_operator_by_pattern(settings, pattern="CCC"):
    return create_caputo_operator(
        settings,
        _create_classic_derivative_dispatcher(pattern)
    )


def _create_classic_derivative_dispatcher(pattern):
    left, center, right = pattern

    stencils = {
        "B": Stencil.backward(1.),
        "C": Stencil.central(1.),
        "F": Stencil.forward(1.),
    }

    return lambda start, end, position: \
        Operator(
            stencils[{start: left, end: right}.get(position, center)]
        )

