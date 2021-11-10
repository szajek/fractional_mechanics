from fdm import Stencil, Operator


def create_inwards_classic_derivative_dispatcher(pattern, span):
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
