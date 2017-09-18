from fractulus.equation import create_riesz_caputo_stencil
from fdm import Operator, Number, Stencil


__all__ = ['create_fractional_deformation_operator']


def create_fractional_deformation_operator(settings):
    alpha = settings.alpha
    p = settings.resolution

    multiplier = p ** (alpha - 1)  # l_ef**(alpha-1) = p**(alpha-1) * 1./h**(1-alpha) <- lef describes by grid span

    return Operator(
        Number(multiplier) * create_riesz_caputo_stencil(settings, increase_order_by=1-alpha),
        Operator(  # todo: replace with Operator.order(n)
            Stencil.central(1.),
        )
    )