from fractulus.equation import create_riesz_caputo_stencil
from fdm import Operator, Number, Stencil, DispatchedOperator


__all__ = ['create_fractional_deformation_operator']


def create_fractional_deformation_operator(settings, dynamic_resolution=None, stencil=None):
    alpha = settings.alpha

    def multiplier(address):
        p = dynamic_resolution(address) if dynamic_resolution is not None else settings.resolution
        return p ** (alpha - 1)  # l_ef**(alpha-1) = p**(alpha-1) * 1./h**(1-alpha) <- lef describes by grid span

    riesz_caputo_stencil = create_riesz_caputo_stencil(settings, increase_order_by=1 - alpha, dynamic_resolution=dynamic_resolution)

    return Operator(
        Number(multiplier) * riesz_caputo_stencil,
        Operator(  # todo: replace with Operator.order(n)
            stencil or Stencil.central(1.),
        )
    )
