from fractulus.equation import create_riesz_caputo_stencil, CaputoSettings
from fdm import Operator, Number, Stencil, DispatchedOperator, DynamicElement


__all__ = ['create_fractional_deformation_operator']


def create_fractional_deformation_operator(settings, stencil=None):
    alpha, lf, p = settings

    multiplier = p ** (alpha - 1)  # l_ef**(alpha-1) = p**(alpha-1) * 1./h**(1-alpha) <- lef describes by grid span

    caputo_stencil = create_riesz_caputo_stencil(settings, increase_order_by=1 - alpha)

    # return Operator(
    #     Number(multiplier) * riesz_caputo_stencil,
    #     Operator(  # todo: replace with Operator.order(n)
    #         stencil or Stencil.central(1.),
    #     )
    # )

    def dispatcher(position):
        if position == DispatchedOperator.Position.START:
            return Operator(  # todo: replace with Operator.order(n)
                stencil or Stencil.forward(1.),
            )
        elif position == DispatchedOperator.Position.CENTER:
            return Operator(  # todo: replace with Operator.order(n)
                stencil or Stencil.central(1.),
            )
        elif position == DispatchedOperator.Position.END:
            return Operator(  # todo: replace with Operator.order(n)
                stencil or Stencil.backward(1.),
            )

    return DispatchedOperator(Number(multiplier) * caputo_stencil, dispatcher)