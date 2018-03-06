from fdm.builder import *
from fdm.utils import create_cached_factory
import fractional_mechanics
import fractulus


def boundary_equation_builder(side, boundary, span):
    _type, opts = boundary
    if _type == BoundaryType.FIXED:
        return fdm.create_linear_system_bc('dirichlet', value=opts.get('value', 0.))
    elif _type == BoundaryType.FREE:
        return fdm.create_linear_system_bc('equation', opts['operator'], free_value=opts.get('value', 0.))
    else:
        raise NotImplementedError


def create_vanish_length_scale_corrector(length, init_value, min_value):
    def correct(point):
        return max(min_value, min(init_value, point.x, length - point.x))

    return correct


def create_step_vanish_length_scale_corrector(length, init_value, min_value, span):
    interval_number = round(length/span)
    span_init_value = round(init_value/span)
    span_min_value = round(min_value/span)

    def correct(point):

        node_address = point.x / span

        span_value = max(
            span_min_value,
            min(
                span_init_value,
                int(node_address) + 1,
                (interval_number + 1) - round(node_address)
            )
        )
        return span_value * span

    return correct


LENGTH_SCALES_CONTROLLERS = {
    'vanish': create_vanish_length_scale_corrector,
    'step_vanish': create_step_vanish_length_scale_corrector,
}


class FractionalStiffnessSchemeFactory(StiffnessSchemeFactory):
    def _create_operators_set(self, stiffness_multiplier):
        fractional_deformation_operator_central = fdm.DynamicElement(
            self._create_fractional_operator_builder(
                self._context['fractional_operator_pattern']['central']
            )
        )
        fractional_deformation_operator_backward = fdm.DynamicElement(
            self._create_fractional_operator_builder(
                self._context['fractional_operator_pattern']['backward']
            )
        )
        fractional_deformation_operator_forward = fdm.DynamicElement(
            self._create_fractional_operator_builder(
                self._context['fractional_operator_pattern']['forward']
            )
        )

        E = self._context['young_modulus']

        fractional_ep_central = fdm.Operator(
            fdm.Stencil.central(span=self._span),
            stiffness_multiplier * fdm.Number(E) * fractional_deformation_operator_central
        )
        fractional_ep_backward = fdm.Operator(
            fdm.Stencil.backward(span=self._span),
            stiffness_multiplier * fdm.Number(E) * fractional_deformation_operator_backward
        )
        fractional_ep_forward = fdm.Operator(
            fdm.Stencil.forward(span=self._span),
            stiffness_multiplier * fdm.Number(E) * fractional_deformation_operator_forward
        )
        fractional_ep_forward_central = fdm.Operator(
            fdm.Stencil.forward(span=self._span),
            stiffness_multiplier * fdm.Number(E) * fractional_deformation_operator_central
        )
        fractional_ep_backward_central = fdm.Operator(
            fdm.Stencil.backward(span=self._span),
            stiffness_multiplier * fdm.Number(E) * fractional_deformation_operator_central
        )

        return {
            'central': fractional_ep_central,
            'forward': fractional_ep_forward,
            'backward': fractional_ep_backward,
            'forward_central': fractional_ep_forward_central,
            'backward_central': fractional_ep_backward_central,
            'central_deformation_operator': fractional_deformation_operator_central,
        }

    def _create_fractional_operator_builder(self, pattern):
        dynamic_lf = self._context['length_scale']

        def dynamic_resolution(point):
            return int(dynamic_lf(point) / self._span) if self._context['resolution'] is None else self._context['resolution']

        def create_stencil(point):
            return fractional_mechanics.create_riesz_caputo_operator_by_pattern(
                fractulus.CaputoSettings(self._context['alpha'], dynamic_lf(point), dynamic_resolution(point)),
                pattern,
                self._span
            ).to_stencil(Point(0))

        def create_id(point):
            return dynamic_lf(point), dynamic_resolution(point)

        return create_cached_factory(create_stencil, create_id)


class FractionalTruss1d(Truss1d):
    def __init__(self, length, nodes_number, stiffness_factory_builder):
        fdm.builder.Truss1d.__init__(self, length, nodes_number, stiffness_factory_builder)

        self._boundary_builder = boundary_equation_builder

        self._context['alpha'] = 0.8
        self._context['resolution'] = None

        default_length_scale = length * 0.1
        self._context['length_scale'] = UniformValueController(default_length_scale)
        self._context['fractional_operator_pattern'] = {
            'central': "CCC",
            'backward': "CCC",
            'forward': "CCC",
        }

    def set_fractional_settings(self, alpha, resolution):
        self._context['alpha'] = alpha
        self._context['resolution'] = resolution
        return self

    def set_length_scale_controller(self, _type, *args, **kwargs):

        if _type in LENGTH_SCALES_CONTROLLERS:
            dynamic_lf = UserValueController(
                LENGTH_SCALES_CONTROLLERS[_type](
                    self._length, *args, **kwargs)
            )
        else:
            dynamic_lf = VALUE_CONTROLLERS[_type](*args, **kwargs)

        self._context['length_scale'] = dynamic_lf
        return self

    def set_fractional_operator_pattern(self, **kwargs):
        self._context['fractional_operator_pattern'].update(kwargs)
        return self

    def create(self):
        return super().create()


def create(length, nodes_number):
    return FractionalTruss1d(length, nodes_number, FractionalStiffnessSchemeFactory)
