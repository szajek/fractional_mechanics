from fdm.builder import *
from fdm.utils import create_cached_factory
import fractional_mechanics
import fractulus


def create_for_truss_1d(length, nodes_number):
    builder = FractionalBuilder1D(length, nodes_number)
    builder.set_stiffness_factory(create_truss_stiffness_operators_set)
    return builder


_builders = {
    'truss1d': create_for_truss_1d
}


def create(type_, *args, **kwargs):
    return _builders[type_](*args, **kwargs)


class FractionalBuilder1D(Builder1d):
    def __init__(self, length, nodes_number):
        fdm.builder.Builder1d.__init__(self, length, nodes_number)

        self._boundary_builder = boundary_equation_builder

        self._context['alpha'] = 0.8
        self._context['resolution'] = None
        self._context['integration_method'] = 'caputo'

        default_length_scale = length * 0.1
        self.length_scale_controller = None
        self.set_length_scale_controller('uniform', default_length_scale)
        self._context['fractional_operator_pattern'] = {
            'central': "CCC",
            'backward': "CCC",
            'forward': "CCC",
        }

    def set_fractional_settings(self, alpha, resolution):
        self._context['alpha'] = alpha
        self._context['resolution'] = resolution
        return self

    def set_fractional_integration_method(self, method):
        self._context['integration_method'] = method
        return self

    def set_length_scale_controller(self, _type, *args, **kwargs):

        if _type in LENGTH_SCALES_CONTROLLERS:
            dynamic_lf = UserValueController(
                self._length, self._nodes_number,
                LENGTH_SCALES_CONTROLLERS[_type](
                    self._length, *args, **kwargs)
            )
        else:
            dynamic_lf = self._create_value_controller(_type, *args, **kwargs)

        self.length_scale_controller = self._context['length_scale_controller'] = dynamic_lf
        return self

    def set_fractional_operator_pattern(self, **kwargs):
        self._context['fractional_operator_pattern'].update(kwargs)
        return self

    def create(self):
        return super().create()

    def _create_stiffness_operators_set(self):
        return self._stiffness_factory(
            self._span, self._get_corrected_young_modulus, self._context['alpha'], self._context['resolution'],
            self._context['length_scale_controller'], self._context['integration_method'],
            self._context['fractional_operator_pattern'])

    @property
    def length_scale(self):
        return self._revolve_for_points(self.length_scale_controller.get)


def create_truss_stiffness_operators_set(span, young_modulus_controller, alpha, resolution, length_scale_controller,
                                         integration_method, fractional_operator_pattern):
    def create_fractional_operator_builder(pattern):
        dynamic_lf = length_scale_controller

        def dynamic_resolution(point):
            return int(dynamic_lf(point) / span) if resolution is None else resolution

        def create_stencil(point):
            return fractional_mechanics.create_riesz_operator_by_pattern(
                integration_method,
                fractulus.Settings(alpha, dynamic_lf(point), dynamic_resolution(point)),
                pattern,
                span
            ).to_stencil(Point(0))

        def create_id(point):
            return dynamic_lf(point), dynamic_resolution(point)

        return create_cached_factory(create_stencil, create_id)

    fractional_deformation_operator_central = fdm.DynamicElement(
        create_fractional_operator_builder(
            fractional_operator_pattern['central']
        )
    )
    fractional_deformation_operator_backward = fdm.DynamicElement(
        create_fractional_operator_builder(
            fractional_operator_pattern['backward']
        )
    )
    fractional_deformation_operator_forward = fdm.DynamicElement(
        create_fractional_operator_builder(
            fractional_operator_pattern['forward']
        )
    )

    E = young_modulus_controller

    fractional_ep_central = fdm.Operator(
        fdm.Stencil.central(span=span),
        fdm.Number(E) * fractional_deformation_operator_central
    )
    fractional_ep_backward = fdm.Operator(
        fdm.Stencil.backward(span=span),
        fdm.Number(E) * fractional_deformation_operator_backward
    )
    fractional_ep_forward = fdm.Operator(
        fdm.Stencil.forward(span=span),
        fdm.Number(E) * fractional_deformation_operator_forward
    )
    fractional_ep_forward_central = fdm.Operator(
        fdm.Stencil.forward(span=span),
        fdm.Number(E) * fractional_deformation_operator_central
    )
    fractional_ep_backward_central = fdm.Operator(
        fdm.Stencil.backward(span=span),
        fdm.Number(E) * fractional_deformation_operator_central
    )

    return {
        'central': fractional_ep_central,
        'forward': fractional_ep_forward,
        'backward': fractional_ep_backward,
        'forward_central': fractional_ep_forward_central,
        'backward_central': fractional_ep_backward_central,
    }


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

