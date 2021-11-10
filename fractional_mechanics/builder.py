from fdm.builder import *
from fdm.utils import create_cached_factory
import fractional_mechanics
import fractulus


def create(type_, *args, **kwargs):
    builder = create_builder()
    return builder(type_, *args, **kwargs)


def create_builder():
    builder = Strategy()
    builder.register('truss1d', create_for_truss_1d)
    builder.register('beam1d', create_for_beam_1d)
    return builder


def create_for_truss_1d(length, nodes_number):
    builder = FractionalBuilder1D(length, nodes_number)
    builder.set_stiffness_factory(create_truss_stiffness_operators_set)

    operator_strategy = create_truss_fractional_operator_dispatcher_strategy()
    builder.set_stiffness_operator_dispatcher(operator_strategy)

    _register_strategy(builder)
    return builder


def create_for_beam_1d(length, nodes_number):
    builder = FractionalBuilder1D(length, nodes_number)
    builder.set_stiffness_factory(create_beam_stiffness_operators_set)

    operator_strategy = create_beam_fractional_operator_dispatcher_strategy()
    builder.set_stiffness_operator_dispatcher(operator_strategy)

    _register_strategy(builder)
    return builder


def _register_strategy(builder):

    virtual_eq_strategy = create_virtual_eq_strategy()
    bcs_eq_strategy = create_bcs_eq_strategy()

    builder.set_virtual_node_equation_strategy(virtual_eq_strategy)
    builder.set_boundary_equation_strategy(bcs_eq_strategy)
    return builder


class FractionalBuilder1D(Builder1d):
    def __init__(self, length, nodes_number):
        fdm.builder.Builder1d.__init__(self, length, nodes_number)

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
    settings_builder = dynamic_settings_builder(span, alpha, length_scale_controller, resolution)

    create_strain_operator_builder = create_fractional_strain_operator_builder(
        span, integration_method, settings_builder)

    fractional_deformation_operator_central = fdm.DynamicElement(
        create_strain_operator_builder(
            fractional_operator_pattern['central']
        )
    )
    fractional_deformation_operator_backward = fdm.DynamicElement(
        create_strain_operator_builder(
            fractional_operator_pattern['backward']
        )
    )
    fractional_deformation_operator_forward = fdm.DynamicElement(
        create_strain_operator_builder(
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


def create_beam_stiffness_operators_set(span, young_modulus_controller, alpha, resolution, length_scale_controller,
                                        integration_method, fractional_operator_pattern):

    settings_builder = dynamic_settings_builder(span, alpha, length_scale_controller, resolution)

    E = young_modulus_controller

    def moment_of_inertia_controller(point):
        return 1.

    I = moment_of_inertia_controller

    central_base_stencils = {
        'A': fdm.Stencil.central(span),
        'B': fdm.Stencil.central(span),
        'C': fdm.Stencil.central(span),
        'D': fdm.Stencil.central(span),
    }
    backward_base_stencils = {
        'A': fdm.Stencil.backward(span),
        'B': fdm.Stencil.backward(span),
        'C': fdm.Stencil.backward(span),
        'D': fdm.Stencil.backward(span),
    }
    forward_base_stencils = {
        'A': fdm.Stencil.forward(span),
        'B': fdm.Stencil.forward(span),
        'C': fdm.Stencil.forward(span),
        'D': fdm.Stencil.forward(span),
    }

    central_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        integration_method, central_base_stencils, settings_builder)
    backward_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        integration_method, backward_base_stencils, settings_builder)
    forward_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        integration_method, forward_base_stencils, settings_builder)

    central_operators = fractional_mechanics.create_beam_stiffness_operators_factory(central_stencils)
    backward_operators = fractional_mechanics.create_beam_stiffness_operators_factory(backward_stencils)
    forward_operators = fractional_mechanics.create_beam_stiffness_operators_factory(forward_stencils)

    def create_operator(operators):
        return fdm.Number(E) * fdm.Number(I) * operators['D']

    return {
        'central': create_operator(central_operators),
        'forward': create_operator(forward_operators),
        'backward': create_operator(backward_operators),
    }


def create_fractional_strain_operator_builder(span, integration_method, settings_builder):
    def build(pattern):
        def create_stencil(point):
            return fractional_mechanics.create_riesz_caputo_strain_operator_by_pattern(
                integration_method,
                settings_builder(point),
                pattern,
                span
            ).to_stencil(Point(0))

        def create_id(point):
            return settings_builder(point)

        return create_cached_factory(create_stencil, create_id)
    return build


def create_riesz_caputo_stencil_builder(integration_method, settings_builder, multiplier=None):
    def create(point):
        settings = settings_builder(point)
        stencil = fractulus.create_riesz_caputo_stencil(integration_method, settings)
        m = multiplier(point) if multiplier else 1.
        return (fdm.Number(m) * stencil).to_stencil(point)

    def create_id(point):
        return settings_builder(point)

    return create_cached_factory(create, create_id)


def dynamic_settings_builder(span, alpha, length_scale_controller, resolution):
    dynamic_lf = length_scale_controller

    def dynamic_resolution(point):
        return int(dynamic_lf(point) / span) if resolution is None else resolution

    def get(point):
        return fractulus.Settings(alpha, dynamic_lf(point), dynamic_resolution(point))
    return get


def create_truss_fractional_operator_dispatcher_strategy():
    operators_dispatcher = Strategy()
    operators_dispatcher.register('standard', create_standard_operator_dispatcher)
    operators_dispatcher.register('minimize_virtual_layer', create_minimize_virtual_layer_dispatcher_for_truss)
    return operators_dispatcher


def create_beam_fractional_operator_dispatcher_strategy():
    operators_dispatcher = Strategy()
    operators_dispatcher.register('standard', create_standard_operator_dispatcher)
    operators_dispatcher.register('minimize_virtual_layer', create_minimize_virtual_layer_dispatcher_for_beam)
    return operators_dispatcher


def create_minimize_virtual_layer_dispatcher_for_truss(*args):
    return _create_minimize_virtual_layer_dispatcher('forward_central', 'backward_central', *args)


def create_minimize_virtual_layer_dispatcher_for_beam(*args):
    return _create_minimize_virtual_layer_dispatcher('forward', 'backward', *args)


def _create_minimize_virtual_layer_dispatcher(forward_name, backward_name, operators, length, span):

    def dispatch(point):
        return {
            Point(0. + span): operators[forward_name],
            Point(length - span): operators[backward_name],
        }.get(point, operators['central'])

    return fdm.DynamicElement(dispatch)


def create_vanish_length_scale_corrector(length, init_value, min_value):
    def correct(point):
        return max(min_value, min(init_value, point.x, length - point.x))

    return correct


def create_step_vanish_length_scale_corrector(length, init_value, min_value, span):
    interval_number = round(length / span)
    span_init_value = round(init_value / span)
    span_min_value = round(min_value / span)

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
