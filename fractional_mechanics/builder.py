import fdm
from fdm.builder import *
from fdm.utils import create_cached_factory
import fractional_mechanics
import fractulus


class FractionalVirtualBoundaryStrategy(enum.Enum):
    BASED_ON_SECOND_DERIVATIVE = 0
    BASED_ON_FOURTH_DERIVATIVE = 1


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
    builder.set_stiffness_factory({
        AnalysisStrategy.UP_TO_DOWN: create_truss_stiffness_operators_up_to_down,
        AnalysisStrategy.DOWN_TO_UP: create_truss_stiffness_operators_down_to_up,
    })
    builder.set_complex_boundary_factory(create_complex_truss_bcs())
    return builder


def create_for_beam_1d(length, nodes_number):
    builder = FractionalBuilder1D(length, nodes_number)
    builder.set_stiffness_factory({
        AnalysisStrategy.UP_TO_DOWN: create_beam_stiffness_operators_up_to_down,
        AnalysisStrategy.DOWN_TO_UP: create_beam_stiffness_operators_down_to_up,
    })
    builder.set_complex_boundary_factory(create_complex_beam_bcs())
    return builder


FractionalStiffnessInput = collections.namedtuple('FractionalStiffnessInput', (
    'mesh', 'length', 'span', 'strategy', 'young_modulus_controller', 'alpha', 'resolution',
    'length_scale_controller', 'moment_of_inertia_controller', 'integration_method', 'fractional_operator_pattern'
))

FractionalBCsInput = collections.namedtuple('FractionalBCsInput', (
    'mesh', 'length', 'span', 'virtual_nodes_strategy', 'alpha', 'resolution', 'length_scale_controller',
    'moment_of_inertia_controller', 'integration_method', 'young_modulus_controller'
))


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

    def _create_stiffness_stencils(self, mesh):
        data = FractionalStiffnessInput(
            mesh, self._length, self._span,
            self._context['stiffness_operator_strategy'],
            self._get_corrected_young_modulus,
            self._context['alpha'],
            self._context['resolution'],
            self._context['length_scale_controller'],
            self.moment_of_inertia_controller,
            self._context['integration_method'],
            self._context['fractional_operator_pattern']
        )

        return self._stiffness_factory[self._analysis_strategy](data)

    def _create_complex_bcs(self, mesh):
        data = FractionalBCsInput(
            mesh, self._length, self._span,
            self._context['virtual_boundary_strategy'],
            self._context['alpha'],
            self._context['resolution'],
            self._context['length_scale_controller'],
            self.moment_of_inertia_controller,
            self._context['integration_method'],
            self._get_corrected_young_modulus,
        )

        return self._complex_boundary_factory(
            self._context['analysis_type'], self._context['boundary'], data)

    @property
    def length_scale(self):
        return self._revolve_for_points(self.length_scale_controller.get)


def create_truss_stiffness_operators_up_to_down(data):
    span = data.span
    resolution = data.resolution
    alpha = data.alpha
    length = data.length

    settings_builder = dynamic_settings_builder(span, alpha, data.length_scale_controller, resolution)

    create_strain_operator_builder = create_fractional_strain_operator_builder(
        span, data.integration_method, settings_builder)

    fractional_deformation_operator_central = fdm.DynamicElement(
        create_strain_operator_builder(
            data.fractional_operator_pattern['central']
        )
    )
    fractional_deformation_operator_backward = fdm.DynamicElement(
        create_strain_operator_builder(
            data.fractional_operator_pattern['backward']
        )
    )
    fractional_deformation_operator_forward = fdm.DynamicElement(
        create_strain_operator_builder(
            data.fractional_operator_pattern['forward']
        )
    )

    E = data.young_modulus_controller

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

    operators = {
        'central': fractional_ep_central,
        'forward': fractional_ep_forward,
        'backward': fractional_ep_backward,
        'forward_central': fractional_ep_forward_central,
        'backward_central': fractional_ep_backward_central,
    }

    def dispatch(point):
        return {
            Point(0. + span): operators['forward_central'],
            Point(length - span): operators['backward_central'],
        }.get(point, operators['central'])

    if data.strategy == 'minimize_virtual_layer':
        return fdm.DynamicElement(dispatch)
    else:  # standard
        return operators['central']


def create_truss_stiffness_operators_down_to_up(data):
    return []


def create_beam_stiffness_operators_up_to_down(data):
    span = data.span
    resolution = data.resolution
    alpha = data.alpha
    length = data.length

    settings_builder = dynamic_settings_builder(span, alpha, data.length_scale_controller, resolution)

    E = data.young_modulus_controller
    I = data.moment_of_inertia_controller

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
        data.integration_method, central_base_stencils, settings_builder)
    backward_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, backward_base_stencils, settings_builder)
    forward_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, forward_base_stencils, settings_builder)

    central_operators = fractional_mechanics.create_beam_stiffness_operators_factory(central_stencils)
    backward_operators = fractional_mechanics.create_beam_stiffness_operators_factory(backward_stencils)
    forward_operators = fractional_mechanics.create_beam_stiffness_operators_factory(forward_stencils)

    def create_operator(operators):
        return fdm.Number(E) * fdm.Number(I) * operators['D']

    operators = {
        'central': create_operator(central_operators),
        'forward': create_operator(forward_operators),
        'backward': create_operator(backward_operators),
    }

    def dispatch(point):
        return {
            Point(0. + span): operators['forward'],
            Point(length - span): operators['backward'],
        }.get(point, operators['central'])

    if data.strategy == 'minimize_virtual_layer':
        return fdm.DynamicElement(dispatch)
    else:  # standard
        return operators['central']


def create_beam_stiffness_operators_down_to_up(data):
    span = data.span
    resolution = data.resolution
    alpha = data.alpha
    mesh = data.mesh

    settings_builder = dynamic_settings_builder(span, alpha, data.length_scale_controller, resolution)

    E = data.young_modulus_controller
    I = data.moment_of_inertia_controller

    EI = fdm.Number(E) * fdm.Number(I)

    central_base_stencils = {
        'A': EI*fdm.Stencil.central(span),
        'B': fdm.Stencil.central(span),
        'C': fdm.Stencil.central(span),
        'D': fdm.Stencil.central(span),
    }
    wide_central_base_stencils = {
        'A': EI*fdm.Stencil.central(2.*span),
        'B': fdm.Stencil.central(2.*span),
        'C': fdm.Stencil.central(2.*span),
        'D': fdm.Stencil.central(2.*span),
    }
    backward_base_stencils = {
        'A': EI*fdm.Stencil.backward(span),
        'B': fdm.Stencil.backward(span),
        'C': fdm.Stencil.backward(span),
        'D': fdm.Stencil.backward(span),
    }
    forward_base_stencils = {
        'A': EI*fdm.Stencil.forward(span),
        'B': fdm.Stencil.forward(span),
        'C': fdm.Stencil.forward(span),
        'D': fdm.Stencil.forward(span),
    }

    central_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, central_base_stencils, settings_builder)
    wide_central_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, wide_central_base_stencils, settings_builder)
    backward_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, backward_base_stencils, settings_builder)
    forward_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, forward_base_stencils, settings_builder)

    def get_stencils(key):
        return [central_stencils[key], wide_central_stencils[key], backward_stencils[key], forward_stencils[key]]

    def null_factory(point):
        return fdm.Stencil.null()

    def build_dispatcher(rules, operator_name):
        def dispatch(point):
            central, wide_central, backward, forward = get_stencils(operator_name)

            if point in rules.null:
                factory = null_factory
            elif point in rules.forward:
                factory = forward
            elif point in rules.backward:
                factory = backward
            elif point in rules.wide_central:
                factory = wide_central
            else:
                factory = central
            return factory(point)
        return dispatch

    class Rules:
        def __init__(self, null=(), backward=(), forward=(), wide_central=()):
            self.null = null
            self.backward = backward
            self.forward = forward
            self.wide_central = wide_central

    def p(idx):
        return Point(idx*span)

    virtual_nodes = mesh.virtual_nodes
    vn = len(virtual_nodes)
    hvn = int(vn/2.)
    n = len(mesh.real_nodes) - 1
    hn = int(n/2.)
    left_virtual_nodes = virtual_nodes[:hvn]
    right_virtual_nodes = virtual_nodes[hvn:]
    real_nodes = mesh.real_nodes

    a_range = hn
    a_rules = Rules(
        null=left_virtual_nodes[-2:] + right_virtual_nodes[-2:],
        forward=left_virtual_nodes[:a_range] + real_nodes[:a_range],
        backward=right_virtual_nodes[:a_range] + real_nodes[-a_range:],
    )
    b_range = hn
    b_rules = Rules(
        null=left_virtual_nodes[-2:] + right_virtual_nodes[-2:],
        # forward=left_virtual_nodes[:b_range] + real_nodes[:b_range],
        # backward=right_virtual_nodes[:b_range] + real_nodes[-b_range:],
    )
    c_range = hn
    c_rules = Rules(
        null=left_virtual_nodes[-2:] + right_virtual_nodes[-2:],
        forward=left_virtual_nodes[:c_range] + real_nodes[:c_range],
        backward=right_virtual_nodes[:c_range] + real_nodes[-c_range:],
    )
    scheme_null = left_virtual_nodes + right_virtual_nodes
    d_range = hn
    d_rules = Rules(
        null=scheme_null,
        # forward=real_nodes[1:d_range],
        # backward=real_nodes[-d_range:-1],
    )

    if data.strategy == 'minimize_virtual_layer':
        return [
            fdm.DynamicElement(build_dispatcher(a_rules, 'A')),
            fdm.DynamicElement(build_dispatcher(b_rules, 'B')),
            fdm.DynamicElement(build_dispatcher(c_rules, 'C')),
            fdm.DynamicElement(build_dispatcher(d_rules, 'D')),
        ]
    else:  # standard
        return central_stencils


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


def create_complex_beam_bcs():
    strategy = Strategy()
    strategy.register(fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, create_beam_statics_bcs)
    strategy.register(fdm.AnalysisType.EIGENPROBLEM, create_beam_eigenproblem_bc)
    return strategy


def create_beam_statics_bcs(boundary, data):
    span = data.span
    alpha = data.alpha
    resolution = data.resolution
    mesh = data.mesh

    begin_node, end_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    begin_displacement_fixed = static_boundary(fdm.Scheme({begin_node: 1.}), 0.)
    end_displacement_fixed = static_boundary(fdm.Scheme({end_node: 1.}), 0.)

    settings_builder = dynamic_settings_builder(span, alpha, data.length_scale_controller, resolution)

    E = data.young_modulus_controller

    def moment_of_inertia_controller(point):
        return 1.

    I = moment_of_inertia_controller

    EI = fdm.Number(E) * fdm.Number(I)

    central_base_stencils = {
        'A': EI*fdm.Stencil.central(span),
        'B': fdm.Stencil.central(span),
        'C': fdm.Stencil.central(span),
        'D': fdm.Stencil.central(span),
    }

    central_stencils = fractional_mechanics.create_beam_stiffness_stencils_factory(
        data.integration_method, central_base_stencils, settings_builder)

    operators = fractional_mechanics.create_beam_stiffness_operators_factory(central_stencils)

    begin_A_scheme = operators['A'].expand(begin_node)
    end_A_scheme = operators['A'].expand(end_node)

    begin_B_scheme = operators['B'].expand(begin_node)
    end_B_scheme = operators['B'].expand(end_node)

    begin_rotation_zero = static_boundary(begin_A_scheme, 0.)
    end_rotation_zero = static_boundary(end_A_scheme, 0.)

    begin_moment_zero = static_boundary(begin_B_scheme, 0.)
    end_moment_zero = static_boundary(end_B_scheme, 0.)

    bcs = []

    left_type = boundary[Side.LEFT].type
    right_type = boundary[Side.RIGHT].type

    if left_type == BoundaryType.FIXED:
        bcs += [
            begin_displacement_fixed,
            begin_rotation_zero,
        ]
    elif left_type == BoundaryType.HINGE:
        bcs += [
            begin_displacement_fixed,
            begin_moment_zero,
        ]

    if right_type == BoundaryType.FIXED:
        bcs += [
            end_displacement_fixed,
            end_rotation_zero,
        ]
    elif right_type == BoundaryType.HINGE:
        bcs += [
            end_displacement_fixed,
            end_moment_zero,
        ]

    def p(s, base=0.):
        return Point(base + span * s)

    left_vbc_stencil = fdm.Stencil({p(-2): -1., p(-1): 4., p(0): -5., p(2): 5., p(3): -4., p(4): 1.})
    right_vbc_stencil = fdm.Stencil({p(2): -1., p(1): 4., p(0): -5., p(-2): 5., p(-3): -4., p(-4): 1.})

    virtual_nodes = mesh.virtual_nodes
    vn = len(virtual_nodes)
    hvn = int(vn / 2.)
    left_virtual_nodes = virtual_nodes[:hvn]
    right_virtual_nodes = virtual_nodes[hvn:]

    bcs += [static_boundary(left_vbc_stencil.expand(node), 0.) for node in left_virtual_nodes[:-2]]
    bcs += [static_boundary(right_vbc_stencil.expand(node), 0.) for node in right_virtual_nodes[:-2]]

    return bcs


def create_beam_eigenproblem_bc(length, span, mesh, boundary):  # todo:
    pass
    return {}


def create_complex_truss_bcs():
    strategy = Strategy()
    strategy.register(fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, create_truss_statics_bcs)
    strategy.register(fdm.AnalysisType.EIGENPROBLEM, create_truss_eigenproblem_bc)
    return strategy


def create_truss_statics_bcs(boundary, data):
    mesh = data.mesh
    span = data.span

    begin_node, end_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    begin_displacement_fixed = static_boundary(fdm.Scheme({begin_node: 1.}), 0.)
    end_displacement_fixed = static_boundary(fdm.Scheme({end_node: 1.}), 0.)

    bcs = []
    if boundary[Side.LEFT].type == BoundaryType.FIXED:
        bcs += [
            begin_displacement_fixed,
        ]

    if boundary[Side.RIGHT].type == BoundaryType.FIXED:
        bcs += [
            end_displacement_fixed,
        ]

    def p(s, base=0.):
        return Point(base + span * s)

    virtual_nodes = mesh.virtual_nodes
    vn = len(virtual_nodes)
    hvn = int(vn / 2.)
    left_virtual_nodes = virtual_nodes[:hvn]
    right_virtual_nodes = virtual_nodes[hvn:]

    if data.virtual_nodes_strategy == VirtualBoundaryStrategy.SYMMETRY:
        symmetry_stencil = fdm.Stencil({p(-1): -1., p(1): 1.})
        bcs += [
            static_boundary(symmetry_stencil.expand(left_virtual_nodes[0]), 0.),
            static_boundary(symmetry_stencil.expand(right_virtual_nodes[0]), 0.)
        ]
    elif data.virtual_nodes_strategy == FractionalVirtualBoundaryStrategy.BASED_ON_SECOND_DERIVATIVE:
        left_vbc_stencil = fdm.Stencil({p(0): -1., p(1): 3., p(2): -3., p(3): 1.})
        right_vbc_stencil = fdm.Stencil({p(-3): 1., p(-2): -3., p(-1): 3., p(0): -1.})

        bcs += [
            static_boundary(left_vbc_stencil.expand(left_virtual_nodes[0]), 0.),
            static_boundary(right_vbc_stencil.expand(right_virtual_nodes[0]), 0.)
        ]
    else:
        raise NotImplementedError

    return bcs


def create_truss_eigenproblem_bc(boundary, data):
    mesh = data.mesh

    begin_node, end_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    begin_displacement_fixed = dynamic_boundary(fdm.Scheme({begin_node: 1.}), fdm.Scheme({}), replace=begin_node)
    end_displacement_fixed = dynamic_boundary(fdm.Scheme({end_node: 1.}), fdm.Scheme({}), replace=end_node)

    bcs = []
    if boundary[Side.LEFT].type == BoundaryType.FIXED:
        bcs += [
            begin_displacement_fixed,
        ]

    if boundary[Side.RIGHT].type == BoundaryType.FIXED:
        bcs += [
            end_displacement_fixed,
        ]

    return bcs


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
