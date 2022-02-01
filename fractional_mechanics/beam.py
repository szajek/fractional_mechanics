import fdm
import fdm.utils
import fractulus


__all__ = 'create_beam_stiffness_stencils_factory', 'create_beam_stiffness_operators_factory'


def create_beam_stiffness_operators_factory(stencils, settings_factory):
    A = create_beam_stiffness_operator_factory(settings_factory, stencils['A'])
    B = create_beam_stiffness_operator_factory(settings_factory, stencils['B'], A)
    C = create_beam_stiffness_operator_factory(settings_factory, stencils['C'], B)
    D = create_beam_stiffness_operator_factory(settings_factory, stencils['D'], C)
    return {'A': A, 'B': B, 'C': C, 'D': D}


def create_beam_stiffness_operator_factory(settings_factory, stencil_factory, element=None):
    def create(point):
        return fdm.Operator(stencil_factory(point), element if element else None)

    def id_builder(point):
        return settings_factory(point)

    factory = fdm.utils.create_cached_factory(create, id_builder)
    return fdm.DynamicElement(factory)


def create_beam_stiffness_stencils_factory(integration_method, base_stencils, settings_factory):
    def operator_factory(factory, base_stencil):
        def create(point):
            return fdm.Operator(factory(point), base_stencil).to_stencil(fdm.Point(0.))

        def id_builder(point):
            return settings_factory(point)

        return fdm.utils.create_cached_factory(create, id_builder)

    return {
        'A': operator_factory(create_A_stencil_factory(integration_method, settings_factory), base_stencils['A']),
        'B': operator_factory(create_B_stencil_factory(integration_method, settings_factory), base_stencils['B']),
        'C': operator_factory(create_C_stencil_factory(integration_method, settings_factory), base_stencils['C']),
        'D': operator_factory(create_D_stencil_factory(integration_method, settings_factory), base_stencils['D']),
    }


def create_A_stencil_factory(integration_method, settings_factory):
    def multiplier(point):
        settings = settings_factory(point)
        return settings.lf ** (settings.alpha - 1.)

    return create_riesz_caputo_stencil_factory(integration_method, settings_factory, multiplier=multiplier)


def create_B_stencil_factory(integration_method, settings_factory):
    def multiplier(point):
        settings = settings_factory(point)
        return settings.lf ** (2*settings.alpha - 2.)

    return create_riesz_caputo_stencil_factory(integration_method, settings_factory, multiplier=multiplier)


def create_C_stencil_factory(integration_method, settings_factory):
    def multiplier(point):
        settings = settings_factory(point)
        return settings.lf ** (settings.alpha - 1.)

    return create_riesz_caputo_stencil_factory(integration_method, settings_factory, multiplier=multiplier)


def create_D_stencil_factory(integration_method, settings_factory):
    return create_riesz_caputo_stencil_factory(integration_method, settings_factory, multiplier=None)


def create_riesz_caputo_stencil_factory(integration_method, settings_factory, multiplier=None):
    def create(point):
        settings = settings_factory(point)
        stencil = fractulus.create_riesz_caputo_stencil(integration_method, settings)
        if multiplier:
            m = multiplier(point)
            stencil = (fdm.Number(m) * stencil).to_stencil(fdm.Point(0.))
        return stencil

    return create
