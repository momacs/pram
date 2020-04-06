"""
Basic reproduction number (R0)
    Varicella            10-12
    Measles              16-18
    Rotavirus            16-25
    Smallpox             3-10
    Spanish flu          2.0 [1.5 - 2.8]
    Seasonal influenza   1.3 [0.9 - 1.8]
    H1N1 swine flu 2019  1.2 - 1.5
"""

from abc import abstractmethod, ABC

from dotmap import DotMap

from .model import Model, ModelConstructionError, MCSolver, ODESolver
from ..rule import TimeAlways, IterAlways, DiscreteInvMarkovChain, ODEDerivatives, ODESystemMass

__all__ = ['SISModel', 'SIRModel', 'SIRSModel', 'SEIRModel', 'SEQIHRModel']


# ----------------------------------------------------------------------------------------------------------------------
# ODE models deriavatives:

class SIRModelDerivatives(ODEDerivatives):
    '''
    Model parameters
        beta  - Transmission rate (or effective contact rate)
        gamma - Recovery rate

    Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
    Royal Society A. 115(772), 700--721.

    http://www.public.asu.edu/~hnesse/classes/sir.html
    '''

    def __init__(self, beta, gamma):
        self.params = DotMap(beta=beta, gamma=gamma)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,I,R = state
            N = sum(state)
            return [
                -p.beta * S * I / N,                # dS/dt
                 p.beta * S * I / N - p.gamma * I,  # dI/dt
                                      p.gamma * I   # dR/dt
            ]
        return fn


class SIRSModelDerivatives(ODEDerivatives):
    pass


class SEIRModelDerivatives(ODEDerivatives):
    '''
    Model parameters
        beta  - Transmission rate (or effective contact rate)
        k     - Progression rate from exposed (latent) to infected
        gamma - Recovery rate
    '''

    def __init__(self, beta, k, gamma):
        self.params = DotMap(beta=beta, k=k, gamma=gamma)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,E,I,R = state
            N = sum(state)
            return [
                -p.beta * S * I / N,                # dS/dt
                 p.beta * S * I / N - p.k     * E,  # dE/dt
                 p.k    * E         - p.gamma * I,  # dI/dt
                                      p.gamma * I   # dR/dt
            ]
        return fn


class SEQIHRModelDerivatives(ODEDerivatives):
    '''
    Model parameters
        beta             - Transmission coefficient
        alpha_n, alpha_q - Rate at which non-quarantined and quarantined individuals become infectious
        delta_n, delta_h - Rate at which non-isolated and isolated individuals become recovered
        mu               - Natural death rate
        chi, phi         - Rate of quarantine and isolation
        rho              - Isolation efficiency [0..1]

    In the source below, parameter names used are:
        alpha_1 instead of alpha_n
        alpha_2 instead of alpha_q
        delta_1 instead of delta_n
        delta_2 instead of delta_h

    "(...) some of the drawbacks of the simple model when used to evaluate intervention policies. We argue that the
    main reason for these problems is due to the simplifying assumption of exponential distributions for the
    disease stages, which is used in the model. This provides a motivation for using more realistic stage
    distributions. Non-exponential distributions have been considered in epidemiological models (see, for
    example, ...). However, none of these studies focuses on the evaluation of intervention policies."

    Feng & Xu (2007) Epidemiological Models with Non-Exponentially Distributed Disease Stages and Applications to
    Disease Control.  Bulletin of Mathematical Biology.
    '''

    def __init__(self, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho):
        self.params = DotMap(beta=beta, alpha_n=alpha_n, alpha_q=alpha_q, delta_n=delta_n, delta_h=delta_h, mu=mu, chi=chi, phi=phi, rho=rho)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,E,Q,I,H,R = state
            N = sum(state)
            return [
                p.mu * N - p.beta * S * (I + (1.0 - p.rho) * H) / N - p.mu * S,             # dS/dt
                p.beta * S * (I + (1.0 - p.rho) * H) / N - (p.chi + p.alpha_n + p.mu) * E,  # dE/dt
                p.chi * E - (p.alpha_q + p.mu) * Q,                                         # dQ/dt
                p.alpha_n * E - (p.phi + p.delta_n + p.mu) * I,                             # dI/dt
                p.alpha_q * Q + p.phi * I - (p.delta_h + p.mu) * H,                         # dH/dt
                p.delta_n * I + p.delta_h * H - p.mu * R                                    # dR/dt
            ]
        return fn


# ----------------------------------------------------------------------------------------------------------------------
# Markov chain models:

class SISModel_MC(DiscreteInvMarkovChain):
    '''
    The SIS epidemiological model without vital dynamics (Markov chain implementation).

    Model parameters:
        beta  - Transmission rate (or effective contact rate)
        gamma - Recovery rate
    '''

    def __init__(self, var, beta, gamma, name='sis-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1.0 - beta, beta], 'i': [gamma, 1.0 - gamma] }, name, t, i, memo)


class SIRModel_MC(DiscreteInvMarkovChain):
    '''
    The SIR epidemiological model without vital dynamics (Markov chain implementation).

    Model parameters:
        beta  - Transmission rate (or effective contact rate)
        gamma - Recovery rate
    '''

    def __init__(self, var, beta, gamma, name='sir-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1.0 - beta, beta, 0.0], 'i': [0.0, 1.0 - gamma, gamma], 'r': [0.0, 0.0, 1.0] }, name, t, i, memo)




class SIRSModel_MC(DiscreteInvMarkovChain):
    '''
    The SIRS epidemiological model without vital dynamics (Markov chain implementation).

    Model parameters:
        beta  - Transmission rate (or effective contact rate)
        gamma - Recovery rate
        alpha - Immunity loss rate (alpha = 0 implies life-long immunity and consequently the SIR model)
    '''

    def __init__(self, var, beta, gamma, alpha=0.0, name='sirs-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1.0 - beta, beta, 0.0], 'i': [0.0, 1.0 - gamma, gamma], 'r': [alpha, 0.0, 1.0 - alpha] }, name, t, i, memo)


class SEIRModel_MC(DiscreteInvMarkovChain):
    '''
    The SEIR epidemiological model without vital dynamics (Markov chain implementation).

    Model parameters:
        beta  - Transmission rate (or effective contact rate)
        k     - Progression rate from exposed (latent) to infected
        gamma - Recovery rate
    '''

    def __init__(self, var, beta, k, gamma, name='seir-model', t=TimeAlways(), i=IterAlways(), memo=None, cb_before_apply=None):
        super().__init__(var, { 's': [1.0 - beta, beta, 0.0, 0.0], 'e': [0.0, 1.0 - k, k, 0.0], 'i': [0.0, 0.0, 1.0 - gamma, gamma], 'r': [0.0, 0.0, 0.0, 1.0] }, name, t, i, memo, cb_before_apply)


class SEQIHRModel_MC(DiscreteInvMarkovChain):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# ODE models:

class SISModel_ODE(ODESystemMass):
    pass


class SIRModel_ODE(ODESystemMass):
    '''
    The SIR epidemiological model without vital dynamics (Markov chain implementation).

    Model parameters:
        beta  - Transmission rate (or effective contact rate)
        gamma - Recovery rate
    '''

    def __init__(self, var, beta, gamma, name='sir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SIRModelDerivatives(beta, gamma), [DotMap(attr={ var:v }) for v in 'sir'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, gamma=None):
        super().set_params(beta=beta, gamma=gamma)


class SIRSModel_ODE(ODESystemMass):
    def __init__(self, var, beta, gamma, alpha=0.0, name='sir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SIRSModelDerivatives(beta, gamma, alpha), [DotMap(attr={ var:v }) for v in 'sirs'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, gamma=None, alpha=None):
        super().set_params(beta=beta, gamma=gamma, alpha=alpha)


class SEIRModel_ODE(ODESystemMass):
    def __init__(self, var, beta, k, gamma, name='seir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SEIRModelDerivatives(beta, k, gamma), [DotMap(attr={ var:v }) for v in 'seir'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, k=None, gamma=None):
        super().set_params(beta=beta, k=k, gamma=gamma)


class SEQIHRModel_ODE(ODESystemMass):
    def __init__(self, var, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho, name='seqihr-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SEQIHRModelDerivatives(beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho), [DotMap(attr={ var:v }) for v in 'seqihr'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, alpha_n=None, alpha_q=None, delta_n=None, delta_h=None, mu=None, chi=None, phi=None, rho=None):
        super().set_params(beta=beta, alpha_n=alpha_n, alpha_q=alpha_q, delta_n=delta_n, delta_h=delta_h, mu=mu, chi=chi, phi=phi, rho=rho)


# ----------------------------------------------------------------------------------------------------------------------
# Main model interfaces:

class SISModel(Model):
    pass


class SIRModel(Model):
    def __init__(self, var, beta, gamma, name='sir-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, MCSolver):
            self.rule = SIRModel_MC (var, beta, gamma, name, t, i, memo)
        elif isinstance(solver, ODESolver):
            self.rule = SIRModel_ODE(var, beta, gamma, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')


class SIRSModel(Model):
    """Main SIRS model interface.

    Notes:
       alpha of 0 implies the SIR model.
    """

    def __init__(self, var, beta, gamma, alpha=0.0, name='sirs-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, MCSolver):
            self.rule = SIRSModel_MC (var, beta, gamma, alpha, name, t, i, memo)
        elif isinstance(solver, ODESolver):
            self.rule = SIRSModel_ODE(var, beta, gamma, alpha, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')


class SEIRModel(Model):
    """Main SEIR model interface.
    """

    def __init__(self, var, beta, k, gamma, name='seir-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None, cb_before_apply=None):
        if isinstance(solver, MCSolver):
            self.rule = SEIRModel_MC (var, beta, k, gamma, name, t, i, memo, cb_before_apply)
        elif isinstance(solver, ODESolver):
            self.rule = SEIRModel_ODE(var, beta, k, gamma, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')


class SEQIHRModel(Model):
    """Main SEQIHR model interface.
    """

    def __init__(self, var, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho, name='seqihr-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, ODESolver):
            self.rule = SEQIHRModel_ODE(var, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')
