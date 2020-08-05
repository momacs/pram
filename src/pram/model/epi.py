# -*- coding: utf-8 -*-
"""Contains epidemiological models code."""

# Basic reproduction number (R0) for several diseases
#     Rotavirus            16-25
#     Measles              16-18
#     Varicella            10-12
#     Smallpox             3-10
#     Spanish flu          2.0 [1.5 - 2.8]
#     Seasonal influenza   1.3 [0.9 - 1.8]
#     H1N1 swine flu 2019  1.2 - 1.5

from abc import abstractmethod, ABC

from dotmap import DotMap

from  .model  import Model, ModelConstructionError, MCSolver, ODESolver
from ..entity import GroupQry
from ..rule   import TimeAlways, IterAlways, DiscreteInvMarkovChain, ODEDerivatives, ODESystemMass

__all__ = ['SEIRModelParams', 'SEI2RModelParams', 'SISModel', 'SIRModel', 'SIRSModel', 'SEIRModel', 'SEQIHRModel']


# ----------------------------------------------------------------------------------------------------------------------
# ODE models deriavatives:

class SIRModelDerivatives(ODEDerivatives):
    """SIR model derivatives.

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - gamma - Recovery rate

    R0 = beta * S0 / gamma,    where S0 is the initial size of the susceptible population

    Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
    Royal Society A. 115(772), 700--721.
    """

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


class SIRDModelDerivatives(ODEDerivatives):  # new and not validated yet, but boy, does it have potential...
    """SIRD model derivatives.

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - gamma - Recovery rate
        - f     - Redovery fraction (i.e., 1 - f agents die)
    """

    def __init__(self, beta, gamma):
        self.params = DotMap(beta=beta, gamma=gamma)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,I,R = state
            N = sum(state)
            return [
                -p.beta * S * I / N,                    # dS/dt
                 p.beta * S * I / N -     p.gamma * I,  # dI/dt
                                      f * p.gamma * I   # dR/dt
            ]
        return fn


class SIRSModelDerivatives(ODEDerivatives):
    def __init__(self, beta, gamma, alpha):
        self.params = DotMap(beta=beta, gamma=gamma, alpha=alpha)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,I,R = state
            N = sum(state)
            return [
                -p.beta * S * I / N               + p.alpha * R,  # dS/dt
                 p.beta * S * I / N - p.gamma * I,                # dI/dt
                                      p.gamma * I - p.alpha * R   # dR/dt
            ]
        return fn


class SEIRModelDerivatives(ODEDerivatives):
    """SEIR model derivatives.

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - kappa - Progression rate from exposed (latent) to infected
        - gamma - Recovery rate

    R0 = beta * S0 / gamma,    where S0 is the initial size of the susceptible population
    """

    def __init__(self, beta, kappa, gamma):
        self.params = DotMap(beta=beta, kappa=kappa, gamma=gamma)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,E,I,R = state
            N = sum(state)
            return [
                -p.beta  * S * I / N,                              # dS/dt
                 p.beta  * S * I / N - p.kappa * E,                # dE/dt
                                       p.kappa * E - p.gamma * I,  # dI/dt
                                                     p.gamma * I   # dR/dt
            ]
        return fn


class SEQIHRModelDerivatives(ODEDerivatives):
    """SEQIHR model derivatives.

    Model parameters:
        - beta             - Transmission coefficient
        - alpha_n, alpha_q - Rate at which non-quarantined and quarantined individuals become infectious
        - delta_n, delta_h - Rate at which non-isolated and isolated individuals become recovered
        - mu               - Natural death rate
        - chi, phi         - Rate of quarantine and isolation
        - rho              - Isolation efficiency [0..1]

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
    """

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
# Model parameters:

class ModelParams(ABC):
    @classmethod
    def by_clinical_obs(cls):
        """
        Instantiates the class using clinical observations instead of model rate parameters.  This is useful because
        typically the former are easier to obtain and more agreed upon.

        The actual number of arguments to the method of an extending class will depend on the number of compartments in
        in the underlying model.  Progression rates (kappa) should be computed based on incubation and symptomatic
        periods (inbub_period and sympt_period).  Recovery rates (gamma) should be computed based on infection
        durations (inf_dur).  Finally, transmission rates (beta) should be based on the R0 formula, which for the SIR
        model (and all descendant models without vital dynamics) takes the form:

            R0 = beta * S0 / gamma,    where S0 is the initial size of the susceptible population
        """

        pass


class SEIRModelParams(ModelParams):
    def __init__(self, beta, kappa, gamma, r0=None):
        self.beta  = beta
        self.kappa = kappa_1
        self.gamma = gamma
        self.r0    = r0

    @classmethod
    def by_clinical_obs(cls, s0, r0, incub_period, sympt_period):
        kappa = 1 / incub_period
        gamma = 1 / inf_dur
        beta  = r0 * gamma / s0

        return cls(beta, kappa, gamma, r0)


class SEI2RModelParams(ModelParams):
    def __init__(self, beta, kappa_1, kappa_2, gamma, r0=None):
        self.beta    = beta
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.gamma   = gamma
        self.r0      = r0

    @classmethod
    def by_clinical_obs(cls, s0, r0, incub_period, asympt_period, inf_dur):
        kappa_1 = 1 / incub_period
        kappa_2 = 1 / asympt_period
        gamma   = 1 / (inf_dur - asympt_period)
        beta    = r0 * gamma / s0

        return cls(beta, kappa_1, kappa_2, gamma, r0)


# ----------------------------------------------------------------------------------------------------------------------
# Markov chain models:

class SISModel_MC(DiscreteInvMarkovChain):
    """SIS model (Markov chain).

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - gamma - Recovery rate
    """

    def __init__(self, var, beta, gamma, name='sis-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1.0 - beta, beta], 'i': [gamma, 1.0 - gamma] }, name, t, i, memo)


class SIRModel_MC(DiscreteInvMarkovChain):
    """SIR model (Markov chain).

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - gamma - Recovery rate
    """

    def __init__(self, var, beta, gamma, name='sir-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1.0 - beta, beta, 0.0], 'i': [0.0, 1.0 - gamma, gamma], 'r': [0.0, 0.0, 1.0] }, name, t, i, memo)


class SIRSModel_MC(DiscreteInvMarkovChain):
    """SIRS model (Markov chain).

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - gamma - Recovery rate
        - alpha - Immunity loss rate (alpha = 0 implies life-long immunity and consequently the SIR model)
    """

    def __init__(self, var, beta, gamma, alpha=0.0, name='sirs-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1.0 - beta, beta, 0.0], 'i': [0.0, 1.0 - gamma, gamma], 'r': [alpha, 0.0, 1.0 - alpha] }, name, t, i, memo)


class SEIRModel_MC(DiscreteInvMarkovChain):
    """SEIR model (Markov chain).

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - k     - Progression rate from exposed (latent) to infected
        - gamma - Recovery rate
    """

    def __init__(self, var, beta, kappa, gamma, name='seir-model', t=TimeAlways(), i=IterAlways(), memo=None, cb_before_apply=None):
        super().__init__(var, { 's': [1.0 - beta, beta, 0.0, 0.0], 'e': [0.0, 1.0 - kappa, kappa, 0.0], 'i': [0.0, 0.0, 1.0 - gamma, gamma], 'r': [0.0, 0.0, 0.0, 1.0] }, name, t, i, memo, cb_before_apply)


class SEQIHRModel_MC(DiscreteInvMarkovChain):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# ODE models:

class SISModel_ODE(ODESystemMass):
    pass


class SIRModel_ODE(ODESystemMass):
    """SIS model (ordinary differential equations).

    Model parameters:
        - beta  - Transmission rate (or effective contact rate)
        - gamma - Recovery rate
    """

    def __init__(self, var, beta, gamma, name='sir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SIRModelDerivatives(beta, gamma), [GroupQry(attr={ var:v }) for v in 'sir'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, gamma=None):
        super().set_params(beta=beta, gamma=gamma)


class SIRSModel_ODE(ODESystemMass):
    def __init__(self, var, beta, gamma, alpha=0.0, name='sir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SIRSModelDerivatives(beta, gamma, alpha), [GroupQry(attr={ var:v }) for v in 'sirs'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, gamma=None, alpha=None):
        super().set_params(beta=beta, gamma=gamma, alpha=alpha)


class SEIRModel_ODE(ODESystemMass):
    def __init__(self, var, beta, kappa, gamma, name='seir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SEIRModelDerivatives(beta, kappa, gamma), [GroupQry(attr={ var:v }) for v in 'seir'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, kappa=None, gamma=None):
        super().set_params(beta=beta, kappa=kappa, gamma=gamma)


class SEQIHRModel_ODE(ODESystemMass):
    def __init__(self, var, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho, name='seqihr-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(SEQIHRModelDerivatives(beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho), [GroupQry(attr={ var:v }) for v in 'seqihr'], name, t, i, dt, memo=memo)

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
    """SIRS model interface.

    Notes:
       ``alpha = 0`` implies the SIR model.
    """

    def __init__(self, var, beta, gamma, alpha=0.0, name='sirs-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, MCSolver):
            self.rule = SIRSModel_MC (var, beta, gamma, alpha, name, t, i, memo)
        elif isinstance(solver, ODESolver):
            self.rule = SIRSModel_ODE(var, beta, gamma, alpha, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')


class SEIRModel(Model):
    """SEIR model interface.
    """

    def __init__(self, var, beta, kappa, gamma, name='seir-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None, cb_before_apply=None):
        if isinstance(solver, MCSolver):
            self.rule = SEIRModel_MC (var, beta, kappa, gamma, name, t, i, memo, cb_before_apply)
        elif isinstance(solver, ODESolver):
            self.rule = SEIRModel_ODE(var, beta, kappa, gamma, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')


class SEQIHRModel(Model):
    """SEQIHR model interface.
    """

    def __init__(self, var, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho, name='seqihr-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, ODESolver):
            self.rule = SEQIHRModel_ODE(var, beta, alpha_n, alpha_q, delta_n, delta_h, mu, chi, phi, rho, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')
