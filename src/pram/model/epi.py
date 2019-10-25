from abc import abstractmethod, ABC

from dotmap import DotMap

from .model import Model, ModelConstructionError, MCSolver, ODESolver
from ..rule import TimeAlways, IterAlways, DiscreteInvMarkovChain, ODEDerivatives, ODESystemMass


# ----------------------------------------------------------------------------------------------------------------------
# def gen_fn_deriv_sir(beta, gamma):
#     def fn(t, state):
#         '''
#         Model parameters
#             beta  - Transmission rate
#             gamma - Recovery rate
#
#         Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
#         Royal Society A. 115(772), 700--721.
#
#         http://www.public.asu.edu/~hnesse/classes/sir.html
#         '''
#
#         S,I,R = state
#         N = S + I + R  # sum(state)
#         return [
#             -beta * S * I / N,              # dS/dt
#              beta * S * I / N - gamma * I,  # dI/dt
#                                 gamma * I   # dR/dt
#         ]
#     return fn


# ----------------------------------------------------------------------------------------------------------------------
class SIRModelDerivatives(ODEDerivatives):
    '''
    Model parameters
        beta  - Transmission rate
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
            N = S + I + R  # sum(state)
            return [
                -p.beta * S * I / N,                # dS/dt
                 p.beta * S * I / N - p.gamma * I,  # dI/dt
                                      p.gamma * I   # dR/dt
            ]
        return fn


# ----------------------------------------------------------------------------------------------------------------------
# def gen_fn_deriv_seqihr(beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho):
#     def fn(t, state):
#         '''
#         Model parameters
#             beta             - Transmission coefficient
#             alpha_n, alpha_q - Rate at which non-quarantined and quarantined individuals become infectious
#             alpha_q, delta_i - Rate at which non-isolated and isolated individuals become recovered
#             mu               - Natural death rate
#             chi, phi         - Rate of quarantine and isolation
#             rho              - Isolation efficiency [0..1]
#
#         In the source below, parameters used are
#             alpha_1 = alpha_n
#             alpha_2 = alpha_q
#             delta_1 = delta_n
#             delta_2 = delta_i
#
#         "(...) some of the drawbacks of the simple model when used to evaluate intervention policies. We argue that the
#         main reason for these problems is due to the simplifying assumption of exponential distributions for the
#         disease stages, which is used in the model. This provides a motivation for using more realistic stage
#         distributions. Non-exponential distributions have been considered in epidemiological models (see, for
#         example, ...). However, none of these studies focuses on the evaluation of intervention policies."
#
#         Feng & Xu (2007) Epidemiological Models with Non-Exponentially Distributed Disease Stages and Applications to
#         Disease Control.  Bulletin of Mathematical Biology.
#         '''
#
#         S,E,Q,I,H,R = state
#         N = S + E + Q + I + H + R  # sum(state)
#         return [
#             mu * N - beta * S * (I + (1 - rho) * H) / N - mu * S,           # dS/dt
#             beta * S * (I + (1 - rho) * H) / N - (chi + alpha_n + mu) * E,  # dE/dt
#             chi * E - (alpha_q + mu) * Q,                                   # dQ/dt
#             alpha_n * E - (phi + delta_n + mu) * I,                         # dI/dt
#             alpha_q * Q + phi * I - (delta_i + mu) * H,                     # dH/dt
#             delta_n * I + delta_i * H - mu * R                              # dR/dt
#
#             # mu * N - beta * S * (I + (1 - rho) * H) / N - mu * S
#             # beta * S * (I + (1 - rho) * H) / N - (chi + alpha_1 + mu) * E
#             # chi * E - (alpha_2 + mu) * Q
#             # alpha_1 * E - (phi + delta_1 + mu) * I
#             # alpha_2 * Q + phi * I - (delta_2 + mu) * H
#             # delta_1 * I + delta_2 * H - mu * R
#         ]
#     return fn


# ----------------------------------------------------------------------------------------------------------------------
class SEQIHRModelDerivatives(ODEDerivatives):
    '''
    Model parameters
        beta             - Transmission coefficient
        alpha_n, alpha_q - Rate at which non-quarantined and quarantined individuals become infectious
        alpha_q, delta_i - Rate at which non-isolated and isolated individuals become recovered
        mu               - Natural death rate
        chi, phi         - Rate of quarantine and isolation
        rho              - Isolation efficiency [0..1]

    In the source below, parameters used are
        alpha_1 = alpha_n
        alpha_2 = alpha_q
        delta_1 = delta_n
        delta_2 = delta_i

    "(...) some of the drawbacks of the simple model when used to evaluate intervention policies. We argue that the
    main reason for these problems is due to the simplifying assumption of exponential distributions for the
    disease stages, which is used in the model. This provides a motivation for using more realistic stage
    distributions. Non-exponential distributions have been considered in epidemiological models (see, for
    example, ...). However, none of these studies focuses on the evaluation of intervention policies."

    Feng & Xu (2007) Epidemiological Models with Non-Exponentially Distributed Disease Stages and Applications to
    Disease Control.  Bulletin of Mathematical Biology.
    '''

    def __init__(self, beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho):
        self.params = DotMap(beta=beta, alpha_n=alpha_n, alpha_q=alpha_q, delta_n=delta_n, delta_i=delta_i, mu=mu, chi=chi, phi=phi, rho=rho)

    def get_fn(self):
        p = self.params
        def fn(t, state):
            S,E,Q,I,H,R = state
            N = S + E + Q + I + H + R  # sum(state)
            return [
                p.mu * N - p.beta * S * (I + (1 - p.rho) * H) / N - p.mu * S,             # dS/dt
                p.beta * S * (I + (1 - p.rho) * H) / N - (p.chi + p.alpha_n + p.mu) * E,  # dE/dt
                p.chi * E - (p.alpha_q + p.mu) * Q,                                       # dQ/dt
                p.alpha_n * E - (p.phi + p.delta_n + p.mu) * I,                           # dI/dt
                p.alpha_q * Q + p.phi * I - (p.delta_i + p.mu) * H,                       # dH/dt
                p.delta_n * I + p.delta_i * H - p.mu * R                                  # dR/dt

                # mu * N - beta * S * (I + (1 - rho) * H) / N - mu * S
                # beta * S * (I + (1 - rho) * H) / N - (chi + alpha_1 + mu) * E
                # chi * E - (alpha_2 + mu) * Q
                # alpha_1 * E - (phi + delta_1 + mu) * I
                # alpha_2 * Q + phi * I - (delta_2 + mu) * H
                # delta_1 * I + delta_2 * H - mu * R
            ]
        return fn


# ----------------------------------------------------------------------------------------------------------------------
class SISModel_MC(DiscreteInvMarkovChain):
    '''
    The SIS epidemiological model without vital dynamics.

    Model parameters:
        beta  - Transmission rate
        gamma - Recovery rate


    ----[ Notation A ]----

    code:
        SISModel('flu', 0.05, 0.10)
    '''

    def __init__(self, var, beta, gamma, name='sis-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1 - beta, beta], 'i': [gamma, 1 - gamma] }, name, t, i, memo)


# ----------------------------------------------------------------------------------------------------------------------
class SIRSModel_MC(DiscreteInvMarkovChain):
    '''
    The SIR(S) epidemiological model without vital dynamics.

    Model parameters:
        beta  - Transmission rate
        gamma - Recovery rate
        alpha - Immunity loss rate (alpha = 0 implies life-long immunity and consequently the SIR model)


    ----[ Notation A ]----

    code:
        SIRSModel('flu', 0.05, 0.20, 0.10)  # SIRS
        SIRSModel('flu', 0.05, 0.20, 0.00)  # SIR
    '''

    def __init__(self, var, beta, gamma, alpha=0.00, name='sirs-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1 - beta, beta, 0.00], 'i': [0.00, 1 - gamma, gamma], 'r': [alpha, 0.00, 1 - alpha] }, name, t, i, memo)


# ----------------------------------------------------------------------------------------------------------------------
class SIRSModel_ODE(ODESystemMass):
    def __init__(self, var, beta, gamma, alpha=0.00, name='sirs-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        # super().__init__(gen_fn_deriv_sir(beta, gamma), [DotMap(attr={ var:v }) for v in 'sir'], name, t, i, dt, memo=memo)
        super().__init__(SIRModelDerivatives(beta, gamma), [DotMap(attr={ var:v }) for v in 'sir'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, gamma=None, alpha=None):
        super().set_params(beta=beta, gamma=gamma, alpha=alpha)


# ----------------------------------------------------------------------------------------------------------------------
class SEQIHRModel_ODE(ODESystemMass):
    def __init__(self, var, beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho, name='seqihr-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        # super().__init__(gen_fn_deriv_seqihr(beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho), [DotMap(attr={ var:v }) for v in 'seqihr'], name, t, i, dt, memo=memo)
        super().__init__(SEQIHRModelDerivatives(beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho), [DotMap(attr={ var:v }) for v in 'seqihr'], name, t, i, dt, memo=memo)

    def set_params(self, beta=None, alpha_n=None, alpha_q=None, delta_n=None, delta_i=None, mu=None, chi=None, phi=None, rho=None):
        super().set_params(beta=beta, alpha_n=alpha_n, alpha_q=alpha_q, delta_n=delta_n, delta_i=delta_i, mu=mu, chi=chi, phi=phi, rho=rho)


# ----------------------------------------------------------------------------------------------------------------------
class SIRSModel(Model):
    def __init__(self, var, beta, gamma, alpha=0.00, name='sirs-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, MCSolver):
            self.rule = SIRSModel_MC(var, beta, gamma, alpha, name, t, i, memo)
        elif isinstance(solver, ODESolver):
            self.rule = SIRSModel_ODE(var, beta, gamma, alpha, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')


# ----------------------------------------------------------------------------------------------------------------------
class SEQIHRModel(Model):
    def __init__(self, var, beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho, name='seqihr-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, ODESolver):
            self.rule = SEQIHRModel_ODE(var, beta, alpha_n, alpha_q, delta_n, delta_i, mu, chi, phi, rho, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')
