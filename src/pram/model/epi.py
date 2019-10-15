from abc import abstractmethod, ABC

from dotmap import DotMap

from .model import Model, ModelConstructionError, MCSolver, ODESolver
from ..rule import TimeAlways, IterAlways, DiscreteInvMarkovChain, ODESystemMass


# ----------------------------------------------------------------------------------------------------------------------
def fn_deriv_sir(beta, gamma):
    def fn(t, state):
        '''
        [1] Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
            Royal Society A. 115(772), 700--721.

        http://www.public.asu.edu/~hnesse/classes/sir.html
        '''

        s,i,r = state
        n = s + i + r
        return [
            -beta * s * i / n,
             beta * s * i / n - gamma * i,
                                gamma * i
        ]
    return fn


# ----------------------------------------------------------------------------------------------------------------------
class SISModel_MC(DiscreteInvMarkovChain):
    '''
    The SIS epidemiological model without vital dynamics.

    Model parameters:
        beta  - transmission rate
        gamma - recovery rate


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
        beta  - transmission rate
        gamma - recovery rate
        alpha - immunity loss rate (alpha = 0 implies life-long immunity)


    ----[ Notation A ]----

    code:
        SIRSModel('flu', 0.05, 0.20, 0.10)  # SIRS
        SIRSModel('flu', 0.05, 0.20, 0.00)  # SIR
    '''

    def __init__(self, var, beta, gamma, alpha=0.00, name='sir-model', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { 's': [1 - beta, beta, 0.00], 'i': [0.00, 1 - gamma, gamma], 'r': [alpha, 0.00, 1 - alpha] }, name, t, i, memo)


# ----------------------------------------------------------------------------------------------------------------------
class SIRSModel_ODE(ODESystemMass):
    def __init__(self, var, beta, gamma, alpha=0.00, name='sir-model', t=TimeAlways(), i=IterAlways(), dt=0.1, memo=None):
        super().__init__(fn_deriv_sir(beta, gamma), [DotMap(attr={ var:v }) for v in 'sir'], name, t, i, dt, memo=memo)

# ----------------------------------------------------------------------------------------------------------------------
class SIRSModel(Model):
    def __init__(self, var, beta, gamma, alpha=0.00, name='sir-model', t=TimeAlways(), i=IterAlways(), solver=MCSolver(), memo=None):
        if isinstance(solver, MCSolver):
            self.rule = SIRSModel_MC(var, beta, gamma, alpha, name, t, i, memo)
        elif isinstance(solver, ODESolver):
            self.rule = SIRSModel_ODE(var, beta, gamma, alpha, name, t, i, solver.dt, memo)
        else:
            raise ModelConstructionError('Incompatible solver')
