# -*- coding: utf-8 -*-
"""Contains abstract and generic model code."""

from abc import abstractmethod, ABC

__all__ = ['ModelConstructionError', 'Model', 'Solver', 'MCSolver', 'ODESolver']


# ----------------------------------------------------------------------------------------------------------------------
class ModelConstructionError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
class Model(ABC):
    """A model base class.

    Args:
        rule (Rule): The group rule implementing the model dynamics.
    """

    def __init__(self, rule):
        self.rule = rule

    def set_params(self, **kwargs):
        self.rule.set_params(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
class Solver(ABC):
    """Model solver base class.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class MCSolver(Solver):
    """Markov chain model solver.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class ODESolver(Solver):
    """Ordinary differential equations system model solver.

    Args:
        dt (float): Numeric integrator time step size.
    """

    def __init__(self, dt=0.1):
        self.dt = dt
