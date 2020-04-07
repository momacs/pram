from abc import abstractmethod, ABC

__all__ = ['ModelConstructionError', 'Model', 'Solver', 'MCSolver', 'ODESolver']


# ----------------------------------------------------------------------------------------------------------------------
class ModelConstructionError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
class Model(ABC):
    def __init__(self, rule):
        self.rule = rule

    def set_params(self, **kwargs):
        self.rule.set_params(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
class Solver(ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class MCSolver(Solver):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class ODESolver(Solver):
    def __init__(self, dt=0.1):
        self.dt = dt
