# -*- coding: utf-8 -*-
"""Contains rule code."""

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import string

from abc             import abstractmethod, ABC
from attr            import attrs, attrib, converters
from collections     import Iterable
from dotmap          import DotMap
from enum            import IntEnum
from scipy.stats     import gamma, lognorm, norm, poisson, rv_discrete
from scipy.integrate import ode, solve_ivp

from .entity import Group, GroupQry, GroupSplitSpec, Site
from .util   import Err, Time as TimeU


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class Iter(ABC):
    """Iteration selector base class."""

    pass


@attrs(slots=True)
class IterAlways(Iter):
    """Iteration selector compatible with all iterations."""

    pass


@attrs(slots=True)
class IterPoint(Iter):
    """Iteration selector compatible with only one iteration.

    Args:
        i (int): Iteration number.
    """

    i: int = attrib(default=0, converter=int)

    def __attrs_post_init__(self):
        if self.i < 0:
            raise ValueError(f'Iteration must be non-negative, but {self.i} passed.')


@attrs(slots=True)
class IterInt(Iter):
    """Iteration selector compatible with a range of iterations.

    Args:
        i0 (int): Lower bound iteration number.
        i1 (int): Upper bound iteration number.
    """

    i0: int = attrib(default=  0, converter=int)
    i1: int = attrib(default=100, converter=int)

    def __attrs_post_init__(self):
        if self.i0 < 0 or self.i1 < 0:
            raise ValueError(f'Iterations must be non-negative, but ({self.i0}, {self.i1}) passed.')


@attrs(slots=True)
class IterSet(Iter):
    """Iteration selector compatible with an enumeration of iterations.

    Args:
        i (set[int]): Iteration numbers.
    """

    i: set = attrib(factory=set, converter=converters.default_if_none(factory=set))

    def __attrs_post_init__(self):
        if any(map(lambda x: x < 0, self.i)):
            raise ValueError(f'All iterations must be non-negative, but {self.i} passed.')


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class Time(ABC):
    """Time selector base class."""

    pass


@attrs(slots=True)
class TimeAlways(Time):
    """Time selector compatible with all times."""

    pass


@attrs(slots=True)
class TimePoint(Time):
    """Time selector compatible with only one time.

    Args:
        t (float): Time.
    """

    t: float = attrib(default=0.00, converter=float)

    def __attrs_post_init__(self):
        if self.t < 0:
            raise ValueError(f'Time must be non-negative, but {self.i} passed.')


@attrs(slots=True)
class TimeInt(Time):
    """Time selector compatible with a range of times.

    Args:
        t0 (float): Lower bound time.
        t1 (float): Upper bound time.
    """

    t0: float = attrib(default= 0.00, converter=float)
    t1: float = attrib(default=24.00, converter=float)

    def __attrs_post_init__(self):
        if self.t0 < 0 or self.t1 < 0:
            raise ValueError(f'Times must be non-negative, but ({self.t0}, {self.t1}) passed.')


@attrs(slots=True)
class TimeSet(Time):
    """Time selector compatible with an enumeration of times.

    Args:
        t (set[int]): Times.
    """

    t: set = attrib(factory=set, converter=converters.default_if_none(factory=set))

    def __attrs_post_init__(self):
        if any(map(lambda x: x < 0, self.t)):
            raise ValueError(f'All times must be non-negative, but {self.t} passed.')


# ----------------------------------------------------------------------------------------------------------------------
# class TimeCtrl(object):
#     pass


# ----------------------------------------------------------------------------------------------------------------------
# class IterCtrl(object):
#     def __init__(self, rule):
#         self.rule = rule


# ----------------------------------------------------------------------------------------------------------------------
class Directive(ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class ForkDirective(Directive):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class Rule(ABC):
    """A group rule.

    A rule that can be applied to a group and may split it into multiple groups.

    A rule will be applied if the simulation timer's time (external to this class) falls within the range defined by
    the time specification 't'.  Every time a rule is applied, it is applied to all groups it is compatible with.  For
    instance, a rule that renders a portion of a group infection-free (i.e., marks it as recovered) can be applied to a
    group of humans currently infected with some infectious disease.  The same rule, however, would not be applied to
    a group of city buses.  Each rule knows how to recognize a compatible group.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    T_UNIT_MS = TimeU.MS.h
    NAME = 'Rule'
    ATTRS = {}  # a dict of attribute names as keys and the list of their values as values

    pop = None
    compile_spec = None

    def __init__(self, name='rule', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        # Err.type(t, 't', Time, True)
        # Err.type(i, 'i', Iter, True)

        self.name = name
        self.group_qry = group_qry
        self.memo = memo

        self._set_t(t)
        self._set_i(i)

        self.rules = []  # the rule's inner rules

        self.t_unit_ms = None  # set via set_t_unit() by the simulation every time it runs
        self.t_mul = 0.00      # ^
        self.T = {}

    def __repr__(self):
        if isinstance(self.t, TimeAlways):
            return '{}(name={}, t=.)'.format(self.__class__.__name__, self.name)

        if isinstance(self.t, TimePoint):
            return '{}(name={}, t={:>4})'.format(self.__class__.__name__, self.name, round(self.t.t, 1))

        if isinstance(self.t, TimeInt):
            return '{}(name={}, t=({:>4},{:>4}))'.format(self.__class__.__name__, self.name, round(self.t.t0, 1), round(self.t.t1, 1))

    def __str__(self):
        if isinstance(self.t, TimeAlways):
            return 'Rule  name: {:16}  t: .'.format(self.name)

        if isinstance(self.t, TimePoint):
            return 'Rule  name: {:16}  t: {:>4}'.format(self.name, round(self.t.t, 1))

        if isinstance(self.t, TimeInt):
            return 'Rule  name: {:16}  t: ({:>4},{:>4})'.format(self.name, round(self.t.t0, 1), round(self.t.t1, 1))

    def _set_i(self, i=None):
        """Decode and set iteration selector.

        Args:
            i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int], optional): Iteration selector.
        """

        ii = isinstance
        if i is None:
            self.i = IterAlways()
        elif ii(i, Iter):
            self.i = i
        elif ii(i, int):
            self.i = IterPoint(i)
        elif (ii(i, list) or ii(i, tuple) or ii(i, np.ndarray)) and len(i) == 2 and ii(i[0], int) and ii(i[1], int):
            self.i = IterInt(i[0], i[1])
        elif ii(i, set):
            self.i = IterSet(i)
        else:
            raise ValueError("Wrong type of the argument 'i' specified.")

    def _set_t(self, t=None):
        """Decode and set time selector.

        Args:
            t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int], optional): Time selector.
        """

        ii = isinstance
        if t is None:
            self.t = TimeAlways()
        elif ii(t, Time):
            self.t = t
        elif ii(t, int):
            self.t = TimePoint(t)
        elif (ii(t, list) or ii(t, tuple) or ii(t, np.ndarray)) and len(t) == 2 and ii(t[0], int) and ii(t[1], int):
            self.t = TimeInt(t[0], t[1])
        elif ii(t, set):
            self.t = TimeSet(t)
        else:
            raise ValueError("Wrong type of the argument 't' specified.")

    @abstractmethod
    def apply(self, pop, group, iter, t):
        """Apply the rule.

        This method is called automatically for every group at every simulation iteration provided the rule is
        compatible with the iteration, time, and the group.

        Args:
            pop (GroupPopulation): Population.
            group (Group): Group.
            iter (int): Iteration.
            time (float): Time.

        Returns:
            Iterable[GroupSplitSpect] or None: Specs groups the current group should be split into.
        """

        pass

    def cleanup(self, pop, group):
        """Rule cleanup.

        Run once at the end of a simulation run.  Symmetrical to :meth:`~pram.rule.Rule.setup`.  Uses group-splitting
        mechanism.

        Args:
            pop (GroupPopulation): Population.
            group (Group): Group.

        Returns:
            Iterable[GroupSplitSpect] or None: Specs groups the current group should be split into.
        """

        pass

    @staticmethod
    def get_rand_name(name, prefix='__', rand_len=12):
        """Generate a random rule name.

        Args:
            prefix (str): Name prefix.
            rand_len (int): Number of random characters to be generated.

        Returns:
            str
        """

        return f'{prefix}{name}_' + ''.join(random.sample(string.ascii_letters + string.digits, rand_len))

    def get_inner_rules(self):
        """Get a rule's inner rules.

        A rule's inner rules allow a rule dependency hierarchy to be specified.

        Returns:
            Iterable[Rule] or None
        """

        return self.rules

    def is_applicable(self, group, iter, t):
        """Verifies if the rule is applicable to the specified group at the specified iteration and time.

        This method is called automatically and a rule will only be applied if this method returns ``True``.

        Args:
            group (Group): Group.
            iter (:class:`~pram.rule.Iter`): Iteration.
            time (:class:`~pram.rule.Time`): Time.

        Returns:
            bool: True if applicable, False otherwise.
        """

        return self.is_applicable_iter(iter) and self.is_applicable_time(t) and self.is_applicable_group(group)

    def is_applicable_iter(self, iter):
        """Verifies if the rule is applicable at the specified iteration.

        Args:
            i (:class:`~pram.rule.Iter`, int, tuple[int,int], set): Iteration selector.

        Returns:
            bool: True if applicable, False otherwise.
        """

        if isinstance(self.i, IterAlways):
            return True
        elif isinstance(self.i, IterPoint):
            return self.i.i == iter
        elif isinstance(self.i, IterInt):
            if self.i.i1 == 0:
                return self.i.i0 <= iter
            else:
                return self.i.i0 <= iter <= self.i.i1
        elif isinstance(self.i, IterSet):
            return iter in self.i.i
        else:
            raise TypeError("Type '{}' used for specifying rule iteration not yet implemented (Rule.is_applicable).".format(type(self.i).__name__))

    def is_applicable_time(self, t):
        """Verifies if the rule is applicable at the specified time.

        Args:
            t (:class:`~pram.rule.Time`, int, tuple[int,int], set): Time selector.

        Returns:
            bool: True if applicable, False otherwise.
        """

        if isinstance(self.t, TimeAlways):
            return True
        elif isinstance(self.t, TimePoint):
            return self.t.t == t
        elif isinstance(self.t, TimeInt):
            return self.t.t0 <= t <= self.t.t1
        elif isinstance(self.t, TimeSet):
            return t in self.i.t
        else:
            raise TypeError("Type '{}' used for specifying rule timing not yet implemented (Rule.is_applicable).".format(type(self.t).__name__))

    def is_applicable_group(self, group):
        """Verifies if the rule is applicable to the the specified group.

        If the :attr:`~pram.rule.Rule.group_qry` is None, the group is applicable automatically.

        Args:
            group (Group): Group.

        Returns:
            bool: True if applicable, False otherwise.
        """

        if not self.group_qry:
            return True
        else:
            return group.matches_qry(self.group_qry)

    def set_params(self):
        """Set rule's parameters."""

        pass

    def set_t_unit(self, ms):
        """Set timer units.

        Args:
            ms (int): Number of milliseconds per timer unit.  Values from :attr:`pram.util.Time.MS <util.Time.MS>` can
                be used for convenience.
        """

        self.t_unit_ms = ms
        self.t_mul = float(self.__class__.T_UNIT_MS) / float(self.t_unit_ms)

    def setup(self, pop, group):
        """Rule setup.

        If a set of attributes and relations should be part of a group's definition, that set should be specified here.
        For example, the following sets the attribute ``did-attend-school-today`` of all the groups::

            return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]

        Each group will be split into a (possibly non-extant) new group via the group-splitting mechanism and the
        entirety of the group's mass will be moved into that new group.

        This method is also where a rule should do any other population initialization required.  For example, a rule
        that introduces a new Site object to the simulation, should make the population object aware of that site like
        so::

            pop.add_site(Site('new-site'))

        Args:
            pop (GroupPopulation): Population.
            group (Group): Group.

        Returns:
            Iterable[GroupSplitSpect] or None: Specs groups the current group should be split into.
        """

        pass

    @staticmethod
    def tp2rv(tp, a=0, b=23):
        """Converts a time probability distribution function (PDF) to a scipy's discrete random variable.

        Args:
            tp (Mapping): Time-probability mapping, e.g., ``{ 1: 0.05, 3: 0.2, 4: 0.25, 5: 0.2, 6: 0.1, 7: 0.2 }``.
            a (int): Lower bound.
            b (int): Upper bound.

        Returns
            scipy.stats.rv_discrete
        """

        return rv_discrete(a,b, values=(tuple(tp.keys()), tuple(tp.values())))


# ----------------------------------------------------------------------------------------------------------------------
class SimRule(Rule, ABC):
    """Simulation rule.

    A simulation rule is run at the end of every simulation iteration (i.e., after the mass transfer for that iteration
    has concluded).  These sort of rules are useful for implementing policy-level intervention.  For example, should
    schools in a county, state, or the entire country be closed in response to an epidemic?  This sort of
    mitigation measure might be in effect after, say, 6% of the population has become infected.

    Upon adding an instance of this class to the :class:`~pram.sim.Simulation` object, the latter will add all
    'self.vars' as simulation variables.  Consequently, this ABC's constructor should ordinarily be called as the first
    step in the child's constructor so that the dict is not overwritten.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Iteration selector.
        memo (str): Description.
    """

    def __init__(self, name='rule', t=TimeAlways(), i=IterAlways(), memo=None):
        # Err.type(t, 't', Time, True)
        # Err.type(i, 'i', Iter, True)

        self.name = name
        self.memo = memo
        self.vars = {}

        self._set_t(t)
        self._set_i(i)

    @abstractmethod
    def apply(self, sim, iter, t):
        """Apply the rule.

        This method is called automatically for every group at every simulation iteration provided the rule is
        compatible with the iteration, time, and the group.

        Args:
            sim (Simulation): Simulation.
            iter (int): Iteration.
            time (float): Time.

        Returns:
            Iterable[GroupSplitSpect] or None: Specs groups the current group should be split into.
        """

        pass

    def is_applicable(self, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable_iter(iter) and super().is_applicable_time(t)


# ----------------------------------------------------------------------------------------------------------------------
class Noop(Rule):
    """NO-OP, i.e., a rule that does not do anything.

    Useful for testing simulations that require no rules (because at least one rule need to be present in a simulation
    in order for groups to be added).

    Args:
        name (str): Name.
    """

    def __init__(self, name='noop'):
        super().__init__(name)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        return None


# ----------------------------------------------------------------------------------------------------------------------
class Sequence(Rule, ABC):
    pass

class Equation(Rule, ABC):
    pass

class DifferenceEquation(Equation, ABC):
    pass

class Event(Rule, ABC):
    pass

class Control(Rule, ABC):
    pass

class Input(Rule, ABC):
    pass

class Stimulus(Rule, ABC):
    pass

class Forcing(Rule, ABC):
    pass

class Disturbance(Rule, ABC):
    pass

class Perturbation(Rule, ABC):
    pass

class Intervention(Rule, ABC):
    pass

class Action(Rule, ABC):
    """A physical action.

    `Wikipedia: Action (physics) <https://en.wikipedia.org/wiki/Action_(physics)>`_
    """

    pass

class Process(Rule, ABC):
    pass

class Model(Rule, ABC):
    pass

class ChaoticMap(Rule, ABC):
    """The simplest dynamical system that has relevant features of chaotic Hamiltonian systems.

    `Wikipedia: List of chaotic maps <https://en.wikipedia.org/wiki/List_of_chaotic_maps>`_
    """

    pass

class System(Rule, ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class GroupMassDecByNumRule(Rule):
    """Decreases the mass of a group by number.

    Args:
        m (float): The mass to remove.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, m, name='grp-mass-dec-by-num', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)
        self.m = m

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        p = min(max(self.m / group.m, 0), 1)
        return [
            GroupSplitSpec(p=    p, attr_set=Group.VOID),
            GroupSplitSpec(p=1 - p)
        ]


# ----------------------------------------------------------------------------------------------------------------------
class GroupMassDecByPropRule(Rule):
    """Decreases the mass of a group by a proportion/probability.

    Args:
        p (float): The proportion of mass to remove.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, p, name='grp-mass-dec-by-prop', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)
        self.p = p

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        return [
            GroupSplitSpec(p=    self.p, attr_set=Group.VOID),
            GroupSplitSpec(p=1 - self.p)
        ]


# ----------------------------------------------------------------------------------------------------------------------
class GroupMassIncByNumRule(Rule):
    """Increases the mass of a group by number.

    Args:
        m (float): The mass to remove.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, m, name='grp-mass-inc-by-num', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)
        self.m = m

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        g = group.copy()
        g.m = self.m
        pop.add_vita_group(g)
        return None


# ----------------------------------------------------------------------------------------------------------------------
class GroupMassIncByPropRule(Rule):
    """Increases the mass of a group by a proportion/probability.

    Args:
        p (float): The proportion of mass to remove.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, p, name='grp-mass-inc-by-prop', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)
        self.p = p

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        g = group.copy()
        g.m = group.m * self.p
        pop.add_vita_group(g)
        return None


# ----------------------------------------------------------------------------------------------------------------------
class StochasticProcess(Rule, ABC):
    """A stochastic process is a collection of random variables indexed by time or space."""

    pass

class MarkovProcess(StochasticProcess, ABC):
    """A stochastic process that satisfies the Markov property."""

    pass

class StationaryProcess(StochasticProcess, ABC):
    """A stochastic process whose unconditional joint probability distribution does not change when shifted in time."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class DiscreteSpacetimeStochasticProcess           (StochasticProcess, ABC): pass
class DiscreteSpaceContinuousTimeStochasticProcess (StochasticProcess, ABC): pass
class ContinuousSpaceDiscreteTimeStochasticProcess (StochasticProcess, ABC): pass
class ContinuousSpacetimeStochasticProcess         (StochasticProcess, ABC): pass


# ----------------------------------------------------------------------------------------------------------------------
class TimeSeriesFn(Rule, ABC):
    """Function-based time series."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class TimeSeriesObs(Rule, ABC):
    """Observation-based time series.

    A rule that replays a time series given by a stochastic vector or a stochastic matrix of explicit observations
    (e.g., historical data).  The following matrix::

            t1 t2 t3
        x1   1  2  3
        x2   4  5  6

    should be provided as::

        { 'x1': [1,2,3], 'x2': [4,5,6] }

    Args:
        x (Mapping[str,Iterable[number]]): Time series to replay.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, x, name='time-series', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)
        self.x = x  # f(t)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        # self.x[iter]
        pass


# ----------------------------------------------------------------------------------------------------------------------
class DistributionProcess(Process, ABC):
    """Generic distribution process.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Iteration selector.
        p_max (float): The maximum proportion of mass that should be converted (i.e., at the mode of the distribution).
            The distribution that describes mass conversion is internally scaled to match that argument.  If 'None', no
            scaling is performed.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, name='distrib-proc', t=TimeAlways(), i=IterAlways(), p_max=None, group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.p_max = p_max
        self.dist = None    # the scipy distribution object (the extending class needs to call set_dist() to set this)
        self.mode = None    # the mode of the distribution (ditto)
        self.p_mult = None  # probability multiplier that scales the distribution to p_max at its mode (ditto)
        self.p = None       # a frozen distribution (ditto)

        self.plot_label = ''

        # Iteration offset:
        if isinstance(self.i, IterAlways):
            self.iter0 = 0
        elif isinstance(self.i, IterPoint):
            self.iter0 = self.i.i
        elif isinstance(self.i, IterInt):
            self.iter0 = self.i.i0

    def get_p(self, iter):
        return self.p(iter - self.iter0)

    def plot(self, figsize=(10,2), fmt='g-', lw=1, label=None, dpi=150, do_legend=True, do_show=True):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        x = np.linspace(self.dist.ppf(0.01), self.dist.ppf(0.99), 100)
        plt.plot(x, self.dist.pdf(x) * self.p_mult, fmt, lw=lw, label=label or self.plot_label)
        if do_legend:
            plt.legend(loc='best', frameon=False)
        if do_show:
            plt.show()
        return fig

    def set_dist(self, dist, mode):
        """Sets the distribution.

        Args:
            dist (scipy.stats.rv_generic): Distribution.
            mode (float): The mode of the distribution.
        """

        self.dist = dist
        self.mode = mode
        self.p_mult = 1.0 if self.p_max is None else self.p_max / self.dist.pdf(self.mode)
        self.p = lambda iter: self.dist.pdf(iter) * self.p_mult


# ----------------------------------------------------------------------------------------------------------------------
class GammaDistributionProcess(DistributionProcess, ABC):
    """Gamma distribution process.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Iteration selector.
        p_max (float, optional): The maximum proportion of mass that should be converted (i.e., at the mode of the distribution).
            The distribution that describes mass conversion is internally scaled to match that argument.  If 'None', no
            scaling is performed.
        a (float): A paramater.
        loc (float): Location parameter.
        scale (float): Scale parameter.
        label (str, optional): Plot label.
        memo (str, optional): Description.
    """

    def __init__(self, name='gamma-distrib-proc', t=TimeAlways(), i=IterAlways(), p_max=None, a=1.0, loc=0.0, scale=1.0, label=None, group_qry=None, memo=None):
        super().__init__(name, t, i, p_max, group_qry, memo)

        self.a = a
        self.loc = loc
        self.scale = scale
        self.plot_label = label or f'Gamma({self.a},{self.scale})'

        self.set_dist(gamma(a=self.a, loc=self.loc, scale=self.scale), (self.a - 1) * self.scale)


# ----------------------------------------------------------------------------------------------------------------------
class NormalDistributionProcess(DistributionProcess, ABC):
    """Normal distribution process.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Iteration selector.
        p_max (float): The maximum proportion of mass that should be converted (i.e., at the mode of the distribution).
            The distribution that describes mass conversion is internally scaled to match that argument.  If 'None', no
            scaling is performed.
        loc (float): Location parameter.
        scale (float): Scale parameter.
        label (str, optional): Plot label.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, name='norm-distrib-proc', t=TimeAlways(), i=IterAlways(), p_max=None, loc=0.0, scale=1.0, label=None, group_qry=None, memo=None):
        super().__init__(name, t, i, p_max, group_qry, memo)

        self.loc = loc
        self.scale = scale
        self.plot_label = label or f'Normal({self.loc},{self.scale})'

        self.set_dist(norm(loc=self.loc, scale=self.scale), self.loc)


# ----------------------------------------------------------------------------------------------------------------------
class FibonacciSeq(Sequence, DifferenceEquation):
    """The Fibonacci numbers sequence.

    This rule updates the designated group attribute with sequential Fibonacci numbers, one number per iteration.

    Args:
        attr (str): The attribute name to store the sequence in.  The state of the sequence is stored in the group's
            attribute space using two extra attributes: ``attr_1`` and ``attr_2``.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, attr='fib', name='fibonacci-seq', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.attr   = attr                             # n
        self.attr_1 = Rule.get_rand_name(f'{attr}_1')  # n-1
        self.attr_2 = Rule.get_rand_name(f'{attr}_2')  # n-2

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        n  = group.ga(self.attr)
        n1 = group.ga(self.attr_1) or 0
        n2 = group.ga(self.attr_2) or 0

        if n1 == 0:
            n_, n1, n2 = 1,1,0
        elif n2 == 0:
            n_, n1, n2 = 1,1,1
        else:
            n_ = n1 + n2
            n2 = n1
            n1 = n_

        return [GroupSplitSpec(p=1.00, attr_set={ self.attr: n_, self.attr_1: n1, self.attr_2: n2 })]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha(self.attr)


# ----------------------------------------------------------------------------------------------------------------------
class CompoundInterstSeq(Sequence, DifferenceEquation):
    """Applies the comound interest formula to the designated attribute.

    The sequence state is store in the rule object and not the group attribute space.  Consequently, one instance of
    the rule should associated with only one group.

    Args:
        attr (str): Attribute name to store the value in.
        r (float): Nominal annual interest rate.
        n (int): Compund frequency in months.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, attr, r, n, name='compund-interst-seq', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.attr = attr
        self.r = r
        self.n = n

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        p0 = group.ga(self.attr)
        t = 1  # we do this every single iteration
        p = p0 * (1 + self.r / self.n) ** (self.n * t)

        return [GroupSplitSpec(p=1.00, attr_set={ self.attr: p })]


# ----------------------------------------------------------------------------------------------------------------------
class ProbabilisticEquation(Equation):
    """An algebraic equation p(t), where $p(t) \in [0,1]$ (i.e., it yields a probability)."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class ODeltaESystem(Rule):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class ODEDerivatives(ABC):
    """Ordinary differential equations system's derivatives.

    This object is one of the parameters to the :class:`~pram.rule.ODESystemMass` constructor.
    """

    def __init__(self):
        self.params = DotMap()

    @abstractmethod
    def get_fn(t, state):
        """Get the variable values of the system given state.

        Args:
            state (Iterable[float]): List of variable values.
        """

        pass

    def set_params(self, **kwargs):
        """Set a subset of superset of the parameters."""

        for (k,v) in kwargs.items():
            if v is not None:
                self.params[k] = v


# ----------------------------------------------------------------------------------------------------------------------
class ODESystem(Rule):
    """Ordinary differential equations system.

    Wraps a numeric integrator to solve a system of ordinary differential equations (ODEs) that do not interact with
    anything else in the simulation.  The non-interactivity stems from the derivatives being kept internally to the
    integrator only.  This is useful when computations need to happen on a step-by-step basis, consistent with the rest
    of the simulation, but the results should not affect the group attribute space.  One consequnce of this
    operationalization is that the rule need to fire only once per iteration, which this rule does.

    Args:
        fn_deriv (Callable): Derivative-computing function.
        y0 (Iterable[float]): Initial condition.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        dt (float): Time step size.
        ni_name (str): Name of numeric integrator.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, fn_deriv, y0, name='ode-system', t=TimeAlways(), i=IterAlways(), dt=1.0, ni_name='zvode', group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.y0 = y0  # initial condition
        self.dt = dt  # TODO: should change with the change in the timer
        self.ni_name = ni_name

        self.ni = ode(fn_deriv).set_integrator(self.ni_name, method='bdf')
        self.ni.set_initial_value(self.y0, 0)

        self.hist = DotMap(t=[], y=[])  # history

        self.iter_last = -1  # keep the last-computed iterations to avoid multiple computations (due to multiple groups)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        if iter != self.iter_last:  # only integrate if not done for this iteration
            self.ni.integrate(self.ni.t + self.dt)

            self.hist.t.append(self.ni.t)
            self.hist.y.append(self.ni.y)

            # print(f'{round(self.ni.t,1)}: {self.ni.y}')

            self.iter_last = iter

        return None

    def get_hist(self):
        """Get the history of times and derivatives computed at those times."""

        t = [0.00] + self.hist.t
        y = [[self.y0[i]] + [y[i].tolist().real for y in self.hist.y] for i in range(len(self.y0))]

        return [t,y]

    def set_params(self, fn_deriv):
        self.ni = ode(fn_deriv).set_integrator(self.ni_name, method='bdf')
        self.ni.set_initial_value(self.hist.y[-1], 0)  # TODO: Test this


# ----------------------------------------------------------------------------------------------------------------------
class ODESystemAttr(Rule):
    """Ordinary differential equations system on group attributes.

    Wraps a numeric integrator to solve a system of ordinary differential equations (ODEs) and store its state in the
    group attribute space.

    Args:
        fn_deriv (Callable): Derivative-computing function.
        y0 (Iterable[float]): Initial condition.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, fn_deriv, y0, name='ode-system', t=TimeAlways(), i=IterAlways(), dt=1.0, ni_name='zvode', memo=None):
        super().__init__(name, t, i, memo)

        self.y0 = y0
        self.y0_val = None  # initial condition
        self.dt = dt  # TODO: should change with the change in the timer
        self.ni_name = ni_name
        self.ni = ode(fn_deriv).set_integrator(ni_name, method='bdf')  # numeric integrator
        # self.ni.set_initial_value(self.y0, 0)

        self.hist = DotMap(t=[], y=[])  # history
        self.f = f

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        # Previous version (no initial condition update after the first one):
        # if self.y0_val is None:
        #     self.y0_val = [group.attr[y] for y in self.y0]
        #     self.ni.set_initial_value(self.y0_val, 0)
        #
        # self.ni.integrate(self.ni.t + self.dt)
        #
        # self.hist.t.append(self.ni.t)
        # self.hist.y.append(self.ni.y)

        self.y0_val = [group.attr[y] for y in self.y0]
        self.ni.set_initial_value(self.y0_val, 0)
        self.ni.integrate(self.dt)

        self.hist.t.append((iter + 1) * self.dt)
        self.hist.y.append(self.ni.y)

        return [GroupSplitSpec(p=1.00, attr_set={ self.y0[i]: self.ni.y[i].real for i in range(len(self.y0))})]

    def get_hist(self):
        """Get the history of times and derivatives computed at those times."""

        t = [0.00] + self.hist.t
        y = [[self.y0_val[i]] + [y[i].tolist().real for y in self.hist.y] for i in range(len(self.y0))]

        return [t,y]

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.ha(self.y0)

    def set_params(self, fn_deriv):
        self.ni = ode(fn_deriv).set_integrator(self.ni_name, method='bdf')


# ----------------------------------------------------------------------------------------------------------------------
class ODESystemMass(Rule):
    """Ordinary differential equations system on group mass.

    Wraps a numeric integrator to solve a system of ordinary differential equations (ODEs) that drive mass shifts.

    The initial state is deterimed by the group landscape.

    The rule splits the congruent group (i.e., the one meeting the search criteria) into the number of parts that
    corresponds to the number of ODEs.  Even though it might seem like different groups should be split differently,
    this solution is correct and stems from the fact that the new group sizes are calculated for the iteration step
    (i.e., for all groups) and not for individual group.

    Args:
        derivatives (ODEDerivatives): Derivatives object.
        group_queries (Iterable[GroupQry]): Group selectors.  The number and order must correspond to the
            ``derivatives``.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        dt (float): Time step size.
        ni_name (str): Name of numeric integrator.
        memo (str, optional): Description.
    """

    def __init__(self, derivatives, group_queries, name='ode-system-mass', t=TimeAlways(), i=IterAlways(), dt=1.0, ni_name='zvode', memo=None):
        super().__init__(name, t, i, memo)

        self.derivatives = derivatives
        self.group_queries = group_queries
        self.y0_mass = None  # initial condition
        self.dt = dt  # TODO: should change with the change in the timer
        self.ni_name = ni_name

        self.ni = ode(self.derivatives.get_fn()).set_integrator(self.ni_name, method='bdf')  # numeric integrator
        # self.ni.set_initial_value(self.y0, 0)

        self.hist = DotMap(t=[], y=[])  # history

        self.iter_last = -1  # keep the last-computed iterations to avoid multiple computations (due to multiple groups)

        self.tmp = DotMap()  # scratchpad to remember computation throughout an iteration (remember, one computation per iteration)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        if iter != self.iter_last:  # only integrate once per iteration
            # Previous version (no initial condition update after the first one):
            # if self.y0_mass is None:
            #     self.y0_mass = [sum([0] + [g.m for g in pop.get_groups(q)]) for q in self.group_queries]
            #     # print(f'y0 : {self.y0_mass}')
            #     self.ni.set_initial_value(self.y0_mass, 0)
            #
            # self.ni.integrate(self.ni.t + self.dt)
            #
            # self.hist.t.append(self.ni.t)
            # self.hist.y.append(self.ni.y)

            self.y0_mass = [sum([0] + [g.m for g in pop.get_groups(q)]) for q in self.group_queries]
            # self.y0_mass = [sum([0] + [g.m for g in pop.get_groups(GroupQry(attr=q.attr))]) for q in self.group_queries]
            self.ni.set_initial_value(self.y0_mass, 0)
            self.ni.integrate(self.dt)

            self.hist.t.append((iter + 1) * self.dt)
            self.hist.y.append(self.ni.y)

            # n  = self.ni.y[0].tolist().real + self.ni.y[1].tolist().real + self.ni.y[2].tolist().real
            self.tmp.m  = sum(self.y0_mass)
            # self.tmp.ps = min(1.00, max(0.00, self.ni.y[0].tolist().real / self.tmp.n))
            # self.tmp.pi = min(1.00, max(0.00, self.ni.y[1].tolist().real / self.tmp.n))
            # self.tmp.pr = 1.00 - self.tmp.ps - self.tmp.pi
            self.tmp.p = [min(1.00, max(0.00, self.ni.y[i].tolist().real / self.tmp.m)) for i in range(len(self.group_queries) - 1)]
            self.tmp.p.append(1.00 - sum(self.tmp.p))

        # return [
        #     GroupSplitSpec(p=self.tmp.ps, attr_set={ 'flu': 's' }),
        #     GroupSplitSpec(p=self.tmp.pi, attr_set={ 'flu': 'i' }),
        #     GroupSplitSpec(p=self.tmp.pr, attr_set={ 'flu': 'r' })
        # ]

        return [GroupSplitSpec(p=self.tmp.p[i], attr_set=self.group_queries[i].attr) for i in range(len(self.group_queries))]

    def get_hist(self):
        """Get the history of times and derivatives computed at those times."""

        t = [0.00] + self.hist.t
        y = [[self.y0_mass[i]] + [y[i].tolist().real for y in self.hist.y] for i in range(len(self.group_queries))]

        return [t,y]

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.ha(self.y0)

    def set_params(self, **kwargs):
        self.derivatives.set_params(**kwargs)
        self.ni = ode(self.derivatives.get_fn()).set_integrator(self.ni_name, method='bdf')


# ----------------------------------------------------------------------------------------------------------------------
class LogisticMap(Sequence, ChaoticMap):
    # x_{n+1} = r x_n (1 - x_n),
    #
    # where $r \n [2,4]$ and $x \in [0,1]$.
    #
    # --------------------------------------------------------------------------------------------------------------------
    #
    # Time domain                : Discrete
    # Space domain               : Real
    # Number of space dimensions : 1
    # Number of parameters       : 1

    pass


# ----------------------------------------------------------------------------------------------------------------------
class LotkaVolterraSystem(ChaoticMap, ODESystem):
    # A classic population dynamics model.
    #
    # a is the natural growth rate of rabbits in the absence of predation,
    # c is the natural death rate of foxes in the absence of food (rabbits),
    # b is the death rate per encounter of rabbits due to predation,
    # e is the efficiency of turning predated rabbits into foxes.
    #
    # --------------------------------------------------------------------------------------------------------------------
    #
    # Time domain                : Continuous
    # Space domain               : Real
    # Number of space dimensions : 3
    # Number of parameters       : 4

    pass


# ----------------------------------------------------------------------------------------------------------------------
class LorenzSystem(ChaoticMap, ODESystem):
    # Atmospheric convection model.
    #
    # A simplified mathematical model for atmospheric convection specified by the Lorenz equations:
    #
    #     dx/dt = sigma * (y - x)
    #     dy/dt = x * (rho - z) - y
    #     dz/dt = xy - beta * z
    #
    # The equations relate the properties of a two-dimensional fluid layer uniformly warmed from below and cooled from above.
    # In particular, the equations describe the rate of change of three quantities with respect to time: x is proportional to
    # the rate of convection, y to the horizontal temperature variation, and z to the vertical temperature variation. The
    # constants sigma, rho, and beta are system parameters proportional to the Prandtl number, Rayleigh number, and certain
    # physical dimensions of the layer itself.
    #
    # From a technical standpoint, the Lorenz system is nonlinear, non-periodic, three-dimensional and deterministic.
    #
    # SRC: https://en.wikipedia.org/wiki/Lorenz_system
    #
    # --------------------------------------------------------------------------------------------------------------------
    #
    # Time domain                : Continuous
    # Space domain               : Real
    # Number of space dimensions : 3
    # Number of parameters       : 3

    pass


# ----------------------------------------------------------------------------------------------------------------------
class ProbabilisticAutomaton(Rule, ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class MarkovChain(MarkovProcess, ProbabilisticAutomaton, ABC):
    """Markov process with a discrete state space."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class DiscreteInvMarkovChain(MarkovChain, DiscreteSpacetimeStochasticProcess, StationaryProcess):
    """Discrete-time time-homogenous Markov chain with finite state space.

    The following sample transition model for the variables named X::

                   x_1^t   x_2^t
        x_1^{t+1}    0.1     0.3
        x_2^{t+1}    0.9     0.7

    should be specified as the following list of stochastic vectors::

        { 'x1': [0.1, 0.9], 'x2': [0.3, 0.7] }

    Definition:

        A stochastic process $X = \{X_n:n \geq 0\}$ on a countable set $S$ is a Markov Chain if, for any $i,j \in S$
        and $n \geq 0$,

        P(X_{n+1} = j | X_0, \ldots, X_n) = P(X_{n+1} = j | X_n)
        P(X_{n+1} = j | X_n = i) = p_{ij}

        The $p_{ij} is the probability that the Markov chain jumps from state $i$ to state $j$.  These transition
        probabilities satisfy $\sum_{j \in S} p_{ij} = 1, i \in S$, and the matrix $P=(pij)$ is the transition matrix
        of the chain.

    Rule notation A::

        code:
            DiscreteInvMarkovChain('flu', { 's': [0.95, 0.05, 0.00], 'i': [0.00, 0.80, 0.10], 'r': [0.10, 0.00, 0.90] })

        init:
            tm = {
                s: [0.95, 0.05, 0.00],
                i: [0.00, 0.80, 0.20],
                r: [0.10, 0.00, 0.90]
            }  # right stochastic matrix

        is-applicable:
            has-attr: flu

        apply:
            curr_state = group.attr.flu
            new_state_sv = tm[]  # stochastic vector
            move-mass:
                new_state_sv[0] -> A:flu = 's'
                new_state_sv[1] -> A:flu = 'i'
                new_state_sv[2] -> A:flu = 'r'

    Rule notation B::

        tm_i = tm[group.attr.flu]
        tm_i[0] -> A:flu = 's'
        tm_i[1] -> A:flu = 'i'
        tm_i[2] -> A:flu = 'r'

    Args:
        attr (str): Name of state attribute.
        tm (Mapping[str,Iterable[float]]): Transition matrix.  Keys correspond to state names and values to lists of
            transition probabilities.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
        cb_before_apply (Callable, optional): Function called before the group is split.  The signature is
            ``fn(group, attr_val, tm)``.
    """

    def __init__(self, attr, tm, name='markov-chain', t=TimeAlways(), i=IterAlways(), memo=None, cb_before_apply=None):
        super().__init__(name, t, i, memo)

        if sum([i for x in list(tm.values()) for i in x]) != float(len(tm)):
            raise ValueError(f"'{self.__class__.__name__}' class: Probabilities in the transition model must add up to 1")

        self.attr = attr
        self.tm = tm
        self.states = list(self.tm.keys())  # simplify and speed-up lookup in apply()
        self.cb_before_apply = cb_before_apply

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        attr_val = group.get_attr(self.attr)
        tm = self.tm.get(attr_val)
        if tm is None:
            raise ValueError(f"'{self.__class__.__name__}' class: Unknown state '{group.get_attr(self.attr)}' for attribute '{self.attr}'")
        if self.cb_before_apply:
            tm = self.cb_before_apply(group, attr_val, tm) or tm
        return [GroupSplitSpec(p=tm[i], attr_set={ self.attr: self.states[i] }) for i in range(len(self.states)) if tm[i] > 0]

    def get_states(self):
        """Get list of states."""

        return self.states

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.has_attr([ self.attr ])


# ----------------------------------------------------------------------------------------------------------------------
class DiscreteVarMarkovChain(MarkovChain, DiscreteSpacetimeStochasticProcess):
    """Discrete-time non-time-homogenous Markov chain with finite state space.

    Rule notation A::

        code:
            fn_flu_s_sv:
                p_inf = n@_{attr.flu = 'i'} / n@
                return [1 - p_inf, p_inf, 0.00]

            TimeVarMarkovChain('flu', { 's': fn_flu_s_sv, 'i': [0.00, 0.80, 0.20], 'r': [0.10, 0.00, 0.90] })

        ext:
            p_inf

        init:
            tm = {
                s: [1 - p_inf, p_inf, 0.00],
                i: [0.00, 0.80, 0.20],
                r: [0.10, 0.00, 0.90]
            }  # right stochastic matrix

        is-applicable:
            has-attr: flu

        apply:
            tm_i = tm[group.attr.flu]
            move-mass:
                tm_i[0] -> A: flu = s
                tm_i[1] -> A: flu = i
                tm_i[2] -> A: flu = r

    Rule notation B::

        tm_i = tm[group.attr.flu]
        tm_i[0] -> A:flu = 's'
        tm_i[1] -> A:flu = 'i'
        tm_i[2] -> A:flu = 'r'

    Args:
        attr (str): Name of state attribute.
        tm (Mapping[str,Iterable[float]]): Transition matrix.  Keys correspond to state names and values to lists of
            transition probabilities.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        memo (str, optional): Description.
    """

    def __init__(self, attr, tm, name='markov-chain', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(name, t, memo)

        for sv in tm.values():  # stochastic vectors or function pointers
            if isinstance(sv, list):
                if sum(sv) != 1.00:
                    raise ValueError(f"'{self.__class__.__name__}' class: Probabilities in the transition model must add up to 1")

        self.attr = attr
        self.tm = tm
        self.states = list(self.tm.keys())  # simplify and speed-up lookup in apply()

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        tm = self.tm.get(group.get_attr(self.attr))
        if tm is None:
            raise ValueError(f"'{self.__class__.__name__}' class: Unknown state '{group.get_attr(self.attr)}' for attribute '{self.attr}'")
        if hasattr(tm, '__call__'):
            tm = tm(pop, group, iter, t)
        return [GroupSplitSpec(p=tm[i], attr_set={ self.attr: self.states[i] }) for i in range(len(self.states)) if tm[i] > 0]

    def get_states(self):
        """Get list of states."""

        return self.states

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.has_attr([ self.attr ])


# ----------------------------------------------------------------------------------------------------------------------
class CotinuousMarkovChain(MarkovChain, DiscreteSpaceContinuousTimeStochasticProcess):
    """Continuous-time Markov chain with finite state space."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class BernoulliScheme(DiscreteInvMarkovChain):
    """Bernoulli scheme or Bernoulli shift is a generalization of the Bernoulli process to more than two possible
    outcomes.

    A Bernoulli scheme is a special case of a Markov chain where the transition probability matrix has identical rows,
    which means that the next state is even independent of the current state (in addition to being independent of the
    past states).

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, attr='state', vals=(0,1), name='bernoulli-process', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(var, { vals[0]: [1-p, p], vals[1]: [1-p, p] }, name, t, i, memo)


# ----------------------------------------------------------------------------------------------------------------------
class BernoulliProcess(BernoulliScheme):
    """
    A Bernoulli process is a sequence of independent trials in which each trial results in a success or failure with
    respective probabilities $p$ and $q=1−p$.

    A Bernoulli scheme with only two possible states is known as a Bernoulli process.

    Binomial Markov Chain.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, p=0.5, attr='state', vals=(0,1), name='bernoulli-process', t=TimeAlways(), i=IterAlways(), memo=None):
        super().__init__(attr, { vals[0]: [1-p, p], vals[1]: [1-p, p] }, name, t, i, memo)


# ----------------------------------------------------------------------------------------------------------------------
class RandomWalk(DiscreteInvMarkovChain):
    """A Markov processes in discreet time.

    The results of discretizing both time and space of a diffusion equation.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class LevyProcess(MarkovProcess):
    """The continuous-time analog of a random walk."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class DiffusionProcess(MarkovProcess):
    """A continuous-time Markov process with almost surely continuous sample paths.

    A solution to a stochastic differential equation (in short form),

        d X_t = \mu(X_t, t)dt + \sigma X_t(X_t, t)dB_t,

    where $B$ is a standard Brownian motion.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class WienerProcess(LevyProcess, DiffusionProcess):
    """Markov processes in continuous time.

    Brownian motion process or Brownian motion.

    A limit of simple random walk.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class OrnsteinUhlenbeckProcess(LevyProcess, DiffusionProcess):
    """A stationary Gauss–Markov mean-reverting process."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class PoissonProcess(LevyProcess):
    """Poisson point process.

    Markov processes in continuous time.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class JumpProcess(PoissonProcess):
    """Jump process.

    A jump process is a type of stochastic process that has discrete movements, called jumps, with random arrival
    times, rather than continuous movement, typically modelled as a simple or compound Poisson process.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class CoxProcess(PoissonProcess):
    """Cox process.

    A point process which is a generalization of a Poisson process where the time-dependent intensity is itself a
    stochastic process.

    Also known as a doubly stochastic Poisson process.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class CellularAutomaton(StochasticProcess):
    """Cellular automaton.

    Cellular automata are a discrete-time dynamical system of interacting entities, whose state is discrete.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class StochasticCellularAutomaton(CellularAutomaton):
    """Probabilistic cellular automata (PCA), random cellular automata, or locally interacting Markov chains."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class InteractingParticleSystem(StochasticProcess):
    """Continuous-time Markov jump processes describing the collective behavior of stochastically interacting
    components.  IPS are the continuous-time analogue of stochastic cellular automata."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class BirthDeathProcess(PoissonProcess):
    """A jump process is a type of stochastic process that has discrete movements, called jumps, with random arrival
    times, rather than continuous movement, typically modelled as a simple or compound Poisson process."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
class BirthDeathProcess(MarkovProcess):
    """Birth death process.

    A special case of continuous-time Markov process where the state transitions are of only two types: "births", which
    increase the state variable by one and "deaths", which decrease the state by one.

    A homogeneous Poisson process is a pure birth process.

    The following infinitesimal generator of the process (\lambda are birth rates and \mu are death rates)::

        | -l_0           l_0             0             0    0  ... |
        |  m_1  -(l_1 + m_1)           l_1             0    0  ... |
        |    0           m_2  -(l_2 + m_2)           l_2    0  ... |
        |    0             0           m_3  -(l_3 + m_3)  l_3  ... |
        |    .             .             .             .    .  ... |
        |    .             .             .             .    .  ... |

    should be provided as::

        [l_0, [l_1, m_1], [l_2, m_2], [l_3, m_3], ..., m_n]

    Sojourn times have exponential p.d.f. :math:`\lambda e^{-\lambda t}`.
    """

    # Birth process examples [1]:
    # - Radioactive transformations
    # - Conversion of fibroge molecules into fibrin molecules follows a birth process (blood clotting)
    #
    # Application areas [1]:
    # - Epidemics
    # - Queues, inventories, and reliability
    # - Production management
    # - Computer communication systems
    # - Neutron propagation
    # - Optics
    # - Chemical reactions
    # - Construction and mining
    # - Compartmental models
    #
    # References:
    # [1] Birth and Death Processes (Chapter 4)
    #     http://neutrino.aquaphoenix.com/ReactionDiffusion/SERC5chap4.pdf
    # [2] https://cs.nyu.edu/mishra/COURSES/09.HPGP/scribe3

    def __init__(self, ig, name='birth-death-process', t=TimeAlways()):
        super().__init__(name, t)

        self.ig = ig  # inifinitesimal generator of the process


# ----------------------------------------------------------------------------------------------------------------------
class SIRModelGillespie(Rule):
    """Gillespie algorithm.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class PoissonIncidenceProcess(PoissonProcess):
    """Homogenous and inhomogenous Poisson point process for disease incidence.

    The non-homogeneity of the Poisson process is handled by allowing the user to specify how the rate (i.e., the
    $\lambda$ parameter) of the process changes over time.  The change to that rate is given as the combination of a
    delta on it and the number of iterations for that delta to take place.  The rate parameter is allowed to change
    gradually (i.e., at every step of the simulation) or in strict increments (e.g., only every 10 iterations) and is
    controlled by the ``inc_smooth`` argument.  The rate is determined by the ``rate_delta_mode`` argument.

    A Cox process, also known as a doubly stochastic Poisson process is a point process which is a generalization of a
    Poisson process where the time-dependent intensity is itself a stochastic process.  The Cox process can be
    implemented by passing as an argument a function which will be called at every iteration to determine the current
    Poisson rate.  ``rate_delta_mode = RateDeltaMode.FN`` must be used in that case.

    An example model given as "AD incidence doubles every five years after 65 yo" can be instantiated by using the
    delta of two and the number of iterations of five.

    Epidemiology of Alzheimer's disease and other forms of dementia in China, 1990-2010: a systematic review and
    analysis.  `PDF <https://www.thelancet.com/action/showPdf?pii=S0140-6736%2813%2960221-4>`_

    `Wikipedia: Incidence (epidemiology) <https://en.wikipedia.org/wiki/Incidence_(epidemiology)>`_

    Rule notation A::

        code:
            PoissonIncidenceProcess('ad', 65, 0.01, 2, 5, rate_delta_mode=PoissonIncidenceProcess.RateDeltaMode.EXP))

        init:
            age_0=65, l_0=0.01, c=2, t_c=5

        is-applicable:
            has-attr: age, ad

        apply:
            if (group.attr.age >= age_0):
                l = double_rate(l)        # l0 * c^{(group.attr.age - age_0) / t_c}
                p_0 = PoissonPMF(l,0)  # ...
                move-mass:
                        p_0 -> A:age = group.attr.age + 1
                    1 - p_0 -> A:age = group.attr.age + 1, A:ad = True

    Rule notation B::

        (group.attr.age >= age_0)
            l = double_rate(l)        # l0 * c^{(group.attr.age - age_0) / t_c}
            p_0 = PoissonPMF(l,0)
            p_0     -> A:age = group.attr.age + 1
            1 - p_0 -> A:age = group.attr.age + 1, A:ad = True

    Args:
        attr (str): Attribute name for the disease.
        age_0 (int): Lower bound cut-off age for getting AD.
        rate_0 (float): Base rate of the Poisson process (i.e., at the cut-off age).
        delta (float): Increase the rate by this factor.
        delta_t (float): Increase the rate every this many time units
        rate_delta_mode (str):
            - ``NC``  - No change (i.e., the Poisson process is stationary)
            - ``ADD`` - Additive
            - ``MUL`` - Multiplicative
            - ``EXP`` - Exponential
            - ``FN``  - User-provided lambda function called at every iteration
        fn_calc_lambda (Callable, optional): Custom lambda function.
        is_smooth (bool): Smooth increment if true, defined increments otherwise.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    RateDeltaMode = IntEnum('RateDeltaMode', 'NC ADD MUL EXP FN')  # incidence (or rate) change mode

    def __init__(self, attr, age_0, rate_0, delta, delta_t, rate_delta_mode=RateDeltaMode.NC, fn_calc_lambda=None, is_smooth=True, name='change-incidence', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.attr = attr            # the attribute name for the disease
        self.age_0 = age_0          # lower bound cut-off age for getting AD
        self.rate_0 = rate_0        # base rate of the Poisson process (i.e., at the cut-off age)
        self.rate = rate_0          # current rate of the Poisson process
        self.delta = delta          # increase the rate by this factor
        self.delta_t = delta_t      # increase the rate every this many time units
        self.is_smooth = is_smooth  # flag: smooth increment? (or happen in the defined increments)

        self.rate_delta_mode = rate_delta_mode
        self.fn_calc_lambda = {
            self.RateDeltaMode.NC  : self.calc_lambda_nc,
            self.RateDeltaMode.ADD : self.calc_lambda_add,
            self.RateDeltaMode.MUL : self.calc_lambda_mul,
            self.RateDeltaMode.EXP : self.calc_lambda_exp,
            self.RateDeltaMode.FN  : rate_delta_mode
        }.get(self.rate_delta_mode)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        return self.get_split_specs(group)

    def calc_lambda_nc(self, age):
        """No rate change (i.e., a stationary Poisson point process)."""

        return self.rate_0

    def calc_lambda_add(self, age):
        """Additive rate change."""

        if self.is_smooth:
            return self.rate_0 + self.delta *            ((age - self.age_0) / self.delta_t)
        else:
            return self.rate_0 + self.delta *  math.floor((age - self.age_0) / self.delta_t)

    def calc_lambda_mul(self, age):
        """Multiplicative rate change."""

        if self.is_smooth:
            return self.rate_0 * self.delta *            ((age - self.age_0) / self.delta_t)
        else:
            return self.rate_0 * self.delta *  math.floor((age - self.age_0) / self.delta_t)

    def calc_lambda_exp(self, age):
        """Exponential rate change."""

        if self.is_smooth:
            return self.rate_0 * self.delta **           ((age - self.age_0) / self.delta_t)
        else:
            return self.rate_0 * self.delta ** math.floor((age - self.age_0) / self.delta_t)

    def calc_lambda_hmm(self, age):
        """Poisson hidden Markov models (PHMM) are special cases of hidden Markov models where a Poisson process has a
        rate which varies in association with changes between the different states of a Markov model."""

        return self.rate_0

    def get_split_specs(self, group, age_inc=1):
        """
        Args:
            group (Group): Group.
            age_inc (int): Number by which to increment the ``age`` attribute.
        """

        age = group.ga('age')

        if age < self.age_0:
            return [GroupSplitSpec(p=1.00, attr_set={ 'age': age + age_inc, self.attr: False })]

        self.rate = self.fn_calc_lambda(age)
        p0 = poisson(self.rate).pmf(0)
        # print(f'n: {round(group.m,2):>12}  age: {age:>3}  l: {l:>3}  p0: {round(p0,2):<4}  p1: {round(1-p0,2):<4}')

        return [
            GroupSplitSpec(p=    p0, attr_set={ 'age': age + age_inc, self.attr: False }),
            GroupSplitSpec(p=1 - p0, attr_set={ 'age': age + age_inc, self.attr: True  })
        ]

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.ha(['age', self.attr])

    def setup(self, pop, group):
        """See :meth:`pram.rule.Rule.setup <Rule.setup()>`."""

        if group.ha('age'):
            return self.get_split_specs(group, 0)
        else:
            return None


# ----------------------------------------------------------------------------------------------------------------------
class OneDecayProcess(Rule):
    """Radioactive one-decay process.

    N(t) = N_0 e^{-\lambda t}
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class SegregationModel(Rule):
    """Segregation model.

   Rule notation A::

        code:
            SegregationModel('team', 2)

        init:
            p_migrate = 0.05

        is-applicable:
            has-attr: team
            has-rel: @

        apply:
            p_team = n@_{attr.team = group.attr.team} / n@  # ...
            if (p_team < 0.5):
                rd p_migrate -> R:@ = get_random_site()
                nc 1 - p_migrate

    Rule notation B::

        p_team = n@_{attr.team = group.attr.team} / n@
        if (p_team < 0.5) p_migrate -> R:@ = get_random_site()

    Args:
        attr (str): Name.
        attr_dom_card (str): Cardinality of the attribute values set.
        p_migrate (str): Probability of the population to migrate if repelled.
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    def __init__(self, attr, attr_dom_card, p_migrate=0.05, name='segregation-model', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.attr = attr
        self.p_migrate = p_migrate           # proportion of the population that will migrate if repelled
        self.p_repel = 1.00 / attr_dom_card  # population will be repelled (i.e., will move) if the site that population is at has a proportion of same self.attr lower than this

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        attr_val = group.get_attr(self.attr)
        site     = group.get_site_at()
        m        = site.get_mass()

        if m == 0:
            return None

        m_team = site.get_mass(GroupQry(attr={ self.attr: attr_val }))
        p_team = m_team / m  # proportion of same self.attr

        if p_team < self.p_repel:
            site_rnd = self.get_random_site(pop, site)
            return [
                GroupSplitSpec(p=    self.p_migrate, rel_set={ Site.AT: site_rnd }),
                GroupSplitSpec(p=1 - self.p_migrate)
            ]
        else:
            return None

    def get_random_site(self, pop, site):
        """Get a random site different than the specified one."""

        s = random.choice(list(pop.sites.values()))
        while s == site:
            s = random.choice(list(pop.sites.values()))
        return s

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.ha([self.attr]) and group.hr([Site.AT])


# ----------------------------------------------------------------------------------------------------------------------
class SEIRModel(Rule, ABC):
    """The SEIR compartmental epidemiological model.
    """

    # Disease states (SEIR)
    #     S Susceptible
    #     E Exposed (i.e., incubation period)
    #     I Infectious (can infect other agents)
    #     R Recovered
    #
    # State transition timing
    #     S  Until E (either by another agent or on import)
    #     E  Random sample
    #     I  Random sample
    #     R  Indefinite

    State = IntEnum('State', 'S E I R')

    ATTR = 'seir-state'

    T_UNIT_MS = TimeU.MS.d
    NAME = 'SEIR model'
    ATTRS = { ATTR: [member.value for name, member in State.__members__.items()] }


    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, name='seir', t=TimeAlways(), susceptibility=1.0, p_start_E=0.05, do_clean=True, memo=None):
        super().__init__(name, t, memo)

        self.susceptibility = susceptibility
        self.p_start_E = p_start_E    # prob of starting in the E state
        self.do_clean = do_clean      # flag: remove traces of the disease when recovered?

    # ------------------------------------------------------------------------------------------------------------------
    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        if iter == 0:
            return [
                GroupSplitSpec(p=    self.p_start_E, attr_set={ self.__class__.ATTR: self.__class__.State.E }),
                GroupSplitSpec(p=1 - self.p_start_E, attr_set={ self.__class__.ATTR: self.__class__.State.S })
            ]

        state = group.get_attr(self.__class__.ATTR)
        return {
            self.__class__.State.S: self.apply_S,
            self.__class__.State.E: self.apply_E,
            self.__class__.State.I: self.apply_I,
            self.__class__.State.R: self.apply_R,
        }.get(state, self.apply_err)(pop, group, iter, t, state)

    # ------------------------------------------------------------------------------------------------------------------
    def apply_S(self, pop, group, iter, t, state):
        return None

    def apply_E(self, pop, group, iter, t, state):
        # s  = self.T['E_rvs']()
        # tE = group.get_attr('tE') or 0
        # p  = min(1.00, tE / s)
        # print(f'E: {round(s,2)}  {tE}  {round(p,2)}')

        tE = group.get_attr('tE') or 0
        if tE < self.T['E_min'] * self.t_mul:
            p = 0.00
        elif tE > self.T['E_max'] * self.t_mul:
            p = 1.00
        else:
            p = self.T['E_rv'].cdf(tE / self.t_mul)

        # print(f'E: {round(p,4)}  {tE}')

        if self.do_clean:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.I }, attr_del=['tE']),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.E, 'tE': tE + 1 })
            ]
        else:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.I, 'tE': tE + 1 }),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.E, 'tE': tE + 1 })
            ]

    def apply_I(self, pop, group, iter, t, state):
        # s  = self.T['I_rvs']()
        # tI = group.get_attr('tI') or 0
        # p  = min(1.00, tI / s)
        # print(f'I: {round(s,2)}  {tI}  {round(p,2)}')

        tI = group.get_attr('tI') or 0
        if tI < self.T['I_min'] * self.t_mul:
            p = 0.00
        elif tI > self.T['I_max'] * self.t_mul:
            # print(iter)
            p = 1.00
        else:
            p = self.T['I_rv'].cdf(tI / self.t_mul)

        # if tI > self.T['I_max'] * self.t_mul:
        #     print(group.n)

        # print(f'I: {round(p,4)}  {tI}')

        if self.do_clean:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.R }, attr_del=['tI']),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.I, 'tI': tI + 1 })
            ]
        else:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.R, 'tI': t + 1 }),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.I, 'tI': t + 1 })
            ]

    def apply_R(self, pop, group, iter, t, state):
        return None

    def apply_err(self, pop, group, iter, t, state):
        raise ValueError(f'Invalid value for attribute {self.__class__.ATTR}: {state}')

    # ------------------------------------------------------------------------------------------------------------------
    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.get_attr(self.__class__.ATTR) in self.__class__.State

    def setup(self, pop, group):
        """See :meth:`pram.rule.Rule.setup <Rule.setup()>`."""

        return [GroupSplitSpec(p=1.0, attr_set={ self.__class__.ATTR: __class__.State.S })]

    def cleanup(self, pop, group):
        """See :meth:`pram.rule.Rule.cleanup <Rule.cleanup()>`."""

        return [GroupSplitSpec(p=1.0, attr_del=['tE', 'tI'])]


# ----------------------------------------------------------------------------------------------------------------------
class SEIRFluModel(SEIRModel):
    """The SEIR model for the influenza.
    """

    # --------------------------------------------------------------------------------------------------------------------
    # This part is taken from the "FRED daily rules" by David Sinclair and John Grefenstette (5 Feb, 2019)
    # --------------------------------------------------------------------------------------------------------------------
    #
    # Disease states (SEIR)
    #     S  Susceptible
    #     E  Exposed (i.e., incubation period)
    #     IS Infectious & Symptomatic (can infect other agents)
    #     IA Infectious & Asymptomatic (can infect other agents)
    #     R  Recovered
    #
    # State transition probability
    #     S  --> E   1.00
    #     E  --> IS  0.67
    #     E  --> IA  0.33
    #     IS --> R   1.00
    #     IA --> R   1.00
    #
    # State transition timing
    #     S      Until E (either by another agent or on import)
    #     E      Median = 1.9, dispersion = 1.23 days (M= 45.60, d=29.52 hours)
    #     IS/IA  Median = 5.0, dispersion = 1.50 days (M=120.00, d=36.00 hours)
    #     R      Indefinite
    #
    # Probability of staying home
    #     IS/IA  0.50
    #
    # Time periods
    #     Drawn from a log-normal distribution
    #         median = exp(mean)
    #     Dispersion is equivalent to the standard deviation, but with range of [median/dispersion, median*dispersion]
    #         E      [1.54, 2.34] days ([36.96,  56.16] hours)
    #         IS/IA  [3.33, 7.50] days ([79.92, 180.00] hours)
    #
    # --------------------------------------------------------------------------------------------------------------------
    # This part is taken from FRED docs on parameters (github.com/PublicHealthDynamicsLab/FRED/wiki/Parameters)
    # --------------------------------------------------------------------------------------------------------------------
    #
    # The median and dispersion of the lognormal distribution are typically how incubation periods and symptoms are
    # reported in the literature (e.g., Lessler, 2009).  They are translated into to the parameters of the lognormal
    # distribution using the formulas:
    #
    #     location = log(median)
    #     scale = 0.5 * log(dispersion)
    #
    # We can expect that about 95% of the draws from this lognormal distribution will fall between (median / dispersion)
    # and (median * dispersion).
    #
    # Example (Lessler, 2009):
    #
    #     influenza_incubation_period_median     = 1.9
    #     influenza_incubation_period_dispersion = 1.81
    #     influenza_symptoms_duration_median     = 5.0
    #     influenza_symptoms_duration_dispersion = 1.5
    #
    # These values give an expected incubation period of about 1 to 3.5 days, and symptoms lasting about 3 to 7.5 days.
    #
    # --------------------------------------------------------------------------------------------------------------------
    # This part is based on my own reading of the following article:
    # Lessler (2009) Incubation periods of acute respiratory viral infections -- A systematic review.
    # --------------------------------------------------------------------------------------------------------------------
    #
    # Influenza A
    #     incubation period median     = 1.9
    #     incubation period dispersion = 1.22
    #
    # Timer
    #     1. Sample once per time step and store the current time in state as group attribute.
    #     2. Sample once and register a timer event with the global simulation timer which will perform the appropriate
    #        mass distribution.

    ATTR = 'flu-state'
    NAME = 'SEIR flu model'

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, t=TimeAlways(), susceptibility=1.0, p_start_E=0.05, do_clean=True, memo=None):
        super().__init__('flu', t, susceptibility, p_start_E, do_clean, memo)

        # self.T['E_median']     = lambda: 1.90 * self.t_mul
        # self.T['E_dispersion'] = 1.23
        # self.T['E_min']        = lambda: self.T['E_median']() / self.T['E_dispersion']
        # self.T['E_max']        = lambda: self.T['E_median']() * self.T['E_dispersion']
        # self.T['I_median']     = lambda: 5.00 * self.t_mul
        # self.T['I_dispersion'] = 1.50
        # self.T['I_min']        = lambda: self.T['I_median']() / self.T['I_dispersion']
        # self.T['I_max']        = lambda: self.T['I_median']() * self.T['I_dispersion']
        #
        # self.T['E_rvs'] = lambda t: min(self.T['E_max'](), max(self.T['E_min'](), lognorm(0.607 * math.log(self.T['E_dispersion']), 0, self.T['E_median']()).pdf(t)))
        # self.T['I_rvs'] = lambda t: min(self.T['I_max'](), max(self.T['I_min'](), lognorm(0.607 * math.log(self.T['I_dispersion']), 0, self.T['I_median']()).pdf(t)))

        self.T['E_median']     = 1.90
        self.T['E_dispersion'] = 1.23
        self.T['E_min']        = self.T['E_median'] / self.T['E_dispersion']
        self.T['E_max']        = self.T['E_median'] * self.T['E_dispersion']

        self.T['I_median']     = 5.00
        self.T['I_dispersion'] = 1.50
        self.T['I_min']        = self.T['I_median'] / self.T['I_dispersion']
        self.T['I_max']        = self.T['I_median'] * self.T['I_dispersion']

        self.T['E_rv'] = lognorm(0.607 * math.log(self.T['E_dispersion']), 0, self.T['E_median'])  # incubation period sampling distribution
        self.T['I_rv'] = lognorm(0.607 * math.log(self.T['I_dispersion']), 0, self.T['I_median'])  # infectious period sampling distribution


# ----------------------------------------------------------------------------------------------------------------------
class GoToRule(Rule):
    """GoTo.

    Changes the location of a group from the designated site to the designated site.  Both of the sites are
    specificed by relation name (e.g., 'store').  The rule will only apply to a group that (a) is currently located at
    the "from" relation and has the "to" relation.  If the "from" argument is None, all groups will qualify as long as
    they have the "to" relation.  The values of the relations need to be of type Site.

    Only one 'from' and 'to' location is handled by this rule (i.e., lists of locations are not supported).

    The group's current location is defined by the 'Site.AT' relation name and that's the relation that this rule
    updates.

    Example use is to compel a portion of agents that are at ``home`` go to ``work`` or vice versa.

    Args:
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    NAME = 'Goto'

    def __init__(self, p, rel_from, rel_to, name='goto', t=TimeAlways(), i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        # Err.type(rel_from, 'rel_from', str, True)
        # Err.type(rel_to, 'rel_to', str)

        self.p = p
        self.rel_from = rel_from  # if None, the rule will not be conditional on current location
        self.rel_to = rel_to

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        return [
            GroupSplitSpec(p=self.p, rel_set={ Site.AT: group.get_rel(self.rel_to) }),
            GroupSplitSpec(p=1 - self.p)
        ]

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        if not super().is_applicable(group, iter, t):
            return False

        # Moving from the designated location:
        if self.rel_from is not None:
            return (
                group.has_sites([self.rel_to, self.rel_from]) and
                group.get_rel(Site.AT) == group.get_rel(self.rel_from)
            )
        # Moving from any location:
        else:
            return group.has_sites(self.rel_to)


# ----------------------------------------------------------------------------------------------------------------------
class GoToAndBackTimeAtRule(Rule):
    """GoTo and back.

    Compels agents to go to the designated site and back; agents stay at the destination for a specificed amount of
    time.

    Both 'to' and 'back' sites need to be specified.  Moreover, the proportion of agents moving to the destination can
    be speficied as a probability distribution function (PDF) as can the time agents should spend at the destination.
    As a consequence of the latter, this rule tracks the time agents spend at the destination via a group attribute
    't_at_attr'.  Furthermore, agents can be forced to the original location at the end of the rule's time interval.
    For example, students can be made go back home at the time schools close, irrespective of how long they have been
    at school.

    This rule is only applicable to groups with both to and from sites and only those that are currently located at the
    latter.

    Args:
        to (str): Name of destination site.
        back (str): Name of source site.
        time_pdf_to (Mapping[int,float]): PDF for going to the destination site.
        time_pdf_back (Mapping[int,float]): PDF for going back to the source site.
        do_force_back (bool): Force the entire population back to the source site?
        name (str): Name.
        t (:class:`~pram.rule.Time`, int, tuple[int,int], set[int]): Compatible time selector.
        i (:class:`~pram.rule.Iter`, int, tuple[int,int], set[int]): Compatible iteration selector.
        group_qry (GroupQry, optional): Compatible group selector.
        memo (str, optional): Description.
    """

    # TODO: Switch from PDF to CDF because it's more natural.

    NAME = 'Goto and back'

    TIME_PDF_TO_DEF   = { 8: 0.5, 12:0.5 }
    TIME_PDF_BACK_DEF = { 1: 0.05, 3: 0.2, 4: 0.25, 5: 0.2, 6: 0.1, 7: 0.1, 8: 0.1 }

    def __init__(self, to='school', back='home', time_pdf_to=None, time_pdf_back=None, t_at_attr='t@', do_force_back=True, name='to-and-back', t=[8,16], i=IterAlways(), group_qry=None, memo=None):
        super().__init__(name, t, i, group_qry, memo)

        self.to   = to
        self.back = back

        self.time_pdf_to   = time_pdf_to   or GoToAndBackTimeAtRule.TIME_PDF_TO_DEF
        self.time_pdf_back = time_pdf_back or GoToAndBackTimeAtRule.TIME_PDF_BACK_DEF

        self.rv_to   = Rule.tp2rv(self.time_pdf_to)
        self.rv_back = Rule.tp2rv(self.time_pdf_back)

        self.t_at_attr = t_at_attr          # name of the attribute that stores time-at the 'to' location
        self.do_force_back = do_force_back  # force all agents to go back at the end of rule time (i.e., 't.t1')?

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        if group.has_rel({ Site.AT: group.get_rel(self.back) }) and not group.has_attr(self.t_at_attr):
            return self.apply_to(group, iter, t)

        if group.has_rel({ Site.AT: group.get_rel(self.to) }):
            return self.apply_back(group, iter, t)

    def apply_to(self, group, iter, t):
        p = self.rv_to.cdf(t)

        return [
            GroupSplitSpec(p=p, attr_set={ self.t_at_attr: 0 }, rel_set={ Site.AT: group.get_rel(self.to) }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_back(self, group, iter, t):
        # TODO: Make this method generalize to arbitrary time steps; currently, 1h increments are asssumed.

        t_at = group.get_attr(self.t_at_attr)
        p = self.rv_back.cdf(t_at)

        if self.do_force_back and t >= self.t.t1:
            p = 1.0

        return [
            GroupSplitSpec(p=p,     attr_set={ self.t_at_attr: (t_at + 1) }, rel_set={ Site.AT: group.get_rel(self.back) }),
            GroupSplitSpec(p=1 - p, attr_set={ self.t_at_attr: (t_at + 1) })
        ]

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.has_sites([self.to, self.back])


# ----------------------------------------------------------------------------------------------------------------------
class ResetRule(Rule):
    NAME = 'Reset'

    def __init__(self, t=5, i=None, attr_del=None, attr_set=None, rel_del=None, rel_set=None, memo=None):
        super().__init__('reset', t, i, memo)

        self.attr_del = attr_del
        self.attr_set = attr_set
        self.rel_del  = rel_del
        self.rel_set  = rel_set

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        return [GroupSplitSpec(p=1.0, attr_del=self.attr_del, attr_set=self.attr_set, rel_del=self.rel_del, rel_set=self.rel_del)]


# ----------------------------------------------------------------------------------------------------------------------
class ResetSchoolDayRule(ResetRule):
    NAME = 'Reset school day'

    def __init__(self, t=5, i=None, attr_del=['t-at-school'], attr_set=None, rel_del=None, rel_set=None, memo=None):
        super().__init__(t, i, attr_del, attr_set, rel_del, rel_set, memo)
        self.name = 'reset-school-day'

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.has_sites(['home', 'school'])


# ----------------------------------------------------------------------------------------------------------------------
class ResetWorkDayRule(ResetRule):
    NAME = 'Reset work day'

    def __init__(self, t=5, i=None, attr_del=None, attr_set=None, rel_del=None, rel_set=None, memo=None):
        super().__init__(t, i, attr_del, attr_set, rel_del, rel_set, memo)
        self.name = 'reset-work-day'

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        return super().is_applicable(group, iter, t) and group.has_sites(['home', 'work'])


# ----------------------------------------------------------------------------------------------------------------------
class RuleAnalyzerTestRule(Rule):
    NAME = 'Rule analyzer test'

    def __init__(self, t=[8,20], i=None, memo=None):
        super().__init__('progress-flu', t, i, memo)

    def an(self, s): return f'b{s}'  # attribute name
    def rn(self, s): return f's{s}'  # relation  name

    def apply(self, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        if group.has_attr({ 'flu-stage': 's' }):
            pass
        elif group.has_attr({ 'flu-stage': 'i' }):
            pass
        elif group.has_attr({ 'flu-stage': 'r' }):
            pass

    def is_applicable(self, group, iter, t):
        """See :meth:`pram.rule.Rule.is_applicable <Rule.is_applicable()>`."""

        g = group
        c01, c02, c03, c04, c05 = 'cc01', 'cc02', 'cc03', 'cc04', 'cc05'  # attribute names stored in local variables
        s01, s02, s03, s04, s05 = 'ss01', 'ss02', 'ss03', 'ss04', 'ss05'  # ^ (relation)

        return (
            super().is_applicable(group, iter, t) and

            g.has_attr('a01') and g.has_attr([ 'a02', 'a03' ]) and g.has_attr({ 'a04':1, 'a05':2 }) and
            g.has_attr(c01) and g.has_attr([ c02, c03 ]) and g.has_attr({ c04:1, c05:2 }) and
            g.has_attr(self.an('01')) and g.has_attr([ self.an('02'), self.an('03') ]) and g.has_attr({ self.an('04'):1, self.an('05'):2 }) and

            g.has_rel('r01') and g.has_rel([ 'r02', 'r03' ]) and g.has_rel({ 'r04':1, 'r05':2 }) and
            g.has_rel(s01) and g.has_rel([ s02, s03 ]) and g.has_rel({ s04:1, s05:2 }) and
            g.has_rel(self.rn('01')) and g.has_rel([ self.rn('02'), self.rn('03') ]) and g.has_rel({ self.rn('04'):1, self.rn('05'):2 })
        )


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluProgressRule(Rule):
    """Describes how a population transitions between the flu states of susceptible, infected, and recovered.

    Rule notation A::

        code:
            SimpleFluProgressRule()

        is-applicable:
            has-attr: flu

        apply:
            if group.attr.flu = 's':
                p_inf = n@_{attr.flu = 'i'} / n@
                move-mass:
                    p_inf -> A:flu = 'i'
            if group.attr.flu = 'i':
                move-mass:
                    0.2 -> A:flu = 'r'
            if group.attr.flu = 'r':
                move-mass:
                    0.1 -> A:flu = 's'

    Rule notation B::

        if (flu = s)
            p_inf = n@_{attr.flu = i} / n@
            p_inf -> A: flu = 'i'
        if (flu = i) 0.2 > A:flu = r
        if (flu = r) 0.1 > A:flu = s
    """

    ATTRS = { 'flu': [ 's', 'i', 'r' ] }
    NAME = 'Simple flu progression model'

    def __init__(self, t=None, i=None, memo=None):
        super().__init__('flu-progress-simple', t, i, memo)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            p_infection = group.get_site_at().get_mass_prop(GroupQry(attr={ 'flu': 'i' }))
            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Infected:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.8, attr_set={ 'flu': 'i' }),
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' })
            ]

        raise ValueError('Unknown flu state')

    def setup(self, pop, group):
        """See :meth:`pram.rule.Rule.setup <Rule.setup()>`."""

        return [
            GroupSplitSpec(p=0.9, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=0.1, attr_set={ 'flu': 'i' })
        ]


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluProgressMoodRule(Rule):
    """Describes how a population transitions between the states of susceptible, infected, and recovered.  Includes the
    inconsequential 'mood' attribute which may improve exposition of how the PRAM framework works.

    Rule notation A::

        code:
            SimpleFluProgressMoodRule()

        is-applicable:
            has-attr: flu

        apply:
            if group.attr.flu = 's':
                p_inf = n@_{attr.flu = 'i'} / n@
                move-mass:
                    p_inf -> A:flu = 'i', A:mood = 'annoyed'
            if group.attr.flu = 'i':
                move-mass:
                    0.2 -> A:flu = 'r', A:mood = 'happy'
                    0.5 ->              A:mood = 'bored'
                    0.3 ->              A:mood = 'annoyed'
            if group.attr.flu = 'r':
                move-mass:
                    0.1 -> A:flu = 's'

    Rule notation B::

        if (flu = s)
            p_inf = n@_{attr.flu = i} / n@
            p_inf -> A:flu = 'i', A:mood = 'annoyed'
        if (flu = i)
            0.2 -> A:flu = 'r', A:mood = 'happy'
            0.5 ->              A:mood = 'bored'
            0.3 ->              A:mood = 'annoyed'
        if (flu = r) 0.1 > A:flu = s
    """

    NAME = 'Simple flu progression model with mood'

    def __init__(self, t=TimeAlways(), memo=None):
        super().__init__('flu-progress-simple', t, memo)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            p_infection = group.get_site_at().get_mass_prop(GroupQry(attr={ 'flu': 'i' }))

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'i', 'mood': 'annoyed' }),
                GroupSplitSpec(p=1 - p_infection)
            ]

        # Infected:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r', 'mood': 'happy'   }),
                GroupSplitSpec(p=0.5, attr_set={             'mood': 'bored'   }),
                GroupSplitSpec(p=0.3, attr_set={             'mood': 'annoyed' })
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.9)
            ]

        print(group.ga('flu'))
        raise ValueError('Unknown flu state')


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluLocationRule(Rule):
    """Describes how student population changes location conditional upon being exposed to the flu.

    Rule notation A::

        code:
            SimpleFluLocationRule()

        is-applicable:
            has-attr: flu
            has-rel: home, school

        apply:
            if group.attr.flu = 'i':
                if group.attr.income = 'l':
                    rd 0.1 -> R:@ = group.rel.home  # rd - redistribute mass
                    nc 0.9                          # nc - no change
                if group.attr.income = 'm':
                    rd 0.6 -> R:@ = group.rel.home
                    nc 0.4
            if group.attr.flu = 'r':
                rd 0.8 -> R:@ = group.rel.school
                nc 0.2

    Rule notation B::

        if (flu = i)
            if (income = l)
                rd 0.1 -> R:@ = home
                nc 0.9
            if (income = m)
                rd 0.6 > R:@ = home
                nc 0.4
        if (flu = r)
            rd 0.8 > R:@ = school
            nc 0.2
    """

    ATTRS = { 'flu': [ 's', 'i', 'r' ], 'income': ['l', 'm'] }
    NAME = 'Simple flu location model'

    def __init__(self, t=TimeAlways(), memo=None):
        super().__init__('flu-location', t, memo)

    def apply(self, pop, group, iter, t):
        """See :meth:`pram.rule.Rule.apply <Rule.apply()>`."""

        # Infected and low income:
        if group.has_attr({ 'flu': 'i', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Infected and medium income:
        if group.has_attr({ 'flu': 'i', 'income': 'm' }):
            return [
                GroupSplitSpec(p=0.6, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.4)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.8, rel_set={ Site.AT: group.get_rel('school') }),
                GroupSplitSpec(p=0.2)
            ]

        return None
