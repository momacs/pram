# -*- coding: utf-8 -*-
"""Contains PRAM simulation code."""

# ----------------------------------------------------------------------------------------------------------------------
#
# Probabilitistic Relational Agent-based Models (PRAMs)
#
# BSD 3-Clause License
#
# Copyright (c) 2018-2020, momacs, University of Pittsburgh
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Contributors
#     Tomek D Loboda  (https://github.com/ubiq-x)                                                     [2018.12.16 - ...]
#         The original author of the present design and implementation
#     Paul R Cohen                                                                             [2018.10.01 - 2019.12.31]
#         The idea of PRAMs
#
# ----------------------------------------------------------------------------------------------------------------------

import ast
import bz2
import datetime
import gc
import gzip
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import random
import sqlite3
import statistics
import time

from collections import namedtuple, Counter
from dotmap      import DotMap
from scipy.stats import gaussian_kde

from .data        import GroupSizeProbe, Probe
from .entity      import Agent, Group, GroupQry, Site
from .model.model import Model
from .pop         import GroupPopulation, GroupPopulationHistory
from .rule        import Rule, SimRule, IterAlways, IterPoint, IterInt
from .util        import Err, FS, Size, Time

__all__ = ['SimulationConstructionError', 'SimulationConstructionWarning', 'Simulation']


# ----------------------------------------------------------------------------------------------------------------------
class SimulationConstructionError(Exception): pass
class SimulationConstructionWarning(Warning): pass


# ----------------------------------------------------------------------------------------------------------------------
class Timer(object):
    """Simulation timer.

    This discrete time is unitless; it is the simulation context that defines the appropriate granularity.  For
    instance, million years (myr) might be appropriate for geological processes while Plank time might be appropriate
    for modeling quantum phenomena.  For example, one interpretation of 'self.t == 4' is that the current simulation
    time is 4am.

    To enable natural blending of rules that operate on different time scales, the number of milliseconds is stored by
    the object.  Because of this, rule times, which define the number of milliseconds in a time unit, can be scaled.

    Apart from time, this timers also stores the iteration count.

    Todo:
        This class and its subclasses are not yet incorporated into the package.
    """

    # TODO: Loop detection currently relies on modulus which does not handle 'step_size' > 1 properly.

    POSIX_DT = datetime.datetime(1970, 1, 1)  # POSIX datetime

    def __init__(self, ms=Time.MS.ms, iter=float('inf'), t0=0, tmin=0, tmax=10, do_disp_zero=True):
        self.ms = ms
        self.i_max = iter
        self.i = None  # set by reset()

        self.t0 = t0         # how time is going to be diplayed
        self.tmin = tmin     # ^
        self.tmax = tmax     # ^
        self.t = t0          # ^
        self.t_loop_cnt = 0  # ^

        self.last_iter_t0 = None
        self.last_iter_t1 = None
        self.last_iter_t  = None

        self.is_running = False

        if self.tmin > self.tmax:
            raise TypeError(f'The min time has to be smaller than the max time.')

        self.do_disp_zero = do_disp_zero  # flag: print empty line on zero?

        self.did_loop_on_last_iter = False  # flag: keeping this avoids printing looping info right before the simulation ends

        self.reset()

    def __repr__(self):
        return '{}(name={}, ms={}  iter={}))'.format(self.__class__.__name__, self.ms, self.i_max)

    def __str__(self):
        return '{}'.format(self.__class__.__name__, self.ms, self.i_max)

    def add_dur(self, dur):
        self.i_max += math.floor(dur / self.ms)

    def add_iter(self, iter):
        self.i_max += iter

    @staticmethod
    def by_ms(ms, **kwargs):
        ''' Returns a Timer object which matched the specified number of milliseconds. '''

        timer = {
            Time.MS.m  : MilsecTimer,
            Time.MS.s  : SecTimer,
            Time.MS.m  : MinTimer,
            Time.MS.h  : HourTimer,
            Time.MS.d  : DayTimer,
            Time.MS.w  : WeekTimer,
            Time.MS.M  : MonthTimer,
            Time.MS.y  : YearTimer
        }.get(ms, None)
        if not timer:
            raise ValueError(f'No timer associated with time unit given by the number of milliseconds: {ms}')
        return timer(**kwargs)

    def get_i(self):
        return self.i

    def get_t(self):
        return self.t

    def get_t_loop_cnt(self):
        return self.t_loop_cnt

    def get_i_left(self):
        return self.i_max - self.i

    def get_t_left(self):
        return self.i_max - self.i

    @staticmethod
    def get_ts():
        return (datetime.datetime.now() - Timer.POSIX_DT).total_seconds() * 1000  # [ms]

    def reset(self):
        self.i = 0
        self.t = self.t0
        self.t_loop_cnt = 0

    def set_dur(self, dur):
        self.i_max = math.floor(dur / self.ms)

    def set_iter(self, iter):
        self.i_max = iter

    def start(self):
        self.is_running = True
        self.last_iter_t0 = Timer.get_ts()

    def step(self):
        self.i += 1
        if self.i > self.i_max:
            self.i = self.i_max
            raise TypeError(f'Timer has reached the maximum value of {self.tmax}.')

        self.t = self.i % self.tmax + self.tmin

        if self.do_disp_zero and self.did_loop_on_last_iter:
            # print(f'\nLoop: {self.t_loop_cnt + 1}')
            self.did_loop_on_last_iter = False

        if self.i > 0 and self.t == self.tmin:
            self.t_loop_cnt += 1
            self.did_loop_on_last_iter = True

        self.last_iter_t1 = Timer.get_ts()
        self.last_iter_t  = self.last_iter_t1 - self.last_iter_t0
        self.last_iter_t0 = Timer.get_ts()
        self.last_iter_t1 = None

    def stop(self):
        self.last_iter_t1 = Timer.get_ts()
        self.last_iter_t  = self.last_iter_t1 - self.last_iter_t0
        self.is_running = False


# ----------------------------------------------------------------------------------------------------------------------
class MilsecTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.ms, iter, 0, 0, 1000, do_disp_zero)


class SecTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.s, iter, 0, 0, 60, do_disp_zero)


class MinTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.m, iter, 0, 0, 60, do_disp_zero)


class HourTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.h, iter, 0, 0, 24, do_disp_zero)


class DayTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.d, iter, 0, 0, 365, do_disp_zero)


class WeekTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.w, iter, 0, 0, 52, do_disp_zero)


class MonthTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.M, iter, 0, 0, 12, do_disp_zero)


class YearTimer(Timer):
    def __init__(self, iter=float('inf'), do_disp_zero=True):
        super().__init__(Time.MS.y, iter, 0, 2000, 10000, do_disp_zero)


# ----------------------------------------------------------------------------------------------------------------------
# class CalMonthTimer(Timer):


# ----------------------------------------------------------------------------------------------------------------------
# class CalDayTimer(Timer):


# ----------------------------------------------------------------------------------------------------------------------
class DynamicRuleAnalyzer(object):
    """Infers group attributes and relations conditioned upon based on running a simulation.

    Analyze rule conditioning after running the simulation.  This is done by processing group attributes and
    relations that the simulation has recorded as accessed by at least one rule.  The evidence of attributes and
    relations having been actually accessed is a strong one.  However, as tempting as it may be to use this
    information to prune groups, it's possible that further simulation iterations depend on other sets of
    attributes and relations.  As a consequence, it is not possible to perform error-free groups auto-pruning based
    on this analysis alone.

    The sets of attributes and relations used (i.e., conditioned on by at least one rule) are not cleared between
    simulation runs, only updated.  This way a simulation will always be aware of what was relevant from the very
    beginning.  This information will also be persisted if the simulation object is serialized to continue
    execution on another system.
    """

    def __init__(self, sim):
        self.sim = sim

        self.attr_used = set()
        self.rel_used  = set()

        self.attr_unused = set()
        self.rel_unused  = set()

    def analyze(self):
        # Get attributes and relations that define groups:
        self.attr_groups = set()
        self.rel_groups  = set()
        for g in self.sim.pop.groups.values():
            for ga in g.attr.keys(): self.attr_groups.add(ga)
            for gr in g.rel.keys():  self.rel_groups. add(gr)

        # Update the set of used (i.e., conditioned on by at least one rule) attributes and relations:
        # self.attr_used.update(getattr(Group, 'attr_used').copy())
        # self.rel_used.update(getattr(Group, 'rel_used').copy())
        self.attr_used.update(getattr(Group, 'attr_used'))
        self.rel_used.update(getattr(Group, 'rel_used'))

        # Do a set-subtraction to get attributes and relations not conditioned on by even one rule:
        self.attr_unused = self.attr_groups - self.attr_used
        self.rel_unused  = self.rel_groups  - self.rel_used


# ----------------------------------------------------------------------------------------------------------------------
class StaticRuleAnalyzer(object):
    """Analyzes the syntax (i.e., abstract syntax trees) of rule objects to identify group attributes and relations
    these rules condition on.

    Apart from attempting to deduce the attributes and rules, this class keeps track of the numbers of recognized and
    unrecognized attributes and relations (compartmentalized by method type, e.g., 'has_attr' and 'get_attr').

    References
        https://docs.python.org/3.6/library/dis.html
        https://docs.python.org/3.6/library/inspect.html
        https://docs.python.org/3.6/library/ast.html

        https://github.com/hchasestevens/astpath
        https://astsearch.readthedocs.io/en/latest

    TODO
        Double check the case of multiple sequential simulation runs.
    """

    def __init__(self):
        self.reset()

    def _dump(self, node, annotate_fields=True, include_attributes=False, indent='  '):
        '''
        Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default
        '''

        if not isinstance(node, ast.AST):
            raise TypeError('expected AST, got %r' % node.__class__.__name__)
        return self._format(node, 0, annotate_fields, include_attributes, indent)

    def _format(self, node, level=0, annotate_fields=True, include_attributes=False, indent='  '):
        '''
        Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default
        '''

        if isinstance(node, ast.AST):
            fields = [(a, self._format(b, level, annotate_fields, include_attributes, indent)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(a, self._format(getattr(node, a), level, annotate_fields, include_attributes, indent))
                               for a in node._attributes])
            return ''.join([
                node.__class__.__name__,
                '(',
                ', '.join(('%s=%s' % field for field in fields)
                           if annotate_fields else
                           (b for a, b in fields)),
                ')'])
        elif isinstance(node, list):
            lines = ['[']
            lines.extend((indent * (level + 2) + self._format(x, level + 2, annotate_fields, include_attributes, indent) + ','
                         for x in node))
            if len(lines) > 1:
                lines.append(indent * (level + 1) + ']')
            else:
                lines[-1] += ']'
            return '\n'.join(lines)
        return repr(node)

    def _analyze_test(self, node):
        if isinstance(node, ast.AST):
            fields = [(a, self._analyze_test(b)) for a,b in ast.iter_fields(node)]
            return ''.join([node.__class__.__name__, '(', ', '.join((b for a,b in fields)), ')'])
        elif isinstance(node, list):
            lines = []
            lines.extend((self._analyze_test(x) + ',' for x in node))
            return '\n'.join(lines)
        return repr(node)

    def _analyze(self, node):
        '''
        Processe a node of the AST recursively looking for method calls that suggest group attribute and relation names
        conditioned on by the rule.  It also updates counts of all known and unknown names (compartmentalized by the
        method name).

        References
            https://docs.python.org/3.6/library/ast.html#abstract-grammar
        '''

        if isinstance(node, ast.AST):
            for _,v in ast.iter_fields(node):
                self._analyze(v)

            if node.__class__.__name__ == 'Call':
                call_args = list(ast.iter_fields(node))[1][1]

                if list(ast.iter_fields(node))[0][1].__class__.__name__ == 'Attribute':
                    attr = list(ast.iter_fields(node))[0][1]
                    attr_name = list(ast.iter_fields(attr))[1][1]

                    if attr_name in ('get_attr', 'get_rel', 'has_attr', 'has_rel', 'ga', 'gr', 'ha', 'hr'):
                        call_args = call_args[0]
                        if call_args.__class__.__name__ == 'Str':
                            if attr_name in ('get_attr', 'has_attr', 'ga', 'ha'):
                                self.attr_used.add(StaticRuleAnalyzer.get_str(call_args))
                            else:
                                self.rel_used.add(StaticRuleAnalyzer.get_str(call_args))
                            self.cnt_rec[attr_name] += 1
                        elif call_args.__class__.__name__ in ('List', 'Dict'):
                            for i in list(ast.iter_fields(call_args))[0][1]:
                                if i.__class__.__name__ == 'Str':
                                    if attr_name in ('get_attr', 'has_attr', 'ga', 'ha'):
                                        self.attr_used.add(StaticRuleAnalyzer.get_str(i))
                                    else:
                                        self.rel_used.add(StaticRuleAnalyzer.get_str(i))
                                    self.cnt_rec[attr_name] += 1
                                else:
                                    self.cnt_unrec[attr_name] += 1
                                    # print(list(ast.iter_fields(i)))
        elif isinstance(node, list):
            for i in node:
                self._analyze(i)

    def analyze_rules(self, rules):
        """
        Can be (and in fact is) called before any groups have been added.
        """

        # (1) Reset the state:
        self.attr_used = set()
        self.rel_used  = set()

        self.attr_unused = set()
        self.rel_unused  = set()

        self.cnt_rec   = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })  # recognized
        self.cnt_unrec = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })  # unrecognized

        # (2) Analyze the rules:
        for r in rules:
            self.analyze_rule(r)

        self.are_rules_done = True

    def analyze_groups(self, groups):
        """
        Should be called after all the groups have been added.
        """

        attr_groups = set()  # attributes defining groups
        rel_groups  = set()  # ^ (relations)
        for g in groups:
            for ga in g.attr.keys(): attr_groups.add(ga)
            for gr in g.rel.keys():  rel_groups. add(gr)

        self.attr_unused = attr_groups - self.attr_used  # attributes not conditioned on by even one rule
        self.rel_unused  = rel_groups  - self.rel_used   # ^ (relations)

        self.are_groups_done = True

    def analyze_rule(self, rule):
        tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef): continue  # skip non-classes

            for node_fn in node.body:
                if not isinstance(node_fn, ast.FunctionDef): continue  # skip non-methods
                self._analyze(node_fn.body)

                # if node_fn.name in ('is_applicable'): print(self._analyze_01(node_fn.body))

    def dump(self, rule):
        tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))
        print(self._dump(tree))

    @staticmethod
    def get_str(node):
        return list(ast.iter_fields(node))[0][1]

    def reset(self):
        self.are_rules_done  = False
        self.are_groups_done = False

        self.attr_used = set()
        self.rel_used  = set()

        self.attr_unused = set()
        self.rel_unused  = set()

        self.cnt_rec   = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })  # recognized
        self.cnt_unrec = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })  # unrecognized


# ----------------------------------------------------------------------------------------------------------------------
class SimulationAdder(object):
    """Simulation element adder.

    A syntactic sugar class that makes the simulation definition API even more flexible and palatable.

    An instance of this class is returned by :meth:`Simulation.add() <pram.sim.Simulation.add>` and enables better
    compartmentalization::

        s = Simulation()
        s.add().
            rules(...).
            probes(...).
            groups(...).
            sites(...)
        s.run(24)

    or::

        (Simulation().
            add().
                rules(...).
                probes(...).
                groups(...).
                sites(...).
                done().
            run(24)
        )

    All methods of this class, apart from :meth:`~pram.sim.SimulationAdder.done`, return ``self`` for method call
    chaining.
    """

    def __init__(self, sim):
        self.sim = sim

    def done(self):
        """Declare being done adding and return to the :class:`~pram.sim.Simulation` object.

        Returns:
            :class:`~pram.sim.Simulation`: The delegating simulation object.
        """

        return self.sim

    def group(self, group):
        """Shortcut to :meth:`Simulation.add_group() <pram.sim.Simulation.add_group>`."""

        self.sim.add_group(group)
        return self

    def groups(self, groups):
        """Shortcut to :meth:`Simulation.add_groups() <pram.sim.Simulation.add_groups>`."""

        self.sim.add_groups(groups)
        return self

    def probe(self, probe):
        """Shortcut to :meth:`Simulation.add_probe() <pram.sim.Simulation.add_probe>`."""

        self.sim.add_probe(probe)
        return self

    def probes(self, probes):
        """Shortcut to :meth:`Simulation.add_probes() <pram.sim.Simulation.add_probes>`."""

        self.sim.add_probes(probes)
        return self

    def rule(self, rule):
        """Shortcut to :meth:`Simulation.add_rule() <pram.sim.Simulation.add_rule>`."""

        self.sim.add_rule(rule)
        return self

    def rules(self, rules):
        """Shortcut to :meth:`Simulation.add_rules() <pram.sim.Simulation.add_rules>`."""

        self.sim.add_rules(rules)
        return self

    def sim_rule(self, rule):
        """Shortcut to :meth:`Simulation.add_sim_rule() <pram.sim.Simulation.add_sim_rule>`."""

        self.sim.add_sim_rule(rule)
        return self

    def sim_rules(self, rules):
        """Shortcut to :meth:`Simulation.add_sim_rules() <pram.sim.Simulation.add_sim_rules>`."""

        self.sim.add_sim_rules(rules)
        return self

    def site(self, site):
        """Shortcut to :meth:`Simulation.add_site() <pram.sim.Simulation.add_site>`."""

        self.sim.add_site(site)
        return self

    def sites(self, sites):
        """Shortcut to :meth:`Simulation.add_sites() <pram.sim.Simulation.add_sites>`."""

        self.sim.add_sites(sites)
        return self


# ----------------------------------------------------------------------------------------------------------------------
class SimulationDBI(object):
    """Simulation database interface.

    A syntactic sugar class that makes the simulation definition API even more flexible and palatable.

    An instance of this class is returned by :meth:`Simulation.dbi() <pram.sim.Simulation.dbi>` and enables better
    compartmentalization::

        from pram.util import PgDB
        ...

        s = Simulation()
        s.add().
            rules(...).
            probes(...)
        s.db(PgDB('localhost', 5432, 'user', 'pwd', 'database')).
            gen_groups(...)
        s.run(24)

    or::

        from pram.util import PgDB
        ...

        (Simulation().
            add().
                rules(...).
                probes(...).
                done().
            db(PgDB('localhost', 5432, 'user', 'pwd', 'database')).
                gen_groups(...).
                done().
            run(24)
        )

    All methods of this class, apart from :meth:`~pram.sim.SimulationAdder.done`, return ``self`` for method call
    chaining.

    Args:
        db (DB): Database management system specific object.
    """

    def __init__(self, sim, db):
        self.sim = sim
        self.db = db

    def done(self):
        """Declare being done interacting with the database interface and return to the :class:`~pram.sim.Simulation`
        object.

        Returns:
            :class:`~pram.sim.Simulation`: The delegating simulation object.
        """

        return self.sim

    def gen_groups(self, tbl, attr_db=[], rel_db=[], attr_fix={}, rel_fix={}, attr_rm=[], rel_rm=[], rel_at=None, limit=0, is_verbose=False):
        """Shortcut to :meth:`Simulation.gen_groups_from_db() <pram.sim.Simulation.gen_groups_from_db>`."""

        self.sim.gen_groups_from_db(self.db, tbl, attr_db, rel_db, attr_fix, rel_fix, attr_rm, rel_rm, rel_at, limit, is_verbose)
        return self


# ----------------------------------------------------------------------------------------------------------------------
class SimulationPlotter(object):
    """Simulation results plotter.

    A syntactic sugar class that makes the simulation results API even more flexible and palatable.

    An instance of this class is returned by :meth:`Simulation.plot() <pram.sim.Simulation.plot>` and enables better
    compartmentalization::

        s = Simulation()
        s.add().
            rules(...).
            probes(...).
            groups(...).
            sites(...)
        s.run(24)
        s.plot().
            group_size(...)

    or::

        (Simulation().
            add().
                rules(...).
                probes(...).
                groups(...).
                sites(...).
                done().
            run(24).
            plot().
                group_size(...).
                done()
        )

    All methods of this class, apart from :meth:`~pram.sim.SimulationAdder.done`, return ``self`` for method call
    chaining.
    """

    def __init__(self, sim):
        self.sim = sim

    def done(self):
        """Declare being done plotting and return to the :class:`~pram.sim.Simulation` object.

        Returns:
            :class:`~pram.sim.Simulation`: The delegating simulation object.
        """

        return self.sim

    def group_size(self, do_log=False, fpath=None, title='Distribution of Group Size', nx=250):
        """Shortcut to :meth:`Simulation.plot_group_size() <pram.sim.Simulation.plot_group_size>`."""

        self.sim.plot_group_size(do_log, fpath, title, nx)
        return self

    def site_size(self, do_log=False, fpath=None, title='Distribution of Group Size', nx=250):
        """Shortcut to :meth:`Simulation.plot_site_size() <pram.sim.Simulation.plot_site_size>`."""

        self.sim.plot_site_size(do_log, fpath, title, nx)
        return self


# ----------------------------------------------------------------------------------------------------------------------
class SimulationSetter(object):
    """Simulation element setter.

    A syntactic sugar class that makes the simulation definition API even more flexible and palletable.

    An instance of this class is returned by :meth:`Simulation.set() <pram.sim.Simulation.set>` and enables better
    compartmentalization::

        s = Simulation()
        s.add().
            rules(...).
            probes(...).
            groups(...).
            sites(...)
        s.set().
            pragma_live_info(True)
        s.run(24)

    or::

        (Simulation().
            add().
                rules(...).
                probes(...).
                groups(...).
                sites(...).
                done().
            set().
                pragma_live_info(True).
                done().
            run(24)
        )

    All methods of this class, apart from :meth:`~pram.sim.SimulationAdder.done`, return ``self`` for method call
    chaining.
    """

    def __init__(self, sim):
        self.sim = sim

    def cb_after_iter(self, fn):
        """Shortcut to :meth:`Simulation.set_cb_after_iter() <pram.sim.Simulation.set_cb_after_iter>`."""

        self.sim.set_cb_after_iter(fn)
        return self

    def cb_before_iter(self, fn):
        """Shortcut to :meth:`Simulation.set_cb_before_iter() <pram.sim.Simulation.set_cb_before_iter>`."""

        self.sim.set_cb_before_iter(fn)
        return self

    def cb_check_work(self, fn):
        """Shortcut to :meth:`Simulation.set_cb_check_work() <pram.sim.Simulation.set_cb_check_work>`."""

        self.sim.set_cb_check_work(fn)
        return self

    def cb_save_state(self, fn):
        """Shortcut to :meth:`Simulation.set_cb_save_state() <pram.sim.Simulation.set_cb_save_state>`."""

        self.sim.set_cb_save_state(fn)
        return self

    def cb_upd_progress(self, fn):
        """Shortcut to :meth:`Simulation.set_cb_upd_progress() <pram.sim.Simulation.set_cb_upd_progress>`."""

        self.sim.set_cb_upd_progress(fn)
        return self

    def done(self):
        """Declare being done setting and return to the :class:`~pram.sim.Simulation` object.

        Returns:
            :class:`~pram.sim.Simulation`: The delegating simulation object.
        """

        return self.sim

    def dur(self, dur):
        """Shortcut to :meth:`Simulation.set_dur() <pram.sim.Simulation.set_dur>`."""

        self.sim.set_dur(dur)
        return self

    def iter_cnt(self, n):
        """Shortcut to :meth:`Simulation.set_iter_cnt() <pram.sim.Simulation.set_iter_cnt>`."""

        self.sim.set_iter_cnt(n)
        return self

    def fn_group_setup(self, fn):
        """Shortcut to :meth:`Simulation.set_fn_group_setup() <pram.sim.Simulation.set_fn_group_setup>`."""

        self.sim.set_fn_group_setup(fn)
        return self

    def pragma(self, name, value):
        """Shortcut to :meth:`Simulation.set_pragma() <pram.sim.Simulation.set_pragma>`."""

        self.sim.set_pragma(name, value)
        return self

    def pragmas(self, analyze=None, autocompact=None, autoprune_groups=None, autostop=None, autostop_n=None, autostop_p=None, autostop_t=None, comp_summary=None, fractional_mass=None, live_info=None, live_info_ts=None, probe_capture_init=None, rule_analysis_for_db_gen=None):
        """Shortcut to :meth:`Simulation.set_pragmas() <pram.sim.Simulation.set_pragmas>`."""

        self.sim.set_pragmas(analyze, autocompact, autoprune_groups, autostop, autostop_n, autostop_p, autostop_t, comp_summary, fractional_mass, live_info, live_info_ts, probe_capture_init, rule_analysis_for_db_gen)
        return self

    def pragma_analyze(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_analyze() <pram.sim.Simulation.set_pragma_analyze>`."""

        self.sim.set_pragma_analyze(value)
        return self

    def pragma_autocompact(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_autocompact() <pram.sim.Simulation.set_pragma_autocompact>`."""

        self.sim.set_pragma_autocompact(value)
        return self

    def pragma_autoprune_groups(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_autoprune_groups() <pram.sim.Simulation.set_pragma_autoprune_groups>`."""

        self.sim.set_pragma_autoprune_groups(value)
        return self

    def pragma_autostop(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_autostop() <pram.sim.Simulation.set_pragma_autostop>`."""

        self.sim.set_pragma_autostop(value)
        return self

    def pragma_autostop_n(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_autostop_n() <pram.sim.Simulation.set_pragma_autostop_n>`."""

        self.sim.set_pragma_autostop_n(value)
        return self

    def pragma_autostop_p(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_autostop_p() <pram.sim.Simulation.set_pragma_autostop_p>`."""

        self.sim.set_pragma_autostop_p(value)
        return self

    def pragma_autostop_t(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_autostop_t() <pram.sim.Simulation.set_pragma_autostop_t>`."""

        self.sim.set_pragma_autostop_t(value)
        return self

    def pragma_comp_summary(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_comp_summary() <pram.sim.Simulation.set_pragma_comp_summary>`."""

        self.sim.set_pragma_comp_summary(value)
        return self

    def pragma_fractional_mass(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_fractional_mass() <pram.sim.Simulation.set_pragma_fractional_mass>`."""

        self.sim.set_pragma_fractional_mass(value)
        return self

    def pragma_live_info(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_live_info() <pram.sim.Simulation.set_pragma_live_info>`."""

        self.sim.set_pragma_live_info(value)
        return self

    def pragma_live_info_ts(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_live_info_ts() <pram.sim.Simulation.set_pragma_live_info_ts>`."""

        self.sim.set_pragma_live_info_ts(value)
        return self

    def pragma_probe_capture_init(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_probe_capture_init() <pram.sim.Simulation.set_pragma_probe_capture_init>`."""

        self.sim.set_pragma_probe_capture_init(value)
        return self

    def pragma_rule_analysis_for_db_gen(self, value):
        """Shortcut to :meth:`Simulation.set_pragma_rule_analysis_for_db_gen() <pram.sim.Simulation.set_pragma_rule_analysis_for_db_gen>`."""

        self.sim.set_pragma_rule_analysis_for_db_gen(value)
        return self

    def rand_seed(self, rand_seed):
        """Shortcut to :meth:`Simulation.set_rand_seed() <pram.sim.Simulation.set_rand_seed>`."""

        self.sim.set_rand_seed(rand_seed)
        return self

    def var(self, name, val):
        """Shortcut to :meth:`Simulation.set_var() <pram.sim.Simulation.set_var>`."""

        self.sim.set_var(name, val)
        return self

    def vars(self, vars):
        """Shortcut to :meth:`Simulation.set_vars() <pram.sim.Simulation.set_vars>`."""

        self.sim.set_vars(vars)
        return self


# ----------------------------------------------------------------------------------------------------------------------
# class SimulationUI(object):
#     Var = namedtuple('Var', 'name val specs')
#
#     def __init__(self):
#         self.vars = []
#
#     def add(self, name, val, specs):
#         self.vars[name] = self.__class__.Var(name, val, specs)
#         return self
#
#     def get(self, name):
#         return self.vars.get[name]
#
#     def get_val(self, name):
#         return self.vars.get[name].val


# ----------------------------------------------------------------------------------------------------------------------
class Simulation(object):
    """A PRAM simulation.

    At this point, this simulation is a discrete-event simulation (DES).  DES models the operation of a system as a
    (discrete) sequence of events in time.  DES can be contrasted with continuous simulation in which the system state
    is changed continuously over time on the basis of a set of differential equations defining the rates of change of
    state variables.

    Args:
        pop_hist_len (int): Maximum lenght of the population history.  Keep memory utilization in mind when using
            positive numbers for this argument.
        traj_id (Any): ID of the trajectory which wraps the simulation object.  That ID should come from the trajectory
            ensemble database.
        random_seed (int, optional): Pseudo-random number generator seed.
        do_keep_mass_flow_specs (bool): Store the last iteration mass flow specs?  See
            :class:`pop.GroupPopulation <pram.pop.GroupPopulation>` and
            :class:`pop.MassFlowSpec <pram.pop.MassFlowSpec>` classes.
    """

    def __init__(self, pop_hist_len=0, traj_id=None, rand_seed=None, do_keep_mass_flow_specs=False):
        self.set_rand_seed(rand_seed)

        self.pid = os.getpid()  # process ID
        self.traj_id = traj_id  # trajectory ID
        self.run_cnt = 0

        self.pop = GroupPopulation(self, pop_hist_len, do_keep_mass_flow_specs)
        self.rules = []
        self.sim_rules = []
        self.probes = []

        self.timer = None  # value deduced in add_group() based on rule timers

        self.is_setup_done = False  # flag
            # ensures simulation setup is performed only once while enabling multiple incremental simulation runs of
            # arbitrary length thus promoting interactivity (a sine qua non for a user interface)

        self.running = DotMap(
            is_running = False,
            progress = 0.0,
            step = 1.0
        )

        self.fn = DotMap(
            group_setup = None  # called before the simulation is run for the very first time
        )

        self.analysis = DotMap(
            rule_static  = StaticRuleAnalyzer(),
            rule_dynamic = DynamicRuleAnalyzer(self)
        )

        self.vars = {}  # simulation variables

        self.reset_cb()
        self.reset_pragmas()
        self.reset_comp_hist()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.rand_seed or ""})'

    def _inf(self, msg):
        if not self.pragma.live_info:
            return

        if self.pragma.live_info_ts:
            print(f'[{datetime.datetime.now()}: info] {msg}')
        else:
            print(f'[info] {msg}')

    def add(self, lst=None):
        """Simulation element adder.

        If the ``lst`` argument is None, this method returns instance of the :class:`~pram.sim.SimulationAdder` class
        that will handle adding simulation elements.  Otherwise, it will add all elements of ``lst`` to the simulation.

        Args:
            lst (Iterable): Combination of objects of the following types: :class:`Group <pram.entity.Group>`,
                :class:`Probe <pram.data.Probe>`, :class:`SimRule <pram.rule.SimRule>`,
                :class:`Rule <pram.rule.Rule>`, :class:`Model <pram.model.model.Model>`, and
                :class:`Site <pram.entity.Site>`.

        Returns:
            SimulationAdder
        """

        if lst:
            for i in lst:
                if isinstance(i, Group):
                    self.add_group(i)
                elif isinstance(i, Probe):
                    self.add_probe(i)
                elif isinstance(i, SimRule):  # must be before Rule
                    self.add_sim_rule(i)
                elif isinstance(i, Rule):
                    self.add_rule(i)
                elif isinstance(i, Model):
                    self.add_rule(i.rule)
                elif isinstance(i, Site):
                    self.add_site(i)
            return self
        else:
            return SimulationAdder(self)

    def add_group(self, group):
        """Adds a group.

        Args:
            group (Group): The group.

        Returns:
            ``self``
        """

        # No rules present:
        if len(self.rules) == 0:
            raise SimulationConstructionError('A group is being added but no rules are present; rules need to be added before groups.')

        # No groups present:
        if len(self.pop.groups) == 0:  # run when the first group is being added (because that marks the end of adding rules)
            if not self.analysis.rule_static.are_rules_done:
                self.analyze_rules_static()

            # Sync simulation and rules timers:
            rule_t_unit_ms = min([r.T_UNIT_MS for r in self.rules])
            self.timer = Timer.by_ms(rule_t_unit_ms, iter=0)

        self.pop.add_group(group)
        return self

    def add_groups(self, groups):
        """Adds groups.

        Args:
            groups (Iterable[Group]): The groups.

        Returns:
            ``self``
        """

        for g in groups:
            self.add_group(g)
        return self

    def add_probe(self, probe):
        """Adds a probe.

        Args:
            probe (Probe): The probe.

        Returns:
            ``self``
        """

        if probe.name in [p.name for p in self.probes]:
            raise SimulationConstructionError(f'Probe with that name ({probe.name}) already exists.')

        self.pop.ar_enc.encode_probe(probe)
        self.probes.append(probe)
        probe.set_pop(self.pop)
        return self

    def add_probes(self, probes):
        """Adds probes.

        Args:
            probes (Iterable[Probe]): The probes.

        Returns:
            ``self``
        """

        for p in probes:
            self.add_probe(p)
        return self

    def add_rule(self, rule):
        """Adds a rule.

        If the rule has any inner rules, all of those are added as well.  An example of an inner rule is a desease
        transmission model that is being acted upon by another rule, e.g., an intervention rule which therefore
        contains it.  Such containing relationship is not being enforced by the framework however.

        An instance of a rule can only be added once.

        Args:
            rule (Rule): The rule.

        Returns:
            ``self``
        """

        if len(self.rules) == 0:
            self._inf('Constructing a PRAM')

        if len(self.pop.groups) > 0:
            raise SimulationConstructionError('A rule is being added but groups already exist; rules need be added before groups.')

        if isinstance(rule, Rule):
            self.add_rules(rule.get_inner_rules())
            if rule not in self.rules:
                self.rules.append(rule)
        elif isinstance(rule, Model):
            self.add_rule(rule.rule)

        # self.rules.append(rule)
        self.analysis.rule_static.analyze_rules(self.rules)  # keep the results of the static rule analysis current

        return self

    def add_rules(self, rules):
        """Adds rules.

        Args:
            rules (Iterable[Rule]): The rules.

        Returns:
            ``self``
        """

        for r in rules:
            self.add_rule(r)
        return self

    def add_sim_rule(self, rule):
        """Adds a simulation rule.

        Args:
            rule (SimRule): The simulation rule.

        Returns:
            ``self``
        """

        self.sim_rules.append(rule)
        self.set_vars(rule.vars)
        return self

    def add_sim_rules(self, rules):
        """Adds simulation rules.

        Args:
            rules (Iterable[SimRule]): The simulation rules.

        Returns:
            ``self``
        """

        for r in rules:
            self.add_sim_rule(r)
        return self

    def add_site(self, site):
        """Adds a site.

        Args:
            site (Site): The site.

        Returns:
            ``self``
        """

        self.pop.add_site(site)
        return self

    def add_sites(self, sites):
        """Adds sites.

        Args:
            sites (Iterable[Site]): The sites.

        Returns:
            ``self``
        """

        self.pop.add_sites(sites)
        return self

    def analyze_rules_static(self):
        """Runs static rule analysis.

        See :class:`~pram.sim.StaticRuleAnalyzer`.

        Returns:
            ``self``
        """

        self._inf('Running static rule analysis')

        self.analysis.rule_static.analyze_rules(self.rules)

        self._inf(f'    Relevant attributes found : {list(self.analysis.rule_static.attr_used)}')
        self._inf(f'    Relevant relations  found : {list(self.analysis.rule_static.rel_used)}')

        return self

    def analyze_rules_dynamic(self):
        """Runs dynamic rule analysis.

        See :class:`~pram.sim.DynamicRuleAnalyzer`.

        Returns:
            ``self``
        """

        self._inf('Running dynamic rule analysis')

        rd = self.analysis.rule_dynamic
        rd.analyze()

        if self.pragma.live_info:
            self._inf(f'    Accessed attributes    : {list(rd.attr_used)}')
            self._inf(f'    Accessed relations     : {list(rd.rel_used)}')
            self._inf(f'    Superfluous attributes : {list(rd.attr_unused)}')
            self._inf(f'    Superfluous relations  : {list(rd.rel_unused )}')
        else:
            if self.pragma.analyze and (len(rd.attr_unused) > 0 or len(rd.rel_unused) > 0):
                print('Based on the most recent simulation run, the following group attributes A and relations R are superfluous:')
                print(f'    A: {list(rd.attr_unused)}')
                print(f'    R: {list(rd.rel_unused )}')

        return self

    def analyze_rules_dynamic_old(self):
        self._inf('Running dynamic rule analysis')

        lr = self.analysis.rule_dynamic
        lr.clear()

        lr.attr_used = getattr(Group, 'attr_used').copy()  # attributes conditioned on by at least one rule
        lr.rel_used  = getattr(Group, 'rel_used').copy()   # relations

        lr.attr_groups = set()  # attributes defining groups
        lr.rel_groups  = set()  # relations
        for g in self.pop.groups.values():
            for ga in g.attr.keys(): lr.attr_groups.add(ga)
            for gr in g.rel.keys():  lr.rel_groups. add(gr)

        lr.attr_unused = lr.attr_groups - lr.attr_used  # attributes not conditioned on by even one rule
        lr.rel_unused  = lr.rel_groups  - lr.rel_used   # relations

        if self.pragma.live_info:
            self._inf(f'    Accessed attributes    : {list(self.analysis.rule_dynamic.attr_used)}')
            self._inf(f'    Accessed relations     : {list(self.analysis.rule_dynamic.rel_used)}')
            self._inf(f'    Superfluous attributes : {list(lr.attr_unused)}')
            self._inf(f'    Superfluous relations  : {list(lr.rel_unused )}')
        else:
            if self.pragma.analyze and (len(lr.attr_unused) > 0 or len(lr.rel_unused) > 0):
                print('Based on the most recent simulation run, the following group attributes A and relations R are superfluous:')
                print(f'    A: {list(lr.attr_unused)}')
                print(f'    R: {list(lr.rel_unused )}')

        return self

    def commit_group(self, group):
        """Finishes adding a new group.

        See :meth:`~pram.sim.Simulation.new_group` for explanation of the mechanism.

        Args:
            group (Group): The group in question.

        Returns:
            ``self``
        """

        self.add_group(group)
        return self

    def compact(self):
        """Compacts the simulation.

        Returns:
            ``self``
        """

        self.pop.compact()
        return self

    def db(self, db):
        """Simulation database interface.

        Args:
            db (DB): Database management system specific object.

        Returns:
            SimulationDBI
        """

        return SimulationDBI(self, db)

    def gen_diagram(self, fpath_diag, fpath_pdf):
        """Generates a simulation diagram.

        Todo:
            Reimplement and extend this method.

        Args:
            fpath_diag(str): Path to the diagram source file.
            fpath_pdf(str): Path to the diagram PDF file.

        Returns:
            ``self``
        """

        # blockdiag sim {
        # diagram = '''
        #     diagram sim {
        #         box [shape = box, label = "box"];
        #         square [shape = square, label = "sq"];
        #         roundedbox [shape = roundedbox, label = "rbox"];
        #         circle [shape = circle, label = "circ"];
        #
        #         box -> square -> roundedbox -> circle;
        #
        #         #pop [shape = actor, label = "pop", stacked, numbered = 1000];
        #         #db [shape = flowchart.database, label = "DB"];
        #
        #         #db -> pop
        #     }'''
        #
        # with open(fpath_diag, 'w') as f:
        #     f.write(diagram)

        node_w = 128
        node_h =  40
        span_w =  64
        span_h =  40
        fontsize = 8

        rules = self.rules

        with open(fpath_diag, 'w') as f:
            f.write( 'diagram sim {')
            f.write( 'orientation = portrait;')
            f.write(f'    node_width = {node_w};')
            f.write(f'    node_height = {node_h};')
            f.write(f'    default_fontsize = {fontsize};')
            # f.write(f'    timeline [shape=box, label="", width={node_w * len(rules) + span_w * (len(rules) - 1)}, height=8, color="#000000"];')

            for (i,r) in enumerate(rules):
                # Rule block:
                if hasattr(r, 'derivatives'):
                    num = f', numbered={len(r.derivatives.params)}'
                else:
                    num = ''

                f.write(f'        rule-{i} [shape=box, label="{r.__class__.__name__}" {num}];')
                f.write(f'        t-{i}    [shape=box, label="", height=8, color="#000000"];')

                # Rule-timeline arc:
                if isinstance(r.i, IterAlways):
                    i0, i1 = 0,0
                elif isinstance(r.i, IterPoint):
                    i0, i1 = r.i.i, r.i.i
                elif isinstance(r.i, IterInt):
                    i0, i1 = r.i.i0, r.i.i1
                f.write(f'        rule-{i} -> t-{i} [label="{i0}"];')

            f.write('}')

        import subprocess
        subprocess.run(['blockdiag', '-Tpdf', fpath_diag])
        subprocess.run(['open', fpath_pdf])

        return self

    def gen_groups_from_db(self, db, schema, tbl, attr_db=[], rel_db=[], attr_fix={}, rel_fix={}, attr_rm=[], rel_rm=[], rel_at=None, limit=0):
        """Generate groups from a database.

        Usage example::

            from pram.util import SQLiteDB
            ...

            s = Simulation()
            s.gen_groups_from_db(
                db       = SQLiteDB('db.sqlite3'),
                schema   = None,
                tbl      = 'people',
                attr_db  = ['age_group'],
                attr_fix = {},
                rel_db   = [
                    GroupDBRelSpec(name='school', col='school_id', fk_schema=None, fk_tbl='schools', fk_col='sp_id', sites=None)
                ],
                rel_fix  = { 'home': Site('home') },
                rel_at   = 'home'
            )
            s.add_rules(...)
            s.add_probes(...)
            s.run(24)

        Args:
            db (DB): Database management system specific object.
            schema (str): Database schema.
            tbl (str): Table name.
            attr_db (Iterable[str]): Group attributes to be retrieved from the database (if extant).
            rel_db (Iterable[GroupDBRelSpec]): Group relation to be retrieved from the database (if extant).
            attr_fix (Mappint[str, Any]): Group attributes to be fixed for every group.
            rel_fix (Mapping[str, Site]): Group relations to be fixed for every group.
            attr_rm (Iterable[str]): Group attributes to NOT be retrieved from the database (overwrites all).
            rel_rm (Iterable[str]): Group relation to NOT be retrieved from the database (overwrites all).
            rel_at (Site, optional): A site to be set as every group's current location.
            limit (int): The maximum number of groups to be generated.  Ordinarily, this is not changed from its
                default value of zero.  It is however useful for testing, especially with very large databases.

        Returns:
            ``self``
        """

        if not self.analysis.rule_static.are_rules_done:
            self.analyze_rules_static()  # by now we know all rules have been added

        self._inf(f"Generating groups from a database ({db.get_name()}; table '{tbl}')")

        if self.pragma.rule_analysis_for_db_gen:
            attr_db.extend(self.analysis.rule_static.attr_used)
            # rel_db  = self.analysis.rule_static.rel_used  # TODO: Need to use entity.GroupDBRelSpec class

        fn_live_info = self._inf if self.pragma.live_info else None
        self.add_groups(Group.gen_from_db(db, schema, tbl, attr_db, rel_db, attr_fix, rel_fix, attr_rm, rel_rm, rel_at, limit, fn_live_info))
        return self

    def gen_groups_from_db_old(self, fpath_db, tbl, attr={}, rel={}, attr_db=[], rel_db=[], rel_at=None, limit=0, fpath=None, is_verbose=False):
        if not self.analysis.rule_static.are_rules_done:
            self.analyze_rules_static()  # by now we know all rules have been added

        self._inf(f"Generating groups from a database ({fpath_db}; table '{tbl}')")

        if self.pragma.rule_analysis_for_db_gen:
            # self._inf('    Using relevant attributes and relations')

            attr_db.extend(self.analysis.rule_static.attr_used)
            # rel_db  = self.analysis.rule_static.rel_used  # TODO: Need to use entity.GroupDBRelSpec class

        # fn_gen = lambda: Group.gen_from_db(fpath_db, tbl, attr, rel, attr_db, rel_db, rel_at, limit)
        # fn_gen = lambda: Group.gen_from_db(self, fpath_db, tbl, attr, rel, attr_db, rel_db, rel_at, limit)
        # groups = FS.load_or_gen(fpath, fn_gen, 'groups', is_verbose)

        groups = Group.gen_from_db(self, fpath_db, tbl, attr, rel, attr_db, rel_db, rel_at, limit)
        self.add_groups(groups)
        return self

    def get_iter(self):
        """Get current iteration.

        Used for parallel execution progress reporting.

        Returns:
            int: If the simulation is running, a non-negative value is returned.  If the simulation is in the
                initial-condition mode (i.e., before at least one call of :meth:`~pram.sim.Simulation.run`), -1 is
                returned.
        """

        return self.timer.i if self.timer.is_running else -1

    def gen_sites_from_db(self, db, schema, tbl, name_col, rel_name=Site.AT, attr=[], limit=0):
        """Generate sites from a database.

        Args:
            db (DB): Database management system specific object.
            schema (str): Schema name.
            tbl (str): Table name.
            name_col (str): Table column storing names of sites.
            rel_name (str): Name of the relation to be associated with each of the sites generated.  For example, if
                hospital sites are being generated, the ``rel_name`` could be set to ``hospital``.
            attr (Iterable[str]): Names of table columns storing attributes to be internalized by the site objects
                being generated.
            limit (int): The maximum number of sites to be generated.  Ordinarily, this is not changed from its default
                value of zero.  It is however useful for testing, especially with very large databases.

        Returns:
            ``self``
        """

        self._inf(f'Generating sites from a database ({fpath_db})')

        self.add_sites(Site.gen_from_db(db, schema, tbl, name_col, rel_name, attr, limit))
        return self

    @staticmethod
    def gen_sites_from_db_old(fpath_db, fn_gen=None, fpath=None, is_verbose=False, pragma_live_info=False, pragma_live_info_ts=False):
        if pragma_live_info:
            if pragma_live_info_ts:
                print(f'[{datetime.datetime.now()}: info] Generating sites from the database ({fpath_db})')
            else:
                print(f'[info] Generating sites from the database ({fpath_db})')

        return FS.load_or_gen(fpath, lambda: fn_gen(fpath_db), 'sites', is_verbose)

    def get_comp_hist(self):
        """Retrieves computational history.

        The computational history dict contains the following items:
        - **mem_iter** (*Iterable[int]): Memory usage per iteration [B].
        - **t_iter** (*Iterable[int]): Time per iteration [ms].
        - **t_sim** (*int*): Total simulation time [ms].

        Returns:
            Mapping[str, Any]
        """

        return self.comp_hist

    def get_pragma(self, name):
        """Return value of the designated pragma.

        Available pragmas, their data types, and their functions are:

        - **analyze** (*bool*): Should static and dynamic rule analyses be performed?
        - **autocompact** (*bool*): Should the simulation be autocompacted after every iteration?
        - **autoprune_groups** (*bool*): Should empty groups be removed after every iteration?
        - **autostop** (*bool*): Should the simulation be stoped after stopping condition has been reached?
        - **autostop_n** (*bool*): Stopping condition: Mass smaller than specified has been transfered.
        - **autostop_p** (*bool*): Stopping condition: Mass proportion smaller than specified has been transfered.
        - **autostop_t** (*bool*):
        - **live_info** (*bool*): Display live info during simulation run?
        - **live_info_ts** (*bool*): Display live info timestamps?
        - **partial_mass** (*bool*): Allow floating point group mass?  Integer is the default.
        - **probe_capture_init** (*bool*): Instruct probes to capture the initial state of the simulation?
        - **rule_analysis_for_db_gen** (*bool*):

        Args:
            name (str): The pragma.

        Returns:
            Any: Depending on the pragma type.
        """

        fn = {
            'analyze'                  : self.get_pragma_analyze,
            'autocompact'              : self.get_pragma_autocompact,
            'autoprune_groups'         : self.get_pragma_autoprune_groups,
            'autostop'                 : self.get_pragma_autostop,
            'autostop_n'               : self.get_pragma_autostop_n,
            'autostop_p'               : self.get_pragma_autostop_p,
            'autostop_t'               : self.get_pragma_autostop_t,
            'comp_summary'             : self.get_pragma_comp_summary,
            'live_info'                : self.get_pragma_live_info,
            'live_info_ts'             : self.get_pragma_live_info_ts,
            'partial_mass'             : self.get_pragma_partial_mass,
            'probe_capture_init'       : self.get_pragma_probe_capture_init,
            'rule_analysis_for_db_gen' : self.get_pragma_rule_analysis_for_db_gen
        }.get(name, None)

        if fn is None:
            raise TypeError(f"Pragma '{name}' does not exist.")

        return fn()

    def get_pragma_analyze(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.analyze

    def get_pragma_autocompact(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.autocompact

    def get_pragma_autoprune_groups(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.autoprune_groups

    def get_pragma_autostop(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.autostop

    def get_pragma_autostop_n(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.autostop_n

    def get_pragma_autostop_p(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.autostop_p

    def get_pragma_autostop_t(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.autostop_t

    def get_pragma_comp_summary(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.comp_summary

    def get_pragma_fractional_mass(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.fractional_mass

    def get_pragma_live_info(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.live_info

    def get_pragma_live_info_ts(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.live_info_ts

    def get_pragma_probe_capture_init(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.probe_capture_init

    def get_pragma_rule_analysis_for_db_gen(self):
        """See :meth:`~pram.sim.Simulation.get_pragma`."""

        return self.pragma.rule_analysis_for_db_gen

    def get_probe(self, name):
        for p in self.probes:
            if p.name == name:
                return p
        return None

    def get_probes(self):
        return self.probes

    def get_state(self):
        """Get the current state of the simulation.

        Returns:
            Mapping[str, Mapping[str, Any]]
        """

        return {
            'sim': {
                'run-cnt': self.run_cnt,
                'timer': {
                    'iter': (self.timer.i if self.timer else 0)
                },
                'run': {
                    'is-running': self.running.is_running,
                    'progress': self.running.progress
                },
                'pragma': {
                    'analyze'                  : self.get_pragma_analyze(),
                    'autocompact'              : self.get_pragma_autocompact(),
                    'autoprune_groups'         : self.get_pragma_autoprune_groups(),
                    'autostop'                 : self.get_pragma_autostop(),
                    'autostop_n'               : self.get_pragma_autostop_n(),
                    'autostop_p'               : self.get_pragma_autostop_p(),
                    'autostop_t'               : self.get_pragma_autostop_t(),
                    'live_info'                : self.get_pragma_live_info(),
                    'live_info_ts'             : self.get_pragma_live_info_ts(),
                    'probe_capture_init'       : self.get_pragma_probe_capture_init(),
                    'rule_analysis_for_db_gen' : self.get_pragma_rule_analysis_for_db_gen()
                }
            },
            'pop': {
                'agent-mass' : self.pop.get_mass(),
                'group-cnt'  : self.pop.get_group_cnt(),
                'site-cnt'   : self.pop.get_site_cnt()
            },
            'probes': {
                'ls': [{ 'name': p.name } for p in self.probes]
            },
            'rules': {
                'ls': [{ 'cls': r.__class__.__name__, 'name': r.__class__.NAME } for r in self.rules],
                'analysis': {
                    'static': {
                        'attr': {
                            'used'   : list(self.analysis.rule_static.attr_used),
                            'unused' : list(self.analysis.rule_static.attr_unused)
                        },
                        'rel': {
                            'used'   : list(self.analysis.rule_static.rel_used),
                            'unused' : list(self.analysis.rule_static.rel_unused)
                        }
                    },
                    'dynamic': {
                        'attr-used'   : list(self.analysis.rule_dynamic.attr_used),
                        'rel-used'    : list(self.analysis.rule_dynamic.rel_used),
                        'attr-unused' : list(self.analysis.rule_dynamic.attr_unused),
                        'rel-unused'  : list(self.analysis.rule_dynamic.rel_unused )
                    }
                }
            }
        }

    def get_var(self, name):
        """Get value of the designated simulation variable.

        Args:
            name (str): The variable.

        Returns:
            Any
        """

        return self.vars.get(name)

    @staticmethod
    def _load(fpath, fn):
        with fn(fpath, 'rb') as f:
            gc.disable()
            sim = pickle.load(f)
            gc.enable()
            return sim

    @staticmethod
    def load(fpath):
        """Deserialize simulation from a file.

        Args:
            fpath (str): Source file path.

        Returns:
            Simulation
        """

        return Simulation._unpickle(fpath, open)

    @staticmethod
    def load_bz2(fpath):
        """Deserialize simulation from a bzip2-compressed file.

        Args:
            fpath (str): Source file path.

        Returns:
            Simulation
        """

        return Simulation._unpickle(fpath, bz2.BZ2File)

    @staticmethod
    def load_gz(fpath):
        """Deserialize simulation from a gzip-compressed file.

        Args:
            fpath (str): Source file path.

        Returns:
            Simulation
        """

        return Simulation._unpickle(fpath, gzip.GzipFile)

    def new_group(self, name=None, m=0.0):
        """Begins adding a new group.

        After calling this method, the interaction is with the newly created group which will not have been added to
        the simulation until :meth:`~pram.sim.Simulation.commit_group` is called, at which point the interaction
        returns to the simulation object.  The :meth:`~pram.sim.Simulation.commit_group` isn't called directly;
        instead, it's the :meth:`Group.done() <pram.entity.Group.done>` which calls it.

        Below is a usage example::

            (Simulation().
                add().
                    rule(...).
                    probe(...).
                    done().
                new_group(1000).                     # a group of a 1000 agents
                    set_attr('income', 'medium').    # with medium income
                    set_rel(Site.AT, Site('home')).  # who are all currently located at a site called 'home'
                    done().                          # back to the Simulation object
                run(12)
            )

        Args:
            name (str, optional): Group name.
            m (float, int): Group mass.

        Returns:
            Group
        """

        return Group(name, m, callee=self)

    def plot(self):
        """Simulation results plotter.

        Returns:
            SimulationPlotter
        """

        return SimulationPlotter(self)

    def plot_group_size(self, fpath=None, title='Distribution of Group Size', nx=250):
        """Plots the distribution of group sizes.

        Args:
            fpath (str, optional): Path to the plot file.
            title (str): Plot title.
            nx (int): Number of x-axis (i.e., the iteration axis) points.

        Returns:
            matplotlib figure object if ``fpath`` is None; ``self`` otherwise
        """

        # Data:
        data = [g.n for g in self.pop.groups.values()]
        density = gaussian_kde(data)
        x = np.linspace(min(data), max(data), nx)
        density.covariance_factor = lambda: .25
        density._compute_covariance()

        # Figure:
        fig = plt.figure(figsize=(12,2))
        if title:
            plt.title(title)
        plt.plot(x, density(x), lw=1, linestyle='-', c='#666666', mfc='none', antialiased=True)  # marker='o', markersize=5
        # plt.legend(['Susceptible', 'Exposed', 'Recovered'], loc='upper right')
        plt.ylabel('Density')
        plt.xlabel('Group size')
        plt.grid(alpha=0.25, antialiased=True)

        # Return:
        if fpath:
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            return self
        else:
            return fig

    def plot_site_size(self, fpath=None, title='Distribution of Site Size', nx=250):
        """Plots the distribution of site sizes.

        Args:
            fpath (str, optional): Path to the plot file.
            title (str): Plot title.
            nx (int): Number of x-axis (i.e., the iteration axis) points.

        Returns:
            matplotlib figure object if ``fpath`` is None; ``self`` otherwise
        """

        # Data:
        data = [g.n for g in self.pop.groups.values()]
        density = gaussian_kde(data)
        x = np.linspace(min(data), max(data), nx)
        density.covariance_factor = lambda: .25
        density._compute_covariance()

        # Figure:
        fig = plt.figure(figsize=(12,2))
        if title:
            plt.title(title)
        plt.plot(x, density(x), lw=1, linestyle='-', c='#666666', mfc='none', antialiased=True)  # marker='o', markersize=5
        plt.ylabel('Density')
        plt.xlabel('Group size')
        plt.grid(alpha=0.25, antialiased=True)

        # Return:
        if fpath:
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            return self
        else:
            return fig

    def rem_probe(self, probe):
        """Removes the designated probe.

        Args:
            probe (Probe): The probe.

        Returns:
            ``self``
        """

        self.probes.discard(probe)
        return self

    def rem_rule(self, rule):
        """Removes the designated rule.

        Args:
            rule (Rule): The rule.

        Returns:
            ``self``
        """


        self.rules.discard(rule)
        return self

    def remote_after(self):
        """Restores the object after remote execution (on a cluster).

        Returns:
            ``self``
        """

        return self

    def remote_before(self):
        """Prepare the object for remote execution (on a cluster).

        Returns:
            ``self``
        """

        if self.traj_id is not None:
            for p in self.probes:
                p.set_traj_id(self.traj_id)
        return self

    def reset_cb(self):
        """Reset all callback functions.

        The following callback functions are available:
        - **after_iter**: Call after iteration.
        - **before_iter**: Call before iteration.
        - **check_work**:
        - **save_state**:
        - **upd_progress**:

        Returns:
            ``self``
        """

        self.cb = DotMap(
            after_iter   = None,
            before_iter  = None,
            check_work   = None,
            save_state   = None,
            upd_progress = None
        )
        return self

    def reset_comp_hist(self):
        """Reset computational history.

        See :meth:`~pram.sim.Simulation.get_comp_hist`.

        Returns:
            ``self``
        """

        self.comp_hist = DotMap(  # computational history
            mem_iter = [],        # memory usage per iteration [B]
            t_iter = [],          # time per iteration [ms]
            t_sim = 0             # total simulation time [ms]
        )
        return self

    def reset_pop(self):
        """Resets population.

        All groups and sites are removed from the simulation and the simulation object is set back in the pre-setup
        state.

        Returns:
            ``self``
        """

        self.run_cnt = 0
        self.is_setup_done = False
        self.pop = GroupPopulation()
        gc.collect()
        return self

    def reset_pragmas(self):
        """Reset all pragmas to their default values.

        See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma = DotMap(
            analyze = True,                  # flag: analyze the simulation and suggest improvements?
            autocompact = False,             # flag: remove empty groups after every iteration?
            autoprune_groups = False,        # flag: remove attributes and relations not referenced by rules?
            autostop = False,                # flag: stop simulation when the n, p, or t condition is met?
            autostop_n = 0,                  #
            autostop_p = 0,                  #
            autostop_t = 10,                 #
            comp_summary = False,            # flag: show computational summary at the end of a simulation run?
            live_info = False,               #
            live_info_ts = False,            #
            fractional_mass = False,         # flag: should fractional mass be allowed?
            probe_capture_init = True,       # flag: let probes capture the pre-run state of the simulation?
            rule_analysis_for_db_gen = True  # flag: should static rule analysis results help form DB groups
        )
        return self

    def reset_probes(self):
        """Removes all probes.

        Returns:
            ``self``
        """

        self.probes = []
        return self

    def reset_rules(self):
        """Removes all group rules.

        Returns:
            ``self``
        """

        self.rules = []
        self.analysis.rule_static.reset()
        return self

    def reset_sim_rules(self):
        """Removes all simulation rules.

        Returns:
            ``self``
        """

        self.sim_rules = []
        return self

    def reset_vars(self):
        """Reset all simulation variables.

        Returns:
            ``self``
        """

        self.vars = {}
        return self

    def run(self, iter_or_dur=1, do_disp_t=False, do_disp_iter=False):
        """Run the simulation.

        Args:
            iter_or_dur (int or str): Number of iterations or a string representation of duration (see
                :meth:`util.Time.dur2ms() <pram.util.Time.dur2ms>`)
            do_disp_t (bool): Display simulation time at every iteration?  Useful for debugging.
            do_disp_iter (bool): Display simulation iteration at every iteration?  Useful for debugging.

        Returns:
            ``self``
        """

        ts_sim_0 = Time.ts()

        # No rules or groups:
        if len(self.rules) == 0:
            print('No rules are present\nExiting')
            return self

        if len(self.pop.groups) == 0:
            print('No groups are present\nExiting')
            return self

        self.pop.freeze()  # masses of groups cannot be changed directly but only via the group-splitting mechanism

        # Decode iterations/duration:
        self._inf('Setting simulation duration')

        self.running.is_running = True
        self.running.progress = 0.0
        if isinstance(iter_or_dur, int):
            self.timer.add_iter(iter_or_dur)
            self.running.step = 1.0 / float(iter_or_dur)
        elif isinstance(iter_or_dur, str):
            self.timer.add_dur(Time.dur2ms(iter_or_dur))
            # self.running.step = ...  # TODO
        else:
            raise ValueError(f'Number of iterations or duration must be an integer or a string: {iter_or_dur}')

        # Rule conditioning 01 -- Init:
        setattr(Group, 'attr_used', set())
        setattr(Group, 'rel_used',  set())

        # Sync simulation and rule timers:
        self._inf('Syncing rule timers')

        for r in self.rules:
            r.set_t_unit(self.timer.ms)

        # Rule setup and simulation compacting:
        if not self.is_setup_done:
            if self.fn.group_setup:
                self._inf('Running group setup')
                self.pop.apply_rules(self.fn.group_setup, 0, self.timer, is_sim_setup=True)

            self._inf('Running rule setup')
            self.pop.apply_rules(self.rules, 0, self.timer, is_rule_setup=True)
            self.is_setup_done = True

        if self.pragma.autocompact:
            self._inf('Compacting the model')
            self.compact()

        # Save last-iter info:
        # self.comp_hist.mem_iter.append(psutil.Process(self.pid).memory_full_info().uss)  # TODO: ray doesn't work with memory_full_info() (access denied)
        self.comp_hist.mem_iter.append(0)
        self.comp_hist.t_iter.append(Time.ts() - ts_sim_0)

        # Force probes to capture the initial state:
        if self.pragma.probe_capture_init and self.run_cnt == 0:
            self._inf('Capturing the initial state')

            for p in self.probes:
                p.run(None, None, self.traj_id)

        # Run the simulation:
        self._inf('Initial population')
        self._inf(f'    Mass   : {"{:,}".format(int(self.pop.get_mass()))}')
        self._inf(f'    Groups : {"{:,}".format(self.pop.get_group_cnt())}')
        self._inf(f'    Sites  : {"{:,}".format(self.pop.get_site_cnt())}')
        self._inf('Running the PRAM')

        self.run_cnt += 1
        self.autostop_i = 0  # number of consecutive iterations the 'autostop' condition has been met for

        self.timer.start()
        for i in range(self.timer.get_i_left()):
            if do_disp_iter:
                print(i)

            ts_iter_0 = Time.ts()

            if self.cb.before_iter is not None:
                self.cb.before_iter(self)

            if self.pragma.live_info:
                self._inf(f'Iteration {self.timer.i + 1} of {self.timer.i_max}')
                self._inf(f'    Group count: {self.pop.get_group_cnt()}')
            elif do_disp_t:
                print(f't:{self.timer.get_t()}')

            # Apply group rules:
            self.pop.apply_rules(self.rules, self.timer.get_i(), self.timer.get_t())
            m_flow = self.pop.last_iter.mass_flow_tot
            m_pop = float(self.pop.get_mass())
            if m_pop > 0:
                p_flow = float(m_flow) / m_pop
            else:
                p_flow = None

            # Apply simulation rules:
            for r in self.sim_rules:
                if r.is_applicable(self.timer.get_i(), self.timer.get_t()):
                    r.apply(self, self.timer.get_i(), self.timer.get_t())

            # Save last-iter info:
            # self.comp_hist.mem_iter.append(psutil.Process(self.pid).memory_full_info().uss)  # TODO: ray doesn't work with memory_full_info() (access denied)
            self.comp_hist.mem_iter.append(0)
            self.comp_hist.t_iter.append(Time.ts() - ts_iter_0)

            # Run probes:
            for p in self.probes:
                p.run(self.timer.get_i(), self.timer.get_t(), self.traj_id)

            # Cleanup the population:
            self.pop.do_post_iter()

            # Advance timer:
            self.timer.step()
            self.running.progress += self.running.step

            # Autostop:
            if self.pragma.autostop and p_flow is not None:
                if m_flow < self.pragma.autostop_n or p_flow < self.pragma.autostop_p:
                    self.autostop_i += 1
                else:
                    self.autostop_i = 0

                if self.autostop_i >= self.pragma.autostop_t:
                    if self.pragma.live_info:
                        self._inf('Autostop condition has been met; population mass transfered during the most recent iteration')
                        self._inf(f'    {m_flow} of {self.pop.get_mass()} = {p_flow * 100}%')
                        self.timer.stop()
                        break
                    else:
                        print('')
                        print('Autostop condition has been met; population mass transfered during the most recent iteration:')
                        print(f'    {m_flow} of {self.pop.get_mass()} = {p_flow * 100}%')
                        self.timer.stop()
                        break

            # Autocompact:
            if self.pragma.autocompact:
                self._inf(f'    Compacting the model')
                self.compact()

            # Callbacks:
            if self.cb.after_iter:
                self.cb.after_iter(self)

            if self.cb.upd_progress:
                self.cb.upd_progress(i, iter_or_dur)

            if self.cb.check_work:
                while not self.cb.check_work():
                    time.sleep(0.1)

        self.timer.stop()

        self._inf(f'Final population info')
        self._inf(f'    Groups: {"{:,}".format(self.pop.get_group_cnt())}')

        # Rule conditioning 02 -- Analyze and cleanup:
        self.analysis.rule_static.analyze_groups(self.pop.groups.values())
        self.analyze_rules_dynamic()

        setattr(Group, 'attr_used', None)
        setattr(Group, 'rel_used',  None)
            # we want this None instead of just calling clear() to prevent dynamic rule analysis picking up on
            # calls to has_attr(), has_rel(), and others that happen via outside rules
        # getattr(Group, 'attr_used').clear()
        # getattr(Group, 'rel_used' ).clear()

        # Rule cleanup and simulation compacting:
        self._inf('Running rule cleanup')

        self.pop.apply_rules(self.rules, 0, self.timer, is_rule_cleanup=True)
        if self.pragma.autocompact:
            self._inf('Compacting the model')
            self.compact()

        self._inf('Finishing simulation')
        self.running.is_running = False
        self.running.progress = 1.0
        self.running.step = 1.0

        # Flush any buffered probe data:
        for p in self.probes:
            if p.persistence is not None:
                p.persistence.flush()

        self.comp_hist.t_sim = Time.ts() - ts_sim_0

        self.run__comp_summary()

        return self

    def run__comp_summary(self):
        """Called by :meth:`~pram.sim.Simulation.run` to display computational summary."""

        if not self.pragma.comp_summary:
            return

        if len(self.comp_hist.mem_iter) == 0:
            print('No computational summary to display')
            return

        mem = {
            'min':    Size.bytes2human(min               (self.comp_hist.mem_iter)),
            'max':    Size.bytes2human(max               (self.comp_hist.mem_iter)),
            'mean':   Size.bytes2human(statistics.mean   (self.comp_hist.mem_iter)),
            'median': Size.bytes2human(statistics.median (self.comp_hist.mem_iter)),
            'stdev':  Size.bytes2human(statistics.stdev  (self.comp_hist.mem_iter)),
            'sum':    Size.bytes2human(sum               (self.comp_hist.mem_iter))
        }
        t = {
            'min':    Time.tsdiff2human(min               (self.comp_hist.t_iter)),
            'max':    Time.tsdiff2human(max               (self.comp_hist.t_iter)),
            'mean':   Time.tsdiff2human(statistics.mean   (self.comp_hist.t_iter)),
            'median': Time.tsdiff2human(statistics.median (self.comp_hist.t_iter)),
            'stdev':  Time.tsdiff2human(statistics.stdev  (self.comp_hist.t_iter))
        }

        print('Computational summary')
        print(f'    Iteration memory : Range: [{mem["min"]}, {mem["max"]}]    Mean (SD): {mem["mean"]} ({mem["stdev"]})    Median: {mem["median"]}    Sum: {mem["sum"]}')
        print(f'    Iteration time   : Range: [{t  ["min"]}, {t  ["max"]}]    Mean (SD): {t  ["mean"]} ({t  ["stdev"]})    Median: {t  ["median"]}')
        print(f'    Simulation time  : {Time.tsdiff2human(self.comp_hist.t_sim)}')

    def _save(self, fpath, fn):
        with fn(fpath, 'wb') as f:
            pickle.dump(self, f)

    def save(self, fpath):
        """Serialize simulation to a file.

        Args:
            fpath (str): Destination file path.

        Returns:
            ``self``
        """

        self._pickle(fpath, open)
        return self

    def save_bz2(self, fpath):
        """Serialize simulation to a bzip2-compressed file.

        Args:
            fpath (str): Destination file path.

        Returns:
            ``self``
        """

        self._pickle(fpath, bz2.BZ2File)
        return self

    def save_gz(self, fpath):
        """Serialize simulation to a gzip-compressed file.

        Args:
            fpath (str): Destination file path.

        Returns:
            ``self``
        """

        self._pickle(fpath, gzip.GzipFile)
        return self

    def save_state(self, mass_flow_specs=None):
        """Call the save simulation state callback function.

        Args:
            mass_flow_specs(MassFlowSpecs, optional): Mass flow specs.

        Returns:
            ``self``
        """

        # # if self.cb.save_state and mass_flow_specs:
        # #     self.cb.save_state(mass_flow_specs)
        #
        # if self.cb.save_state:
        #     self.cb.save_state(mass_flow_specs)
        #
        # if self.traj is None:
        #     return self
        #
        # if self.timer.i > 0:  # we check timer not to save initial state of a simulation that's been run before
        #     self.traj.save_state(None)
        # else:
        #     self.traj.save_state(mass_flow_specs)

        if self.cb.save_state:
            self.cb.save_state([{
                'type'            : 'state',
                'host_name'       : None,
                'host_ip'         : None,
                'traj_id'         : self.traj_id,  # self.traj.id if self.traj else None,
                'iter'            : self.timer.i if self.timer.is_running else -1,
                'pop_m'           : self.pop.get_mass(),
                'groups'          : [{ 'hash': g.get_hash(), 'm': g.m, 'attr': g.attr, 'rel': g.rel } for g in self.pop.groups.values()],  # self.pop.get_groups()
                # 'mass_flow_specs' : mass_flow_specs if self.timer.i > 0 else None
                'mass_flow_specs' : mass_flow_specs
            }])

        return self

    def set(self):
        """Simulation element setter.

        Returns:
            SimulationSetter
        """

        return SimulationSetter(self)

    def set_cb_after_iter(self, fn):
        """Set the callback function.

        See :meth:`~pram.sim.Simulation.reset_cb`.

        Args:
            fn (Callable): The function.

        Returns:
            ``self``
        """

        self.cb.after_iter = fn
        return self

    def set_cb_before_iter(self, fn):
        """Set the callback function.

        See :meth:`~pram.sim.Simulation.reset_cb`.

        Args:
            fn (Callable): The function.

        Returns:
            ``self``
        """

        self.cb.before_iter = fn
        return self

    def set_cb_check_work(self, fn):
        """Set the callback function.

        See :meth:`~pram.sim.Simulation.reset_cb`.

        Args:
            fn (Callable): The function.

        Returns:
            ``self``
        """

        self.cb.check_work = fn
        return self

    def set_cb_save_state(self, fn):
        """Set the callback function.

        See :meth:`~pram.sim.Simulation.reset_cb`.

        Args:
            fn (Callable): The function.

        Returns:
            ``self``
        """

        self.cb.save_state = fn
        return self

    def set_cb_upd_progress(self, fn):
        """Set the callback function.

        See :meth:`~pram.sim.Simulation.reset_cb`.

        Args:
            fn (Callable): The function.

        Returns:
            ``self``
        """

        self.cb.upd_progress = fn
        return self

    def set_fn_group_setup(self, fn):
        """Set the group setup function.

        The group setup function is the last function that is called before the simulatin is deemed ready for
        execution.  An example of how that function could be used is to make a certain proportion of population
        infected with a disease in a epidemiological modeling setting.

        Args:
            fn (Callable): The function.

        Returns:
            ``self``
        """

        self.fn.group_setup = fn
        return self

    def set_pragmas(self, analyze=None, autocompact=None, autoprune_groups=None, autostop=None, autostop_n=None, autostop_p=None, autostop_t=None, comp_summary=None, fractional_mass=None, live_info=None, live_info_ts=None, probe_capture_init=None, rule_analysis_for_db_gen=None):
        """Sets values of multiple pragmas.

        See :meth:`~pram.sim.Simulation.get_pragma`.

        Args:
            Names of all pragmas; values set only when not None (which is also the default value).

        Returns:
            ``self``
        """

        if analyze                  is not None: self.set_pragma_analyze(analyze),
        if autocompact              is not None: self.set_pragma_autocompact(autocompact),
        if autoprune_groups         is not None: self.set_pragma_autoprune_groups(autoprune_groups),
        if autostop                 is not None: self.set_pragma_autostop(autostop),
        if autostop_n               is not None: self.set_pragma_autostop_n(autostop_n),
        if autostop_p               is not None: self.set_pragma_autostop_p(autostop_p),
        if autostop_t               is not None: self.set_pragma_autostop_t(autostop_t),
        if comp_summary             is not None: self.set_pragma_comp_summary(comp_summary),
        if fractional_mass          is not None: self.set_pragma_fractional_mass(fractional_mass),
        if live_info                is not None: self.set_pragma_live_info(live_info),
        if live_info_ts             is not None: self.set_pragma_live_info_ts(live_info_ts),
        if probe_capture_init       is not None: self.set_pragma_probe_capture_init(probe_capture_init),
        if rule_analysis_for_db_gen is not None: self.set_pragma_rule_analysis_for_db_gen(rule_analysis_for_db_gen)

        return self

    def set_pragma(self, name, value):
        """Set value of the designated pragma.

        See :meth:`~pram.sim.Simulation.get_pragma`.

        Args:
            name(str): Pragma.
            value(Any): Value.

        Returns:
            ``self``
        """

        fn = {
            'analyze'                  : self.set_pragma_analyze,
            'autocompact'              : self.set_pragma_autocompact,
            'autoprune_groups'         : self.set_pragma_autoprune_groups,
            'autostop'                 : self.set_pragma_autostop,
            'autostop_n'               : self.set_pragma_autostop_n,
            'autostop_p'               : self.set_pragma_autostop_p,
            'autostop_t'               : self.set_pragma_autostop_t,
            'live_info'                : self.set_pragma_live_info,
            'live_info_ts'             : self.set_pragma_live_info_ts,
            'probe_capture_init'       : self.set_pragma_probe_capture_init,
            'rule_analysis_for_db_gen' : self.set_pragma_rule_analysis_for_db_gen
        }.get(name, None)

        if fn is None:
            raise TypeError(f"Pragma '{name}' does not exist.")

        fn(value)
        return self

    def set_pragma_analyze(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.analyze = value
        return self

    def set_pragma_autocompact(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.autocompact = value
        return self

    def set_pragma_autoprune_groups(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.autoprune_groups = value
        return self

    def set_pragma_autostop(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.autostop = value
        return self

    def set_pragma_autostop_n(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.autostop_n = value
        return self

    def set_pragma_autostop_p(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.autostop_p = value
        return self

    def set_pragma_autostop_t(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.autostop_t = value
        return self

    def set_pragma_fractional_mass(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.fractional_mass = value
        return self

    def set_pragma_comp_summary(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.comp_summary = value
        return self

    def set_pragma_live_info(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.live_info = value
        return self

    def set_pragma_live_info_ts(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.live_info_ts = value
        return self

    def set_pragma_probe_capture_init(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.probe_capture_init = value
        return self

    def set_pragma_rule_analysis_for_db_gen(self, value):
        """See :meth:`~pram.sim.Simulation.get_pragma`.

        Returns:
            ``self``
        """

        self.pragma.rule_analysis_for_db_gen = value
        return self

    def set_rand_seed(self, rand_seed=None):
        """Set pseudo-random generator seed.

        Both the ``random`` and ``numpy`` generators' seeds are set.

        Args:
            rand_seed (int, optional): The seed.

        Returns:
            ``self``
        """

        self.rand_seed = rand_seed
        if self.rand_seed is not None:
            random.seed(self.rand_seed)
            np.random.seed(self.rand_seed)
        return self

    def set_var(self, name, val):
        """Set value of a simulation variable.

        Args:
            name (str): The variable.
            val (any): The value.

        Returns:
            ``self``
        """

        self.vars[name] = val
        return self

    def set_vars(self, vars):
        """Set values of multiple simulation variables.

        Args:
            vars (Mapping[str, Any]): A dictionary of variable name-value pairs.

        Returns:
            ``self``
        """

        for k,v in vars.items():
            self.vars[k] = v
        return self

    def show_rule_analysis(self):
        """Display the results of both static and dynamic rule analyses.

        See :class:`~pram.sim.StaticRuleAnalyzer` and :class:`~pram.sim.DynamicRuleAnalyzer`.

        Returns:
            ``self``
        """

        self.show_static_rule_analysis()
        self.show_dynamic_rule_analysis()

        return self

    def show_rule_analysis_dynamic(self):
        """Display the results of dynamic rule analyses.

        See :class:`~pram.sim.DynamicRuleAnalyzer`.

        Returns:
            ``self``
        """

        rd = self.analysis.rule_dynamic

        print( 'Most recent simulation run')
        print( '    Used')
        print(f'        Attributes : {list(rd.attr_used)}')
        print(f'        Relations  : {list(rd.rel_used)}')
        print( '    Groups')
        print(f'        Attributes : {list(rd.attr_groups)}')
        print(f'        Relations  : {list(rd.rel_groups)}')
        print( '    Superfluous')
        print(f'        Attributes : {list(rd.attr_unused)}')
        print(f'        Relations  : {list(rd.rel_unused)}')

        return self

    def show_rule_analysis_static(self):
        """Display the results of static rule analyses.

        See :class:`~pram.sim.StaticRuleAnalyzer`.

        Returns:
            ``self``
        """

        ra = self.analysis.rule_static

        print( 'Rule analyzer')
        print( '    Used')
        print(f'        Attributes : {list(ra.attr_used)}')
        print(f'        Relations  : {list(ra.rel_used)}')
        print( '    Superfluous')
        print(f'        Attributes : {list(ra.attr_unused)}')
        print(f'        Relations  : {list(ra.rel_unused)}')
        # print( '    Counts')
        # print(f'        Recognized   : get_attr:{ra.cnt_rec["get_attr"]} get_rel:{ra.cnt_rec["get_rel"]} has_attr:{ra.cnt_rec["has_attr"]} has_rel:{ra.cnt_rec["has_rel"]}')
        # print(f'        Unrecognized : get_attr:{ra.cnt_unrec["get_attr"]} get_rel:{ra.cnt_unrec["get_rel"]} has_attr:{ra.cnt_unrec["has_attr"]} has_rel:{ra.cnt_unrec["has_rel"]}')

        return self

    def show_summary(self, do_header=True, n_groups=8, n_sites=8, n_rules=8, n_probes=8, end_line_cnt=(0,0)):
        """Display simulation summary.

        Args:
            do_header (bool): Display header?
            n_groups (int): Maximum number of groups to be displayed.
            n_sites (int): Maximum number of groups to be displayed.
            n_rules (int): Maximum number of groups to be displayed.
            n_probes (int): Maximum number of groups to be displayed.
            end_line_cnt (tuple[int,int]): The number of endline characters before and after the summary.

        Returns:
            ``self``
        """

        print('\n' * end_line_cnt[0], end='')

        if do_header:
            print( 'Simulation')
            print(f'    Random seed: {self.rand_seed}')
            print( '    Timing')
            print(f'        Timer: {self.timer}')
            print( '    Population')
            print(f'        Mass        : {"{:,.2f}".format(round(self.pop.get_mass(), 1))}')
            print(f'        Groups      : {"{:,}".format(self.pop.get_group_cnt())}')
            print(f'        Groups (ne) : {"{:,}".format(self.pop.get_group_cnt(True))}')
            print(f'        Sites       : {"{:,}".format(self.pop.get_site_cnt())}')
            print(f'        Rules       : {"{:,}".format(len(self.rules))}')
            print(f'        Probes      : {"{:,}".format(len(self.probes))}')

            if self.pragma.analyze:
                print( '    Static rule analysis')
                print( '        Used')
                print(f'            Attributes : {list(self.analysis.rule_static.attr_used)}')
                print(f'            Relations  : {list(self.analysis.rule_static.rel_used)}')
                print( '        Superfluous')
                print(f'            Attributes : {list(self.analysis.rule_static.attr_unused)}')
                print(f'            Relations  : {list(self.analysis.rule_static.rel_unused)}')
                print( '    Dynamic rule analysis')
                print( '        Used')
                print(f'            Attributes : {list(self.analysis.rule_dynamic.attr_used)}')
                print(f'            Relations  : {list(self.analysis.rule_dynamic.rel_used)}')
                # print('        Dynamic - Groups')
                # print(f'            Attributes : {list(self.analysis.rule_dynamic.attr_groups)}')
                # print(f'            Relations  : {list(self.analysis.rule_dynamic.rel_groups)}')
                print( '        Superfluous')
                print(f'            Attributes : {list(self.analysis.rule_dynamic.attr_unused)}')
                print(f'            Relations  : {list(self.analysis.rule_dynamic.rel_unused)}')

        if n_groups > 0:
            if len(self.pop.groups) > 0: print(f'    Groups ({"{:,}".format(len(self.pop.groups))})\n' + '\n'.join(['        {}'.format(g) for g in list(self.pop.groups.values())[:n_groups]]))
            if len(self.pop.groups) > n_groups: print('        ...')

        if n_sites > 0:
            if len(self.pop.sites)  > 0: print(f'    Sites ({"{:,}".format(len(self.pop.sites))})\n'   + '\n'.join(['        {}'.format(s) for s in list(self.pop.sites.values())[:n_sites]]))
            if len(self.pop.sites) > n_sites: print('        ...')

        if n_rules > 0:
            if len(self.rules)      > 0: print(f'    Rules ({"{:,}".format(len(self.rules))})\n'       + '\n'.join(['        {}'.format(r) for r in self.rules[:n_rules]]))
            if len(self.rules) > n_rules: print('        ...')

        if n_probes > 0:
            if len(self.probes)     > 0: print(f'    Probes ({"{:,}".format(len(self.probes))})\n'     + '\n'.join(['        {}'.format(p) for p in self.probes[:n_probes]]))
            if len(self.probes) > n_probes: print('        ...')

        print('\n' * end_line_cnt[1], end='')

        return self


# ----------------------------------------------------------------------------------------------------------------------
# @ray.remote
# class SimulationRemote(Simulation):
#     pass
