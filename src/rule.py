from abc import abstractmethod, ABC
from attr import attrs, attrib
from entity import AttrFluStatus, GroupQry, GroupSplitSpec, Site
from util import Err


@attrs(slots=True)
class Time(object):
    pass


@attrs(slots=True)
class TimePoint(Time):
    # TODO: Restrict valid value range to [0-24).

    t: float = attrib(default=-1.0, converter=float)


@attrs(slots=True)
class TimeInt(Time):
    '''
    A time interval a rule is active in.
    '''

    # TODO: Restrict valid value range to [0-24).

    t0: float = attrib(default=-1.0, converter=float)
    t1: float = attrib(default=-1.0, converter=float)


# ----------------------------------------------------------------------------------------------------------------------
class Rule(ABC):
    '''
    A rule that can be applied to a group and may augment that group or split it into multiple subgroups.

    A rule will be applied if the simulation timer's time (external to this class) falls within the range defined by
    the time specification 't'.  Every time a rule is applied, it is applied to all groups it is compatible with.  For
    instance, a rule that renders a portion of a group infection-free (i.e., marks it as recovered) can be applied to a
    group of humans currently infected with some infectious disease.  The same rule, however, would not be applied to
    a group of city buses.  Each rule knows how to recognize a compatible group.
    '''

    __slots__ = ('name', 't', 'memo')

    DEBUG_LVL = 0  # 0=none, 1=normal, 2=full

    pop = None

    def __init__(self, name, t, memo=None):
        '''
        t: Time
        '''

        Err.type(t, 't', Time)

        self.name = name
        self.t = t
        self.memo = memo

    def __repr__(self):
        if isinstance(self.t, TimePoint):
            return '{}(name={}, t={:>4})'.format(self.__class__.__name__, self.name, round(self.t.t, 1))

        if isinstance(self.t, TimeInt):
            return '{}(name={}, t=({:>4},{:>4}))'.format(self.__class__.__name__, self.name, round(self.t.t0, 1), round(self.t.t1, 1))

    def __str__(self):
        if isinstance(self.t, TimePoint):
            return 'Rule  name: {:16}  t: {:>4}'.format(self.name, round(self.t.t, 1))

        if isinstance(self.t, TimeInt):
            return 'Rule  name: {:16}  t: ({:>4},{:>4})'.format(self.name, round(self.t.t0, 1), round(self.t.t1, 1))

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1: print(msg)

    @abstractmethod
    def apply(self, pop, group, t):
        pass

    def is_applicable(self, t):
        ''' Verifies if the rule is applicable given the context. '''

        if isinstance(self.t, TimePoint):
            return self.t.t == t

        if isinstance(self.t, TimeInt):
            return self.t.t0 <= t <= self.t.t1

        raise TypeError("Type '{}' used for specifying rule timing not yet implemented (Rule.is_applicable).".format(type(self.t).__name__))

    @staticmethod
    def setup(pop, group):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class GotoRule(Rule):
    '''
    Changes the location of a group from the designated site to the designated site.  Both of the sites are
    specificed by relation name (e.g., 'store').  The rule will only apply to a group that (a) is currently located at
    the "from" relation and has the "to" relation.  If the "from" argument is None, all groups will qualify as long as
    they have the "to" relation.  The values of the relations need to be of type Site.

    Only one "from" and "to" location is handled by this rule.  Lists of locations are not handled.

    The group's current location is defined by the 'Site.AT' relation name and that's the relation that this
    rule updated.

    Example uses:
        - Compel agents that are at 'home' go to 'work' or vice versa
    '''

    __slots__ = ('p', 'rel_from', 'rel_to')

    def __init__(self, t, p, rel_from, rel_to, memo=None):
        super().__init__('goto', t, memo)

        Err.type(rel_from, 'rel_from', str, True)
        Err.type(rel_to, 'rel_to', str)

        self.p = p
        self.rel_from = rel_from  # if None, the rule will not be conditional on current location
        self.rel_to = rel_to

    def __repr__(self):
        return '{}(name={}, t=({:>4},{:>4}), p={}, rel_from={}, rel_to={})'.format(self.__class__.__name__, self.name, round(self.t.t0, 1), round(self.t.t1, 1), self.p, self.rel_from, self.rel_to)

    def __str__(self):
        return 'Rule  name: {:16}  t: ({:>4},{:>4})  p: {}  rel: {} --> {}'.format(self.name, round(self.t.t0, 1), round(self.t.t1, 1), self.p, self.rel_from, self.rel_to)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        return [
            GroupSplitSpec(p=self.p, rel_set={ Site.AT: group.get_rel(self.rel_to) }),
            GroupSplitSpec(p=1 - self.p)
        ]

    def is_applicable(self, group, t):
        if not super().is_applicable(t): return False

        # Moving from the designated location only:
        if self.rel_from is not None:
            return (
                group.has_rel(self.rel_to) and Err.type(group.get_rel(self.rel_to), 'self.rel_to', Site) and
                group.has_rel(self.rel_from) and Err.type(group.get_rel(self.rel_from), 'self.rel_from', Site) and
                group.get_rel(Site.AT) == group.get_rel(self.rel_from))
        # Moving from any location:
        else:
            return group.has_rel(self.rel_to) and Err.type(group.get_rel(self.rel_to), 'self.rel_to', Site)


# ----------------------------------------------------------------------------------------------------------------------
class ResetDayRule(Rule):
    __slots__ = ()

    def __init__(self, t, memo=None):
        super().__init__('reset-day', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],

    def is_applicable(self, group, t):
        return super().is_applicable(t)


# ----------------------------------------------------------------------------------------------------------------------
class AttendSchoolRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,16), memo=None):
        super().__init__('attend-school', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        if group.has_rel({ Site.AT: group.get_rel('home') }) and (not group.has_attr('did-attend-school-today') or group.has_attr({ 'did-attend-school-today': False })):
            return self.apply_at_home(group, t)

        if group.has_rel({ Site.AT:  group.get_rel('school') }):
            return self.apply_at_school(group, t)

    def apply_at_home(self, group, t):
        if t < 8 or t > 12:
            return

        p = { 8:0.50, 9:0.50, 10:0.50, 11:0.50, 12:1.00 }.get(t, 0.00)  # TODO: Provide these as a CDF
            # prob of goint to school = f(time of day)

        return [
            GroupSplitSpec(p=p, attr_set={ 'did-attend-school-today': True, 't-at-school': 0 }, rel_set={ Site.AT: group.get_rel('school') }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_at_school(self, group, t):
        t_at_school = group.get_attr('t-at-school')
        p = { 0: 0.00, 1:0.05, 2:0.05, 3:0.25, 4:0.50, 5:0.70, 6:0.80, 7:0.90, 8:1.00 }.get(t_at_school, 1.00) if t < self.t.t1 else 1.00
            # prob of going home = f(time spent at school)

        return [
            GroupSplitSpec(p=p, attr_set={ 't-at-school': (t_at_school + 1) }, rel_set={ Site.AT: group.get_rel('home') }),
            GroupSplitSpec(p=1 - p, attr_set={ 't-at-school': (t_at_school + 1) })
        ]

        # TODO: Give timer information to the rule so it can appropriate determine time passage (e.g., to add it to 't-at-school' above).

    def is_applicable(self, group, t):
        return (
            super().is_applicable(t) and
            group.has_attr({ 'is-student': True }) and
            group.has_rel(['home', 'school']))

    @staticmethod
    def setup(pop, group):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],


# ----------------------------------------------------------------------------------------------------------------------
class AttendSchool02Rule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,16), memo=None):
        super().__init__('attend-school-02', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        if group.has_rel({ Site.AT: group.get_rel('home') }) and (not group.has_attr('did-attend-school-today') or group.has_attr({ 'did-attend-school-today': False })):
            return self.apply_at_home(group, t)

        if group.has_rel({ Site.AT:  group.get_rel('school') }):
            return self.apply_at_school(group, t)

    def apply_at_home(self, group, t):
        if t < 8 or t > 12:
            return

        p = { 8:0.50, 9:0.50, 10:0.50, 11:0.50, 12:1.00 }.get(t, 0.00)  # TODO: Provide these as a CDF
            # prob of goint to school = f(time of day)

        return [
            GroupSplitSpec(p=p, attr_set={ 'did-attend-school-today': True, 't-at-school': 0 }, rel_set={ Site.AT: group.get_rel('school') }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_at_school(self, group, t):
        t_at_school = group.get_attr('t-at-school')
        p = { 0: 0.00, 1:0.05, 2:0.05, 3:0.25, 4:0.50, 5:0.70, 6:0.80, 7:0.90, 8:1.00 }.get(t_at_school, 1.00) if t < self.t.t1 else 1.00
            # prob of going home = f(time spent at school)

        return [
            GroupSplitSpec(p=p, attr_set={ 't-at-school': (t_at_school + 1) }, rel_set={ Site.AT: group.get_rel('home') }),
            GroupSplitSpec(p=1 - p, attr_set={ 't-at-school': (t_at_school + 1) })
        ]

    def is_applicable(self, group, t):
        return (
            super().is_applicable(t) and
            group.has_attr({ 'is-student': True }) and
            group.has_rel(['home', 'school']))

    @staticmethod
    def setup(pop, group):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,20), memo=None):  # 8am - 8pm
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        p_infection = 0.05

        if group.get_attr('flu-status') == AttrFluStatus.no:
            return [
                GroupSplitSpec(p=p_infection,     attr_set={ 'flu-status': AttrFluStatus.asympt }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu-status': AttrFluStatus.no     })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.asympt:
            return [
                GroupSplitSpec(p=0.80, attr_set={ 'flu-status': AttrFluStatus.sympt }),
                GroupSplitSpec(p=0.20, attr_set={ 'flu-status': AttrFluStatus.no    })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.sympt:
            return [
                GroupSplitSpec(p=0.20, attr_set={ 'flu-status': AttrFluStatus.asympt }),
                GroupSplitSpec(p=0.75, attr_set={ 'flu-status': AttrFluStatus.sympt  }),
                GroupSplitSpec(p=0.05, attr_set={ 'flu-status': AttrFluStatus.no     })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-status'.")

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_attr([ 'flu-status' ])


# ----------------------------------------------------------------------------------------------------------------------
class GotoHomeFluRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(1,500), memo=None):
        super().__init__('goto-home-flu', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        p = 1.00
        return [
            GroupSplitSpec(p=p, rel_set={ Site.AT: group.get_rel('home') }),
            GroupSplitSpec(p=1 - p)
        ]

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_rel({ Site.AT: group.get_rel('school') })


# ----------------------------------------------------------------------------------------------------------------------
class GotoSchoolFluRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(1,500), memo=None):
        super().__init__('goto-school-flu', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        if group.has_attr({ 'flu-status': AttrFluStatus.no }):
            p = 1.00
        else:
            if group.has_rel({ 'school': Site('school-a').get_hash()}):
                p = 0.50
            else:
                p = 0.10

        return [
            GroupSplitSpec(p=p, rel_set={ Site.AT: group.get_rel('school') }),
            GroupSplitSpec(p=1 - p)
        ]

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_rel({ Site.AT: group.get_rel('home') })


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFlu02Rule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(1,500), memo=None):
        super().__init__('progress-flu-day', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        site = pop.sites[group.get_rel(Site.AT)]
        n = site.get_pop_size()
        n_infectious = (
            sum([g.n for g in site.get_groups_here(GroupQry(rel={ 'flu-status': AttrFluStatus.asympt }))]) +
            sum([g.n for g in site.get_groups_here(GroupQry(rel={ 'flu-status': AttrFluStatus.sympt }))]))
        # p_infection = n_infectious / n

        # print('* {}: {} / {}'.format(site.name, n_infectious, n))

        p_infection = 0.05

        if group.get_attr('flu-status') == AttrFluStatus.no:
            return [
                GroupSplitSpec(p=p_infection,     attr_set={ 'flu-status': AttrFluStatus.sympt }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu-status': AttrFluStatus.no })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.sympt:
            return [
                GroupSplitSpec(p=0.75, attr_set={ 'flu-status': AttrFluStatus.sympt }),
                GroupSplitSpec(p=0.05, attr_set={ 'flu-status': AttrFluStatus.no })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-status'.")

    def is_applicable(self, group, t):
        return (
            super().is_applicable(t) and
            group.has_attr([ 'flu-status' ]) and
            not group.has_rel({ Site.AT: group.get_rel('home') }))  # not at home


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print(GotoRule(TimeInt(8,10), 0.5, 'home', 'work'))
    print(ProgressFluRule())
