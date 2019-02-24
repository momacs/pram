from abc import abstractmethod, ABC
from attr import attrs, attrib

from .entity import GroupQry, GroupSplitSpec, Site
from .util import Err


@attrs(slots=True)
class Time(object):
    pass


@attrs(slots=True)
class TimePoint(Time):
    # TODO: Restrict valid value range to [0-24).

    t: float = attrib(default=0.00, converter=float)


@attrs(slots=True)
class TimeInt(Time):
    '''
    A time interval a rule is active in.
    '''

    # TODO: Restrict valid value range to [0-24).

    t0: float = attrib(default= 0.00, converter=float)
    t1: float = attrib(default=24.00, converter=float)


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

    @abstractmethod
    def apply(self, pop, group, iter, t):
        pass

    def is_applicable(self, iter, t):
        ''' Verifies if the rule is applicable given the context. '''

        if isinstance(self.t, TimePoint):
            return self.t.t == t

        if isinstance(self.t, TimeInt):
            return self.t.t0 <= t <= self.t.t1

        raise TypeError("Type '{}' used for specifying rule timing not yet implemented (Rule.is_applicable).".format(type(self.t).__name__))

    @staticmethod
    def setup(pop, group):
        '''
        A rule's setup place.  If the rule relies on groups having a certain set of attributes and relations, this is
        where they should be set.  For example, a rule might set an attribute of all the groups like so:

            return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]

        This reuses the group splitting mechanism; here, each group will be split into a (possibly non-extant) new
        group and the entirety of the group's mass will be moved into that new group.

        This is also where a rule should do any other population initialization required.  For example, a rule that may
        introduce a new Site object to the simulation, should make the population object aware of that site like so:

            pop.add_site(Site('new-site'))

        Every rule's setup() method is called only once by Simulation.run() method before a simulation run commences.
        '''

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

    def __init__(self, t, p, rel_from, rel_to, memo=None):
        super().__init__('goto', t, memo)

        Err.type(rel_from, 'rel_from', str, True)
        Err.type(rel_to, 'rel_to', str)

        self.p = p
        self.rel_from = rel_from  # if None, the rule will not be conditional on current location
        self.rel_to = rel_to

    def __repr__(self):
        return '{}(name={}, t={}, p={}, rel_from={}, rel_to={})'.format(self.__class__.__name__, self.name, self.t.t, self.p, self.rel_from, self.rel_to)

    def __str__(self):
        return 'Rule  name: {:16}  t: {}  p: {}  rel: {} --> {}'.format(self.name, self.t, self.p, self.rel_from, self.rel_to)

    def apply(self, pop, group, iter, t):
        return [
            GroupSplitSpec(p=self.p, rel_set={ Site.AT: group.get_rel(self.rel_to) }),
            GroupSplitSpec(p=1 - self.p)
        ]

    def is_applicable(self, group, iter, t):
        if not super().is_applicable(iter, t): return False

        # Moving from the designated location:
        if self.rel_from is not None:
            return (
                group.has_rel(self.rel_to) and Err.type(group.get_rel(self.rel_to), 'self.rel_to', Site) and
                group.has_rel(self.rel_from) and Err.type(group.get_rel(self.rel_from), 'self.rel_from', Site) and
                group.get_rel(Site.AT) == group.get_rel(self.rel_from))
        # Moving from any location:
        else:
            return group.has_rel(self.rel_to) and Err.type(group.get_rel(self.rel_to), 'self.rel_to', Site)
