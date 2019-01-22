from abc import abstractmethod
from attr import attrs, attrib
from entity import AttrFluStatus, GroupSplitSpec, Site
from util import Err


@attrs(slots=True)
class TimeSpec:
    '''
    A time specification for a rule.
    '''

    t0: float = attrib(default=-1.0, converter=float)
    t1: float = attrib(default=-1.0, converter=float)


class Rule(object):
    '''
    A rule that can be applied to a group and may augment that group or split it into multiple subgroups.

    A rule will be applied if the simulation timer's time (external to this class) falls within the range defined by
    the time specification 't'.  Every time a rule is applied, it is applied to all groups it is compatible with.  For
    instance, a rule that renders a portion of a group infection-free (i.e., marks it as recovered) can be applied to a
    group of humans currently infected with some infectious disease.  The same rule, however, would not be applied to
    a group of city buses.  Each rule knows how to recognize a compatible group.
    '''

    DEBUG_LVL = 0  # 0=none, 1=normal, 2=full

    pop = None

    def __init__(self, name, t_spec):
        '''
        t_spec: TimeSpec
        '''

        if not isinstance(t_spec, TimeSpec):
            raise TypeError(Err.type('t_spec', 'TimeSpec'))

        self.name = name
        self.t_spec = t_spec

    def __repr__(self):
        return '{}(name={}, t=({:>4},{:>4}))'.format(self.__class__.__name__, self.name, round(self.t_spec.t0, 1), round(self.t_spec.t1, 1))

    def __str__(self):
        return 'Rule  name: {:32}  t=({:>4},{:>4})'.format(self.name, round(self.t_spec.t0, 1), round(self.t_spec.t1, 1))

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1: print(msg)

    @abstractmethod
    def apply(self):
        pass

    def is_applicable(self, t):
        ''' Verifies if the rule is applicable given the context. '''

        return self.t_spec.t0 <= t <= self.t_spec.t1


# ======================================================================================================================
# class RuleGoto_BySite(Rule):
#     '''
#     Changes the location of the group from the designated site to the designated site.  Both of the sites are
#     specificed by name (e.g., 'store-315').  The rule will only apply to a group that (a) is currently located at the
#     "from" site and has the .
#     However, if the "from" site is None, all groups will qualify.
#     '''
#
#     def __init__(self, t_spec, p, site_from, site_to):
#         super().__init__('goto--by-site', t_spec)
#
#         self.p = p
#         self.site_from = site_from
#         self.site_to = site_to
#
#     def apply(self, group, t):
#         if not self.is_applicable(group, t): return
#
#         self._debug('rule.apply: {} (from:{} to:{})'.format(self.name, self.site_from, self.site_to))
#
#         return (
#             GroupSplitSpec(p=self.p, rel_upd={ Site.DEF_REL_NAME: Site(self.site_to).get_hash() }),
#             GroupSplitSpec(p=1 - self.p)
#         )
#
#     def is_applicable(self, group, t):
#         # Moving from a paricular location:
#         if self.site_from is not None:
#             return (
#                 super().is_applicable(t) and
#                 group.has_rel(self.site_to) and
#                 group.has_rel(self.site_from) and
#                 group.get_rel(Site.DEF_REL_NAME) == Site(self.site_from).get_hash()
#             )
#         # Moving from any location:
#         else:
#             return (
#                 super().is_applicable(t) and
#                 group.has_rel(self.site_to)
#             )


# ======================================================================================================================
class RuleGoto(Rule):
    '''
    Changes the location of a group from the designated site to the designated site.  Both of the sites are
    specificed by relation name (e.g., 'store').  The rule will only apply to a group that (a) is currently located at
    the "from" relation and has the "to" relation.  If the "from" argument is None, all groups will qualify as long as
    they have the "to" relation.  The values of the relations need to be of type Site.

    Only one "from" and "to" location is handled by this rule.  Lists of locations are not handled.

    The group's current location is defined by the 'Site.DEF_REL_NAME' relation name and that's the relation that this
    rule updated.

    Example uses:
        - Compel agents that are at 'home' go to 'work' or vice versa
    '''

    def __init__(self, t_spec, p, rel_from, rel_to):
        super().__init__('goto--by-rel', t_spec)

        # if self.rel_from is not None and not isinstance(self.rel_from, Site):
        #     raise TypeError(Err.type('rel_from', 'Site', True))
        #
        # if not isinstance(self.rel_to, Site):
        #     raise TypeError(Err.type('rel_from', 'Site'))

        Err.type(rel_from, 'rel_from', str, True)
        Err.type(rel_to, 'rel_to', str)

        self.p = p
        self.rel_from = rel_from  # if None, the rule will not be conditional on current location
        self.rel_to = rel_to

    def apply(self, group, t):
        if not self.is_applicable(group, t): return

        self._debug('rule.apply: {} (from:{} to:{})'.format(self.name, self.rel_from, self.rel_to))

        return (
            GroupSplitSpec(p=self.p, rel_upd={ Site.DEF_REL_NAME: group.get_rel(self.rel_to) }),
            GroupSplitSpec(p=1 - self.p)
        )

    def is_applicable(self, group, t):
        if not super().is_applicable(t): return False

        # Moving from the designated location only:
        if self.rel_from is not None:
            return (
                group.has_rel(self.rel_to) and Err.type(group.get_rel(self.rel_to), 'self.rel_to', Site) and
                group.has_rel(self.rel_from) and Err.type(group.get_rel(self.rel_from), 'self.rel_from', Site) and
                group.get_rel(Site.DEF_REL_NAME) == group.get_rel(self.rel_from)
            )
        # Moving from any location:
        else:
            return group.has_rel(self.rel_to) and Err.type(group.get_rel(self.rel_to), 'self.rel_to', Site)


# ======================================================================================================================
class RuleProgressFlu(Rule):
    def __init__(self, t_spec=TimeSpec(8,20)):  # 8am - 8pm
        super().__init__('progress-flu', t_spec)

    def apply(self, group, t):
        if not self.is_applicable(group, t): return None

        self._debug('rule.apply: {} to {}'.format(self.name, group.name))

        p_infection = 0.05

        if group.get_attr('flu-status') == AttrFluStatus.no:
            return (
                GroupSplitSpec(p=p_infection,     attr_upd={ 'flu-status': AttrFluStatus.asympt }),
                GroupSplitSpec(p=1 - p_infection, attr_upd={ 'flu-status': AttrFluStatus.no     })
            )
        elif group.get_attr('flu-status') == AttrFluStatus.asympt:
            return (
                GroupSplitSpec(p=0.80, attr_upd={ 'flu-status': AttrFluStatus.sympt }),
                GroupSplitSpec(p=0.20, attr_upd={ 'flu-status': AttrFluStatus.no    })
            )
        elif group.get_attr('flu-status') == AttrFluStatus.sympt:
            return (
                GroupSplitSpec(p=0.20, attr_upd={ 'flu-status': AttrFluStatus.asympt }),
                GroupSplitSpec(p=0.75, attr_upd={ 'flu-status': AttrFluStatus.sympt  }),
                GroupSplitSpec(p=0.05, attr_upd={ 'flu-status': AttrFluStatus.no     })
            )
        else:
            raise ValueError("Invalid value for attribute 'flu-status'.")

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_attr([ 'flu-status' ])


# ======================================================================================================================
if __name__ == '__main__':
    print(RuleGoto(TimeSpec(8,10), 0.5, 'home', 'work'))
    print(RuleProgressFlu())
