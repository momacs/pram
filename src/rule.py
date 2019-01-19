from abc import abstractmethod
from entity import AttrFluStatus, GroupSplitSpec, Site


class Rule(object):
    '''
    A rule that can be applied to a group and may augment that group or split it into multiple subgroups.

    A rule will be applied if the simulation timer's time (external to this class) falls within the range defined by
    the (t0, t1) interval.  Every time a rule is applied, it is applied to all groups it is compatible with.  For
    instance, a rule that renders a portion of a group infection-free (i.e., marks it as recovered) can be applied to a
    group of humans currently infected with some infectious disease.  The same rule, however, would not be applied to
    a group of city buses.  Each rule knows how to recognize a compatible group.
    '''

    DEBUG_LVL = 0  # 0=none, 1=normal, 2=full

    pop = None

    def __init__(self, name, t0, t1):
        self.name = name
        self.t0 = t0
        self.t1 = t1

    def __repr__(self):
        return '{}(name={}, t0={}, t1={})'.format(self.__class__.__name__, self.name, self.t0, self.t1)

    def __str__(self):
        return 'Rule  name: {:16}  t=({:2}, {:2})'.format(self.name, self.t0, self.t1)

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1: print(msg)

    @abstractmethod
    def apply(self):
        pass

    def is_applicable(self, t):
        ''' Verifies if the rule is applicable given the context. '''

        return self.t0 <= t <= self.t1


# ======================================================================================================================
class RuleGoHome(Rule):
    def __init__(self, t0=16, t1=22):  # 4pm - 10pm
        super().__init__('go-home', t0, t1)

    def apply(self, group, t):
        if not self.is_applicable(group, t): return

        self._debug('rule.apply: {}'.format(self.name))

        p = 0.4 if t < self.t1 else 1.0
        return (
            GroupSplitSpec(p=p,     rel_upd={ Site.DEF_REL_NAME: Site('home').get_hash() }),
            GroupSplitSpec(p=1 - p, rel_upd={ Site.DEF_REL_NAME: Site('work').get_hash() })
        )

    def is_applicable(self, group, t):
        # print('HOME {} : {}'.format(group.get_rel(Site.DEF_REL_NAME), Site('work').get_hash()))
        return super().is_applicable(t) and group.get_rel(Site.DEF_REL_NAME) == Site('work').get_hash()


# ======================================================================================================================
class RuleGoToWork(Rule):
    def __init__(self, t0=8, t1=12):  # 8am - 12 noon
        super().__init__('go-to-work', t0, t1)

    def apply(self, group, t):
        if not self.is_applicable(group, t): return

        self._debug('rule.apply: {}'.format(self.name))

        p = 0.4
        return (
            GroupSplitSpec(p=p,     rel_upd={ Site.DEF_REL_NAME: Site('work').get_hash() }),
            GroupSplitSpec(p=1 - p, rel_upd={ Site.DEF_REL_NAME: Site('home').get_hash() })
        )

    def is_applicable(self, group, t):
        # print('WORK {} : {}'.format(group.get_rel(Site.DEF_REL_NAME), Site('home').get_hash()))
        return super().is_applicable(t) and group.get_rel(Site.DEF_REL_NAME) == Site('home').get_hash()


# ======================================================================================================================
class RuleProgressFlu(Rule):
    def __init__(self, t0=8, t1=20):  # 8am - 8pm
        super().__init__('progress-flu', t0, t1)

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
    print(RuleProgressFlu())
    print(RuleGoHome())
    print(RuleGoToWork())
