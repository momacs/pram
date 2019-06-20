'''
Rules used throughout the 'sim' test simulations package.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.rule   import GoToRule, Rule, TimeInt, TimePoint
from pram.util   import Err


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    '''
    Time-homogenous Markov chain with a finite state space with the initial state distribution:

        flu: (1 0 0)

    and the transition model:

              s     i     r
        s  0.95  0.00  0.10
        i  0.05  0.50  0
        r  0     0.50  0.90
    '''

    def __init__(self, t=TimeInt(8,20), memo=None):
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, iter, t):
        p = 0.05  # prob of infection

        if group.get_attr('flu') == 's':
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=p,     attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.00,  attr_set={ 'flu': 'r' })
            ]
        elif group.get_attr('flu') == 'i':
            return [
                GroupSplitSpec(p=0.00, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'r' })
            ]
        elif group.get_attr('flu') == 'r':
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu': 'r' })
            ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu' ])


# ----------------------------------------------------------------------------------------------------------------------
class ProgressAndTransmitFluRule(Rule):
    '''
    This rule makes the following assumptions:

        1. There are three sequential stages of the flu:
           - Susceptible
           - Infectious
           - Recovered
        2. An agent cannot transition back to the previous flu stage; it can only remain in its current flu stage or
           advance to the next.  Symptomatic agents are allowed to make a transition to the "no-flu" stage effectively
           modeling recovery.
        3. Agents who are either asymptomatic or symptomatic are considered infectious.  Some communicable diseases may
           become less infectious once the host becomes symptomatic, but this rule does not model that (although it
           could).
        4. Recovered agents are not marked as immune and are therefore as likely to concieve the flu as other agents.
    '''

    def __init__(self, t=TimeInt(8,20), p_infection_min=0.01, p_infection_max=0.95, memo=None):  # 8am - 8pm
        super().__init__('progress-flu', t, memo)

        self.p_infection_min = p_infection_min
        self.p_infection_max = p_infection_max

    def apply(self, pop, group, iter, t):
        p = self.p_infection_min

        if group.has_rel({ Site.AT: group.get_rel('school') }):
            site = group.get_rel(Site.AT)

            na = site.get_pop_size(GroupQry(attr={ 'flu': 'i' }))
            ns = site.get_pop_size(GroupQry(attr={ 'flu': 'r' }))

            p = self.get_p_infection_site(na, ns)

        if group.get_attr('flu') == 's':
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=p,     attr_set={ 'flu': 'i' })
            ]
        elif group.get_attr('flu') == 'i':
            return [
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 's' })
            ]
        elif group.get_attr('flu') == 'r':
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu': 'r' })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu'")

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu' ])

    def get_p_infection_site(self, na, ns):
        ''' Agent density based formula for infection. '''

        return min(self.p_infection_max, self.p_infection_min * ((na + ns) / 2))
