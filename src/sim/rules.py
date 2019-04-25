'''
Rules used throughout the 'sim' test simulations package.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.entity import AttrFluStage, GroupQry, GroupSplitSpec, Site
from pram.rule   import GoToRule, Rule, TimeInt, TimePoint
from pram.util   import Err


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    def __init__(self, t=TimeInt(8,20), memo=None):
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, iter, t):
        # An equivalent Dynamic Bayesian net has the initial state distribution:
        #
        #     flu-stage: (1 0 0)
        #
        # and the transition model:
        #
        #           N     A     S
        #     N  0.95  0.00  0.10
        #     A  0.05  0.50  0
        #     S  0     0.50  0.90

        p = 0.05  # prob of infection

        if group.get_attr('flu-stage') == AttrFluStage.NO:
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=p,     attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.00,  attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.ASYMPT:
            return [
                GroupSplitSpec(p=0.00, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.SYMPT:
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu-stage' ])


# ----------------------------------------------------------------------------------------------------------------------
class ProgressAndTransmitFluRule(Rule):
    '''
    This rule makes the following assumptions:

        1. There are three sequential stages of the flu (per AttrFluStage):
           - No flu
           - Asymptomatic
           - Symptomatic
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

            na = site.get_pop_size(GroupQry(attr={ 'flu-stage': AttrFluStage.ASYMPT }))
            ns = site.get_pop_size(GroupQry(attr={ 'flu-stage': AttrFluStage.SYMPT  }))

            p = self.get_p_infection_site(na, ns)

        if group.get_attr('flu-stage') == AttrFluStage.NO:
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=p,     attr_set={ 'flu-stage': AttrFluStage.ASYMPT })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.ASYMPT:
            return [
                GroupSplitSpec(p=0.50, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.SYMPT:
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu-stage': AttrFluStage.NO    }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu-stage': AttrFluStage.SYMPT })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-stage'.")

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu-stage' ])

    def get_p_infection_site(self, na, ns):
        ''' Agent density based formula for infection. '''

        return min(self.p_infection_max, self.p_infection_min * ((na + ns) / 2))
