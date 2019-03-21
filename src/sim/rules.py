'''
Rules used throughout the 'sim' simulations-testing package.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.entity import AttrFluStage, GroupSplitSpec, Site
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint


# ----------------------------------------------------------------------------------------------------------------------
class RuleAnalyzerTestRule(Rule):
    '''
    A rule used for testing the syntactic rule analysis.

    This rules is only used in development and is not meant to be added to actual simulations.  The component tested
    via this rule is the 'sim.RuleAnalyzer' class.
    '''

    def __init__(self, t=TimeInt(8,20), memo=None):
        super().__init__('progress-flu', t, memo)

    def an(self, s): return f'b{s}'  # attribute name
    def rn(self, s): return f's{s}'  # relation  name

    def apply(self, group, iter, t):
        if group.has_attr({ 'flu-stage': AttrFluStage.NO }):
            pass
        elif group.has_attr({ 'flu-stage': AttrFluStage.ASYMPT }):
            pass
        elif group.has_attr({ 'flu-stage': AttrFluStage.SYMPT }):
            pass

    def is_applicable(self, group, iter, t):
        g = group
        c01, c02, c03, c04, c05 = 'cc01', 'cc02', 'cc03', 'cc04', 'cc05'  # attribute names stored in local variables
        s01, s02, s03, s04, s05 = 'ss01', 'ss02', 'ss03', 'ss04', 'ss05'  # ^ (relation)

        return (
            super().is_applicable(iter, t) and

            g.has_attr('a01') and g.has_attr([ 'a02', 'a03' ]) and g.has_attr({ 'a04':1, 'a05':2 }) and
            g.has_attr(c01) and g.has_attr([ c02, c03 ]) and g.has_attr({ c04:1, c05:2 }) and
            g.has_attr(self.an('01')) and g.has_attr([ self.an('02'), self.an('03') ]) and g.has_attr({ self.an('04'):1, self.an('05'):2 }) and

            g.has_rel('r01') and g.has_rel([ 'r02', 'r03' ]) and g.has_rel({ 'r04':1, 'r05':2 }) and
            g.has_rel(s01) and g.has_rel([ s02, s03 ]) and g.has_rel({ s04:1, s05:2 }) and
            g.has_rel(self.rn('01')) and g.has_rel([ self.rn('02'), self.rn('03') ]) and g.has_rel({ self.rn('04'):1, self.rn('05'):2 })
        )


# ----------------------------------------------------------------------------------------------------------------------
class ResetSchoolDayRule(Rule):
    def __init__(self, t, memo=None):
        super().__init__('reset-day', t, memo)

    def apply(self, pop, group, iter, t):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],

    def is_applicable(self, group, iter, t):
        return super().is_applicable(iter, t)


# ----------------------------------------------------------------------------------------------------------------------
class AttendSchoolRule(Rule):
    def __init__(self, t=TimeInt(8,16), memo=None):
        super().__init__('attend-school', t, memo)

    def apply(self, pop, group, iter, t):
        if group.has_rel({ Site.AT: group.get_rel('home') }) and (not group.has_attr('did-attend-school-today') or group.has_attr({ 'did-attend-school-today': False })):
            return self.apply_at_home(group, iter, t)

        if group.has_rel({ Site.AT: group.get_rel('school') }):
            return self.apply_at_school(group, iter, t)

    def apply_at_home(self, group, iter, t):
        p = { 8:0.50, 9:0.50, 10:0.50, 11:0.50, 12:1.00 }.get(t, 0.00)  # TODO: Provide these as a CDF
            # prob of going to school = f(time of day)

        return [
            GroupSplitSpec(p=p, attr_set={ 'did-attend-school-today': True, 't-at-school': 0 }, rel_set={ Site.AT: group.get_rel('school') }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_at_school(self, group, iter, t):
        t_at_school = group.get_attr('t-at-school')
        p = { 0: 0.00, 1:0.05, 2:0.05, 3:0.25, 4:0.50, 5:0.70, 6:0.80, 7:0.90, 8:1.00 }.get(t_at_school, 1.00) if t < self.t.t1 else 1.00
            # prob of going home = f(time spent at school)

        return [
            GroupSplitSpec(p=p, attr_set={ 't-at-school': (t_at_school + 1) }, rel_set={ Site.AT: group.get_rel('home') }),
            GroupSplitSpec(p=1 - p, attr_set={ 't-at-school': (t_at_school + 1) })
        ]

        # TODO: Give timer information to the rule so it can appropriate determine time passage (e.g., to add it to 't-at-school' above).

    def is_applicable(self, group, iter, t):
        return (
            super().is_applicable(iter, t) and
            group.has_rel(['home', 'school']))

    @staticmethod
    def setup(pop, group):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    def __init__(self, t=TimeInt(8,20), memo=None):  # 8am - 8pm
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
        return super().is_applicable(iter, t) and group.has_attr([ 'flu-stage' ])


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

    The above assumption state the mechanisms implemented in the current simulation.  They do not however define limits
    on the PRAM simulation framework.
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
        return super().is_applicable(iter, t) and group.has_attr([ 'flu-stage' ])

    def get_p_infection_site(self, na, ns):
        ''' Agent density based formula for infection. '''

        return min(self.p_infection_max, self.p_infection_min * ((na + ns) / 2))
