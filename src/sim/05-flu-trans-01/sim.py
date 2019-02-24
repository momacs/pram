'''
A simulation of the flu transmission model in a population of school-attenting agents.  Specifically, the transmission
modeled as a mechanism with a dual underpinning:

1. Sontaneous (extraneous).
The disease afflicts an agent with a small unconditional probability.  This accounts for unmodeled influences, such as
the introduction of the virus from outside of the simulation.

2. Proximy based.
The probability of an agent getting infected increases with the number of nearby agents who are infectious.  The
infection probability increaes as a function of the environment and once the maximum number of infectious agents is
reached, it does not increase any more.  This simulates the intuition that even in a densly populated location only a
certain number of agents can be nearby (or get in contact with) an agent.

For simplicity, infection probability increases only when agents are at school.

Irrespective of the infection mode, an infected agent goes through the flu stages specified by the progression model.
In this simulation, agents attend school irrespective of being sick or not which allows us to draw some interesting
conclusions.  Five schools of different sizes are modeled.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from collections import namedtuple

from pram.sim    import Simulation
from pram.entity import AttrFluStage, GroupQry, GroupSplitSpec, Site
from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint


# ----------------------------------------------------------------------------------------------------------------------
class ResetSchoolDayRule(Rule):
    def __init__(self, t, memo=None):
        super().__init__('reset-day', t, memo)

    def apply(self, pop, group, iter, t):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(iter, t)


# ----------------------------------------------------------------------------------------------------------------------
class AttendSchoolFluRule(Rule):
    def __init__(self, t=TimeInt(8,16), memo=None):
        super().__init__('attend-school', t, memo)

    def apply(self, pop, group, iter, t):
        if group.has_rel({ Site.AT: group.get_rel('home') }) and (not group.has_attr('did-attend-school-today') or group.has_attr({ 'did-attend-school-today': False })):
            return self.apply_at_home(group, iter, t)

        if group.has_rel({ Site.AT:  group.get_rel('school') }):
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


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init, sites, and probes:

rand_seed = 1928
sim_dur_days = 1

Spec = namedtuple('Spec', ('name', 'n'))
specs = [
    Spec('0',    50),
    Spec('1',   100),
    Spec('2',   200),
    Spec('3',   300),
    Spec('4',   500),
    Spec('5',  5000),
    Spec('6', 50000)
]

dir = os.path.dirname(__file__)
fpath_db = os.path.join(dir, f'probes-{sim_dur_days}d.sqlite3')

probe_persistance = ProbePersistanceDB(fpath_db)

probes_grp_size_flu_school = []

sites = {
    'home' : Site('home')
}

for s in specs:
    # Site:
    site_name = f'school-{s.name}'
    site = Site(site_name)
    sites[site_name] = site

    # Probe:
    probes_grp_size_flu_school.append(
        GroupSizeProbe(
            name=f'flu-{s.name}',
            queries=[
                GroupQry(attr={ 'flu-stage': AttrFluStage.NO     }, rel={ 'school': site }),
                GroupQry(attr={ 'flu-stage': AttrFluStage.ASYMPT }, rel={ 'school': site }),
                GroupQry(attr={ 'flu-stage': AttrFluStage.SYMPT  }, rel={ 'school': site })
            ],
            qry_tot=GroupQry(rel={ 'school': site }),
            var_names=['pn', 'pa', 'ps', 'nn', 'na', 'ns'],
            persistance=probe_persistance,
            memo=f'Flu at school {s.name.upper()}'
        )
    )

probe_grp_size_flu  = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage,   msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')
probe_grp_size_site = GroupSizeProbe.by_rel ('site', Site.AT,    sites.values(), msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across sites')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

sim = Simulation(6,1,24 * sim_dur_days, rand_seed=rand_seed)
for s in specs:
    (sim.new_group(s.name, s.n).
        set_attr('is-student', True).
        set_attr('flu-stage', AttrFluStage.NO).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites[f'school-{s.name}']).
        commit()
    )

(sim.
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(AttendSchoolFluRule()).
    add_rule(ProgressAndTransmitFluRule()).
    add_probes(probes_grp_size_flu_school).
    run()
)
