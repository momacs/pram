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

from collections import namedtuple

from pram.data   import ProbeMsgMode, ProbePersistenceDB, GroupSizeProbe
from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, GoToAndBackTimeAtRule, ResetSchoolDayRule
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class ProgressAndTransmitFluRule(Rule):
    def __init__(self, t=[8,20], p_infection_min=0.001, p_infection_max=0.01):
        super().__init__('progress-flu', t)

        self.p_infection_min = p_infection_min
        self.p_infection_max = p_infection_max

    def apply(self, pop, group, iter, t):
        if group.is_at_site_name('school'):
            p = group.get_rel('school').get_mass_prop(GroupQry(attr={ 'flu': 'i' }))
        else:
            p = self.p_infection_min

        if group.get_attr('flu') == 's':
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=p,     attr_set={ 'flu': 'i' })
            ]
        elif group.get_attr('flu') == 'i':
            return [
                GroupSplitSpec(p=0.95, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.05, attr_set={ 'flu': 'r' })
            ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu' ])


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init, sites, and probes:

sim_dur_days = 3

Spec = namedtuple('Spec', ('name', 'm'))
school_specs = [
    Spec('s0',    50),
    Spec('s1',   100),
    Spec('s2',   200),
    Spec('s3',   300),
    Spec('s4',   500),
    Spec('s5',  5000),
    Spec('s6', 50000)
]

dpath_cwd = os.path.dirname(__file__)
fpath_db  = os.path.join(dpath_cwd, f'probes-{sim_dur_days}d.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

probe_persistence = ProbePersistenceDB(fpath_db)

probes_grp_size_flu_school = []

sites = { 'home': Site('home') }

for s in school_specs:
    # Site:
    site_name = f'school-{s.name}'
    site = Site(site_name)
    sites[site_name] = site

    # Probe:
    probes_grp_size_flu_school.append(
        GroupSizeProbe(
            name=f'flu-{s.name}',
            queries=[
                GroupQry(attr={ 'flu': 's' }, rel={ 'school': site }),
                GroupQry(attr={ 'flu': 'i' }, rel={ 'school': site }),
                GroupQry(attr={ 'flu': 'r' }, rel={ 'school': site })
            ],
            qry_tot=GroupQry(rel={ 'school': site }),
            var_names=['ps', 'pi', 'pr', 'ns', 'ni', 'nr'],
            persistence=probe_persistence,
            memo=f'Flu at school {s.name.upper()}'
        )
    )

probe_grp_size_flu  = GroupSizeProbe.by_attr('flu', 'flu',    ['s', 'i', 'r'], msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')
probe_grp_size_site = GroupSizeProbe.by_rel ('site', Site.AT, sites.values(),  msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across sites')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

if __name__ == '__main__':
    sim = (Simulation().
        # add_rule(ResetSchoolDayRule(7)).
        add_rule(GoToAndBackTimeAtRule()).
        add_rule(ProgressAndTransmitFluRule()).
        add_probes(probes_grp_size_flu_school)
        # add_probe(probe_grp_size_flu)
    )

    for s in school_specs:
        (sim.new_group(s.name, s.m).
            set_attr('flu', 's').
            set_rel(Site.AT,  sites['home']).
            set_rel('home',   sites['home']).
            set_rel('school', sites[f'school-{s.name}']).
            done()
        )

    sim.run(24 * sim_dur_days)
