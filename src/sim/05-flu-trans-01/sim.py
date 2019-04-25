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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from collections import namedtuple

from pram.data   import ProbeMsgMode, ProbePersistanceDB, GroupSizeProbe
from pram.entity import AttrFluStage, GroupQry, GroupSplitSpec, Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimeInt, TimePoint
from pram.sim    import Simulation

from rules import ProgressAndTransmitFluRule


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init, sites, and probes:

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

dpath_cwd = os.path.dirname(__file__)
fpath_db  = os.path.join(dpath_cwd, f'probes-{sim_dur_days}d.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

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

sim = (Simulation().
    set_iter_cnt(24 * sim_dur_days).
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(GoToAndBackTimeAtRule()).
    add_rule(ProgressAndTransmitFluRule()).
    add_probes(probes_grp_size_flu_school)
)

for s in specs:
    (sim.new_group(s.n, s.name).
        set_attr('flu-stage', AttrFluStage.NO).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites[f'school-{s.name}']).
        done()
    )

sim.run()
