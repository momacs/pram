'''
This simulation scrutinizes the previous simulation (i.e., 05-flu-trans-01) by performing sensitivity analysis on the
following simultation parameters:

- The probability of spontanoeous flu infection
- The agent density based formula for infection
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np

from collections import namedtuple

from pram.data   import Probe, ProbePersistanceDB, GroupSizeProbe
from pram.entity import AttrFluStage, GroupQry, GroupSplitSpec, Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimeInt, TimePoint
from pram.sim    import HourTimer, Simulation

from rules import ProgressAndTransmitFluRule


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init, sites, and probes:

sim_dur_days = 7

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
            memo=f'Flu at school {s.name.upper()}'
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

p_inf_lst = np.arange(0.01, 0.1, 0.025).tolist()

flu_rule = ProgressAndTransmitFluRule()

def run_sim(p_lst):
    for p in p_lst:
        sim = (Simulation().
            set_timer(HourTimer(6)).
            set_iter_cnt(24 * sim_dur_days).
            add_rule(ResetSchoolDayRule(TimePoint(7))).
            add_rule(AttendSchoolRule()).
            add_rule(flu_rule).
            add_probes(probes_grp_size_flu_school)
        )

        for s in specs:
            (sim.new_group(s.n, s.name).
                set_attr('is-student', True).
                set_attr('flu-stage', AttrFluStage.NO).
                set_rel(Site.AT,  sites['home']).
                set_rel('home',   sites['home']).
                set_rel('school', sites[f'school-{s.name}']).
                done()
            )

        setattr(flu_rule, 'p_infection_min', p)
        for p in probes_grp_size_flu_school:
            p.set_consts([Probe.Const('p_inf_min', 'float', str(flu_rule.p_infection_min))])
            p.set_persistance(probe_persistance)

        sim.run()

# run_sim(p_inf_lst)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Profile:

# import cProfile
# cProfile.run('run_sim(p_inf_lst)', f'restats-{sim_dur_days}d')

# import pstats
# pstats.Stats('restats-7d-flush-every-1'). sort_stats('time', 'cumulative').print_stats(10)
# pstats.Stats('restats-7d-flush-every-10').sort_stats('time', 'cumulative').print_stats(20)
# pstats.Stats('restats-7d-flush-every-16').sort_stats('time', 'cumulative').print_stats(20)
# pstats.Stats('restats-7d-flush-every-25').sort_stats('time', 'cumulative').print_stats(20)
# pstats.Stats('restats-7d-flush-every-50').sort_stats('time', 'cumulative').print_stats(20)
