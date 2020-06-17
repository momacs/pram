'''
This simulation scrutinizes the previous simulation (i.e., 05-flu-trans-01) by performing sensitivity analysis on the
following simultation parameters:

- The probability of spontanoeous flu infection
- The agent density based formula for infection
'''

import numpy as np
import os

from collections import namedtuple

from pram.data   import Probe, ProbePersistenceDB, GroupSizeProbe, Const
from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, GoToAndBackTimeAtRule, ResetSchoolDayRule
from pram.sim    import HourTimer, Simulation


# ----------------------------------------------------------------------------------------------------------------------
class ProgressAndTransmitFluRule(Rule):
    def __init__(self, t=[8,20], p_infection_min=0.01, p_infection_max=0.95):
        super().__init__('progress-flu', t)

        self.p_infection_min = p_infection_min
        self.p_infection_max = p_infection_max

    def apply(self, pop, group, iter, t):
        p = self.p_infection_min
        if group.is_at_site_name('school'):
            p = group.get_rel(Site.AT).get_mass_prop(GroupQry(attr={ 'flu': 'i' }))

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


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init, sites, and probes:

sim_dur_days = 7

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
            memo=f'Flu at school {s.name.upper()}'
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

if __name__ == '__main__':
    p_inf_lst = np.arange(0.01, 0.1, 0.025).tolist()

    flu_rule = ProgressAndTransmitFluRule()

    def run_sim(p_lst):
        for p in p_lst:
            sim = (Simulation().
                # add_rule(ResetSchoolDayRule(7)).
                add_rule(GoToAndBackTimeAtRule()).
                add_rule(flu_rule).
                add_probes(probes_grp_size_flu_school)
            )

            for s in school_specs:
                (sim.new_group(s.name, s.m).
                    set_attr('is-student', True).
                    set_attr('flu', 's').
                    set_rel(Site.AT,  sites['home']).
                    set_rel('home',   sites['home']).
                    set_rel('school', sites[f'school-{s.name}']).
                    done()
                )

            setattr(flu_rule, 'p_infection_min', p)
            for p in probes_grp_size_flu_school:
                p.set_consts([Const('p_inf_min', 'float', str(flu_rule.p_infection_min))])
                p.set_persistence(probe_persistence)

            sim.run(24 * sim_dur_days)

    run_sim(p_inf_lst)


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
