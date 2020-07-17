'''
A single infection wave with the city closedown intervention.  Social distancing compliance sampled randomly.

Social distancing compliance for young and old people are set according to the results of Kosta's beta regression
analysis.
'''

from pram.data        import Probe, ProbePersistenceDB, ProbePersistenceMem, ProbePersistenceMode, Var
from pram.entity      import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.model.model import ODESolver
from pram.model.epi   import SEIRModel, SEI2RModelParams
from pram.rule        import Rule, SimRule
from pram.sim         import Simulation
from pram.util        import Size, SQLiteDB, Time

import gc
import gzip
import math
import os
import pickle
import psutil

from ext_data      import p_covid19_fat_by_age_group_comb
from ext_group_qry import gq_S, gq_E, gq_IA, gq_IS, gq_R
from ext_probe     import PopProbe
from ext_rule      import DailyBehaviorRule, DiseaseRule2, ClosedownIntervention, ReopenIntervention


import signal,sys
def signal_handler(signal, frame): sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ----------------------------------------------------------------------------------------------------------------------
sim_name = 'sim-03b'

n_weeks = 8

do_sim         = True  # set to False to plot only (probe DB must exist)
do_plot        = True
do_profile_sim = False
do_profile_res = False

do_school      = True
do_work        = False
do_school_work = False

disease_name = 'COVID-19'

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
# Population and environment source:

locale_db = SQLiteDB(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny.sqlite3'))
locale_db.open_conn()


# ----------------------------------------------------------------------------------------------------------------------
# Probe persistence:

persistence = None
persistence = ProbePersistenceMem()
# persistence = ProbePersistenceDB(os.path.join(os.path.dirname(__file__), 'out', sim_name, 'probe.sqlite3'), mode=ProbePersistenceMode.OVERWRITE)


# ----------------------------------------------------------------------------------------------------------------------
# def contact(group, attr_val, tm):
#     if attr_val == 'S':
#         if group.is_at_site_name('school'):
#             m_i = group.get_site_at().get_groups_mass(GroupQry(attr={ disease_name: 'I' }))
#             if m_i == 0:
#                 return [1.0, 0.0, 0.0, 0.0]  # no infected agents nearby, therefore no chance for new infections
#
#             m   = group.get_site_at().get_groups_mass()
#             p_i = min(1.0, m_i * r0 * 1000 / m)  # prob. of infection
#             # print(p_i)
#
#             return [1.0 - p_i, p_i, 0.0, 0.0]
#         else:
#             return [1.0, 0.0, 0.0, 0.0]  # infections only happen at school
#
#     return None


# ----------------------------------------------------------------------------------------------------------------------
# Simulation:

prop_pop_inf = 0.0001  # fractional mass: 20 students and 59 workers; integer mass:  _ students and  7 workers
prop_pop_inf = 0.0005  # fractional mass:  _ students and  _ workers; integer mass: 34 students and 61 workers

def grp_setup(pop, group):
    return [
        GroupSplitSpec(p=1 - prop_pop_inf, attr_set={ disease_name: 'S'  }),
        GroupSplitSpec(p=    prop_pop_inf, attr_set={ disease_name: 'IA' })
    ]

sim = (Simulation(pop_hist_len=0).
    set().
        pragma_analyze(False).
        pragma_autocompact(True).
        pragma_comp_summary(True).
        pragma_fractional_mass(False).
        pragma_live_info(False).
        fn_group_setup(grp_setup).
        done().
    add([
        DailyBehaviorRule('school', p_home_IS=0.90),
        DailyBehaviorRule('work',   p_home_IS=0.90),
        # DiseaseRule('school', r0=1.8, p_E_IA=0.40, p_IA_IS=0.40, p_IS_R=0.40, p_home_E=0.025, p_social_E=0.05, soc_dist_comp_young=0.66, soc_dist_comp_old=0.45, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
        # DiseaseRule('work',   r0=1.8, p_E_IA=0.40, p_IA_IS=0.40, p_IS_R=0.40, p_home_E=0.025, p_social_E=0.05, soc_dist_comp_young=0.66, soc_dist_comp_old=0.45, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
        DiseaseRule2('school', SEI2RModelParams.by_clinical_obs(s0=2, r0=1.8, incub_period=5.1 * 2, asympt_period=2.5 * 2, inf_dur=5.0 * 2), p_home_E=0.0, p_social_E=0.95, soc_dist_comp_young=0.66, soc_dist_comp_old=0.45, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
        DiseaseRule2('work',   SEI2RModelParams.by_clinical_obs(s0=2, r0=1.8, incub_period=5.1 * 2, asympt_period=2.5 * 2, inf_dur=5.0 * 2), p_home_E=0.0, p_social_E=0.95, soc_dist_comp_young=0.66, soc_dist_comp_old=0.45, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
        ClosedownIntervention(prop_pop_IS_threshold=0.01),
        ReopenIntervention(prop_pop_IS_threshold=0.02),
        PopProbe(persistence),
    ])
)

if (do_sim or do_profile_sim) and (do_school or do_school_work):
    sim.gen_groups_from_db(locale_db, schema=None, tbl='people', rel_at='home', limit=0,
        attr_fix = {},
        rel_fix  = { 'home': site_home },
        attr_db  = ['age_group'],
        rel_db   = [
            GroupDBRelSpec(name='school', col='school_id', fk_schema=None, fk_tbl='schools', fk_col='sp_id', sites=None)
        ]
    )

if (do_sim or do_profile_sim) and (do_work or do_school_work):
    sim.gen_groups_from_db(locale_db, schema=None, tbl='people', rel_at='home', limit=0,
        attr_fix = {},
        rel_fix  = { 'home': site_home },
        attr_db  = ['age_group'],
        rel_db   = [
            GroupDBRelSpec(name='work', col='work_id', fk_schema=None, fk_tbl='workplaces', fk_col='sp_id', sites=None)
        ],
        # attr_rm  = ['age_group']  # must remove manually because it will be pulled automatically due to rules conditioning on it
    )

if do_sim and not do_profile_sim:
    sim.run(2 * 7 * n_weeks)  # (two events a day) * (days) * (weeks)


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

if do_plot and persistence:
    quantity = 'p'  # 'm' or 'p'
    series = [
        # { 'var': f'{quantity}_S', 'lw': 1.50, 'ls': 'solid', 'dashes': (4,8), 'marker': '+', 'color': 'blue',   'ms': 0, 'lbl': 'S' },
        { 'var': f'{quantity}_E', 'lw': 1.50, 'ls': 'solid', 'dashes': (1,0), 'marker': '+', 'color': 'orange', 'ms': 0, 'lbl': 'E' },
        { 'var': f'{quantity}_I', 'lw': 1.50, 'ls': 'solid', 'dashes': (5,1), 'marker': '*', 'color': 'red',    'ms': 0, 'lbl': 'I' },
        # { 'var': f'{quantity}_R', 'lw': 1.50, 'ls': 'solid', 'dashes': (5,6), 'marker': '|', 'color': 'green',  'ms': 0, 'lbl': 'R' },
        { 'var': f'{quantity}_X', 'lw': 1.50, 'ls': 'solid', 'dashes': (1,2), 'marker': 'x', 'color': 'black',  'ms': 0, 'lbl': 'X' }
    ]
    sim.probes[0].plot(series, ylabel='Population mass', xlabel='Iteration (half-days from start of infection)', figsize=(12,4), subplot_b=0.15)


# ----------------------------------------------------------------------------------------------------------------------
# Profile:

if do_profile_sim or do_sim:
    import cProfile
    import pstats

    if do_profile_sim and not do_sim:
        # cProfile.run('sim.run(2*7*3)', os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-01'))
        # cProfile.run('sim.run(2*7*3)', os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-02'))
        # cProfile.run('sim.run(2*7*3)', os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-03'))
        # cProfile.run('sim.run(2*7*3)', os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-04'))
        # cProfile.run('sim.run(2*7*3)', os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-work-01'))
        pass

    if do_profile_res:
        # pstats.Stats(os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-01')).sort_stats('time', 'cumulative').print_stats(10)  # tot: 18.191
        # pstats.Stats(os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-02')).sort_stats('time', 'cumulative').print_stats(10)  # tot: 18.155
        # pstats.Stats(os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-03')).sort_stats('time', 'cumulative').print_stats(10)
        # pstats.Stats(os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-school-04')).sort_stats('time', 'cumulative').print_stats(10)
        # pstats.Stats(os.path.join(os.path.dirname(__file__), 'restats', sim_name, '-work-01')).  sort_stats('time', 'cumulative').print_stats(10)
        pass


# 6301ms --> 5828ms = Speed up of 5-10%   [attr and rel encoder]
# 6301ms --> 5530ms = Speed up of 10-15%  [attr and rel encoder + making queries global]
