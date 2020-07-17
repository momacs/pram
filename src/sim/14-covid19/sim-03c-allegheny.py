'''
A single infection wave as a trajectory ensemble with the city closedown intervention.  Social distancing compliance
sampled randomly.
'''

import gc
import gzip
import math
import os
import pickle
import psutil

from scipy.stats import truncnorm

from pram.data        import Probe, ProbePersistenceDB, ProbePersistenceMem, ProbePersistenceMode, Var
from pram.entity      import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.model.model import ODESolver
from pram.model.epi   import SEIRModel, SEI2RModelParams
from pram.rule        import Rule, SimRule
from pram.sim         import Simulation
from pram.util        import Size, SQLiteDB, Time
from pram.traj        import Trajectory, TrajectoryEnsemble, ClusterInf

from ext_data      import p_covid19_fat_by_age_group_comb
from ext_group_qry import gq_S, gq_E, gq_IA, gq_IS, gq_R
from ext_probe     import PopProbe
from ext_rule      import DailyBehaviorRule, DiseaseRule2, ClosedownIntervention


import signal,sys
def signal_handler(signal, frame): sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ----------------------------------------------------------------------------------------------------------------------
sim_name = 'sim-03c'

n_weeks = 4

do_sim            = True  # set to False to plot only (probe DB must exist)
do_plot           = True
do_rm_traj_ens_db = True

do_school      = True
do_work        = False
do_school_work = False

disease_name = 'COVID-19'

fpath_pop_db      = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny.sqlite3')
fpath_traj_ens_db = os.path.join(os.path.dirname(__file__), 'out', sim_name, 'te.sqlite3')

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
def TN(a,b, mu, sigma, n=None):
    return truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(n)


# ----------------------------------------------------------------------------------------------------------------------
# Population and environment source:

locale_db = SQLiteDB(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny.sqlite3'))
locale_db.open_conn()


# ----------------------------------------------------------------------------------------------------------------------
# Probe persistence:

persistence = None
# persistence = ProbePersistenceMem()
# persistence = ProbePersistenceDB(os.path.join(os.path.dirname(__file__), 'out', sim_name, 'probe.sqlite3'), mode=ProbePersistenceMode.OVERWRITE)


# ----------------------------------------------------------------------------------------------------------------------
# Simulation:

prop_pop_inf = 0.0001  # fractional mass: 20 students and 59 workers; integer mass:  _ students and  7 workers
prop_pop_inf = 0.0005  # fractional mass:  _ students and  _ workers; integer mass: 34 students and 61 workers

def grp_setup(pop, group):
    return [
        GroupSplitSpec(p=1 - prop_pop_inf, attr_set={ disease_name: 'S'  }),
        GroupSplitSpec(p=    prop_pop_inf, attr_set={ disease_name: 'IA' })
    ]

if do_rm_traj_ens_db and os.path.isfile(fpath_traj_ens_db):
    os.remove(fpath_traj_ens_db)

te = TrajectoryEnsemble(fpath_traj_ens_db)
# te = TrajectoryEnsemble(fpath_traj_ens_db, cluster_inf=ClusterInf(num_cpus=12, memory=1000*1024*1024, object_store_memory=1000*1024*1024))

if te.is_db_empty:
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectories([
        Trajectory(
            (Simulation(pop_hist_len=0).
                set().
                    pragma_analyze(False).
                    pragma_autocompact(True).
                    pragma_comp_summary(False).
                    pragma_fractional_mass(False).
                    pragma_live_info(False).
                    fn_group_setup(grp_setup).
                    done().
                add([
                    DailyBehaviorRule('school', p_home_IS=0.90),
                    DailyBehaviorRule('work',   p_home_IS=0.90),
                    # DiseaseRule('school', r0=1.8, p_E_IA=0.40, p_IA_IS=0.40, p_IS_R=0.40, p_home_E=0.025, p_social_E=0.05, soc_dist_comp_young=sdc, soc_dist_comp_old=sdc, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
                    # DiseaseRule('work',   r0=1.8, p_E_IA=0.40, p_IA_IS=0.40, p_IS_R=0.40, p_home_E=0.025, p_social_E=0.05, soc_dist_comp_young=sdc, soc_dist_comp_old=sdc, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
                    DiseaseRule2('school', SEI2RModelParams.by_clinical_obs(s0=2, r0=1.8, incub_period=5.1 * 2, asympt_period=2.5 * 2, inf_dur=5.0 * 2), p_home_E=0.0, p_social_E=0.95, soc_dist_comp_young=0.66, soc_dist_comp_old=0.45, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
                    DiseaseRule2('work',   SEI2RModelParams.by_clinical_obs(s0=2, r0=1.8, incub_period=5.1 * 2, asympt_period=2.5 * 2, inf_dur=5.0 * 2), p_home_E=0.0, p_social_E=0.95, soc_dist_comp_young=0.66, soc_dist_comp_old=0.45, p_fat_by_age_group=p_covid19_fat_by_age_group_comb),
                    ClosedownIntervention(prop_pop_IS_threshold=0.03),
                    PopProbe(persistence, do_print=False),
                ])
            )
        ) for sdc in TN(0.0, 1.0, 0.5, 0.15, 5)
    ])

if te.is_db_empty and do_sim and (do_school or do_school_work):
    for t in te.traj.values():
        t.get_sim().gen_groups_from_db(locale_db, schema=None, tbl='people', rel_at='home', limit=0,
            attr_fix = {},
            rel_fix  = { 'home': site_home },
            attr_db  = ['age_group'],
            rel_db   = [
                GroupDBRelSpec(name='school', col='school_id', fk_schema=None, fk_tbl='schools', fk_col='sp_id', sites=None)
            ]
        )

if te.is_db_empty and do_sim and (do_work or do_school_work):
    for t in te.traj.values():
        t.get_sim().gen_groups_from_db(locale_db, schema=None, tbl='people', rel_at='home', limit=0,
            attr_fix = {},
            rel_fix  = { 'home': site_home },
            attr_db  = ['age_group'],
            rel_db   = [
                GroupDBRelSpec(name='work', col='work_id', fk_schema=None, fk_tbl='workplaces', fk_col='sp_id', sites=None)
            ],
            # attr_rm  = ['age_group']  # must remove manually because it will be pulled automatically due to rules conditioning on it
        )

if te.is_db_empty and do_sim:
    te.run(2 * 7 * n_weeks)  # (two events a day) * (days) * (weeks)


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

if do_plot and not te.is_db_empty:
    series = [{ 'var': 'p_E', 'lbl': 'E' }, { 'var':f'p_I', 'lbl': 'I' }]
    te.plot_mass_locus_line_probe((1200,300), os.path.join(os.path.dirname(__file__), 'out', sim_name, 'plot-line-probe.png'), 'probe_pop', series, opacity_min=0.15)
