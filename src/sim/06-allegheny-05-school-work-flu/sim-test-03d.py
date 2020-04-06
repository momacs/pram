'''
The Flu SEIR model on the Allegheny County data.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import pram.util as util

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistenceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, Rule, SEIRModel, TimeAlways, TimePoint
from pram.sim    import Simulation


import signal
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

rand_seed = 1928

pragma_live_info = True
pragma_live_info_ts = False

fpath_db_in = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Sites:

sites = Simulation.gen_sites_from_db(
    fpath_db_in,
    lambda fpath_db: {
        'hosp'    : Site.gen_from_db(fpath_db, 'hospitals',  'hosp_id', 'hospital', ['workers', 'physicians', 'beds']),
        'home_gq' : Site.gen_from_db(fpath_db, 'gq',         'sp_id',   'home',     ['gq_type', 'persons']),
        'home'    : Site.gen_from_db(fpath_db, 'households', 'sp_id',   'home',     ['hh_income']),
        'school'  : Site.gen_from_db(fpath_db, 'schools',    'sp_id',   'school',   []),
        'work'    : Site.gen_from_db(fpath_db, 'workplaces', 'sp_id',   'work',     [])
    },
    pragma_live_info=pragma_live_info,
    pragma_live_info_ts=pragma_live_info_ts
)

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probes:

n_schools = 8
few_schools = [sites['school'][k] for k in list(sites['school'].keys())[:n_schools]]

probe_grp_size_schools = GroupSizeProbe('school', [GroupQry(rel={ Site.AT: s }) for s in few_schools], msg_mode=ProbeMsgMode.DISP)

fpath_db = os.path.join(os.path.dirname(__file__), 'out-test-03c.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

pp = ProbePersistenceDB(fpath_db)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

(Simulation().
    set().
        rand_seed(rand_seed).
        pragma_autocompact(True).
        pragma_live_info(pragma_live_info).
        pragma_live_info_ts(pragma_live_info_ts).
        pragma_rule_analysis_for_db_gen(True).
        done().
    add().
        rule(SEIRModel()).
        rule(ResetSchoolDayRule(TimePoint(7))).
        rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
        # rule(AttrRule()).
        probe(probe_grp_size_schools).
        done().
    gen_groups_from_db(
        fpath_db_in,
        tbl      = 'people',
        attr_fix = {},
        rel_fix  = { 'home': site_home },
        attr_db  = [],
        rel_db   = [
            GroupDBRelSpec('school', 'school_id', sites['school'])
        ],
        rel_at   = 'home'
    ).
    run(1).
    summary()
)
