'''
The third simulation testing constructing a simulation from a database.  The population of interst is the synthetic
population of the Allegheny county.  Apart from the aspects tested by the previous simulations, this simulation adds
the flu progression and fly transmission models developed earlier (see below).  The data produced is visualized
elsewhere (res.py).

Based on:
  sim/04-flu-prog/
  sim/05-flu-trans-01/
  sim/05-flu-trans-02/
  sim/06-allegheny-02-db-school-large/
'''

import gc
import gzip
import os
import pickle
import sys

from pram.data   import ProbePersistenceDB, GroupSizeProbe
from pram.entity import GroupDBRelSpec, GroupQry, Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimePoint
from pram.sim    import Simulation


import signal
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

fpath_db_in  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny.sqlite3')
fpath_sites  = os.path.join(os.path.dirname(__file__), 'allegheny-sites.pickle.gz')
fpath_groups = os.path.join(os.path.dirname(__file__), 'allegheny-groups.pickle.gz')

do_remove_file_sites  = False
do_remove_file_groups = False

if do_remove_file_sites and os.path.isfile(fpath_sites):
    os.remove(fpath_sites)

if do_remove_file_groups and os.path.isfile(fpath_groups):
    os.remove(fpath_groups)


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
    fpath_sites
)

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probes:

fpath_db_out = os.path.join(os.path.dirname(__file__), 'out.sqlite3')

if os.path.isfile(fpath_db_out):
    os.remove(fpath_db_out)

pp = ProbePersistenceDB(fpath_db_out, flush_every=1)

probe_school_pop_size = GroupSizeProbe(
    name='school-pop-size',
    queries=[GroupQry(rel={ Site.AT: s }) for s in sites['school'].values()],
    persistence=pp,
    var_names=
        [f'p{i}' for i in range(len(sites['school']))] +
        [f'n{i}' for i in range(len(sites['school']))],
    memo=f'Population size across all schools'
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

(Simulation().
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
    add_probe(probe_school_pop_size).
    gen_groups_from_db(
        fpath_db_in,
        tbl='people',
        attr_fix={},
        rel_fix={ 'home': site_home },
        attr_db=[],
        rel_db=[GroupDBRelSpec('school', 'school_id', sites['school'])],
        rel_at='home'
    ).
    summary().
    run(3, do_disp_t=True)
)
