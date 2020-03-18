import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import gc
import gzip
import os
import pickle
import signal
import sys

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistenceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import ProgressFluRule, TimeInt, TimePoint
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

def signal_handler(signal, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


dpath_res    = os.path.join(os.sep, 'Volumes', 'd', 'pitt', 'sci', 'pram', 'res', 'fred')
dpath_cwd    = os.path.dirname(__file__)
fpath_db_in  = os.path.join(dpath_res, 'allegheny.sqlite3')
fpath_sites  = os.path.join(dpath_cwd, 'allegheny-sites.pickle.gz')
fpath_groups = os.path.join(dpath_cwd, 'allegheny-groups.pickle.gz')

do_remove_file_sites  = False
do_remove_file_groups = False

if do_remove_file_sites and os.path.isfile(fpath_sites):
    os.remove(fpath_sites)

if do_remove_file_groups and os.path.isfile(fpath_groups):
    os.remove(fpath_groups)


# ----------------------------------------------------------------------------------------------------------------------
# (1) Sites:

sites = Simulation().gen_sites_from_db(
    fpath_db_in,
    lambda fpath_db: {
        'school' : Site.gen_from_db(fpath_db, 'schools',    'sp_id', 'school', []),
        'work'   : Site.gen_from_db(fpath_db, 'workplaces', 'sp_id', 'work',   [])
    },
    fpath_sites
)

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probes:

fpath_db_out = os.path.join(dpath_cwd, 'sim.sqlite3')

if os.path.isfile(fpath_db_out):
    os.remove(fpath_db_out)

pp = ProbePersistenceDB(fpath_db_out, flush_every=1)

probe_school_pop_size = GroupSizeProbe(
    name='school',
    queries=[GroupQry(rel={ Site.AT: s }) for s in sites['school'].values()],
    persistence=pp,
    var_names=
        [f'p{i}' for i in range(len(sites['school']))] +
        [f'n{i}' for i in range(len(sites['school']))],
    memo=f'Population size across all schools'
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

(Simulation(7,1,10).
    # add_rule(ResetDayRule(TimePoint(7), attr_del=['t-at-school'])).
    add_rule(ProgressFluRule()).
    add_probe(probe_school_pop_size).
    gen_groups_from_db(
        fpath_db_in,
        tbl      = 'people',
        attr_fix = {},
        rel_fix  = { 'home': site_home },
        attr_db  = [],
        rel_db   = [
            GroupDBRelSpec('school', 'school_id', sites['school'])
        ],
        rel_at   = 'home',
        fpath    = fpath_groups
    ).
    summary().
    run(do_disp_t=True)
)
