import gc
import gzip
import os
import pickle
import sys

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint
from pram.sim    import Simulation
from pram.util   import Size

from rules import ResetSchoolDayRule, AttendSchoolRule


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

rand_seed = 1928

dpath_res    = os.path.join(os.sep, 'Volumes', 'd', 'pitt', 'sci', 'pram', 'res', 'fred')
dpath_cwd    = os.path.dirname(__file__)
fpath_db_in  = os.path.join(dpath_res, 'allegheny.sqlite3')
fpath_sites  = os.path.join(dpath_cwd, 'fred-flu-sites.pickle.gz')
fpath_groups = os.path.join(dpath_cwd, 'fred-flu-groups.pickle.gz')

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
        'hosp'    : Site.gen_from_db(fpath_db, 'hospitals',  'hosp_id', 'hospital', ['workers', 'physicians', 'beds']),
        'home_gq' : Site.gen_from_db(fpath_db, 'gq',         'sp_id',   'home',     ['gq_type', 'persons']),
        'home'    : Site.gen_from_db(fpath_db, 'households', 'sp_id',   'home',     ['hh_income']),
        'school'  : Site.gen_from_db(fpath_db, 'schools',    'sp_id',   'school',   []),
        'work'    : Site.gen_from_db(fpath_db, 'workplaces', 'sp_id',   'work',     [])
    },
    fpath_sites
)


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probes:

n_schools = 8
few_schools = [sites['school'][k] for k in list(sites['school'].keys())[:n_schools]]

probe_grp_size_schools = GroupSizeProbe('school', [GroupQry(rel={ Site.AT: s }) for s in few_schools], msg_mode=ProbeMsgMode.DISP)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Groups and simulation:

(Simulation(7,1,10, rand_seed=rand_seed).
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(AttendSchoolRule()).
    add_probe(probe_grp_size_schools).
    gen_groups_from_db(
        fpath_db_in,
        tbl='people',
        attr={},
        rel={},
        attr_db=[],
        rel_db=[
            GroupDBRelSpec('home', 'sp_hh_id', sites['home']),
            GroupDBRelSpec('school', 'school_id', sites['school'])
        ],
        rel_at='home',
        fpath=fpath_groups
    ).
    summary().
    run()
)
