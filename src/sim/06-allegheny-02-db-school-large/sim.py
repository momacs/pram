'''
The second simulation testing constructing a simulation from a database.  The population of interst is the synthetic
population of the Allegheny county.  The simulation reuses school-attending rules developed earlier (see below) to
demonstrate that they scale to this more realistic scenario.  Moreover, it also employs probe database persistance to
keep track of vital characteristics over a larger number of sites of interests (specifically, schools).  This data
saved this way is visualized elsewhere (res.py).

Furthermore, this simulation is used to investigate the concept of group parsymony.  Specifically, in one simulation
the agent population is allowed to live in their respective households while in another simulation they are modeled to
live in one big home site.  As anticipated, because neither of the rules conditions on the home agents live in, the
results of the simulations are identical.  What is not identical is the time and space efficiency of the two
simulations.  Using the Allegheny county data which contains information on 350 schools, the parsimonous simulation
runs about 385 times faster than it's "bloated" counterpart.  And that is with some aspects of the population not even
considered (e.g., sex and age).

Based on: sim/06-allegheny-01-db-school
'''

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


from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
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
fpath_sites  = os.path.join(dpath_cwd, 'allegheny-sites.pickle.gz')
fpath_groups = os.path.join(dpath_cwd, 'allegheny-groups.pickle.gz')

do_remove_file_sites  = False
do_remove_file_groups = True

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

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probes:

fpath_db_out_school_pop_size = os.path.join(dpath_cwd, 'school-pop-size.sqlite3')

if os.path.isfile(fpath_db_out_school_pop_size):
    os.remove(fpath_db_out_school_pop_size)

probe_persistance = ProbePersistanceDB(fpath_db_out_school_pop_size, flush_every=1)

probes_school_pop_size = GroupSizeProbe(
    name='school-pop-size',
    queries=[GroupQry(rel={ Site.AT: s }) for s in sites['school'].values()],
    persistance=probe_persistance,
    var_names=
        [f'p{i}' for i in range(len(sites['school']))] +
        [f'n{i}' for i in range(len(sites['school']))],
    memo=f'Population size across all schools'
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

(Simulation(7,1,10, rand_seed=rand_seed).
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(AttendSchoolRule()).
    add_probe(probes_school_pop_size).
    gen_groups_from_db(
        fpath_db_in,
        tbl='people',
        attr={},
        rel={ 'home': site_home },
        # rel={},
        attr_db=[],
        rel_db=[
            # GroupDBRelSpec('home', 'sp_hh_id', sites['home']),
            GroupDBRelSpec('school', 'school_id', sites['school'])
        ],
        rel_at='home',
        fpath=fpath_groups
    ).
    summary().
    run(do_disp_t=True)
)
