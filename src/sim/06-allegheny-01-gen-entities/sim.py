'''
The first simulation testing constructing a simulation from a database.  The population of interst is the synthetic
population of the Allegheny county.  The simulation reuses school-attending rules developed earlier (see below) to
demonstrate that they scale to this more realistic scenario.

Based on: sim/03-attend-school
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import gc
import gzip
import os
import pickle
import signal
import sys


from pram.data   import ProbeMsgMode, GroupSizeProbe
from pram.entity import Group, GroupDBRelSpec, GroupQry, Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimePoint
from pram.sim    import HourTimer, Simulation
from pram.util   import Size


# ----------------------------------------------------------------------------------------------------------------------
def inf(name, o, do_calc_size=False):
    print(f'{name}: {len(o)}  {"(" + Size.bytes2human(Size.get_size(o)) + ")" if do_calc_size else ""}')


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

def signal_handler(signal, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


dpath_res    = os.path.join(os.sep, 'Volumes', 'd', 'pitt', 'sci', 'pram', 'res', 'fred')
dpath_cwd    = os.path.dirname(__file__)
fpath_db     = os.path.join(dpath_res, 'allegheny.sqlite3')
fpath_sites  = os.path.join(dpath_cwd, 'allegheny-sites.pickle.gz')
fpath_groups = os.path.join(dpath_cwd, 'allegheny-groups.pickle.gz')

do_remove_file_sites  = False
do_remove_file_groups = False

do_calc_size_sites  = False
do_calc_size_groups = False


# ----------------------------------------------------------------------------------------------------------------------
# (1) Sites:

if do_remove_file_sites and os.path.isfile(fpath_sites):
    os.remove(fpath_sites)

if os.path.isfile(fpath_sites):
    print('Loading sites... ', end='')
    with gzip.GzipFile(fpath_sites, 'rb') as f:
        gc.disable()
        sites = pickle.load(f)
        gc.enable()
    print('done.')
else:
    print('Generating sites... ', end='')
    sites = {
        'hosp'    : Site.gen_from_db(fpath_db, 'hospitals',  'hosp_id', 'hospital', ['workers', 'physicians', 'beds']),
        'home_gq' : Site.gen_from_db(fpath_db, 'gq',         'sp_id',   'home',     ['gq_type', 'persons']),
        'home'    : Site.gen_from_db(fpath_db, 'households', 'sp_id',   'home',     ['hh_income']),
        'school'  : Site.gen_from_db(fpath_db, 'schools',    'sp_id',   'school',   []),
        'work'    : Site.gen_from_db(fpath_db, 'workplaces', 'sp_id',   'work',     [])
    }
    print('done.')

    print('Saving sites... ', end='')
    with gzip.GzipFile(fpath_sites, 'wb') as f:
        pickle.dump(sites, f)
    print('done.')

inf('hosp    ', sites['hosp'],    do_calc_size_sites)
inf('home_gq ', sites['home_gq'], do_calc_size_sites)
inf('home    ', sites['home'],    do_calc_size_sites)
inf('school  ', sites['school'],  do_calc_size_sites)
inf('work    ', sites['work'],    do_calc_size_sites)

if do_calc_size_sites:
    print(f'sites  {Size.bytes2human(Size.get_size(sites))}')


# --- Select attributes (as in code above) ---
# hosp    : 67  (38.9K)
# home_gq : 195  (111.7K)
# home    : 533919  (298.9M)
# school  : 350  (195.9K)
# work    : 71137  (37.8M)
#
# sites  337.0M
# file  10.4M


# ----------------------------------------------------------------------------------------------------------------------
# (2) Groups:

# Number of agents:
#     SELECT COUNT(*) AS n FROM people
#     1,188,112  (100%)

# Number of agents who have a home:
#     SELECT COUNT(*) AS n FROM people WHERE sp_hh_id IS NOT NULL
#     1,188,112  (100%)

# Number of agents who attend school:
#     SELECT COUNT(*) AS n FROM people WHERE school_id IS NOT NULL
#     200,169  (16.85%)

# Number of agents who work:
#     SELECT COUNT(*) AS n FROM people WHERE work_id IS NOT NULL
#     586,012  (49.32%)

# Number of agents who attend school and work:
#     SELECT COUNT(*) AS n FROM people WHERE school_id IS NOT NULL AND work_id IS NOT NULL
#     10,681  (0.90%)

# Number of agents grouped by household:
#     SELECT COUNT(*) AS n, school_id FROM people WHERE school_id IS NOT NULL GROUP BY school_id ORDER BY n DESC
#     13	12080382
#     13	12083294
#     13	12085842
#     ...
#     533,919 rows  (44.94%)

# Number of agents grouped by school:
#     SELECT COUNT(*) AS n, sp_hh_id FROM people WHERE sp_hh_id IS NOT NULL GROUP BY sp_hh_id ORDER BY n DESC
#     9314	450066968.0
#     8949	450086847.0
#     3838	450067304.0
#     2746	450063600.0
#     2239	450109054.0
#     2070	450143076.0
#     1970	450107620.0
#     1954	450102513.0
#     1843	450140462.0
#     1815	450148402.0
#     ...
#     350 rows  (0.03%)

# Number of agents grouped by workplace:
#     SELECT COUNT(*) AS n, work_id FROM people WHERE work_id IS NOT NULL GROUP BY work_id ORDER BY n DESC
#     7917	513991798.0
#     7179	513990956.0
#     7159	513988354.0
#     6512	513988701.0
#     5899	513975710.0
#     5414	513977354.0
#     5126	513980750.0
#     4120	514020145.0
#     3724	513976064.0
#     3587	513991501.0
#     ...
#     71,137 rows  (5.99%)

# Number of agents grouped household and school:
#     SELECT COUNT(*) AS n, sp_hh_id, school_id FROM people WHERE sp_hh_id IS NOT NULL AND school_id IS NOT NULL GROUP BY sp_hh_id, school_id
#     1	11815531	450086847.0
#     1	11815536	450144545.0
#     ...
#     164,459  (13.84%)


if do_remove_file_groups and os.path.isfile(fpath_groups):
    os.remove(fpath_groups)

if os.path.isfile(fpath_groups):
    print('Loading groups... ', end='')
    with gzip.GzipFile(fpath_groups, 'rb') as f:
        gc.disable()
        groups = pickle.load(f)
        gc.enable()
    print('done.')
else:
    print('Generating groups... ', end='')
    groups = Group.gen_from_db(
        fpath_db,
        tbl='people',
        attr={},
        rel={},
        attr_db=[],
        rel_db=[
            GroupDBRelSpec('home', 'sp_hh_id', sites['home']),
            GroupDBRelSpec('school', 'school_id', sites['school'])
        ],
        rel_at='home'
    )
    print('done.')

    print('Saving groups... ', end='')
    with gzip.GzipFile(fpath_groups, 'wb') as f:
        pickle.dump(groups, f)
    print('done.')

inf('groups', groups, do_calc_size_groups)


# groups: 164,459  (175.2M)
# file  5.1M


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

n_schools = 8
few_schools = [sites['school'][k] for k in list(sites['school'].keys())[:n_schools]]

probe_grp_size_schools = GroupSizeProbe('school', [GroupQry(rel={ Site.AT: s }) for s in few_schools], msg_mode=ProbeMsgMode.DISP)

(Simulation().
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(GoToAndBackTimeAtRule()).
    add_probe(probe_grp_size_schools).
    add_groups(groups).
    run(10)
)
