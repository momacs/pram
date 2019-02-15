import gc
import gzip
import os
import pickle
import sys

from pram.data   import GroupSizeProbe
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint
from pram.sim    import Simulation
from pram.util   import Size


# ----------------------------------------------------------------------------------------------------------------------
def inf(name, o, do_calc_size=False):
    print(f'{name}: {len(o)}  ({Size.bytes2human(Size.get_size(o)) if do_calc_size else "."})')


# ----------------------------------------------------------------------------------------------------------------------
class ResetDayRule(Rule):
    __slots__ = ()

    def __init__(self, t, memo=None):
        super().__init__('reset-day', t, memo)

    def apply(self, pop, group, t):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],

    def is_applicable(self, group, t):
        return super().is_applicable(t)


# ----------------------------------------------------------------------------------------------------------------------
class AttendSchoolRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,16), memo=None):
        super().__init__('attend-school', t, memo)

    def apply(self, pop, group, t):
        if group.has_rel({ Site.AT: group.get_rel('home') }) and (not group.has_attr('did-attend-school-today') or group.has_attr({ 'did-attend-school-today': False })):
            return self.apply_at_home(group, t)

        if group.has_rel({ Site.AT:  group.get_rel('school') }):
            return self.apply_at_school(group, t)

    def apply_at_home(self, group, t):
        p = { 8:0.50, 9:0.50, 10:0.50, 11:0.50, 12:1.00 }.get(t, 0.00)  # TODO: Provide these as a CDF
            # prob of going to school = f(time of day)

        return [
            GroupSplitSpec(p=p, attr_set={ 'did-attend-school-today': True, 't-at-school': 0 }, rel_set={ Site.AT: group.get_rel('school') }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_at_school(self, group, t):
        t_at_school = group.get_attr('t-at-school')
        p = { 0: 0.00, 1:0.05, 2:0.05, 3:0.25, 4:0.50, 5:0.70, 6:0.80, 7:0.90, 8:1.00 }.get(t_at_school, 1.00) if t < self.t.t1 else 1.00
            # prob of going home = f(time spent at school)

        return [
            GroupSplitSpec(p=p, attr_set={ 't-at-school': (t_at_school + 1) }, rel_set={ Site.AT: group.get_rel('home') }),
            GroupSplitSpec(p=1 - p, attr_set={ 't-at-school': (t_at_school + 1) })
        ]

        # TODO: Give timer information to the rule so it can appropriate determine time passage (e.g., to add it to 't-at-school' above).

    def is_applicable(self, group, t):
        return (
            super().is_applicable(t) and
            # group.has_attr({ 'is-student': True }) and
            group.has_rel(['home', 'school']))

    @staticmethod
    def setup(pop, group):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

rand_seed = 1928

dpath_res    = os.path.join(os.sep, 'Volumes', 'd', 'pitt', 'sci', 'pram', 'res', 'fred')
fpath_db     = os.path.join(dpath_res, 'flu.sqlite3')
fpath_sites  = os.path.join(dpath_res, 'flu-sites.pickle.gz')
fpath_groups = os.path.join(dpath_res, 'flu-groups.pickle.gz')

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
        attr=[],
        rel=[
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

probe_grp_size_schools = GroupSizeProbe('school', [GroupQry(rel={ Site.AT: v }) for v in few_schools])

# sys.exit(0)

(Simulation(6,1,16, rand_seed=rand_seed).
    add_groups(groups).
    add_rule(ResetDayRule(TimePoint(7))).
    add_rule(AttendSchoolRule()).
    add_probe(probe_grp_size_schools).
    run()
)
