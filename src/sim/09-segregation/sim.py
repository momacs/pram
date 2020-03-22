'''
Segregation model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import random

from scipy.stats import poisson

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import SegregationModel
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
# (1) Simulation (two locations)

loc = [Site('a'), Site('b')]

probe_loc = GroupSizeProbe.by_rel('loc', Site.AT, loc, msg_mode=ProbeMsgMode.DISP)
probe_sim = GroupSizeProbe(
    name='sim',
    queries=[
        GroupQry(attr={ 'team': 'blue' }, rel={ Site.AT: loc[0] }),
        GroupQry(attr={ 'team': 'red'  }, rel={ Site.AT: loc[0] }),
        GroupQry(attr={ 'team': 'blue' }, rel={ Site.AT: loc[1] }),
        GroupQry(attr={ 'team': 'red'  }, rel={ Site.AT: loc[1] })
    ],
    qry_tot=None,
    msg_mode=ProbeMsgMode.DISP
)

(Simulation().
    set().
        pragma_autocompact(True).
        pragma_live_info(False).
        done().
    add([
        SegregationModel('team', len(loc)),
        Group(m=200, attr={ 'team': 'blue' }, rel={ Site.AT: loc[0] }),
        Group(m=300, attr={ 'team': 'blue' }, rel={ Site.AT: loc[1] }),
        Group(m=100, attr={ 'team': 'red'  }, rel={ Site.AT: loc[0] }),
        Group(m=400, attr={ 'team': 'red'  }, rel={ Site.AT: loc[1] }),
        probe_loc  # the distribution should tend to 50%-50%
        # probe_sim  # mass should tend to move towards two of the four sites
    ]).
    run(100)
)


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation (arbitrary numer of locations)

# loc = [Site('a'), Site('b'), Site('c')]
#
# probe_loc = GroupSizeProbe.by_rel('site', Site.AT, loc, msg_mode=ProbeMsgMode.DISP)
#
# def grp_setup(pop, group):
#     p = [random.random() for _ in range(len(loc))]
#     p = [i / sum(p) for i in p]
#     return [GroupSplitSpec(p=p[i], rel_set={ Site.AT: loc[i] }) for i in range(len(loc))]
#
# (Simulation().
#     set().
#         # rand_seed(1928).
#         pragma_autocompact(True).
#         pragma_live_info(False).
#         fn_group_setup(grp_setup).
#         done().
#     add([
#         SegregationModel('team', len(loc)),
#         Group(n=500, attr={ 'team': 'blue' }),
#         Group(n=500, attr={ 'team': 'red'  }),
#         probe_loc
#     ]).
#     run(10)
# )


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation (synthetic Allegheny County population)

# fpath_db = os.path.join(os.path.dirname(__file__), 'db', 'allegheny-students.sqlite3')
#
# # sites = [Site(450066968), Site(450086847), Site(450066968)]
# # probe_loc = GroupSizeProbe.by_rel('site', Site.AT, sites, msg_mode=ProbeMsgMode.DISP)
#
# sites = [Site(450066968)]
# probe_loc = GroupSizeProbe(
#     name='sex',
#     queries=[
#         GroupQry(attr={ 'sex': 'F' }, rel={ Site.AT: Site(450066968) }),
#         GroupQry(attr={ 'sex': 'M' }, rel={ Site.AT: Site(450066968) })
#     ],
#     qry_tot=GroupQry(rel={ Site.AT: Site(450066968) }),
#     msg_mode=ProbeMsgMode.DISP
# )
#
# (Simulation().
#     set().
#         # rand_seed(1928).
#         pragma_autocompact(True).
#         pragma_live_info(False).
#         done().
#     add().
#         rule(SegregationModel('sex', len(['F','M']))).
#         # rule(SegregationModel('race', len([1,2,3,5,6,7,8,9]))).
#         probe(probe_loc).
#         done().
#     db(fpath_db).
#         gen_groups(
#             tbl      = 'students',
#             attr_db  = ['sex'],
#             # attr_db  = ['race'],
#             rel_db   = [GroupDBRelSpec(name=Site.AT, col='school_id')],
#             attr_fix = {},
#             rel_fix  = {}
#         ).
#         done().
#     run(5)
#     # summary(False, 1024,0,0,0, (1,0))
# )
