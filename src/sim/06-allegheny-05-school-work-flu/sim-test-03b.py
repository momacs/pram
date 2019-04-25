'''
Simple flu and school model.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import pram.util as util

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, Site
from pram.rule   import FluLocationRule, ProgressFluSimpleRule
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

fpath_db = os.path.join(os.path.dirname(__file__), 'out-test-03b.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

pp = ProbePersistanceDB(fpath_db)

home     = Site('home')
school_l = Site('school-l')
school_m = Site('school-m')

probe_grp_size_flu_school_l = GroupSizeProbe(
    name='flu-school-l',
    queries=[
        GroupQry(attr={ 'flu': 's' }, rel={ 'school': school_l }),
        GroupQry(attr={ 'flu': 'e' }, rel={ 'school': school_l }),
        GroupQry(attr={ 'flu': 'r' }, rel={ 'school': school_l })
    ],
    qry_tot=GroupQry(rel={ 'school': school_l }),
    persistance=pp,
    var_names=['ps', 'pe', 'pr', 'ns', 'ne', 'nr'],
    memo='Population size across flu states for low-income school'
)

probe_grp_size_flu_school_m = GroupSizeProbe(
    name='flu-school-m',
    queries=[
        GroupQry(attr={ 'flu': 's' }, rel={ 'school': school_m }),
        GroupQry(attr={ 'flu': 'e' }, rel={ 'school': school_m }),
        GroupQry(attr={ 'flu': 'r' }, rel={ 'school': school_m })
    ],
    qry_tot=GroupQry(rel={ 'school': school_m }),
    persistance=pp,
    var_names=['ps', 'pe', 'pr', 'ns', 'ne', 'nr'],
    memo='Population size across flu states for medium-income school'
)

(Simulation().
    set().
        rand_seed(rand_seed).
        pragma_autocompact(True).
        pragma_live_info(True).
        pragma_live_info_ts(False).
        pragma_rule_analysis_for_db_gen(True).
        done().
    add().
        rule(ProgressFluSimpleRule()).
        rule(FluLocationRule()).
        probe(probe_grp_size_flu_school_l).
        probe(probe_grp_size_flu_school_m).
        done().
    add_group(Group('g1', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group('g2',  50, attr={ 'flu': 'e', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group('g3', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group('g4',  50, attr={ 'flu': 'e', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group('g5', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    add_group(Group('g6',  50, attr={ 'flu': 'e', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    add_group(Group('g7', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    add_group(Group('g8',  50, attr={ 'flu': 'e', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    run(10).
    summary(True, 0,0,0,0, (1,0))
)
