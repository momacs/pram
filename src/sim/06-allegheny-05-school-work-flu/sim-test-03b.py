'''
Simple flu and school model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import pram.util as util

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistenceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, Site
from pram.rule   import SimpleFluLocationRule, SimpleFluProgressRule, SimpleFluProgressMoodRule
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

fpath_db_out = os.path.join(os.path.dirname(__file__), 'sim-test-03b.sqlite3')

if os.path.isfile(fpath_db_out):
    os.remove(fpath_db_out)

pp = ProbePersistenceDB(fpath_db_out)

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
    persistence=pp,
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
    persistence=pp,
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
        rule(SimpleFluProgressRule()).
        # rule(SimpleFluProgressMoodRule()).
        rule(SimpleFluLocationRule()).
        probe(probe_grp_size_flu_school_l).
        probe(probe_grp_size_flu_school_m).
        done().
    add_group(Group(m=450, attr={ 'flu': 's', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group(m= 50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group(m=450, attr={ 'flu': 's', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group(m= 50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home})).
    add_group(Group(m=450, attr={ 'flu': 's', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    add_group(Group(m= 50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    add_group(Group(m=450, attr={ 'flu': 's', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    add_group(Group(m= 50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})).
    run(10).
    summary(True, 0,0,0,0, (1,0))
)
