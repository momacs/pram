'''
A simulation testing a group-state-aware school attending rule (AttendSchoolRule; defined elsewhere).
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint
from pram.sim    import Simulation

from rules import ResetSchoolDayRule, AttendSchoolRule


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

sites = {
    'home'     : Site('home'),
    'school-a' : Site('school-a'),
    'school-b' : Site('school-b')
}

probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across sites')

# (1) Run the simulation without interruptions:
s = (Simulation(7,1,10, rand_seed=rand_seed).
    add_rule(ResetSchoolDayRule(TimePoint(7))).
    add_rule(AttendSchoolRule()).
    add_probe(probe_grp_size_site).
    new_group('0', 500).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-a']).
        commit().
    new_group('1', 500).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-b']).
        commit().
    run()
    # .summary(False, 64,0,0,0, (0,1))  # reveal groups created during the simulation
)

# (2) Run the simulation revealing mass distribution detail:
# (Simulation(7,1,10, rand_seed=rand_seed).
#     add_rule(ResetSchoolDayRule(TimePoint(7))).
#     add_rule(AttendSchoolRule()).
#     add_probe(probe_grp_size_site).
#     new_group('0', 500).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-a']).
#         commit().
#     new_group('1', 500).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-b']).
#         commit().
#     set_pragma_analyze(False).
#            summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(5).summary(False, 64,0,0,0, (0,1))
# )
