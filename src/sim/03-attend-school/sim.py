'''
A simulation testing a group-state-aware school attending system.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimePoint
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
sites = { s:Site(s) for s in ['home', 'school-a', 'school-b']}

probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across sites')

# (1) Run the simulation without interruptions:
(Simulation().
    # set_pragma_live_info(True).
    add().
        rule(ResetSchoolDayRule(TimePoint(7))).
        rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
        probe(probe_grp_size_site).
        done().
    new_group(500).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-a']).
        done().
    new_group(500).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-b']).
        done().
    run(18)
    # .summary(False, 64,0,0,0, (0,1))  # reveal groups created during the simulation
)

# (2) Run the simulation revealing mass distribution detail:
# (Simulation().
#     add_rule(ResetSchoolDayRule(TimePoint(7))).
#     add_rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
#     add_probe(probe_grp_size_site).
#     new_group(500).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-a']).
#         done().
#     new_group(500).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-b']).
#         done().
#     set_pragma_analyze(False).
#            summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(1).summary(False, 64,0,0,0, (0,1)).
#     run(5).summary(False, 64,0,0,0, (0,1))
# )
