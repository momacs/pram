''' '''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimePoint
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
sites = {s:Site(s) for s in ['home', 'school-a', 'school-b']}

probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across sites')

# s = (Simulation().
#     add().
#         rule(ResetSchoolDayRule(TimePoint(7))).
#         rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
#         probe(probe_grp_size_site).
#         done().
#     new_group(500).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-a']).
#         done().
#     new_group(500).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-b']).
#         done()
# )

s = (Simulation().
        set().
            pragma_autocompact(True).
            pragma_live_info(True).
            pragma_live_info_ts(False).
            done()
)
s.get_state()
