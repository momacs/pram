'''
A simulation of the flu transmission model (testing a location-awareness).
'''

from pram.sim    import Simulation
from pram.entity import AttrFluStage, GroupQry, GroupSplitSpec, Site
from pram.data   import GroupSizeProbe
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint


rand_seed = 1928


# ----------------------------------------------------------------------------------------------------------------------
sites = {
    'home'     : Site('home'),
    'school-a' : Site('school-a'),
    'school-b' : Site('school-b')
}

probe_grp_size_flu  = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, memo='Mass distribution across flu stages')
probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), memo='Mass distribution across sites')

(Simulation(1,1,14, rand_seed=rand_seed).  # 14-day simulation
    new_group('A', 500).
        set_attr('flu-stage', AttrFluStage.NO).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-a']).
        commit().
    new_group('B', 500).
        set_attr('flu-stage', AttrFluStage.NO).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-b']).
        commit().
    add_rule(GotoSchoolFluRule()).
    add_rule(GotoHomeFluRule()).
    add_rule(ProgressFlu02Rule()).
    # add_probe(probe_grp_size_flu).
    add_probe(probe_grp_size_site).
    # summary((False, True, False, False, False), (0,1)).
    # run(1).summary((False, True, False, False, False), (1,1)).
    # run(1).summary((False, True, False, False, False), (1,1)).
    # run(1).summary((False, True, False, False, False), (1,1)).
    # run(1).summary((False, True, False, False, False), (1,1))
    # run().summary((False, True, False, False, False), (1,1))
    run()
)
