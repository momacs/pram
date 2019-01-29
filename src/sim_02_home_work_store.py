'''
A simulation of agents going from home to work and then, sometimes, to a store and back home again.
'''


from pram.sim import Simulation
from pram.entity import GroupQry, Site
from pram.data import GroupSizeProbe
from pram.rule import AttrFluStatus, TimeInt, GotoRule


rand_seed = 1928


# ----------------------------------------------------------------------------------------------------------------------
sites = {
    'home'    : Site('home'),
    'work-a'  : Site('work-a'),
    'work-b'  : Site('work-b'),
    'work-c'  : Site('work-c'),
    'store-a' : Site('store-a'),
    'store-b' : Site('store-b')
}

probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), memo='Mass distribution across sites')

(Simulation(6,1,24, rand_seed=rand_seed).
    new_group('g0', 1000).
        set_attr('flu-status', AttrFluStatus.no).
        set_rel(Site.AT, sites['home']).
        set_rel('home',  sites['home']).
        set_rel('work',  sites['work-a']).
        set_rel('store', sites['store-a']).
        commit().
    new_group('g1', 1000).
        set_attr('flu-status', AttrFluStatus.no).
        set_rel(Site.AT, sites['home']).
        set_rel('home',  sites['home']).
        set_rel('work',  sites['work-b']).
        set_rel('store', sites['store-b']).
        commit().
    new_group('g2', 100).
        set_attr('flu-status', AttrFluStatus.no).
        set_rel(Site.AT, sites['home']).
        set_rel('home',  sites['home']).
        set_rel('work',  sites['work-c']).
        commit().
    add_rule(GotoRule(TimeInt( 8,12), 0.4, 'home',  'work',  'Some agents leave home to go to work')).
    add_rule(GotoRule(TimeInt(16,20), 0.4, 'work',  'home',  'Some agents return home from work')).
    add_rule(GotoRule(TimeInt(16,21), 0.2, 'home',  'store', 'Some agents go to a store after getting back home')).
    add_rule(GotoRule(TimeInt(17,23), 0.3, 'store', 'home',  'Some shopping agents return home from a store')).
    add_rule(GotoRule(TimeInt(24,24), 1.0, 'store', 'home',  'All shopping agents return home after stores close')).
    add_rule(GotoRule(TimeInt( 2, 2), 1.0, None,    'home',  'All still-working agents return home')).
    add_probe(probe_grp_size_site).
    summary((True, True, True, True, True), (0,1)).
    run().
    summary((False, True, False, False, False), (1,1)).
    run(4)
)
