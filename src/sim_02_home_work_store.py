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

probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])

(Simulation(6,1,24, rand_seed=rand_seed).
    add_sites(sites.values()).
    new_group('g0', 1000).
        set_attr('flu-status', AttrFluStatus.no).
        set_rel(Site.AT, sites['home'].get_hash()).
        set_rel('home',  sites['home'].get_hash()).
        set_rel('work',  sites['work-a'].get_hash()).
        set_rel('store', sites['store-a'].get_hash()).
        commit().
    new_group('g1', 1000).
        set_attr('flu-status', AttrFluStatus.no).
        set_rel(Site.AT, sites['home'].get_hash()).
        set_rel('home',  sites['home'].get_hash()).
        set_rel('work',  sites['work-b'].get_hash()).
        set_rel('store', sites['store-b'].get_hash()).
        commit().
    new_group('g2', 100).
        set_attr('flu-status', AttrFluStatus.no).
        set_rel(Site.AT, sites['home'].get_hash()).
        set_rel('home',  sites['home'].get_hash()).
        set_rel('work',  sites['work-c'].get_hash()).
        commit().
    add_rule(GotoRule(TimeInt( 8,12), 0.4, 'home',  'work',  'some agents leave home to go to work')).
    add_rule(GotoRule(TimeInt(16,20), 0.4, 'work',  'home',  'some agents return home from work')).
    add_rule(GotoRule(TimeInt(16,21), 0.2, 'home',  'store', 'some agents go to a store after getting back home')).
    add_rule(GotoRule(TimeInt(17,23), 0.3, 'store', 'home',  'some shopping agents return home from a store')).
    add_rule(GotoRule(TimeInt(24,24), 1.0, 'store', 'home',  'all shopping agents return home after stores close')).
    add_rule(GotoRule(TimeInt( 2, 2), 1.0, None,    'home',  'all still-working agents return home')).
    add_probe(probe_grp_size_loc).
    summary((True, True, True, True, True), (0,1)).
    run().
    summary((False, True, False, False, False), (1,0))
)
