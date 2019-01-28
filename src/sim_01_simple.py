'''
Several simple simulations testing various aspects of PRAM.
'''


from pram.sim import Simulation
from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.data import GroupSizeProbe
from pram.rule import AttrFluStatus, GotoRule, Rule, TimeInt


rand_seed = 1928


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,20), memo=None):  # 8am - 8pm
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, t):
        if not self.is_applicable(group, t): return None

        p_infection = 0.05

        if group.get_attr('flu-status') == AttrFluStatus.no:
            return [
                GroupSplitSpec(p=p_infection,     attr_set={ 'flu-status': AttrFluStatus.asympt }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu-status': AttrFluStatus.no     })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.asympt:
            return [
                GroupSplitSpec(p=0.80, attr_set={ 'flu-status': AttrFluStatus.sympt }),
                GroupSplitSpec(p=0.20, attr_set={ 'flu-status': AttrFluStatus.no    })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.sympt:
            return [
                GroupSplitSpec(p=0.20, attr_set={ 'flu-status': AttrFluStatus.asympt }),
                GroupSplitSpec(p=0.75, attr_set={ 'flu-status': AttrFluStatus.sympt  }),
                GroupSplitSpec(p=0.05, attr_set={ 'flu-status': AttrFluStatus.no     })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-status'.")

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_attr([ 'flu-status' ])


# ----------------------------------------------------------------------------------------------------------------------
# (1) Simulations testing the basic operations on groups and rules:

sites = { 'home': Site('home'), 'work': Site('work-a') }

probe_grp_size_flu = GroupSizeProbe('flu', [GroupQry(attr={ 'flu-status': fs }) for fs in AttrFluStatus])

# (1.1) A single-group, single-rule (1g.1r) simulation:
s = Simulation(6,1,16, rand_seed=rand_seed)
s.create_group(1000, { 'flu-status': AttrFluStatus.no }, {})
s.add_rule(ProgressFluRule())
s.add_probe(probe_grp_size_flu)
s.run()

# (1.2) A single-group, two-rule (1g.2r) simulation:
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.3) As above (1g.2r), but with reversed rule order (which should not, and does not, change the results):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_rule(ProgressFluRule())
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.4) A two-group, two-rule (2g.2r) simulation (the groups don't interact via rules):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, attr={ 'flu-status': AttrFluStatus.no })
# s.create_group(2000, rel={ Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.5) A two-group, two-rule (2g.2r) simulation (the groups do interact via one rule):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.create_group(2000, { 'flu-status': AttrFluStatus.no })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.6) A two-group, three-rule (2g.3r) simulation (same results, as expected):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.create_group(2000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash() })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'home', 'work'))
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_flu)
# s.run()


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulations testing rule interactions:

# sites = { 'home': Site('home'), 'work': Site('work-a') }
#
# probe_grp_size_flu = GroupSizeProbe('flu', [GroupQry(attr={ 'flu-status': fs }) for fs in AttrFluStatus])
# probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])

# (2.1) Antagonistic rules overlapping entirely in time (i.e., goto-home and goto-work; converges to a stable distribution):
# s = Simulation(9,1,14, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, {}, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'home', 'work'))
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_loc)
# # s.run(1).summary((False, True, False, False, False), (1,1))
# # s.run(1).summary((False, True, False, False, False), (1,1))
# # s.run(1).summary((False, True, False, False, False), (1,1))
# # s.run(1).summary((False, True, False, False, False), (1,1))
# s.run()

# (2.2) Antagonistic rules overlapping mostly in time (i.e., goto-home and hoto-work; second rule "wins"):
# s = Simulation(9,1,14, rand_seed=rand_seed)
# s.add_sites(sites.values())
# s.create_group(1000, {}, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_loc)
# s.run()


# ----------------------------------------------------------------------------------------------------------------------
# (3) A multi-group, multi-rule, multi-site simulations:

# sites = {
#     'home'    : Site('home'),
#     'work-a'  : Site('work-a'),
#     'work-b'  : Site('work-b'),
#     'work-c'  : Site('work-c'),
#     'store-a' : Site('store-a'),
#     'store-b' : Site('store-b')
# }
#
# probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])
#
# (Simulation(6,1,24, rand_seed=rand_seed).
#     add_sites(sites.values()).
#     new_group('g0', 1000).
#         set_attr('flu-status', AttrFluStatus.no).
#         set_rel(Site.AT, sites['home'].get_hash()).
#         set_rel('home',  sites['home'].get_hash()).
#         set_rel('work',  sites['work-a'].get_hash()).
#         set_rel('store', sites['store-a'].get_hash()).
#         commit().
#     new_group('g1', 1000).
#         set_attr('flu-status', AttrFluStatus.no).
#         set_rel(Site.AT, sites['home'].get_hash()).
#         set_rel('home',  sites['home'].get_hash()).
#         set_rel('work',  sites['work-b'].get_hash()).
#         set_rel('store', sites['store-b'].get_hash()).
#         commit().
#     new_group('g2', 100).
#         set_attr('flu-status', AttrFluStatus.no).
#         set_rel(Site.AT, sites['home'].get_hash()).
#         set_rel('home',  sites['home'].get_hash()).
#         set_rel('work',  sites['work-c'].get_hash()).
#         commit().
#     add_rule(GotoRule(TimeInt( 8,12), 0.4, 'home',  'work',  'some agents leave home to go to work')).
#     add_rule(GotoRule(TimeInt(16,20), 0.4, 'work',  'home',  'some agents return home from work')).
#     add_rule(GotoRule(TimeInt(16,21), 0.2, 'home',  'store', 'some agents go to a store after getting back home')).
#     add_rule(GotoRule(TimeInt(17,23), 0.3, 'store', 'home',  'some shopping agents return home from a store')).
#     add_rule(GotoRule(TimeInt(24,24), 1.0, 'store', 'home',  'all shopping agents return home after stores close')).
#     add_rule(GotoRule(TimeInt( 2, 2), 1.0, None,    'home',  'all still-working agents return home')).
#     add_probe(probe_grp_size_loc).
#     summary((False, True, False, False, False), (0,1)).
#     run().
#     run(4))
