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

        if group.get_attr('flu-status') == AttrFluStatus.NO:
            return [
                GroupSplitSpec(p=p_infection,     attr_set={ 'flu-status': AttrFluStatus.ASYMPT }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu-status': AttrFluStatus.NO     })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.ASYMPT:
            return [
                GroupSplitSpec(p=0.80, attr_set={ 'flu-status': AttrFluStatus.SYMPT }),
                GroupSplitSpec(p=0.20, attr_set={ 'flu-status': AttrFluStatus.NO    })
            ]
        elif group.get_attr('flu-status') == AttrFluStatus.SYMPT:
            return [
                GroupSplitSpec(p=0.20, attr_set={ 'flu-status': AttrFluStatus.ASYMPT }),
                GroupSplitSpec(p=0.75, attr_set={ 'flu-status': AttrFluStatus.SYMPT  }),
                GroupSplitSpec(p=0.05, attr_set={ 'flu-status': AttrFluStatus.NO     })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-status'.")

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_attr([ 'flu-status' ])


# ----------------------------------------------------------------------------------------------------------------------
# (1) Simulations testing the basic operations on groups and rules:

sites = {
    'home': Site('h'),
    'work': Site('w')
}

probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-status', AttrFluStatus, memo='Mass distribution across flu status')

# (1.1) A single-group, single-rule (1g.1r) simulation:
s = Simulation(6,1,16, rand_seed=rand_seed)
s.create_group(1000, { 'flu-status': AttrFluStatus.NO }, {})
s.add_rule(ProgressFluRule())
s.add_probe(probe_grp_size_flu)
s.run()

# (1.2) A single-group, two-rule (1g.2r) simulation:
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.create_group(1000, { 'flu-status': AttrFluStatus.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.3) As above (1g.2r), but with reversed rule order (which should not, and does not, change the results):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.create_group(1000, { 'flu-status': AttrFluStatus.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_rule(ProgressFluRule())
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.4) A two-group, two-rule (2g.2r) simulation (the groups don't interact via rules):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.create_group(1000, attr={ 'flu-status': AttrFluStatus.NO })
# s.create_group(2000, rel={ Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.5) A two-group, two-rule (2g.2r) simulation (the groups do interact via one rule):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.create_group(1000, { 'flu-status': AttrFluStatus.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.create_group(2000, { 'flu-status': AttrFluStatus.NO })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.run()

# (1.6) A two-group, three-rule (2g.3r) simulation (same results, as expected):
# s = Simulation(6,1,16, rand_seed=rand_seed)
# s.create_group(1000, { 'flu-status': AttrFluStatus.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.create_group(2000, { 'flu-status': AttrFluStatus.NO }, { Site.AT: sites['home'], 'home': sites['home'] })
# s.add_rule(ProgressFluRule())
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'home', 'work'))
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_flu)
# s.run()


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulations testing rule interactions:

# sites = {
#     'home': Site('h'),
#     'work': Site('w')
# }
#
# probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-status', AttrFluStatus, memo='Mass distribution across flu status')
# probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), memo='Mass distribution across sites')

# (2.1) Antagonistic rules overlapping entirely in time (i.e., goto-home and goto-work; converges to a stable distribution):
# s = Simulation(9,1,14, rand_seed=rand_seed)
# s.create_group(1000, {}, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'home', 'work'))
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_site)
# # s.run(1).summary((False, True, False, False, False), (1,1))
# # s.run(1).summary((False, True, False, False, False), (1,1))
# # s.run(1).summary((False, True, False, False, False), (1,1))
# # s.run(1).summary((False, True, False, False, False), (1,1))
# s.run()

# (2.2) Antagonistic rules overlapping mostly in time (i.e., goto-home and hoto-work; second rule "wins"):
# s = Simulation(9,1,14, rand_seed=rand_seed)
# s.create_group(1000, {}, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] })
# s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_site)
# s.run()
