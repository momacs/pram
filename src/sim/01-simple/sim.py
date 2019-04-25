'''
Several simple simulations testing various early-development aspects of PRAM.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 'rules' module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import AttrFluStage, Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import GoToRule, TimeInt
from pram.sim    import Simulation

from rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

sites = { 'home': Site('h'), 'work': Site('w') }

probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stage')
probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across sites')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Simulations testing the basic operations on groups and rules:

# (1.1) A single-group, single-rule (1g.1r) simulation:
s = Simulation()
s.add_rule(ProgressFluRule())
s.add_probe(probe_grp_size_flu)
s.add_group(Group('g0', 1000, { 'flu-stage': AttrFluStage.NO }))
s.run(24)

# (1.2) A single-group, two-rule (1g.2r) simulation:
# s = Simulation()
# s.add_rule(ProgressFluRule())
# s.add_rule(GoToRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.add_group(Group('g0', 1000, { 'flu-stage': AttrFluStage.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.run(24)

# (1.3) As above (1g.2r), but with reversed rule order (which should not, and does not, change the results):
# s = Simulation()
# s.add_rule(GoToRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_rule(ProgressFluRule())
# s.add_probe(probe_grp_size_flu)
# s.add_group(Group('g0', 1000, { 'flu-stage': AttrFluStage.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.run(24)

# (1.4) A two-group, two-rule (2g.2r) simulation (the groups don't interact via rules):
# s = Simulation()
# s.add_rule(ProgressFluRule())
# s.add_rule(GoToRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.add_group(Group('g0', 1000, attr={ 'flu-stage': AttrFluStage.NO }))
# s.add_group(Group('g1', 2000, rel={ Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.run(24)

# (1.5) A two-group, two-rule (2g.2r) simulation (the groups do interact via one rule):
# s = Simulation()
# s.add_rule(ProgressFluRule())
# s.add_rule(GoToRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_probe(probe_grp_size_flu)
# s.add_group(Group('g0', 1000, { 'flu-stage': AttrFluStage.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.add_group(Group('g1', 2000, { 'flu-stage': AttrFluStage.NO }))
# s.run(24)

# (1.6) A two-group, three-rule (2g.3r) simulation (same results, as expected):
# s = Simulation()
# s.add_rule(ProgressFluRule())
# s.add_rule(GoToRule(TimeInt(10,22), 0.4, 'home', 'work'))
# s.add_rule(GoToRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_flu)
# s.add_group(Group('g0', 1000, { 'flu-stage': AttrFluStage.NO }, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.add_group(Group('g1', 2000, { 'flu-stage': AttrFluStage.NO }, { Site.AT: sites['home'], 'home': sites['home'] }))
# s.run(24)


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulations testing rule interactions:

# (2.1) Antagonistic rules overlapping entirely in time (i.e., goto-home and goto-work; converges to a stable distribution):
# s = Simulation()
# s.add_rule(GoToRule(TimeInt(10,22), 0.4, 'home', 'work'))
# s.add_rule(GoToRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_site)
# s.add_group(Group('g0', 1000, {}, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.set_pragma_analyze(False)
# s.run(1).summary(False, end_line_cnt=(1,1))
# s.run(1).summary(False, end_line_cnt=(1,1))
# s.run(1).summary(False, end_line_cnt=(1,1))
# s.run(1).summary(False, end_line_cnt=(1,1))
# s.run(24)

# (2.2) Antagonistic rules overlapping mostly in time (i.e., goto-home and hoto-work; second rule "wins"):
# s = Simulation()
# s.add_rule(GoToRule(TimeInt(10,16), 0.4, 'home', 'work'))
# s.add_rule(GoToRule(TimeInt(10,22), 0.4, 'work', 'home'))
# s.add_probe(probe_grp_size_site)
# s.add_group(Group('g0', 1000, {}, { Site.AT: sites['home'], 'home': sites['home'], 'work': sites['work'] }))
# s.run(24)
