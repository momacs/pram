'''
Tests of simulation construction exceptions associated with (1) improper order of adding rules and groups and
(2) superfluous attributes or relations.

Based on the simulation from: sim/04-flu-prog/sim.py
'''

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import AttrFluStage, Group, GroupSplitSpec, Site
from pram.rule   import GoToRule, GoToAndBackTimeAtRule, TimeInt
from pram.sim    import Simulation, SimulationConstructionError, SimulationConstructionWarning, TimeHour

from rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
sites = { 'home': Site('h'), 'work': Site('w') }

probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')


# ----------------------------------------------------------------------------------------------------------------------
# (1) A proper simulation:

print('(1)')
s = (Simulation(TimeHour(6), 16).
    add_rule(ProgressFluRule()).
    add_rule(GoToAndBackTimeAtRule()).
    add_probe(probe_grp_size_flu).
    new_group('0', 1000).
        set_attr('flu-stage', AttrFluStage.NO).
        commit()
)

print(f'Attributes : {s.last_run.attr_used}')
print(f'Relations  : {s.last_run.rel_used}')

print()


# ----------------------------------------------------------------------------------------------------------------------
# (2) Improper simulations (rendered proper automatically):

# (2.1) A group has superfluous attributes and relations, but automatic group pruning saved the day:
print('(2.1)')
(Simulation(TimeHour(6), 16).
    set_pragma_autoprune_groups(True).
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    new_group('0', 1000).
        set_attr('flu-stage', AttrFluStage.NO).
        set_attr('age', 20).             # superfluous attribute
        set_rel('home', sites['home']).  # superfluous relation
        commit().
    summary(False, 8,0,0,0).             # confirm that the group has been pruned
    run()                                # confirm that the simulation runs
)
print()

# (2.2) Like the simulation above, but two groups are added.  These two groups, when pruned, turn out identical;
#       consequently, agent population mass gets automatically cumulated in the only resulting group.
print('(2.2)')
(Simulation(TimeHour(6), 16).
    set_pragma_autoprune_groups(True).
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    new_group('0', 500).
        set_attr('flu-stage', AttrFluStage.NO).
        set_attr('age', 20).             # superfluous attribute
        commit().
    new_group('1', 500).
        set_attr('flu-stage', AttrFluStage.NO).
        set_rel('home', sites['home']).  # superfluous relation
        commit().
    summary(False, 8,0,0,0).             # confirm that only one pruned group exists
    run()                                # confirm that the simulation runs
)
print()


# ----------------------------------------------------------------------------------------------------------------------
# (3) Improper simulations (that will not run):

# (3.1) Problem: Attempting to add a rule after having added a group:
print('(3.1)')
try:
    (Simulation(TimeHour(6), 16).
        add_rule(ProgressFluRule()).
        add_probe(probe_grp_size_flu).
        new_group('0', 1000).
            set_attr('flu-stage', AttrFluStage.NO).
            commit().
        add_rule(GoToRule(TimeInt(10,16), 0.4, 'home', 'work'))
    )
except SimulationConstructionError as e:
    print(e)
print()

# (3.2) Problem: Attempting to add a group with no rules present:
print('(3.2)')
try:
    (Simulation(TimeHour(6), 16).
        new_group('0', 1000).
            set_attr('flu-stage', AttrFluStage.NO).
            commit().
        add_rule(ProgressFluRule()).
        add_probe(probe_grp_size_flu)
    )
except SimulationConstructionError as e:
    print(e)
print()

# (3.3) Problem: A group has superfluous attributes and relations:
print('(3.3)')
try:
    (Simulation(TimeHour(6), 16).
        add_rule(ProgressFluRule()).
        add_probe(probe_grp_size_flu).
        new_group('0', 1000).
            set_attr('flu-stage', AttrFluStage.NO).
            set_attr('age', 20).             # superfluous attribute
            set_rel('home', sites['home']).  # superfluous relation
            commit().
        run()  # an attempt to run the simulation forces the warning to manifest itself
    )
except SimulationConstructionWarning as w:
    print(w)
