# A predator-prey model of mice and hawks

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 'rules' module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, GoToRule, TimeInt
from pram.sim    import Simulation
from pram.util   import Time

# Constants taken from Czajkowsk/Schillak paper 'Iterative Solutions of the Lotka-Volterra Equations
# MICE_REPRODUCTION_RATE changed from 1 to 0.75 to show a better curve; this is likely only necessary because of the REPRODUCTION_PER_PREY problem
MICE_REPRODUCTION_RATE = 0.75
PREDATION_COEFFICIENT = 0.25
#PROBLEM: This is not possible to clearly model in pram, since we recognize all mass as homogeneous
# this simulation, due to the semantics defined below for what units of carbon represent, recognizes this constant to basically be 0.2, since one hawk is 5x the carbon of one mouse
REPRODUCTION_PER_PREY = 2
PREDATOR_MORTALITY = 0.4

TIME_COEFFICIENT = 1

class MiceBirthRule(Rule):

    def __init__(self, t=TimeInt(0,12), memo=None):
        super().__init__('mice-birth', t, memo)
        self.T_UNIT_MS = Time.MS.M

    def apply(self, pop, group, iter, t):

        if group.attr['form'] == 'c':

            mouse_group = pop.get_group(GroupQry({'form' : "m"}))

            p = (mouse_group.n*MICE_REPRODUCTION_RATE)/group.n #mice_group_size/pop_size

            p = p * TIME_COEFFICIENT

            return [
                GroupSplitSpec(p=p, attr_set={'form': "m"}),
                GroupSplitSpec(p=1 - p)
            ]

        else:
            return [GroupSplitSpec(p=1)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.n > 0


class PredationRule(Rule):

    def __init__(self, t=TimeInt(0,12), memo=None):
        super().__init__('hawk-eats-mice', t, memo)
        self.T_UNIT_MS = Time.MS.M

    def apply(self, pop, group, iter, t):
        mouse_group = pop.get_group(GroupQry({'form' : "m"}))
        mouse_population = mouse_group.n

        hawk_group = pop.get_group(GroupQry({'form' : 'h'}))
        hawk_population = hawk_group.n/5

        appetite = mouse_population*hawk_population*PREDATION_COEFFICIENT

        if group.attr['form'] == 'h':
            p = PREDATOR_MORTALITY

            p = p * TIME_COEFFICIENT

            return [
                GroupSplitSpec(p=p, attr_set={'form': "c"}),
                GroupSplitSpec(p=1 - p)
            ]

        elif group.attr['form'] == 'm':
            if appetite > mouse_population:
                appetite = mouse_population

            p = appetite/mouse_population

            p = p * TIME_COEFFICIENT

            return [
                GroupSplitSpec(p=p, attr_set={'form': "h"}),
                GroupSplitSpec(p=1 - p)
            ]
        else:
            return [GroupSplitSpec(p=1)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.n > 0 and (group.attr['form'] == 'h' or group.attr['form'] == 'm')



class CompetingPredationRule(Rule):

    def __init__(self, t=TimeInt(0,12), memo=None):
        super().__init__('snake-eats-mice', t, memo)
        self.T_UNIT_MS = Time.MS.M

    def apply(self, pop, group, iter, t):
        mouse_group = pop.get_group(GroupQry({'form' : "m"}))
        mouse_population = mouse_group.n

        snake_group = pop.get_group(GroupQry({'form' : 's'}))
        snake_population = snake_group.n/5

        appetite = mouse_population*snake_population*PREDATION_COEFFICIENT

        if group.attr['form'] == 's':
            p = PREDATOR_MORTALITY

            p = p * TIME_COEFFICIENT

            return [
                GroupSplitSpec(p=p, attr_set={'form': "c"}),
                GroupSplitSpec(p=1 - p)
            ]

        elif group.attr['form'] == 'm':
            if appetite > mouse_population:
                appetite = mouse_population

            p = appetite/mouse_population

            p = p * TIME_COEFFICIENT

            return [
                GroupSplitSpec(p=p, attr_set={'form': "s"}),
                GroupSplitSpec(p=1 - p)
            ]
        else:
            return [GroupSplitSpec(p=1)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.n > 0 and (group.attr['form'] == 's' or group.attr['form'] == 'm')


class HuntingInterventionRule(Rule):
    def __init__(self, t=TimeInt(0,12), memo=None):
        super().__init__('hunter-kills-snakes', t, memo)
        self.T_UNIT_MS = Time.MS.M

    def apply(self, pop, group, iter, t):
        p = 0

        if group.n >= pop.get_group(GroupQry({'form' : "m"})).n/4:
            p = 0.1

        return [
            GroupSplitSpec(p=p, attr_set={"form": "c"}),
            GroupSplitSpec(p=1 - p)
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.n > 0 and group.attr['form'] == 's'


#probe including snakes
probe_grp_size_attr = GroupSizeProbe('form', [GroupQry(attr={ 'form': 'm' }), GroupQry(attr={ 'form': 'h' }), GroupQry(attr={ 'form': 's' })], msg_mode=ProbeMsgMode.DISP)

#probe not including snakes
#probe_grp_size_attr = GroupSizeProbe('form', [GroupQry(attr={ 'form': 'm' }), GroupQry(attr={ 'form': 'h' })], msg_mode=ProbeMsgMode.DISP)


# 1 carbon => 1 mouse
# 5 carbon => 1 hawk
# 3 carbon => 1 snake

#1750 mice
mice = Group("Mice", 1750)
mice.attr['form'] = "m"

#100 hawks
hawks = Group("Hawks", 500)
hawks.attr['form'] = "h"

#100 snakes
snakes = Group("Snakes", 500)
snakes.attr['form'] = "s"

#the endless void of nature
nature = Group("Nature", 1e100)
nature.attr['form'] = "c"

# Run simulation of 10 years
(Simulation().
    add_rule(MiceBirthRule()).
    add_rule(CompetingPredationRule()).
    add_rule(PredationRule()).
    #add_rule(CompetingPredationRule()).
    #add_rule(HuntingInterventionRule()).
    add_probe(probe_grp_size_attr).
    add_group(mice).
    add_group(hawks).
    add_group(snakes).
    add_group(nature).
    summary(False, end_line_cnt=(0,1)).
    run(120)
)
