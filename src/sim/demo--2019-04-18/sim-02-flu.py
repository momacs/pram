from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation

from util.probes02 import probe_flu_at


# ----------------------------------------------------------------------------------------------------------------------
class FluProgressRule(Rule):
    def __init__(self):
        super().__init__('flu-progress', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total    population at the group's current location
            n_i = at.get_pop_size(GroupQry(attr={ 'flu': 'i' }))  # infected population at the group's current location

            p_infection = float(n_i) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'i', 'mood': 'annoyed' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Infected:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r', 'mood': 'happy'   }),
                GroupSplitSpec(p=0.5, attr_set={ 'flu': 'i', 'mood': 'bored'   }),
                GroupSplitSpec(p=0.3, attr_set={ 'flu': 'i', 'mood': 'annoyed' })
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' })
            ]


# ----------------------------------------------------------------------------------------------------------------------
class FluLocationRule(Rule):
    def __init__(self):
        super().__init__('flu-location', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Infected and poor:
        if group.has_attr({ 'flu': 'i', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Infected and rich:
        if group.has_attr({ 'flu': 'i', 'income': 'm' }):
            return [
                GroupSplitSpec(p=0.6, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.4)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.8, rel_set={ Site.AT: group.get_rel('school') }),
                GroupSplitSpec(p=0.2)
            ]

        return None


# ----------------------------------------------------------------------------------------------------------------------
home     = Site('home')
school_l = Site('school-l')
school_m = Site('school-m')


(Simulation().
    set().
        rand_seed(1928).
        pragma_autocompact(True).
        pragma_live_info(True).
        done().
    add([
        FluProgressRule(),
        FluLocationRule(),
        probe_flu_at(school_l),
        probe_flu_at(school_m),
        Group('g1', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
        Group('g2',  50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
        Group('g3', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
        Group('g4',  50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
        Group('g5', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home }),
        Group('g6',  50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home }),
        Group('g7', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home }),
        Group('g8',  50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home })
    ]).
    run(100).
    summary(True, 0,0,0,0, (1,0))
)


# ----------------------------------------------------------------------------------------------------------------------
# After 100 iterations
#     Low    income school - 25% of infected kids
#     Medium income school -  6% of infected kids
