'''
An example of how a population of agents can be generated from the recorded population-level mass dynamics.

Building on the previous and very simple simulation, this simulation uses two rules, eight groups, and the dynamics is
a result of conditioning on group's current location (which is encoded via a Site object).  Notably, the initial eight
groups are quickly split into the total of 32 groups (already at iteration three).  A further increase in complexity
(but also realism) comes from the feedback in the system.  Namely, susceptible agents become infected with the
probability dependant on the number of infected agents around them.  That number changes from iteration to iteration
and results in the dampened harmonic oscillation in the mass locus evident on the plot.

The simulation also involves irrelevant attributes, i.e., those no rule conditions on.  Those attributes have no causal
relationship with mass dynamics and should therefore be ignored by a mechanism extraction algorithm; if they are not
ignored, then the algorithm can likely be improved.
'''

import numpy as np
import os

from pram.entity      import Group, GroupQry, GroupSplitSpec, Site
from pram.model.model import MCSolver
from pram.model.epi   import SIRSModel
from pram.rule        import Rule
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'sim-02.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
class FluProgressRule(Rule):
    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            # at  = group.get_rel(Site.AT)
            # n   = at.get_pop_size()                               # total    population at the group's current location
            # n_i = at.get_pop_size(GroupQry(attr={ 'flu': 'i' }))  # infected population at the group's current location
            #
            # p_infection = float(n_i) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            p_infection = group.get_rel(Site.AT).get_mass_prop(GroupQry(attr={ 'flu': 'i' }))

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

# if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:
    te.add_trajectories([
        Trajectory(
            (Simulation().
                # set_pragma_live_info(True).
                add([
                    FluProgressRule(),
                    FluLocationRule(),
                    Group('g1', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
                    Group('g2',  50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
                    Group('g3', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
                    Group('g4',  50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home }),
                    Group('g5', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home }),
                    Group('g6',  50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home }),
                    Group('g7', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home }),
                    Group('g8',  50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home })
                ])
            )
        )
    ])
    te.run(100)


# ----------------------------------------------------------------------------------------------------------------------
# te.traj[1].plot_mass_locus_line((1200,300), os.path.join(os.path.dirname(__file__), 'sim-02.png'))

# A single agent:
agent = te.traj[1].gen_agent(3)
print(agent)
print(agent['rel'][Site.AT][0])  # the name and has of the site is retrieved from the DB correctly

# Population of agents:
print(te.traj[1].gen_agent_pop(2,2))  # two-agent population simulated for two iterations
