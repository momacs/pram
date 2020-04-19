'''
A test of the mass transfer graph.
'''

from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
class FluProgressRule(Rule):
    def __init__(self):
        super().__init__('flu-progress', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            p_infection = group.get_mass_at(GroupQry(attr={ 'flu': 'i' }))
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


s = (Simulation().
    add([
        FluProgressRule(),
        FluLocationRule(),
        Group(m=450, attr={ 'flu': 's', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group(m= 50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group(m=450, attr={ 'flu': 's', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group(m= 50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group(m=450, attr={ 'flu': 's', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home}),
        Group(m= 50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home}),
        Group(m=450, attr={ 'flu': 's', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home}),
        Group(m= 50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})
    ])
)


# ----------------------------------------------------------------------------------------------------------------------
def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

te = TrajectoryEnsemble().add_trajectory(Trajectory(s)).run(100)

# te.traj[1].plot_mass_flow_time_series(filepath=get_out_dir('_plot-ts.png'), iter_range=(-1,10), v_prop=False, e_prop=True)
te.traj[1].plot_mass_locus_streamgraph((1200,300), get_out_dir('_plot-steam.png'))
# te.traj[1].plot_heatmap((800,800), get_out_dir('_plot-heatmap.png'), (-1,20))
