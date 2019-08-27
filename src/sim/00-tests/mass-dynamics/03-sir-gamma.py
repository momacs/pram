'''
A test of the mass transfer graph for a system with two interacting components: the SIR model implemented as an ODE
system driving population mass flow and (2) a gamma perturbation process.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


import math
import matplotlib.pyplot as plt

from dotmap      import DotMap
from scipy.stats import gamma

from pram.entity import Group, GroupSplitSpec
from pram.rule   import ODESystemMass, TimeAlways, Process
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
sir_beta  = 0.20  # 0.05  # transmission rate
sir_gamma = 0.02  # 0.50  # recovery rate
# sir_alpha = 0.10  # immunity loss rate (alpha = 0 implies life-long immunity)

def f_sir_model(t, state):
    '''
    [1] Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
        Royal Society A. 115(772), 700--721.
    '''

    s,i,r = state
    n = s + i + r
    return [
        -sir_beta * s * i / n,
         sir_beta * s * i / n - sir_gamma * i,
                                sir_gamma * i
    ]


# ----------------------------------------------------------------------------------------------------------------------
class FluGammaProcess(Process):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='flu-gamma-proc')
        self.p = lambda iter: gamma(a=5.0, loc=5.0, scale=100.0).pdf(iter - 1000) * 500

    def apply(self, pop, group, iter, t):
        p = self.p(iter)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and 1000 <= iter <= 3000 and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
s = (Simulation().
    # set_pragma_autocompact(True).
    add([
        ODESystemMass(f_sir_model, [DotMap(attr={ 'flu': 's' }), DotMap(attr={ 'flu': 'i' }), DotMap(attr={ 'flu': 'r' })], dt=0.1),
        FluGammaProcess(),
        Group(m=950, attr={ 'flu': 's' }),
        Group(m= 50, attr={ 'flu': 'i' })
    ])
)


# ----------------------------------------------------------------------------------------------------------------------
def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

te = (TrajectoryEnsemble().
    add_trajectory(Trajectory('lotka-volterra-gamma', None, s)).
    set_group_names([
        (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
        (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
        (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
    ]).
    run(3000)
)

# te.traj[1].plot_mass_flow_time_series(filepath=get_out_dir('_plot.png'), iter_range=(-1,20), v_prop=False, e_prop=True)
te.traj[1].plot_mass_locus_streamgraph((1200,600), get_out_dir('_plot.png'), do_sort=True)
# te.traj[1].plot_heatmap((800,800), get_out_dir('_plot.png'), (-1,20))
