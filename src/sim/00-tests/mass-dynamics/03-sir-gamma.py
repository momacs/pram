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

from pram.entity      import Group, GroupSplitSpec
from pram.model.model import ODESolver
from pram.model.epi   import SIRSModel
from pram.rule        import Process
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', '03-sir-gamma.sqlite3')

def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

group_names = [
    (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
    (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
    (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
]


# ----------------------------------------------------------------------------------------------------------------------
class FluGammaProcess(Process):
    def __init__(self):
        super().__init__(name='flu-gamma-proc')
        self.p = lambda iter: gamma(a=5.0, loc=5.0, scale=100.0).pdf(iter - 1000) * 500

    def apply(self, pop, group, iter, t):
        p = self.p(iter)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and 1000 <= iter <= 3000 and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:  # generate simulation data if the trajectory ensemble database is empty
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectory(
        (Simulation().
            add([
                SIRSModel('flu', beta=0.20, gamma=0.02, solver=ODESolver()),
                FluGammaProcess(),
                Group(m=950, attr={ 'flu': 's' }),
                Group(m= 50, attr={ 'flu': 'i' })
            ])
        )
    )
    te.set_group_names(group_names)
    te.run(3000)


# te.traj[1].plot_mass_flow_time_series(filepath=get_out_dir('_plot.png'), iter_range=(-1,10), v_prop=False, e_prop=True)
# te.traj[1].plot_mass_locus_streamgraph((1200,600), get_out_dir('_plot.png'))
# te.traj[1].plot_heatmap((800,800), get_out_dir('_plot.png'), (-1,20))

# te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'flu': 's' }), iter_range=(-1, -1))
# te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'flu': 'i' }), iter_range=(-1, 4000))
# te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'flu': 'r' }), iter_range=(-1, 4000))
