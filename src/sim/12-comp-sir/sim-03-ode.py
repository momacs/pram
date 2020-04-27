'''
Multiple composite SIR models.

This is the simulation of two overlapping and time-unbounded SIR models, A and B.  Model B starts with a random time
delay and one of its parameters (gamma) random.  Additional gamma flu process is added as well to further demonstrate
how more complicated systems can be composed from modeling primitives.

This system uses ordinary differential equations solver to implement the SIR models.
'''

import matplotlib.pyplot as plt
import os

from dotmap import DotMap
from scipy.stats import truncnorm, uniform

from pram.entity      import Group, GroupSplitSpec
from pram.model.model import ODESolver
from pram.model.epi   import SIRModel
from pram.rule        import GammaDistributionProcess, IterAlways, IterInt, TimeAlways
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble, ClusterInf


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', 'sir-comp.sqlite3')

def U(a,b, n=None):
    return uniform(a,a+b).rvs(n)

def TN(a,b, mu, sigma, n=None):
    return truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(n)

group_names = [
    (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
    (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
    (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
]


# ----------------------------------------------------------------------------------------------------------------------
# A gamma distribution process rule:

class FluProcess(GammaDistributionProcess):
    def apply(self, pop, group, iter, t):
        p = self.get_p(iter)
        return [
            GroupSplitSpec(p=p, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=1-p)
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)
# te = TrajectoryEnsemble(fpath_db, cluster_inf=ClusterInf(address='auto'))
# te = TrajectoryEnsemble(fpath_db, cluster_inf=ClusterInf(num_cpus=6, memory=500*1024*1024, object_store_memory=500*1024*1024, include_webui=True))

if te.is_db_empty:
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectories([
        Trajectory(
            (Simulation().
                add([
                    SIRModel('flu', beta=0.10, gamma=0.05, solver=ODESolver()),
                    SIRModel('flu', beta=0.50, gamma=U(0.01, 0.15), i=[int(900 + TN(0,600, 100,100)), 0], solver=ODESolver()),
                    FluProcess(i=[2000,0], p_max=None, a=5.0, scale=flu_proc_scale),
                    Group(m=950, attr={ 'flu': 's' }),
                    Group(m= 50, attr={ 'flu': 'i' })
                ])
            )
        ) for flu_proc_scale in U(40,60, 3)
    ])
    te.set_group_names(group_names)
    te.run(3000)


# Visualize:
te.plot_mass_locus_line     ((1200,300), os.path.join(os.path.dirname(__file__), 'out', '_plot-line.png'), opacity_min=0.2)
te.plot_mass_locus_line_aggr((1200,300), os.path.join(os.path.dirname(__file__), 'out', '_plot-iqr.png'))
