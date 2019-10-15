'''
Multiple composite SIR models.

This is the simulation of two overlapping and time-unbounded SIR models, A and B.  Model B starts with a random time
delay and one of its parameters (gamma) random.  Additional gamma flu process is added as well to further demonstrate
how more complicated systems can be composed from modeling primitives.

This system uses time-invariant Markov chain solver to implement the SIR models.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import matplotlib.pyplot as plt

from dotmap import DotMap
from scipy.stats import truncnorm, uniform

from pram.entity      import Group, GroupSplitSpec
from pram.model.model import MCSolver
from pram.model.epi   import SIRSModel
from pram.rule        import ODESystemMass, GammaDistributionProcess
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', 'sir-comp-demo.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions (to simplify the code below):

def get_out_fpath(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

def U(a,b, n=None):
    return uniform(a,a+b).rvs(n)

def TN(a,b, loc, scale, n=None):
    return truncnorm(a, b, loc, scale).rvs(n)

group_names = [
    (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
    (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
    (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
]


# ----------------------------------------------------------------------------------------------------------------------
# A gamma distribution process rule:

class MakeSusceptibleProcess(GammaDistributionProcess):
    def apply(self, pop, group, iter, t):
        p = self.get_p(iter)
        return [
            GroupSplitSpec(p=p, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=1-p)
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# Simulations:

if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:  # generate simulation data if the trajectory ensemble database is empty
    te.add_trajectories([
        Trajectory(
            (Simulation().
                add([
                    SIRSModel('flu', beta=0.10, gamma=0.05, solver=MCSolver()),
                    SIRSModel('flu', beta=0.50, gamma=U(0.01, 0.15), i=[5 + TN(0, 50, 5.0, 10.0), 0], solver=MCSolver()),
                    MakeSusceptibleProcess(i=[50,0], a=3.0, scale=flu_proc_scale),
                    Group(m=1000, attr={ 'flu': 's' })
                ])
            )
        ) for flu_proc_scale in U(1.0, 5.0, 3)
    ])
    te.set_group_names(group_names)
    te.run(120)

te.plot_mass_locus_line((1200,300), get_out_fpath('_plot-line.png'), opacity_min=0.2)
te.plot_mass_locus_line_aggr((1200,300), get_out_fpath('_plot-iqr.png'))
