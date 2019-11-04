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
from pram.rule        import ODESystemMass, GammaDistributionProcess, IterAlways, IterInt, TimeAlways
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', 'sir-comp.sqlite3')

def get_out_fpath(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

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
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:  # generate simulation data if the trajectory ensemble database is empty
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectories([
        Trajectory(
            (Simulation().
                add([
                    SIRSModel('flu', beta=0.10, gamma=0.05, solver=MCSolver()),
                    SIRSModel('flu', beta=0.50, gamma=uniform(loc=0.01, scale=0.14).rvs(), i=IterInt(5 + truncnorm(0, 50, 5.0, 10.0).rvs(), 0), solver=MCSolver()),
                    FluProcess(i=IterInt(50,0), p_max=None, a=3.0, scale=scale),
                    Group(m=1000, attr={ 'flu': 's' })
                ])
            )
        ) for scale in uniform(loc=1.0, scale=5.0).rvs(5)
    ])
    te.set_group_names(group_names)
    te.run(120)


te.plot_mass_locus_line((1200,300), get_out_fpath('_plot-line.png'), opacity_min=0.2)
# te.plot_mass_locus_line_aggr((1200,300), get_out_fpath('_plot-iqr.png'))
