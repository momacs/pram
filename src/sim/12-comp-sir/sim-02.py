'''
Multiple composite SIR models.

This is the simulation of two overlapping and time-unbounded SIR models, A and B.  Model B starts with a random time
delay and one of its parameters (gamma) random.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import matplotlib.pyplot as plt

from dotmap import DotMap
from scipy.stats import gamma, uniform

from pram.entity import Group, GroupSplitSpec
from pram.rule   import ODESystemMass, GammaDistributionProcess, IterAlways, IterInt, TimeAlways
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', 'sir-x2.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# The SIR models generator (helper):

def make_sir(beta, gamma, t=TimeAlways(), i=IterAlways(), dt=0.1):
    def f_sir_model(t, state):
        '''
        [1] Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
            Royal Society A. 115(772), 700--721.

        http://www.public.asu.edu/~hnesse/classes/sir.html
        '''

        s,i,r = state
        n = s + i + r
        return [
            -beta * s * i / n,
             beta * s * i / n - gamma * i,
                                gamma * i
        ]

    return ODESystemMass(f_sir_model, [DotMap(attr={ 'flu': 's' }), DotMap(attr={ 'flu': 'i' }), DotMap(attr={ 'flu': 'r' })], t=t, i=i, dt=dt)


# ----------------------------------------------------------------------------------------------------------------------
# The actual SIR models used:

sir_a = make_sir(0.10, 0.05, i=IterAlways(), dt=0.1)
sir_b = make_sir(0.50, uniform(loc=0.01, scale=0.14).rvs(), i=IterInt(900 + gamma(a=5.0, loc=5.0, scale=25.0).rvs(), 0), dt=0.1)


# ----------------------------------------------------------------------------------------------------------------------
# The recurrent flu process:

class RecurrentFluProcess(GammaDistributionProcess):
    def apply(self, pop, group, iter, t):
        p = self.get_p(iter)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# Generate data:

# if os.path.isfile(fpath_db): os.remove(fpath_db)
#
# te = (
#     TrajectoryEnsemble(fpath_db).
#         add_trajectories([
#             Trajectory(
#                 sim=(Simulation().
#                     add([
#                         sir_a,
#                         sir_b,
#                         RecurrentFluProcess(i=IterInt(2000,0), p_max=gamma_proc_p_max, a=5.0, scale=50.0),
#                         Group(m=950, attr={ 'flu': 's' }),
#                         Group(m= 50, attr={ 'flu': 'i' })
#                     ])
#                 )
#             ) for gamma_proc_p_max in uniform(loc=0.75, scale=0.20).rvs(5)
#         ]).
#         set_group_names([
#             (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
#             (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
#             (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
#         ]).
#         run(4000)
# )


# ----------------------------------------------------------------------------------------------------------------------
# Load data:

te = TrajectoryEnsemble(fpath_db).stats()


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

def get_out_dir(filename): return os.path.join(os.path.dirname(__file__), 'out', filename)

te.plot_mass_locus_line((1200,300), get_out_dir('_plot-line.png'), iter_range=(-1, -1), nsamples=0)
te.plot_mass_locus_line_aggr((1200,300), get_out_dir('_plot-iqr.png'), iter_range=(-1, -1))
