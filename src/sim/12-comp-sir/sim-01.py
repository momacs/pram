'''
Multiple composite SIR models.

This simulation builds up a more a more complicated example of two SIR models interacting.
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
# Scratchpad:

# from scipy.stats import gamma, truncnorm, uniform
# dist = gamma(a=5.0, loc=0.0, scale=5.0)
# print(dist.mean())
# # print(dist.median())
# # print(dist.pdf([20, 25, 26]))
# # print(dist.pdf(dist.median()))
# print(dist.ppf([0,0.99]))
# sys.exit(1)


# ----------------------------------------------------------------------------------------------------------------------
# Plot distributions used:

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import gamma, truncnorm, uniform
#
# # dist, label = gamma(a=5.0, loc=0.0, scale=35.0), 'Gamma(5,35)'
# # dist, label = truncnorm(a=0.01, b=0.09, loc=0.5, scale=0.5), 'TruncNormal(a=0.01, b=0.09, m=0.5, s=0.5)'
# # dist, label = uniform(loc=0.01, scale=0.14), 'Uniform(0.01, 0.15)'
# dist, label = gamma(a=5.0, loc=0.0, scale=50.0), 'Gamma(5,50)'
#
# fig = plt.figure(figsize=(10,2), dpi=150)
# x = np.linspace(dist.ppf(0.00), dist.ppf(0.99), 100)
# plt.plot(x, dist.pdf(x), 'g-', lw=1, label=label)
# plt.legend(loc='best', frameon=False)
# plt.show()
# sys.exit(1)

# ----
# import matplotlib.pyplot as plt
# from scipy.stats import gamma
# import pandas as pd
# g = gamma(a=5.0, loc=0.0, scale=35.0)
# fig = plt.figure(figsize=(10,2), dpi=150)
# pd.Series(g.rvs(1000)).plot(kind='density')
# plt.show()
# sys.exit(1)

# ----
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# pd.Series(np.random.gamma(2.0, 50.0, 1000)).plot(kind='density')
# plt.show()
# sys.exit(1)


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

# 01 SIR A1:
# sir_a1 = make_sir(0.10, 0.05, i=IterAlways(), dt=0.1)

# 02 SIR B1:
# sir_b1 = make_sir(0.50, 0.02, i=IterAlways(), dt=0.1)

# 03 Two non-overlapping and time-bounded SIR models:
# sir_a1 = make_sir(0.10, 0.05, i=IterInt(500,1000), dt=0.1)
# sir_b1 = make_sir(0.50, 0.02, i=IterInt(2000,4000), dt=0.1)

# 04 Two overlapping and time-unbounded SIR models (B1 starts with a fixed delay):
# sir_a1 = make_sir(0.10, 0.05, i=IterAlways(), dt=0.1)
# sir_b1 = make_sir(0.50, 0.02, i=IterInt(1000,0), dt=0.1)

# 05 Two overlapping and time-unbounded SIR models (B1 starts with a fixed delay):
# sir_a1 = make_sir(0.10, 0.05, i=IterAlways(), dt=0.1)
# def make_sir_b1_fixed_delay(iter0):
#     return make_sir(0.50, 0.02, i=IterInt(iter0,0), dt=0.1)

# 06 Two overlapping and time-unbounded SIR models (B1 starts with a random delay):
# sir_a1 = make_sir(0.10, 0.05, i=IterAlways(), dt=0.1)
# def make_sir_b1_random_delay(iter0=900, iter0_dist=gamma(a=5.0, loc=0.0, scale=35.0)):
#     iter0_rnd = iter0_dist.rvs()
#     # print(f'rand-iter: {iter0_rnd}')
#     return make_sir(0.50, 0.02, i=IterInt(iter0 + iter0_rnd,0), dt=0.1)

# 07 Two overlapping and time-unbounded SIR models (B1 starts with a random delay and a random 'gamma' parameter):
sir_a1 = make_sir(0.10, 0.05, i=IterAlways(), dt=0.1)
def make_sir_b1_random_delay_gamma(iter0=900, iter0_dist=gamma(a=5.0, loc=5.0, scale=25.0), gamma=uniform(loc=0.01, scale=0.14)):
    iter0_rnd = iter0_dist.rvs()
    gamma_rnd = gamma.rvs()
    return make_sir(0.50, gamma_rnd, i=IterInt(iter0 + iter0_rnd,0), dt=0.1)


# ----------------------------------------------------------------------------------------------------------------------
# The recurrent flu process:

class RecurrentFluProcess(GammaDistributionProcess):
    def apply(self, pop, group, iter, t):
        p = self.get_p(iter)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


# RecurrentFluProcess(a=5.0, scale=50.0).plot()
# RecurrentFluProcess(p_max=0.5, a=5.0, scale=50.0).plot()
# RecurrentFluProcess(p_max=1.0, a=5.0, scale=50.0).plot()
# sys.exit(1)


# ----------------------------------------------------------------------------------------------------------------------
from pram.rule import NormalDistributionProcess
class ProcessN(NormalDistributionProcess):
    def apply(self, pop, group, iter, t):
        return None

# ProcessN().plot()
# ProcessN(p_max=1.0).plot()
# sys.exit(1)


# ----------------------------------------------------------------------------------------------------------------------
# Generate data:

if os.path.isfile(fpath_db): os.remove(fpath_db)

te = (
    TrajectoryEnsemble(fpath_db).
        set_pragma_memoize_group_ids(True).
        add_trajectories([
            Trajectory(
                sim=(Simulation().
                    add([
                        sir_a1,
                        # sir_b1,
                        # make_sir_b1_fixed_delay(iter0),
                        # make_sir_b1_random_delay(iter0),
                        # make_sir_b1_random_delay(),
                        # make_sir_b1_random_delay(iter0_dist=gamma(a=5.0, loc=50.0, scale=25.0)),
                        make_sir_b1_random_delay_gamma(),

                        RecurrentFluProcess(i=IterInt(2000,0), p_max=gamma_proc_p_max, a=5.0, scale=50.0),
                        Group(m=950, attr={ 'flu': 's' }),
                        Group(m= 50, attr={ 'flu': 'i' })
                    ])
                )
            # ) for _ in range(5)
            # ) for iter0 in [900, 950, 1000, 1050, 1100]
            # ) for gamma_proc_p_max in uniform(loc=0.75, scale=0.20).rvs(1)
            ) for gamma_proc_p_max in [1.00]
        ]).
        set_group_names([
            (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
            (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
            (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
        ]).
        run(4000)
)


# ----------------------------------------------------------------------------------------------------------------------
# Load data:

# te = TrajectoryEnsemble(fpath_db).stats()


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

def get_out_dir(filename): return os.path.join(os.path.dirname(__file__), 'out', filename)

# te.traj[1].plot_mass_locus_line((1200,300), get_out_dir('_plot.png'), iter_range=(-1, -1))

# te.traj[1].plot_mass_locus_streamgraph((1200,600), get_out_dir('_plot.png'), iter_range=(-1, -1), do_sort=True)

# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'),                         sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), iter_range=(999,   -1), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), iter_range=( -1,  999), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), iter_range=(999, 1999), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), iter_range=(999, 2999), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), iter_range=(999, 3999), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), iter_range=(999, 4999), sampling_rate=100, do_sort=True)

# te.traj[1].plot_mass_locus_scaleogram((16,8), get_out_dir('_plot.png'), iter_range=(-1, 4999), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_scaleogram((16,8), get_out_dir('_plot.png'),                        sampling_rate=100, do_sort=True)

# te.traj[1].plot_mass_locus_spectrogram((16,8), get_out_dir('_plot.png'), sampling_rate=None, win_len=100, noverlap=75, do_sort=True)

# te.traj[1].plot_mass_locus_recurrence((12,12), get_out_dir('_plot.png'), iter_range=(-1, 4000), neighbourhood=pyrqa.neighbourhood.FixedRadius(), embedding_dimension=1, time_delay=2)

# te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'flu': 's' }), iter_range=(-1, 4000))
# te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'flu': 'i' }), iter_range=(-1, 4000))
# te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'flu': 'r' }), iter_range=(-1, 4000))


# te.plot_mass_locus_line((1200,300), get_out_dir('_plot-line.png'), iter_range=(-1, -1), nsamples=0)
# te.plot_mass_locus_line_aggr((1200,300), get_out_dir('_plot-iqr.png'), iter_range=(-1, -1))

# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(-1, -1), nsamples=10, n_iter_per_rot=0, do_sort=True)
# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(-1, -1), nsamples=10, n_iter_per_rot=1000, do_sort=True)
# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(999, -1), nsamples=10, n_iter_per_rot=1000, do_sort=True)

# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(999, -1), nsamples=10, n_iter_per_rot=379, do_sort=True)
# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(999, -1), nsamples=10, n_iter_per_rot=1845, do_sort=True)
