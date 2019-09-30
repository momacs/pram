'''
A test of the mass transfer graph.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


import pyrqa


from scipy.stats import beta

from dotmap      import DotMap
from scipy.stats import gamma as gamma

from pram.entity import Group, GroupSplitSpec
from pram.rule   import Process, ODESystemMass, TimeAlways
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', '06-sir-gamma-beta.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# Plot the gamma distribution used later on:

# import numpy as np
# import matplotlib.pyplot as plt
#
# from scipy.stats import gamma
#
# print(gamma(a=5.0, loc=5.0, scale=25.0).pdf(100) * 125)
# print(gamma.pdf(x=100, a=5.0, loc=5.0, scale=25.0) * 125)
#
# x = np.linspace(0, 2000, 10000)
# fig = plt.figure(figsize=(10,2), dpi=150)
# plt.plot(x, gamma(a=5.0, loc=5.0, scale=25.0).pdf(x) * 125)
# plt.show()
# sys.exit(1)


# ----------------------------------------------------------------------------------------------------------------------
# The SIR model:

sir_beta  = 0.20  # transmission rate
sir_gamma = 0.02  # recovery rate

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
# The random flu beta process:

class FluRandomBetaProcess(Process):
    def __init__(self):
        super().__init__('flu-random-beta-proc', TimeAlways())

    def apply(self, pop, group, iter, t):
        p = beta.rvs(a=2.0, b=25.0, loc=0.0, scale=1.0)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# The recurrent flu gamma process:

class RecurrentFluGammaProcess(Process):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='rec-flu-gamma-proc')
        self.p = lambda iter: gamma(a=5.0, loc=5.0, scale=25.0).pdf(iter % 1000) * 125

    def apply(self, pop, group, iter, t):
        p = self.p(iter)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and iter >= 1000 and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# Generate:

# if os.path.isfile(fpath_db):
#     os.remove(fpath_db)
#
# te = (
#     TrajectoryEnsemble(fpath_db).
#         add_trajectories([
#             Trajectory(
#                 sim=(Simulation().
#                     add([
#                         ODESystemMass(f_sir_model, [DotMap(attr={ 'flu': 's' }), DotMap(attr={ 'flu': 'i' }), DotMap(attr={ 'flu': 'r' })], dt=0.1),
#                         FluRandomBetaProcess(),
#                         RecurrentFluGammaProcess(),
#                         Group(m=950, attr={ 'flu': 's' }),
#                         Group(m= 50, attr={ 'flu': 'i' }),
#                     ])
#                 )
#             ) for _ in range(1)
#         ]).
#         set_group_names([
#             (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
#             (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
#             (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
#         ]).
#         run(1000)
# )


# ----------------------------------------------------------------------------------------------------------------------
# Load:

te = TrajectoryEnsemble(fpath_db).stats()


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

# te.traj[1].plot_mass_locus_streamgraph((1200,600), get_out_dir('_plot.png'), iter_range=(-1, 2000), do_sort=True)
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

# te.plot_mass_locus_line((2400,600), get_out_dir('_plot.png'), iter_range=(-1, -1), nsamples=10, do_sort=True)
# te.plot_mass_locus_line_aggr((2400,600), get_out_dir('_plot.png'), iter_range=(-1, 2000), do_sort=True)

# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(-1, -1), nsamples=10, n_iter_per_rot=0, do_sort=True)
# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(-1, -1), nsamples=10, n_iter_per_rot=1000, do_sort=True)
# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(999, -1), nsamples=10, n_iter_per_rot=1000, do_sort=True)

# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(999, -1), nsamples=10, n_iter_per_rot=379, do_sort=True)
# te.plot_mass_locus_polar((12,12), get_out_dir('_plot.png'), iter_range=(999, -1), nsamples=10, n_iter_per_rot=1845, do_sort=True)
