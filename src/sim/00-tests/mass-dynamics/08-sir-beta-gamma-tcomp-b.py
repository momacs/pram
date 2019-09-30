'''
A test of the mass transfer graph.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


import math
import numpy as np

from scipy.stats import beta

from dotmap      import DotMap
from scipy.stats import gamma as gamma

from pram.entity import Group, GroupSplitSpec
from pram.rule   import Process, ODESystemMass, TimeAlways
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', '07-sir-gamma-beta-tcomp.sqlite3')

n_traj = 10
n_iter = 7000


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
# Variable frequency sine wave:

def var_frq_sine_wave(A, y_shift, f_start, f_end, interval, n_steps):
    ''' SRC: https://stackoverflow.com/questions/19771328/sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time '''

    b = np.log(f_end/f_start) / interval
    a = 2 * math.pi * f_start / b

    y = np.zeros(n_steps)

    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        g_t = a * np.exp(b * t)
        y[i] = A * np.sin(g_t) + y_shift

    return y


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
# The SIRS time-compression process:

# class SIRSTimeCompressProcess(Process):
#     def __init__(self):
#         super().__init__('sir-time-compress-proc', TimeAlways())
#         self.last_applied_iter = None
#
#     def apply(self, pop, group, iter, t):
#         if self.last_applied_iter != iter:
#             global sir_beta, sir_gamma
#             sir_beta  += 0.10
#             sir_gamma += 0.05
#             self.last_applied_iter = iter
#
#         return None
#
#     def is_applicable(self, group, iter, t):
#         return (iter >= 2000) and (iter % 1000 == 0)

class SIRSTimeCompressProcess(Process):
    def __init__(self):
        super().__init__('sir-time-compress-proc', TimeAlways())
        self.wave = var_frq_sine_wave(1, 0, 1, (n_iter / 1000) + 3, 1, n_iter + 1)
        self.last_applied_iter = None

    def apply(self, pop, group, iter, t):
        self.last_applied_iter = self.last_applied_iter or iter  # prevent None
        print(f'{iter:6} {iter - self.last_applied_iter:6}: t-comp')

        global sir_beta, sir_gamma
        sir_beta  = min(1.00, sir_beta  + 0.08)
        sir_gamma = min(1.00, sir_gamma + 0.03)
        self.last_applied_iter = iter

        return None

    def is_applicable(self, group, iter, t):
        return (
            super().is_applicable(group, iter, t) and
            iter >= 1000 and
            self.last_applied_iter != iter and
            self.wave[iter] * self.wave[iter + 1] < 0  # the wave crosses the x axis (i.e., the function changes sign)
        )


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
        self.p_iter_0 = None
        self.p = lambda iter: gamma(a=5.0, loc=5.0, scale=25.0).pdf(iter - self.p_iter_0) * 125
        self.wave = var_frq_sine_wave(1, 0, 1, n_iter / 1000 + 2, 1, n_iter + 1)
        self.last_applied_iter = None

    def apply(self, pop, group, iter, t):
        if self.wave[iter] * self.wave[iter + 1] < 0:  # the wave crosses the x axis (i.e., the function changes sign)
            self.p_iter_0 = iter

            self.last_applied_iter = self.last_applied_iter or iter  # prevent None
            print(f'{iter:6} {iter - self.last_applied_iter:6}: gamma')
            self.last_applied_iter = iter

        if self.p_iter_0 is not None:
            p = self.p(iter)
            return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]
        else:
            return None

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


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
#                         # SIRSTimeCompressProcess(),
#                         Group(m=950, attr={ 'flu': 's' }),
#                         Group(m= 50, attr={ 'flu': 'i' }),
#                     ])
#                 )
#             ) for _ in range(n_traj)
#         ]).
#         set_group_names([
#             (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
#             (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
#             (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
#         ]).
#         run(n_iter)
# )


# ----------------------------------------------------------------------------------------------------------------------
# Load:

te = TrajectoryEnsemble(fpath_db).stats()


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

# te.traj[1].plot_mass_locus_streamgraph((1200,600), get_out_dir('_plot.png'), iter_range=(-1, 2000), do_sort=True)
# te.traj[1].plot_mass_locus_fft((1200,200), get_out_dir('_plot.png'), sampling_rate=100, do_sort=True)
# te.traj[1].plot_mass_locus_spectrogram((16,8), get_out_dir('_plot.png'), sampling_rate=None, win_len=100, do_sort=True)
# te.traj[1].plot_mass_locus_scaleogram((16,8), get_out_dir('_plot.png'), sampling_rate=100, do_sort=True)

# te.plot_mass_locus_line((2400,600), get_out_dir('_plot.png'), iter_range=(-1, -1), nsamples=10, do_sort=True)
# te.plot_mass_locus_line_aggr((2400,600), get_out_dir('_plot.png'), iter_range=(-1, -1), do_sort=True)
