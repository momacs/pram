import entropy
import math
import numpy as np

from dotmap      import DotMap
from scipy.stats import beta, gamma

from pram.entity import Group, GroupSplitSpec
from pram.rule   import Process, ODESystemMass, TimeAlways
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# import importlib
# sim = importlib.import_module('07-sir-beta-gamma-tcomp')
# from sim import *


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', '07-sir-gamma-beta-tcomp.sqlite3')


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
# The recurrent flu gamma process with time compression (i.e., occuring with increasing frequency):

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
import matplotlib.pyplot as plt
import sobi

te = TrajectoryEnsemble(fpath_db).stats()
signal = te.traj[1].get_signal()
s = signal.series


# signal.plot_autocorr((16,6), filename=None)
# sys.exit(0)


# The Signal class tests:
# print(f'S: {s[0,0]} {s[0,-1]}')
# print(f'I: {s[1,0]} {s[1,-1]}')
# print(f'R: {s[2,0]} {s[2,-1]}')  # nan as the 1st value confirmed
# sys.exit(0)


# plt.plot(s[0])
# plt.plot(s[1])
# plt.plot(S[2])
# plt.show()
# sys.exit(0)


# from pandas.plotting import autocorrelation_plot
# fig, axes = plt.subplots(3,1,figsize=(16,3), dpi=100)
# autocorrelation_plot(s[0,1:], ax=axes[0])
# autocorrelation_plot(s[1,1:], ax=axes[1])
# autocorrelation_plot(s[2,1:], ax=axes[2])
# plt.show()
# sys.exit(0)


# from statsmodels.tsa.stattools import acf, pacf
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# fig, axes = plt.subplots(2,1,figsize=(16,3), dpi=100)
# plot_acf(s[0], lags=2000, ax=axes[0])
# plot_pacf(s[0], lags=2000, ax=axes[1])
# plt.show()
# sys.exit(0)


# SOBI:
# S,A,W = sobi.sobi(s[:,1:], num_lags=None, eps=1.0e-6, random_order=True)


# print(S.shape)
# print(A)


# plt.plot(ts['s']['i'], ts['s']['m'])
# plt.plot(S[0])
# plt.plot(S[1])
# plt.plot(S[2])
# plt.show()
# sys.exit(0)


# si = 0
#
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('iter')
# ax1.set_ylabel('X', color=color)
# ax1.plot(s[si], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('S', color=color)
# ax2.plot(S[si], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# plt.show()


# Entropy:
print(entropy.perm_entropy(s[0], order=3, normalize=True))                 # Permutation entropy
print(entropy.spectral_entropy(s[0], 100, method='welch', normalize=True)) # Spectral entropy
print(entropy.svd_entropy(s[0], order=3, delay=1, normalize=True))         # Singular value decomposition entropy
print(entropy.app_entropy(s[0], order=2, metric='chebyshev'))              # Approximate entropy
print(entropy.sample_entropy(s[0], order=2, metric='chebyshev'))           # Sample entropy


fpath_db = os.path.join(os.path.dirname(__file__), 'data', '06-sir-gamma-beta.sqlite3')
te = TrajectoryEnsemble(fpath_db).stats()
s = te.traj[1].get_signal().series
print(entropy.app_entropy(s[0], order=2, metric='chebyshev'))              # Approximate entropy
