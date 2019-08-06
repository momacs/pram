'''
The SIR model implemented as an ODE system driving population mass flow.


A time-point event introduces perturbation into the system.  That perturbation takes the form of a spike in infections.
One plausible cause behind such an event is large number of agents being in a close proximity, e.g., a national
celebration, a sports event, or a natural disaster which forces people to relocate to large shelter areas.

As an alternative to the time-point perturbation, a flu spike process models a flu-spike perturbation acting over a
time span.  That process keeps interrupting the "S to R via I" conversion attempted by the SIR model.  That
interruption keeps happening because recovered agents are converted back into suspectible at every iteration.  The
magnitude of the process is linearily or exponentially diminishing in time (depending on the process because in fact
there are two) and at some point it no longer affect the system which allows the SIR model to operated on its own.

A real-life phenomenon that this process might be representing is a simulation of a bio-engineered super-virus of the
flu.  That virus mutates in a manner of hours effectively introducing a new generation of itself into the population
almost continuously which results in a large portion of the population being susceptible for extended period of time.
The diminishing effect might represent the response of a boosted immune system (possibly via a nanobot counter-measure);
the immune system eventually out-adapts the fast-mutating virus leading to lasting herd immunity.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import math
import matplotlib.pyplot as plt

from dotmap      import DotMap
from scipy.stats import gamma as gamma

from pram.data   import GroupSizeProbe, ProbePersistanceMem, ProbeMsgMode
from pram.entity import Group, GroupQry, GroupSplitSpec
from pram.rule   import ODESystemMass, Event, TimeAlways, Process
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (0) Preliminaries:

# (0.1) Plot the gamma distribution used later on:

# import numpy as np
# import matplotlib.pyplot as plt
#
# from scipy.stats import gamma
#
# print(gamma(a=5.0, loc=5.0, scale=100.0).pdf(400) * 500)
# print(gamma.pdf(x=400, a=5.0, loc=5.0, scale=100.0) * 500)
#
# x = np.linspace(0, 2000, 10000)
# fig = plt.figure(figsize=(10,2), dpi=150)
# plt.plot(x, gamma(a=5.0, loc=5.0, scale=100.0).pdf(x) * 500)
# plt.show()
# exit(1)


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init:

# (1.1) The SIR model:

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


# (1.2) Decay model:
decay_lambda = 1.0 / 20.0

def f_decay(t, state):
    n1, n2 = state
    return [-decay_lambda * n1, decay_lambda * n1]


# (1.3) Flu event:
class FluEvent(Event):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='flu-evt')

    def apply(self, pop, group, iter, t):
        return [
            GroupSplitSpec(p=0.80, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=0.20)
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and iter == 1000 and group.ha({ 'flu': 'r' })


# (1.4) Linearly decreasing flu process:
class FluLinProcess(Process):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='flu-lin-proc')

    def apply(self, pop, group, iter, t):
        p = (2000 - iter) / 1000
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1 - p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and 1000 <= iter <= 2000 and group.ha({ 'flu': 'r' })


# (1.5) Exponentially increasing flu process:
class FluExpProcess(Process):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='flu-exp-proc')

    def apply(self, pop, group, iter, t):
        p = math.exp(-decay_lambda * (2000 - iter))
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and 1000 <= iter <= 2000 and group.ha({ 'flu': 'r' })


# (1.6) Gamma flu process:
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
# (2) Probe:

probe = GroupSizeProbe(
    name='flu',
    queries=[GroupQry(attr={ 'flu': 's' }), GroupQry(attr={ 'flu': 'i' }), GroupQry(attr={ 'flu': 'r' })],
    qry_tot=None,
    var_names=['ps', 'pi', 'pr', 'ns', 'ni', 'nr'],
    persistance=ProbePersistanceMem(),
    msg_mode=ProbeMsgMode.NONE
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

s = (Simulation().
    set_pragma_autocompact(True).
    add([
        ODESystemMass(f_sir_model, [DotMap(attr={ 'flu': 's' }), DotMap(attr={ 'flu': 'i' }), DotMap(attr={ 'flu': 'r' })], dt=0.1),
        # FluEvent(),
        # FluLinProcess(),
        # FluExpProcess(),
        FluGammaProcess(),
        # ODESystemMass(f_decay, [DotMap(attr={ 'flu': 'r' }), DotMap(attr={ 'flu': 's' })], dt=0.1),
        Group(m=950, attr={ 'flu': 's' }),
        Group(m= 50, attr={ 'flu': 'i' }),
        probe
    ]).
    run(3000)
)


# ----------------------------------------------------------------------------------------------------------------------
# (4) Results:

# Time series plot (group mass):
cmap = plt.get_cmap('tab20')
series = [
    { 'var': 'ps', 'lw': 2, 'linestyle': '--', 'marker': '+', 'color': cmap(0), 'markersize': 0, 'lbl': 'Susceptible' },
    { 'var': 'pi', 'lw': 2, 'linestyle': '-',  'marker': 'o', 'color': cmap(4), 'markersize': 0, 'lbl': 'Infectious'  },
    { 'var': 'pr', 'lw': 2, 'linestyle': ':',  'marker': 'x', 'color': cmap(6), 'markersize': 0, 'lbl': 'Recovered'   }
]
probe.plot(series, fpath_fig=None, figsize=(24,8), legend_loc='upper right', dpi=150)


# Time series plot (numeric integrator history):
# h = s.rules[0].get_hist()
# plt.plot(h[0], h[1][0], 'b-', h[0], h[1][1], 'g-', h[0], h[1][2], 'r-')  # S-blue, I-green, R-red
# plt.show()

# Phase plot (numeric integrator history):
# h = s.rules[0].get_hist()
# plt.plot(h[1][0], h[1][1], 'k-')
# plt.show()
