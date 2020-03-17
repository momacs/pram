'''
The SIR model implemented as an ODE system driving population mass flow.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import matplotlib.pyplot as plt

from dotmap import DotMap

from pram.data   import GroupSizeProbe, ProbePersistenceMem, ProbeMsgMode
from pram.entity import Group, GroupQry
from pram.rule   import ODESystemMass
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init:

beta  = 0.20  # 0.05  # transmission rate
gamma = 0.02  # 0.50  # recovery rate
# alpha = 0.10  # immunity loss rate (alpha = 0 implies life-long immunity)

def f_sir_model(t, state):
    '''
    [1] Kermack WO & McKendrick AG (1927) A Contribution to the Mathematical Theory of Epidemics. Proceedings of the
        Royal Society A. 115(772), 700--721.
    '''

    s,i,r = state
    n = s + i + r
    return [
        -beta * s * i / n,
         beta * s * i / n - gamma * i,
                            gamma * i
    ]


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probe:

probe = GroupSizeProbe(
    name='flu',
    queries=[GroupQry(attr={ 'flu': 's' }), GroupQry(attr={ 'flu': 'i' }), GroupQry(attr={ 'flu': 'r' })],
    qry_tot=None,
    var_names=['ps', 'pi', 'pr', 'ns', 'ni', 'nr'],
    persistence=ProbePersistenceMem(),
    msg_mode=ProbeMsgMode.NONE
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

s = (Simulation().
    set_pragma_autocompact(True).
    add([
        ODESystemMass(f_sir_model, [DotMap(attr={ 'flu': 's' }), DotMap(attr={ 'flu': 'i' }), DotMap(attr={ 'flu': 'r' })], dt=0.1),
        Group(m=950, attr={ 'flu': 's' }),
        Group(m= 50, attr={ 'flu': 'i' }),
        probe
    ]).
    run(1000)
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
