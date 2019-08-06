'''
A simulation of biological half-life, that is the time it takes for a substance to be removed (or transformed) by a
biological process.

The biological decay of two substances, A and B, is modeled with substance A decaying faster than subtance B.

One interesting observation is that even though the decay ordinary differential equation specifies only the amount of
undecayed substance, in PRAM we need to privide two derivatives because we move mass between two groups.  The two
resulting equations are inverts of one another.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import matplotlib.pyplot as plt

from dotmap import DotMap

from pram.data   import GroupSizeProbe, ProbePersistanceMem, ProbeMsgMode
from pram.entity import Group, GroupQry, GroupSplitSpec
from pram.rule   import ODESystemMass, Event, TimeAlways, Process
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init:

l_a = 1.0
l_b = 1.0 / 20.0

def f_decay_a(t, state):
    n1, n2 = state
    return [-l_a * n1, l_a * n1]

def f_decay_b(t, state):
    n1, n2 = state
    return [-l_b * n1, l_b * n1]


# ----------------------------------------------------------------------------------------------------------------------
# (2) Probe:

probe_u = GroupSizeProbe(
    name='sub',  # substance
    queries=[GroupQry(attr={ 'sub-a': 'u' }), GroupQry(attr={ 'sub-b': 'u' })],
    qry_tot=None,
    var_names=['pau', 'pbu', 'nau', 'nbu'],  # pau - proportion of substance A undecayed, etc.
    persistance=ProbePersistanceMem(),
    msg_mode=ProbeMsgMode.NONE
)

probe_d = GroupSizeProbe(
    name='sub',  # substance
    queries=[GroupQry(attr={ 'sub-a': 'd' }), GroupQry(attr={ 'sub-b': 'd' })],
    qry_tot=None,
    var_names=['pad', 'pbd', 'nad', 'nbd'],  # pad - proportion of substance A decayed, etc.
    persistance=ProbePersistanceMem(),
    msg_mode=ProbeMsgMode.NONE
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

s = (Simulation().
    set_pragma_autocompact(True).
    add([
        ODESystemMass(f_decay_a, [DotMap(attr={ 'sub-a': 'u' }), DotMap(attr={ 'sub-a': 'd' })], dt=0.1),
        ODESystemMass(f_decay_b, [DotMap(attr={ 'sub-b': 'u' }), DotMap(attr={ 'sub-b': 'd' })], dt=0.1),
        Group(m=1000, attr={ 'sub-a': 'u' }),
        Group(m=1000, attr={ 'sub-b': 'u' }),
        probe_u,
        probe_d
    ]).
    run(1000)
)


# ----------------------------------------------------------------------------------------------------------------------
# (4) Results:

# Time series plot (group mass) -- Undecayed:
# series = [
#     { 'var': 'pau', 'lw': 2, 'linestyle': '-',  'marker': '+', 'color': 'red',  'markersize': 0, 'lbl': 'Substance A' },
#     { 'var': 'pbu', 'lw': 2, 'linestyle': '--', 'marker': 'o', 'color': 'blue', 'markersize': 0, 'lbl': 'Substance B' }
# ]
# probe_u.plot(series, fpath_fig=None, figsize=(20,4), legend_loc='upper right', dpi=150)

# Time series plot (group mass) -- Decayed:
series = [
    { 'var': 'pad', 'lw': 2, 'linestyle': '-',  'marker': '+', 'color': 'red',  'markersize': 0, 'lbl': 'Substance A' },
    { 'var': 'pbd', 'lw': 2, 'linestyle': '--', 'marker': 'o', 'color': 'blue', 'markersize': 0, 'lbl': 'Substance B' }
]
probe_d.plot(series, fpath_fig=None, figsize=(20,4), legend_loc='lower right', dpi=150)
