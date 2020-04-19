'''
A simulation implementing the SIRS model of infectious disease spread in a population and in particular demonstrating
probe plotting.
'''

from pram.data      import ProbePersistenceMem, GroupSizeProbe
from pram.entity    import Group
from pram.model.epi import SIRSModel
from pram.sim       import Simulation


# ----------------------------------------------------------------------------------------------------------------------
p = GroupSizeProbe.by_attr('flu', 'flu', ['s', 'i', 'r'], persistence=ProbePersistenceMem())

(Simulation().
    add_probe(p).
    add_rule(SIRSModel('flu', beta=0.05, gamma=0.50, alpha=0.10)).
    add_group(Group(m=1000, attr={ 'flu': 's' })).
    run(100)
)

series = [
    { 'var': 'p0', 'lw': 0.75, 'ls': 'solid',  'marker': 'o', 'color': 'red',   'ms': 0, 'lbl': 'S' },
    { 'var': 'p1', 'lw': 0.75, 'ls': 'dashed', 'marker': '+', 'color': 'blue',  'ms': 0, 'lbl': 'I' },
    { 'var': 'p2', 'lw': 0.75, 'ls': 'dotted', 'marker': 'x', 'color': 'green', 'ms': 0, 'lbl': 'R' }
]
p.plot(series)

    # series = [
    #     { 'var': f'{quantity}_S', 'lw': 1.50, 'ls': 'solid', 'dashes': (4,8), 'marker': '+', 'color': 'blue',   'ms': 0, 'lbl': 'S' },
    #     { 'var': f'{quantity}_E', 'lw': 1.50, 'ls': 'solid', 'dashes': (1,0), 'marker': '+', 'color': 'orange', 'ms': 0, 'lbl': 'E' },
    #     { 'var': f'{quantity}_I', 'lw': 1.50, 'ls': 'solid', 'dashes': (5,1), 'marker': '*', 'color': 'red',    'ms': 0, 'lbl': 'I' },
    #     { 'var': f'{quantity}_R', 'lw': 1.50, 'ls': 'solid', 'dashes': (5,6), 'marker': '|', 'color': 'green',  'ms': 0, 'lbl': 'R' },
    #     { 'var': f'{quantity}_X', 'lw': 1.50, 'ls': 'solid', 'dashes': (1,2), 'marker': 'x', 'color': 'black',  'ms': 0, 'lbl': 'X' }
    # ]
    # sim.probes[0].plot(series, ylabel='Population mass', figsize=(12,4), subplot_b=0.15)

# print(p.get_data())
