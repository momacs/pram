'''
A simulation implementing the SIRS model of infectious disease spread in a population and in particular demonstrating
probe plotting.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


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
    { 'var': 'p0', 'lw': 0.75, 'linestyle': '-',  'marker': 'o', 'color': 'red',   'markersize': 0, 'lbl': 'S' },
    { 'var': 'p1', 'lw': 0.75, 'linestyle': '--', 'marker': '+', 'color': 'blue',  'markersize': 0, 'lbl': 'I' },
    { 'var': 'p2', 'lw': 0.75, 'linestyle': ':',  'marker': 'x', 'color': 'green', 'markersize': 0, 'lbl': 'R' }
]
p.plot(series, figsize=(16,3))

# print(p.get_data())
