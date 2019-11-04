'''
An example of how a population of agents can be generated from the recorded population-level mass dynamics.

For simplicity, a one-rule simulation with no relations and only three groups is used.  Specifically, it is the SIR
model implemented as a Markov chain.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


import numpy as np

from pram.entity      import Group
from pram.model.model import MCSolver
from pram.model.epi   import SIRSModel
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'sim-01.sqlite3')

group_names = [
    (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
    (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
    (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
]

def get_out_fpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# ----------------------------------------------------------------------------------------------------------------------
# if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:
    te.add_trajectories([
        Trajectory(
            (Simulation().
                add([
                    SIRSModel('flu', 0.10, 0.50, 0.00, solver=MCSolver()),
                    Group(m=900, attr={ 'flu': 's' }),
                    Group(m=100, attr={ 'flu': 'i' })
                ])
            )
        )
    ])
    te.set_group_names(group_names)
    te.run(100)


# ----------------------------------------------------------------------------------------------------------------------
# te.traj[1].plot_mass_locus_line((1200,300), get_out_fpath('sim-01.png'))

print(te.traj[1].gen_agent(4))        # four-iteration simulation
print(te.traj[1].gen_agent_pop(2,3))  # two-agent population simulated for three iterations
