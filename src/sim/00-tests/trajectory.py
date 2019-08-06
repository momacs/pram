import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np

from pram.entity import Group
from pram.rule   import FibonacciSeq
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble
from pram.rule   import SIRSModel


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

fpath_db = os.path.join(os.path.dirname(__file__), f'trajectory.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Create the database and run a trajectory ensemble:

# if os.path.isfile(fpath_db):
#     os.remove(fpath_db)
#
# te = (
#     TrajectoryEnsemble(fpath_db).
#         add_trajectories([
#             Trajectory(
#                 f'SIR: b={round(beta,2)}',
#                 None,
#                 (
#                     Simulation().
#                     add([
#                         SIRSModel('flu', beta, 0.50, 0.00),
#                         Group(m=1000, attr={ 'flu': 's' })
#                     ])
#                 )
#             ) for beta in np.arange(0.01, 0.03, 0.01)
#         ]).
#         run(10)
# )


# ----------------------------------------------------------------------------------------------------------------------
# (2) Restore a trajectory ensemble from the database:

te = (
    TrajectoryEnsemble(fpath_db).
    run(5)
)

# for t in te.traj.values():
#     print(t.id)
#     t.plot_states((1600,900), os.path.join(os.path.dirname(__file__), 'mass-dynamics', 'states.png'))
