import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np

from pram.entity import Group
from pram.rule   import FibonacciSeq
from pram.sim    import Simulation, Trajectory, TrajectoryEnsemble
from pram.rule   import SIRSModel


fpath_db = os.path.join(os.path.dirname(__file__), f'trajectory.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

(
    TrajectoryEnsemble(fpath_db).
    add_trajectories([
        Trajectory(f'SIRS: b={round(b,2)}', None, (Simulation().add([SIRSModel('flu', beta=round(b,2), gamma=0.50, alpha=0.10), Group(n=1000, attr={ 'flu': 's' })]))) for b in np.arange(0.01, 0.06, 0.01)
    ]).
    run(2)
)
