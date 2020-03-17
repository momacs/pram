import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np

from pram.entity    import Group
from pram.model.epi import SIRSModel
from pram.sim       import Simulation
from pram.traj      import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'mass-dynamics', 'out', filename)

fpath_db = os.path.join(os.path.dirname(__file__), f'trajectory.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Create the database and run a trajectory ensemble:

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

te = (TrajectoryEnsemble(fpath_db).
    add_trajectories([
        Trajectory(
            (Simulation().
                add([
                    SIRSModel('flu', beta, 0.50, 0.00),
                    Group(m=1000, attr={ 'flu': 's' })
                ])
            ), f'SIR: b={round(beta,2)}'
        ) for beta in [0.05]  # np.arange(0.05, 0.06, 0.01)
    ]).
    set_group_names([
        (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
        (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
        (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
    ]).
    run(100)
)


# ----------------------------------------------------------------------------------------------------------------------
# (2) Restore a trajectory ensemble from the database for additional runs:

# TrajectoryEnsemble(fpath_db).run(20)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Restore a trajectory ensemble from the database for analysis:

# te = TrajectoryEnsemble(fpath_db)

# te.traj[1].plot_mass_flow_time_series(filepath=get_out_dir('states.png'), iter_range=(-1,10), v_prop=False, e_prop=True)
# te.traj[1].plot_group_mass_streamgraph((900,600), get_out_dir('group-streamgraph.png'), (-1,50))
# te.traj[1].plot_heatmap((800,800), get_out_dir('heatmap.png'), (-1,20))
