'''
A test of the mass transfer graph.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


from pram.entity import Group
from pram.rule   import SIRSModel
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

te = (
    TrajectoryEnsemble().
        add_trajectories([
            Trajectory(f'SIR: b={round(beta,2)}', None,
                (Simulation().
                    add([
                        SIRSModel('flu', beta, 0.50, 0.10),
                        Group(m=1000, attr={ 'flu': 's' })
                    ])
                )
            ) for beta in [0.05]
        ]).
        set_group_names([
            (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
            (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
            (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
        ]).
        run(100)
)

# te.traj[1].plot_mass_flow_time_series(filepath=get_out_dir('_plot.png'), iter_range=(-1,7), v_prop=False, e_prop=True)
te.traj[1].plot_mass_locus_streamgraph((900,600), get_out_dir('_plot.png'), do_sort=True)
# te.traj[1].plot_heatmap((800,800), get_out_dir('_plot.png'), (-1,20))
