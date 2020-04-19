'''
A test of the mass transfer graph.
'''

from pram.entity    import Group
from pram.model.epi import SIRSModel
from pram.sim       import Simulation
from pram.traj      import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

te = (
    TrajectoryEnsemble().
        add_trajectories([
            Trajectory(
                sim=(Simulation().
                    add([
                        SIRSModel('flu', beta, 0.50, 0.10),
                        Group(m=1000, attr={ 'flu': 's' })
                    ])
                ),
                name=f'SIR: b={round(beta,2)}'
            ) for beta in [0.05, 0.10]
        ]).
        set_group_names([
            (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
            (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
            (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
        ]).
        run(100)
)

# te.plot_mass_locus_line     ((1200,300), get_out_dir('_plot-line.png'), iter_range=(-1, -1))
# te.plot_mass_locus_line_aggr((1200,300), get_out_dir('_plot-aggr.png'), iter_range=(-1, -1))
# te.traj[1].plot_mass_locus_streamgraph((900,600), get_out_dir('_plot.png'), do_sort=True)
# te.traj[1].plot_heatmap((800,800), get_out_dir('_plot-heatmap.png'), (-1,20))
