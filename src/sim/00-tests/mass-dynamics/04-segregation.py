'''
A test of the mass transfer graph for the segregation model.
'''

import math
import random

from scipy.stats import poisson

from pram.entity import Group, Site
from pram.rule   import SegregationModel
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
loc = [Site('a'), Site('b')]

s = (Simulation().
    set().
        pragma_autocompact(True).
        pragma_live_info(False).
        done().
    add([
        SegregationModel('team', len(loc)),
        Group(m=200, attr={ 'team': 'blue' }, rel={ Site.AT: loc[0] }),
        Group(m=300, attr={ 'team': 'blue' }, rel={ Site.AT: loc[1] }),
        Group(m=100, attr={ 'team': 'red'  }, rel={ Site.AT: loc[0] }),
        Group(m=400, attr={ 'team': 'red'  }, rel={ Site.AT: loc[1] })
    ])
)


# ----------------------------------------------------------------------------------------------------------------------
te = (TrajectoryEnsemble().
    add_trajectory(Trajectory('segregation', None, s)).
    set_group_names([
        (0, 'Blue A', Group.gen_hash(attr={ 'team': 'blue' }, rel={ Site.AT: loc[0] })),
        (1, 'Blue B', Group.gen_hash(attr={ 'team': 'blue' }, rel={ Site.AT: loc[1] })),
        (2, 'Red A',  Group.gen_hash(attr={ 'team': 'red'  }, rel={ Site.AT: loc[0] })),
        (3, 'Red B',  Group.gen_hash(attr={ 'team': 'red'  }, rel={ Site.AT: loc[1] }))
    ]).
    run(100)
)

def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

# te.traj[1].plot_mass_flow_time_series(filepath=get_out_dir('_plot.png'), iter_range=(-1,7), v_prop=False, e_prop=True)
# te.traj[1].plot_mass_locus_streamgraph((900,400), get_out_dir('_plot.png'))
# te.traj[1].plot_heatmap((800,800), get_out_dir('_plot.png'), (-1,20))

te.traj[1].plot_mass_locus_recurrence((16,8), get_out_dir('_plot.png'), Group.gen_hash(attr={ 'team': 'blue' }, rel={ Site.AT: loc[0] }), iter_range=(-1, 4000))
