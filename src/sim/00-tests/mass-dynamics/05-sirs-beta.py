'''
A test of the mass transfer graph.
'''

from scipy.stats import beta

from pram.entity import Group, GroupSplitSpec
from pram.rule   import Process, SIRSModel, TimeAlways
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', '05-sirs-beta.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# Plot the histogram of random samples from the beta distribution used later on:

# import numpy as np
# import matplotlib.pyplot as plt
#
# from scipy.stats import beta
#
# print(beta.rvs(a=2.0, b=25.0, loc=0.0, scale=1.0))
#
# fig = plt.figure(figsize=(10,2), dpi=150)
# plt.hist(beta.rvs(a=2.0, b=25.0, loc=0.0, scale=1.0, size=100000), bins=200)
# plt.show()
# sys.exit(0)


# ----------------------------------------------------------------------------------------------------------------------
# Flu random beta process:

class FluRandomBetaProcess(Process):
    def __init__(self):
        super().__init__('flu-random-beta-proc', TimeAlways())

    def apply(self, pop, group, iter, t):
        p = beta.rvs(a=2.0, b=25.0, loc=0.0, scale=1.0)
        return [GroupSplitSpec(p=p, attr_set={ 'flu': 's' }), GroupSplitSpec(p=1-p)]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha({ 'flu': 'r' })


# ----------------------------------------------------------------------------------------------------------------------
# Test iteration range normalization:
#
# te = TrajectoryEnsemble(fpath_db)
# print(te.normalize_iter_range())
# print(te.normalize_iter_range((-2,-1)))
# print(te.normalize_iter_range((-1,-2)))
# print(te.normalize_iter_range((3,-1)))
# print(te.normalize_iter_range((-1,10)))
# print(te.normalize_iter_range((3,10)))
# print(te.normalize_iter_range((30,10)))
# sys.exit(0)


# ----------------------------------------------------------------------------------------------------------------------
# Generate:

# if os.path.isfile(fpath_db):
#     os.remove(fpath_db)
#
# te = (
#     TrajectoryEnsemble(fpath_db).
#         add_trajectories([
#             Trajectory(
#                 sim=(Simulation().
#                     add([
#                         SIRSModel('flu', 0.2, 0.5, 0.05),
#                         FluRandomBetaProcess(),
#                         Group(m=1000, attr={ 'flu': 's' })
#                     ])
#                 )
#             ) for _ in range(500)
#         ]).
#         set_group_names([
#             (0, 'S', Group.gen_hash(attr={ 'flu': 's' })),
#             (1, 'I', Group.gen_hash(attr={ 'flu': 'i' })),
#             (2, 'R', Group.gen_hash(attr={ 'flu': 'r' }))
#         ]).
#         run(50)
# )


# ----------------------------------------------------------------------------------------------------------------------
# Load:

te = TrajectoryEnsemble(fpath_db).stats()


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

def get_out_dir(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

# te.traj[1].plot_mass_locus_streamgraph((1200,600), get_out_dir('_plot.png'), do_sort=True)
# te.traj[1].plot_mass_locus_freq((12,6), get_out_dir('_plot.png'), do_sort=True)

# te.plot_mass_locus_line((1200,600), get_out_dir('_plot.png'), nsamples=100)
# te.plot_mass_locus_line_aggr((1200,600), get_out_dir('_plot.png'))
