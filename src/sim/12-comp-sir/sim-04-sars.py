'''
A simulation modeling a COVID-19 outbreak via the SEQIHR model.  A quarantine intervention interacts with the epidemic
reducing the number of infected; the sooner it is undertaken the lower the infected count.

This system uses ordinary differential equations solver to implement the SEQIHR model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from scipy.stats import truncnorm, uniform

from pram.entity      import Group
from pram.model.model import ODESolver
from pram.model.epi   import SEQIHRModel
from pram.rule        import ODESystemMass, Intervention
from pram.sim         import Simulation
from pram.traj        import Trajectory, TrajectoryEnsemble
from pram.util        import Err


# ----------------------------------------------------------------------------------------------------------------------
fpath_db = os.path.join(os.path.dirname(__file__), 'data', 'seqihr.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
def TN(a,b, mu, sigma, n=None):
    return truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(n)

group_names = [
    (0, 'S', Group.gen_hash(attr={ 'sars': 's' })),
    (1, 'E', Group.gen_hash(attr={ 'sars': 'e' })),
    (2, 'Q', Group.gen_hash(attr={ 'sars': 'q' })),
    (3, 'I', Group.gen_hash(attr={ 'sars': 'i' })),
    (4, 'H', Group.gen_hash(attr={ 'sars': 'h' })),
    (5, 'R', Group.gen_hash(attr={ 'sars': 'r' }))
]


# ----------------------------------------------------------------------------------------------------------------------
# A quarantine intervention rule:

class SARSQuarantineIntervention(Intervention):
    def __init__(self, seqihr_model, chi, i):
        Err.type(seqihr_model, 'seqihr_model', SEQIHRModel)

        super().__init__(i=i)
        self.seqihr_model = seqihr_model
        self.chi = chi

        self.rules.append(self.seqihr_model)

    def apply(self, pop, group, iter, t):
        self.seqihr_model.set_params(chi=self.chi)


# ----------------------------------------------------------------------------------------------------------------------
# Simulations:

if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectories([
        (Simulation().
            add([
                SARSQuarantineIntervention(
                    SEQIHRModel('sars', beta=0.80, alpha_n=0.75, alpha_q=0.40, delta_n=0.01, delta_h=0.03, mu=0.01, chi=0.01, phi=0.20, rho=0.75, solver=ODESolver()),
                    chi=0.99,
                    i=intervention_onset
                ),
                Group(m=95000, attr={ 'sars': 's' }),
                Group(m= 5000, attr={ 'sars': 'e' })
            ])
        ) for intervention_onset in TN(30,120, 75,100, 5)
    ])
    te.set_group_names(group_names)
    te.run(400)

# te.plot_mass_locus_line     ((1200,300), os.path.join(os.path.dirname(__file__), 'out', 'plot-line.png'), col_scheme='tableau10', opacity_min=0.35)
# te.plot_mass_locus_line_aggr((1200,300), os.path.join(os.path.dirname(__file__), 'out', 'plot-ci.png'),   col_scheme='tableau10')

# fpath_diag = os.path.join(os.path.dirname(__file__), 'out', 'sim-04.diag')
# fpath_pdf  = os.path.join(os.path.dirname(__file__), 'out', 'sim-04.pdf')
# te.traj[2].sim.gen_diagram(fpath_diag, fpath_pdf)
