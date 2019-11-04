'''
A simulation modeling the SARS outbreak via the SEQIHR model.  A quarantine intervention interacts with the epidemic
reducing the number of infected; the sooner it is undertaken the lower the infected count.

"Between November 2002 and July 2003, an outbreak of SARS in southern China caused an eventual 8,098 cases, resulting in
774 deaths reported in 37 countries, with the majority of cases in China and Hong Kong (9.6% fatality rate) according
to the World Health Organization."

[https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome]

This system uses ordinary differential equations solver to implement the SEQIHR model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import matplotlib.pyplot as plt

from dotmap import DotMap
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
def get_out_fpath(filename):
    return os.path.join(os.path.dirname(__file__), 'out', filename)

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

class SARSQuarantineIntervention(Intervention):  # extends the Intervention primitive
    def __init__(self, seqihr_model, chi, i):
        Err.type(seqihr_model, 'seqihr_model', SEQIHRModel)

        super().__init__(i=i)
        self.seqihr_model = seqihr_model
        self.chi = chi

    def apply(self, pop, group, iter, t):
        self.seqihr_model.set_params(chi=self.chi)

    def get_inner_models(self):
        return [self.seqihr_model]


# ----------------------------------------------------------------------------------------------------------------------
if os.path.isfile(fpath_db): os.remove(fpath_db)

te = TrajectoryEnsemble(fpath_db)

if te.is_db_empty:  # generate simulation data if the trajectory ensemble database is empty
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectories([
        (Simulation().
            add([
                SARSQuarantineIntervention(
                    SEQIHRModel('sars', beta=0.80, alpha_n=0.75, alpha_q=0.40, delta_n=0.01, delta_i=0.03, mu=0.01, chi=0.01, phi=0.20, rho=0.75, solver=ODESolver()),
                    chi=0.99,
                    i=intervention_onset
                ),
                Group(m=950, attr={ 'sars': 's' }),
                Group(m= 50, attr={ 'sars': 'e' })
            ])
        ) for intervention_onset in TN(30,120, 75,100, 1)  # a 10-trajectory ensemble
    ])
    te.set_group_names(group_names)
    te.run(400)

# Visualize:
te.plot_mass_locus_line     ((1200,300), get_out_fpath('_plot-line.png'), col_scheme='tableau10', opacity_min=0.2)
# te.plot_mass_locus_line_aggr((1200,300), get_out_fpath('_plot-iqr.png'),  col_scheme='tableau10')
