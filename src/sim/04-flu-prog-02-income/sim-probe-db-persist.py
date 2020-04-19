'''
A simulation implementing the flu progression model.  This version tests probe database persistence.
'''

import os
import sys

from pram.sim    import Simulation
from pram.entity import AttrFluStage, GroupQry, GroupSplitSpec
from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistenceDB
from pram.rule   import Rule, TimeInt

from rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

dpath_cwd = os.path.dirname(__file__)
fpath_db  = os.path.join(dpath_cwd, f'sim.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

probe = GroupSizeProbe(
    name='flu',
    queries=[
        GroupQry(attr={ 'flu-stage': AttrFluStage.NO     }),
        GroupQry(attr={ 'flu-stage': AttrFluStage.ASYMPT }),
        GroupQry(attr={ 'flu-stage': AttrFluStage.SYMPT  })
    ],
    qry_tot=None,
    var_names=['pn', 'pa', 'ps', 'nn', 'na', 'ns'],
    persistence=ProbePersistenceDB(fpath_db)
)

(Simulation(6,1,16, rand_seed=rand_seed).
    add_rule(ProgressFluRule()).
    add_probe(probe).
    new_group('0', 1000).
        set_attr('flu-stage', AttrFluStage.NO).
        commit().
    run()
)
