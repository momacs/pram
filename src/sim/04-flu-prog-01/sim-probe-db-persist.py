'''
A simulation implementing the flu progression model.  This version tests probe database persistence.
'''

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.sim    import Simulation
from pram.entity import GroupQry, GroupSplitSpec
from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistenceDB
from pram.rule   import Rule, TimeInt

from rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
dpath_cwd = os.path.dirname(__file__)
fpath_db  = os.path.join(dpath_cwd, f'sim.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

probe = GroupSizeProbe(
    name='flu',
    queries=[
        GroupQry(attr={ 'flu': 's' }),
        GroupQry(attr={ 'flu': 'i' }),
        GroupQry(attr={ 'flu': 'r' })
    ],
    qry_tot=None,
    var_names=['pn', 'pa', 'ps', 'nn', 'na', 'ns'],
    persistence=ProbePersistenceDB(fpath_db)
)

(Simulation().
    add_rule(ProgressFluRule()).
    add_probe(probe).
    new_group('0', 1000).
        set_attr('flu', 's').
        done().
    run(16)
)
