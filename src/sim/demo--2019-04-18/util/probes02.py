import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from pram.data   import GroupSizeProbe, ProbePersistanceDB
from pram.entity import GroupQry, Site


# ----------------------------------------------------------------------------------------------------------------------
fpath_out = os.path.join(os.path.dirname(__file__), '..', 'out', '02-flu.sqlite3')

if os.path.isfile(fpath_out):
    os.remove(fpath_out)

pp = ProbePersistanceDB(fpath_out)


# ----------------------------------------------------------------------------------------------------------------------
def probe_flu_at(school, name=None):
    return GroupSizeProbe(
        name=name or school.name,
        queries=[
            GroupQry(attr={ 'flu': 's' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'e' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'r' }, rel={ 'school': school })
        ],
        qry_tot=GroupQry(rel={ 'school': school }),
        persistance=pp,
        var_names=['ps', 'pe', 'pr', 'ns', 'ne', 'nr']
    )
