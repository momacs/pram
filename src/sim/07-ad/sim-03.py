'''
An Alzheimer's Disease simulation which models the following scenario: "AD incidence doubles every 5 years after 65 yo."
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import PoissonIncidenceProcess
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
# (1) Simulation (test population):

(Simulation().
    set().
        pragma_autocompact(True).
        pragma_live_info(False).
        done().
    add().
        rule(PoissonIncidenceProcess('ad', 65, 0.01, 2, 5, rate_delta_mode=PoissonIncidenceProcess.RateDeltaMode.EXP)).
        group(Group(m=100, attr={ 'age': 60 })).
        done().
    run(20).
    summary(False, 1024,0,0,0, (1,0))
)


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation (synthetic Allegheny County population):

# fpath_db = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny.sqlite3')
#
# (Simulation().
#     set().
#         pragma_autocompact(True).
#         pragma_live_info(False).
#         done().
#     add().
#         rule(PoissonIncidenceProcess('ad', 65, 0.01, 2, 5, rate_delta_mode=PoissonIncidenceProcess.RateDeltaMode.EXP)).
#         done().
#     db(fpath_db).
#         gen_groups(tbl='people', attr_db=[ 'age' ]).
#         done().
#     run(20).
#     summary(False, 1024,0,0,0, (1,0))
# )
