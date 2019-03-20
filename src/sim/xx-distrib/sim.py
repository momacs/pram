'''
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np

from pram.entity import GroupSplitSpec
from pram.rule   import Rule, TimeInt, TimePoint
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
# Init:

rand_seed = 1928

dpath_res    = os.path.join(os.sep, 'Volumes', 'd', 'pitt', 'sci', 'pram', 'res', 'fred')
dpath_cwd    = os.path.dirname(__file__)
fpath_db_in  = os.path.join(dpath_res, 'allegheny.sqlite3')
fpath_groups = os.path.join(dpath_cwd, 'allegheny-groups.pickle.gz')

do_remove_file_groups = True

if do_remove_file_groups and os.path.isfile(fpath_groups):
    os.remove(fpath_groups)


# ----------------------------------------------------------------------------------------------------------------------
class XRule(Rule):
    def __init__(self, t=TimeInt(0,23), memo=None):
        super().__init__('think-and-eat', t, memo)

    def apply(self, pop, group, iter, t):
        pass

    def is_applicable(self, group, iter, t):
        return super().is_applicable(iter, t)


# ----------------------------------------------------------------------------------------------------------------------
(Simulation(0,1,20, rand_seed=rand_seed).
    add_rule().
    gen_groups_from_db(
        fpath_db_in,
        tbl='people',
        attr={},
        rel={},
        attr_db=[],
        rel_db=[],
        rel_at=None,
        fpath=fpath_groups
    ).
    summary().
    run()
)
