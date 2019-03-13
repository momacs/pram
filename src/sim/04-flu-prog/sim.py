'''
A simulation implementing the flu progression model.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.sim    import Simulation
from pram.entity import AttrFluStage, GroupSplitSpec
from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.rule   import Rule, TimeInt

from rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')

(Simulation(6,1,16, rand_seed=rand_seed).
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    new_group('0', 1000).
        set_attr('flu-stage', AttrFluStage.NO).
        commit().
    run()
)
