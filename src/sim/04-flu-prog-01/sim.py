'''
A simulation implementing the flu progression model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import ProbeMsgMode, GroupSizeProbe
from pram.entity import AttrFluStage
from pram.sim    import Simulation

from rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')

(Simulation().
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    new_group(1000).
        set_attr('flu-stage', AttrFluStage.NO).
        done().
    run(24)
)
