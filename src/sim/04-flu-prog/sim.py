'''
A simulation implementing the flu progression model.
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.sim    import Simulation
from pram.entity import AttrFluStage, GroupSplitSpec
from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.rule   import Rule, TimeInt


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,20), memo=None):  # 8am - 8pm
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, iter, t):
        # An equivalent Dynamic Bayesian net has the initial state distribution:
        #
        #     flu-stage: (1 0 0)
        #
        # and the transition model:
        #
        #           N     A     S
        #     N  0.95  0.00  0.10
        #     A  0.05  0.50  0
        #     S  0     0.50  0.90

        p = 0.05  # prob of infection

        if group.get_attr('flu-stage') == AttrFluStage.NO:
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=p,     attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.00,  attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.ASYMPT:
            return [
                GroupSplitSpec(p=0.00, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.SYMPT:
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-stage'.")

    def is_applicable(self, group, iter, t):
        return super().is_applicable(iter, t) and group.has_attr([ 'flu-stage' ])


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')

(Simulation(6,1,16, rand_seed=rand_seed).
    new_group('0', 1000).
        set_attr('flu-stage', AttrFluStage.NO).
        commit().
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    run()
)
