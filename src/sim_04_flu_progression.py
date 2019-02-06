'''
A simulation progression of the flu.
'''

from pram.sim    import Simulation
from pram.entity import AttrFluStage, GroupSplitSpec
from pram.data   import GroupSizeProbe
from pram.rule   import Rule, TimeInt


rand_seed = 1928


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,20), memo=None):  # 8am - 8pm
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, t):
        # An equivalent Dynamic Bayesian net has the initial state distribution:
        #
        #     flu-stage: (1 0 0)
        #
        # and the transition model:
        #
        #           N     A     S
        #     N  0.95  0.20  0.05
        #     A  0.05  0     0.20
        #     S  0     0.80  0.75

        p = 0.05  # prob of infection

        if group.get_attr('flu-stage') == AttrFluStage.NO:
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=p,     attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.00,  attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.ASYMPT:
            return [
                GroupSplitSpec(p=0.20, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.80, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        elif group.get_attr('flu-stage') == AttrFluStage.SYMPT:
            return [
                GroupSplitSpec(p=0.05, attr_set={ 'flu-stage': AttrFluStage.NO     }),
                GroupSplitSpec(p=0.20, attr_set={ 'flu-stage': AttrFluStage.ASYMPT }),
                GroupSplitSpec(p=0.75, attr_set={ 'flu-stage': AttrFluStage.SYMPT  })
            ]
        else:
            raise ValueError("Invalid value for attribute 'flu-stage'.")

    def is_applicable(self, group, t):
        return super().is_applicable(t) and group.has_attr([ 'flu-stage' ])


# ----------------------------------------------------------------------------------------------------------------------
probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, memo='Mass distribution across flu stages')

(Simulation(6,1,16, rand_seed=rand_seed).
    new_group('0', 1000).
        set_attr('flu-stage', AttrFluStage.NO).
        commit().
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    run()
)
