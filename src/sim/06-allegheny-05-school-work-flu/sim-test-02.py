'''
A test simulation involving the SEIR flu model that is run on an hourly basis (instead of the default hourly
basis).
'''

import pram.util as util

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, Site
from pram.rule   import Rule, SEIRFluRule, TimeAlways
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class HourRule(Rule):
    T_UNIT_MS = util.Time.MS.h

    def __init__(self):
        super().__init__('hour-rule', TimeAlways())

    def apply(self, pop, group, iter, t):
        pass


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928
probe_grp_size_flu = GroupSizeProbe.by_attr('flu', SEIRFluRule.ATTR, SEIRFluRule.State, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu states')

(Simulation().
    set().
        rand_seed(rand_seed).
        pragma_autocompact(True).
        pragma_autostop(True).
        pragma_autostop_t(5).
        pragma_autostop_n(1).
        # pragma_autostop_p(0.001).
        done().
    add().
        rule(SEIRFluRule()).
        rule(HourRule()).
        probe(probe_grp_size_flu).
        done().
    new_group(1000).
        done().
    summary(True, 0,0,0,0, (0,1)).
    run('8d')
    .summary(False, 8,0,0,0, (1,0))
)

# (Simulation().
#     set().
#         # iter_cnt(24 * 3).
#         rand_seed(rand_seed).
#         pragma_analyze(False).
#         pragma_autocompact(True).
#         done().
#     add().
#         rule(SEIRFluRule()).
#         rule(HourRule()).
#         probe(probe_grp_size_flu).
#         done().
#     new_group(1000).
#         done().
#     run().summary(False, 8,0,0,0).
#     run().summary(False, 8,0,0,0).
#     run().summary(False, 8,0,0,0).
#     run().summary(False, 8,0,0,0).
#     run().summary(False, 8,0,0,0)
# )
