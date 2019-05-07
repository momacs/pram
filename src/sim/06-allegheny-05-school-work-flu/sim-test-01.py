'''
A test simulation involving the SEIR flu model in isolation.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, Site
from pram.rule   import SEIRFluRule
from pram.sim    import Simulation


rand_seed = 1928
probe_grp_size_flu = GroupSizeProbe.by_attr('flu', SEIRFluRule.ATTR, SEIRFluRule.State, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu states')

(Simulation().
    set().
        rand_seed(rand_seed).
        done().
    add().
        rule(SEIRFluRule()).
        probe(probe_grp_size_flu).
        done().
    new_group(1000).
        done().
    summary(True, 0,0,0,0, (0,1)).
    run(16).
    compact().
    summary(False, 8,0,0,0, (1,0))
)

# (Simulation().
#     set().
#         rand_seed(rand_seed).
#         pragma_analyze(False).
#         pragma_autocompact(True).
#         done().
#     add().
#         rule(SEIRFluRule()).
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
