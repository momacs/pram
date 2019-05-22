'''
A simulation implementing am infectious disease spread model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import ProbePersistanceDB, ProbeMsgMode, GroupSizeProbe
from pram.entity import AttrFluStage, Group, GroupQry, GroupSplitSpec
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    def __init__(self, t=TimeAlways(), memo=None):
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, iter, t):
        # Time-homogenous Markov chain with a finite state space.
        #
        # Initial state distribution:
        #     flu: (1 0 0)
        #
        # Transition model:
        #           S     I     R
        #     S  0.95  0     0.10
        #     I  0.05  0.50  0
        #     R  0     0.50  0.90

        p = 0.05  # prob of infection

        if group.get_attr('flu') == 's':
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=p,     attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.00,  attr_set={ 'flu': 'r' })
            ]
        elif group.get_attr('flu') == 'i':
            return [
                GroupSplitSpec(p=0.00, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'r' })
            ]
        elif group.get_attr('flu') == 'r':
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu': 'r' })
            ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu' ])


# ----------------------------------------------------------------------------------------------------------------------
# (Simulation().
#     add_rule(ProgressFluRule()).
#     add_probe(GroupSizeProbe.by_attr('flu', 'flu', ['s', 'i', 'r'], msg_mode=ProbeMsgMode.DISP)).
#     add_group(Group(n=1000, attr={ 'flu': 's' })).
#     run(48)
# )


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
    var_names=['ps', 'pi', 'pr', 'ns', 'ni', 'nr'],
    persistance=ProbePersistanceDB(fpath_db)
)

(Simulation().
    add_rule(ProgressFluRule()).
    add_probe(probe).
    add_group(Group(n=1000, attr={ 'flu': 's' })).
    run(48)
)
