'''
A simulation implementing the SIRS model of infectious disease spread in a population.  A time-point perturbation is
added to demonstrate how larger simulations can be composed from individual models.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import ProbePersistanceDB, ProbeMsgMode, GroupSizeProbe
from pram.entity import Group, GroupQry, GroupSplitSpec
from pram.rule   import SIRSModel, Event, TimeAlways
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class FluSpikeEvent(Event):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='flu-spike-evt')

    def apply(self, pop, group, iter, t):
        return [
            GroupSplitSpec(p=0.90, attr_set={ 'flu': 'i' }),
            GroupSplitSpec(p=0.10)
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and iter == 30 and (group.ha({ 'flu': 's' }) or group.ha({ 'flu': 'r' }))


(Simulation().
    add_probe(GroupSizeProbe.by_attr('flu', 'flu', ['s', 'i', 'r'], msg_mode=ProbeMsgMode.DISP)).
    add_rule(SIRSModel('flu', beta=0.05, gamma=0.50, alpha=0.10)).
    add_rule(FluSpikeEvent()).
    add_group(Group(n=1000, attr={ 'flu': 's' })).
    run(48)
)


# ----------------------------------------------------------------------------------------------------------------------
# dpath_cwd = os.path.dirname(__file__)
# fpath_db  = os.path.join(dpath_cwd, f'sim.sqlite3')
#
# if os.path.isfile(fpath_db):
#     os.remove(fpath_db)
#
# probe = GroupSizeProbe(
#     name='flu',
#     queries=[
#         GroupQry(attr={ 'flu': 's' }),
#         GroupQry(attr={ 'flu': 'i' }),
#         GroupQry(attr={ 'flu': 'r' })
#     ],
#     qry_tot=None,
#     var_names=['ps', 'pi', 'pr', 'ns', 'ni', 'nr'],
#     persistance=ProbePersistanceDB(fpath_db)
# )
#
# (Simulation().
#     add_rule(SIRSModel('flu', beta=0.05, gamma=0.50, alpha=0.10)).
#     add_rule(FluSpikeEvent()).
#     add_probe(probe).
#     add_group(Group(n=1000, attr={ 'flu': 's' })).
#     run(48)
# )
