'''
A simulation implementing an infectious disease progression model.  The model is given as a time-homogenous Markov
chain with a finite state space with the initial state distribution:

    flu: (1 0 0)

and the transition model:
          s     i     r
    s  0.95  0     0.10
    i  0.05  0.50  0
    r  0     0.50  0.90
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import ProbePersistanceDB, ProbeMsgMode, GroupSizeProbe
from pram.entity import AttrFluStage, Group, GroupQry, GroupSplitSpec
from pram.rule   import MCRule, TimeAlways
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
(Simulation().
    add_rule(MCRule('flu', { 's': [0.95, 0.05, 0.00], 'i': [0.00, 0.50, 0.50], 'r': [0.10, 0.00, 0.90] })).
    add_probe(GroupSizeProbe.by_attr('flu', 'flu', ['s', 'i', 'r'], msg_mode=ProbeMsgMode.DISP)).
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
#     add_rule(MCRule('flu', { 's': [0.95, 0.05, 0.00], 'i': [0.00, 0.50, 0.50], 'r': [0.10, 0.00, 0.90] })).
#     add_probe(probe).
#     add_group(Group(n=1000, attr={ 'flu': 's' })).
#     run(48)
# )
