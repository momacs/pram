''' Testing built-in serialization of the Simulation object. '''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pickle

from pram.data   import ProbeMsgMode, GroupSizeProbe
from pram.entity import AttrFluStage
from pram.sim    import Simulation

from sim.rules import ProgressFluRule


# ----------------------------------------------------------------------------------------------------------------------
fpath     = os.path.join(os.path.dirname(__file__), 'sim.pickle')
fpath_bz2 = os.path.join(os.path.dirname(__file__), 'sim.pickle.bz2')
fpath_gz  = os.path.join(os.path.dirname(__file__), 'sim.pickle.gz')

probe_grp_size_flu = GroupSizeProbe.by_attr('flu', 'flu-stage', AttrFluStage, msg_mode=ProbeMsgMode.DISP, memo='Mass distribution across flu stages')

s01 = (Simulation().
    add_rule(ProgressFluRule()).
    add_probe(probe_grp_size_flu).
    new_group(1000).
        set_attr('flu-stage', AttrFluStage.NO).
        done()
)

# Run and save:
s01.run(10)
s01.run(2)
s01.pickle(fpath)

# Load and run:
s02 = Simulation.unpickle(fpath)

print(id(s01) == id(s02))
print(s01 == s02)
print(s01 is s02)
print(f'Run count: {s01.run_cnt} {s02.run_cnt}')
print(f'Iteration: {s01.timer.i} {s02.timer.i}')

s02.run(2)

# BZ2:
s02.pickle_bz2(fpath_bz2)
s03 = Simulation.unpickle_bz2(fpath_bz2)

print(f'Run count: {s02.run_cnt} {s03.run_cnt}')
print(f'Iteration: {s02.timer.i} {s03.timer.i}')

s03.run(2)

# GunZip
s03.pickle_gz(fpath_gz)
s04 = Simulation.unpickle_gz(fpath_gz)

print(f'Run count: {s03.run_cnt} {s04.run_cnt}')
print(f'Iteration: {s03.timer.i} {s04.timer.i}')

s04.run(2)


# ----------------------------------------------------------------------------------------------------------------------
# Conclusion: All three versions of the serialization work.
