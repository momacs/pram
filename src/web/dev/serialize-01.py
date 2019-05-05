''' Testing serializing the Simulation object. '''

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
fpath = os.path.join(os.path.dirname(__file__), 'sim.pickle')

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

with open(fpath, 'wb') as f:
    pickle.dump(s01, f)

# Load and run:
with open(fpath, 'rb') as f:
    s02 = pickle.load(f)

print(id(s01) == id(s02))
print(s01 == s02)
print(s01 is s02)
print(f'Run count: {s01.run_cnt} {s02.run_cnt}')
print(f'Iteration: {s01.timer.i} {s02.timer.i}')

s02.run(2)


# ----------------------------------------------------------------------------------------------------------------------
# Conclusion: The above Simulation object can be serialized and deserialized between runs without issues.
