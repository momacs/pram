import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.rule import Rule, TimeAlways
from pram.sim  import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class AttrRule(Rule):
    def __init__(self):
        super().__init__('attr-rule', TimeAlways())

    def apply(self, pop, group, iter, t):
        if iter > 10:                     # dynamic rule analysis will capture 'income' on the 11th iteration
            if group.has_attr('income'):  # static rule analysis will always capture 'income'
                pass


# ----------------------------------------------------------------------------------------------------------------------
(Simulation().
    add_rule(AttrRule()).
    new_group(100).
        set_attr('income', 50).
        set_attr('age', 22).
        done().
    run(5).
    summary(True, 0,0,0,0)
)


# ----------------------------------------------------------------------------------------------------------------------
# Key points
#     Static rule analysis  - Form groups
#     Dynamic rule analysis - Alert to attributes the modeler might have missed
