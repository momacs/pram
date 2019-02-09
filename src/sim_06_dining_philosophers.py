'''
A simulation of the Dining Philosophers problem demonstrating the allocation and release of shared resources.  The five
philosophers ('p') and five forks ('f') are distinguished by approximate clock position.

This simulation does not run in a concurrent manner; if it did, the use of synchronization mechanisms would be
required.  That is not the point here.  The point is to experiment with the Resource entity.  Deadlocks are avoided by
not allowing philosophers to pick up either fork before checking both can be picked up.  Putting down a fork is always
allowed.

For simplicity, this simulation assumes that at each simulation step philosophers want to think or eat with the same
probability.
'''

import numpy as np

from pram.sim    import Simulation
from pram.data   import GroupSizeProbe
from pram.entity import GroupSplitSpec, Resource
from pram.rule   import Rule, TimeInt, TimePoint

from enum import IntEnum


rand_seed = 1928


# ----------------------------------------------------------------------------------------------------------------------
class TERule(Rule):
    '''
    Think-and-Eat rule (or "What do philosophers do when they're not asleep?").
    '''

    __slots__ = ('p_think', 'p_eat', 'is_verbose')

    STATE = 'state'
    FORK_L = 'fork.l'
    FORK_R = 'fork.r'

    State = IntEnum('State', 'THINKING EATING')

    def __init__(self, t=TimeInt(0,23), p_think=0.5, p_eat=0.5, is_verbose=False, memo=None):
        super().__init__('think-and-eat', t, memo)

        self.p_think = p_think
        self.p_eat   = p_eat
        self.is_verbose = is_verbose

    def apply(self, pop, group, t):
        return {
            self.State.THINKING : self.apply_thinking,
            self.State.EATING   : self.apply_eating
        }.get(group.get_attr(self.STATE))(pop, group, t)

    def apply_eating(self, pop, group, t):
        if np.random.random_sample() <= self.p_think:
            return

        if self.is_verbose: print('{:5} is done eating'.format(group.name))

        group.get_rel(self.FORK_L).release(1)
        group.get_rel(self.FORK_R).release(1)

        return [GroupSplitSpec(p=1.0, attr_set={ self.STATE: self.State.THINKING})]

    def apply_thinking(self, pop, group, t):
        if np.random.random_sample() <= self.p_eat:
            return

        if self.is_verbose: print('{:5} is done thinking; '.format(group.name), end='')

        fl = group.get_rel(self.FORK_L)
        fr = group.get_rel(self.FORK_R)

        if not fl.can_accommodate_all(1) or not fr.can_accommodate_all(1):
            if self.is_verbose: print('cannot eat'.format(fl.name, fr.name))
            return

        fl.allocate_all(1)
        fr.allocate_all(1)

        if self.is_verbose: print('begins to eat')

        return [GroupSplitSpec(p=1.0, attr_set={ self.STATE: self.State.EATING})]

    def is_applicable(self, group, t):
        return (
            super().is_applicable(t) and
            group.get_size() > 0 and
            group.has_rel(self.FORK_L) and
            group.has_rel(self.FORK_R)
        )


# ----------------------------------------------------------------------------------------------------------------------
forks = {
    'f1'  : Resource('f1'),
    'f4'  : Resource('f4'),
    'f6'  : Resource('f6'),
    'f8'  : Resource('f8'),
    'f11' : Resource('f11'),
}

(Simulation(0,1,20, do_disp_t=True, rand_seed=rand_seed).  # a 20-hour marathon
    new_group('p0',  1).set_attr(TERule.STATE, TERule.State.THINKING).set_rel(TERule.FORK_L, forks['f1' ]).set_rel(TERule.FORK_R, forks['f11']).commit().
    new_group('p2',  1).set_attr(TERule.STATE, TERule.State.THINKING).set_rel(TERule.FORK_L, forks['f4' ]).set_rel(TERule.FORK_R, forks['f1' ]).commit().
    new_group('p5',  1).set_attr(TERule.STATE, TERule.State.THINKING).set_rel(TERule.FORK_L, forks['f6' ]).set_rel(TERule.FORK_R, forks['f4' ]).commit().
    new_group('p7',  1).set_attr(TERule.STATE, TERule.State.THINKING).set_rel(TERule.FORK_L, forks['f8' ]).set_rel(TERule.FORK_R, forks['f6' ]).commit().
    new_group('p10', 1).set_attr(TERule.STATE, TERule.State.THINKING).set_rel(TERule.FORK_L, forks['f11']).set_rel(TERule.FORK_R, forks['f8' ]).commit().
    add_rule(TERule(is_verbose=True)).
    # run(1).summary((False, True, False, False, False), (0,1)).
    # run(1).summary((False, True, False, False, False), (0,1)).
    # run(1).summary((False, True, False, False, False), (0,1)).
    # run(1).summary((False, True, False, False, False), (0,1)).
    # run(1).summary((False, True, False, False, False), (0,1))
    run()
)
