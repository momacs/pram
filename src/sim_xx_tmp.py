from pram.sim    import Simulation
from pram.data   import GroupSizeProbe
from pram.entity import Site
from pram.rule   import Rule, TimeInt, TimePoint

from enum import IntEnum


rand_seed = 1928

State = IntEnum('State', 'THINKING EATING')

T_THINK_MAX = 5
T_EAT_MAX   = 5


# ----------------------------------------------------------------------------------------------------------------------
class ThinkAndEatRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(0,23), memo=None):
        super().__init__('think-and-eat', t, memo)

    def apply(self, pop, group, t):
        pass

    def is_applicable(self, group, t):
        return super().is_applicable(t)


# ----------------------------------------------------------------------------------------------------------------------
forks = {
    'f1'  : Site('f1'),
    'f4'  : Site('f4'),
    'f6'  : Site('f6'),
    'f8'  : Site('f8'),
    'f11' : Site('f11')
}

# probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), memo='Mass distribution across sites')

(Simulation(1,1,14, rand_seed=rand_seed).
    new_group('p0',  1).set_attr('state', State.THINKING).set_rel('fork.l', forks['f1' ]).set_rel('fork.r', forks['f11']).commit().
    new_group('p2',  1).set_attr('state', State.THINKING).set_rel('fork.l', forks['f4' ]).set_rel('fork.r', forks['f1' ]).commit().
    new_group('p5',  1).set_attr('state', State.THINKING).set_rel('fork.l', forks['f6' ]).set_rel('fork.r', forks['f4' ]).commit().
    new_group('p7',  1).set_attr('state', State.THINKING).set_rel('fork.l', forks['f8' ]).set_rel('fork.r', forks['f6' ]).commit().
    new_group('p10', 1).set_attr('state', State.THINKING).set_rel('fork.l', forks['f11']).set_rel('fork.r', forks['f8' ]).commit().
    add_rule(ThinkAndEatRule()).
    run(1).summary((False, True, False, False, False), (1,1))
)
