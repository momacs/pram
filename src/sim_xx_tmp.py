from pram.sim  import Simulation
from pram.data import GroupSizeProbe
from pram.rule import Rule, TimeInt, TimePoint

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
        return super().is_applicable(t) and group.has_attr([ 'flu-stage' ])


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
    add_sites(forks).
    new_group('p0',  1).set_attr('state', State.THINKING).set_rel('fork.l', sites['f1' ]).set_rel('fork.r', sites['f11']).commit().
    new_group('p2',  1).set_attr('state', State.THINKING).set_rel('fork.l', sites['f4' ]).set_rel('fork.r', sites['f1' ]).commit().
    new_group('p5',  1).set_attr('state', State.THINKING).set_rel('fork.l', sites['f6' ]).set_rel('fork.r', sites['f4' ]).commit().
    new_group('p7',  1).set_attr('state', State.THINKING).set_rel('fork.l', sites['f8' ]).set_rel('fork.r', sites['f6' ]).commit().
    new_group('p10', 1).set_attr('state', State.THINKING).set_rel('fork.l', sites['f11']).set_rel('fork.r', sites['f8' ]).commit().
    add_rule(ThinkAndEatRule()).
    # add_probe(probe_grp_size_site).
    # summary((False, True, False, False, False), (0,1)).
    # run(1).summary((False, True, False, False, False), (1,1)).
    # run(1).summary((False, True, False, False, False), (1,1)).
    # run(1).summary((False, True, False, False, False), (1,1)).
    # run(1).summary((False, True, False, False, False), (1,1))
    # run().summary((False, True, False, False, False), (1,1))
    # run()
)




# from pram.entity import Site
# from pram.pop    import GroupPopulation
#
# class GeoSite(Site):
#     def __init__(self, name, geo_coord=(0.0, 0.0), pop=None):
#         super().__init__(name, pop=pop)
#         self.geo_coord = geo_coord
#
#     def print_location(self):
#         print('{} is on Earth.'.format(self.name))  # 'self.name' is a Site's property
#
# s = GeoSite('Pittsburgh', (40.440624, -79.995888), GroupPopulation())
# s.get_pop_size()    # method from Site
# s.print_location()  # method from GeoSite



# from pram.entity import GroupQry, Site
# from pram.pop import GroupPopulation
# s = Site('store')
# s.set_pop(GroupPopulation())
# s.get_pop_size()
# s.get_groups_here(GroupQry({ 'is-student': True }, { 'location': Site('pitt') }))



# from pram.entity import Group, GroupQry, Site
# from pram.pop    import GroupPopulation
#
# pop = GroupPopulation()
#
# s = Site('Pittsburgh Symphony Orchestra', pop=pop)
#
# g1 = Group(n=1000, attr={ 'is-student': True                 }, rel={ Site.AT: s })
# g2 = Group(n= 200, attr={ 'is-student': True, 'major': 'CS'  }, rel={ Site.AT: s })
# g3 = Group(n=  30, attr={ 'is-student': True, 'major': 'EPI' }, rel={ Site.AT: s })
# g4 = Group(n=   4, attr={ 'is-student': False                }, rel={ Site.AT: s })
#
# pop.add_groups([g1, g2, g3, g4])
#
# n1 = sum([g.n for g in s.get_groups_here()])
# n2 = sum([g.n for g in s.get_groups_here(GroupQry())])
# n3 = s.get_pop_size()
# print('{} {} {}'.format(n1, n2, n3))
#
# n4 = sum([g.n for g in s.get_groups_here(GroupQry({ 'is-student': True }))])
# print('{}'.format(n4))
#
# n5 = sum([g.n for g in s.get_groups_here() if g.has_attr('major')])
# print('{}'.format(n5))
#
# n6 = sum([g.n for g in s.get_groups_here(GroupQry({ 'is-student': True, 'major': 'CS' }))])
# print('{}'.format(n6))
#
# n7 = sum([g.n for g in s.get_groups_here(GroupQry({ 'is-student': True, 'major': 'PHIL' }))])
# print('{}'.format(n7))
