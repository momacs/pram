



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