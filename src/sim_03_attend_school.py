'''
A simulation testing a group-state-aware school attending rule.
'''

from pram.sim    import Simulation
from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.data   import GroupSizeProbe
from pram.rule   import GotoRule, Rule, TimeInt, TimePoint


rand_seed = 1928


# ----------------------------------------------------------------------------------------------------------------------
class ResetDayRule(Rule):
    __slots__ = ()

    def __init__(self, t, memo=None):
        super().__init__('reset-day', t, memo)

    def apply(self, pop, group, t):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],

    def is_applicable(self, group, t):
        return super().is_applicable(t)


# ----------------------------------------------------------------------------------------------------------------------
class AttendSchoolRule(Rule):
    __slots__ = ()

    def __init__(self, t=TimeInt(8,16), memo=None):
        super().__init__('attend-school', t, memo)

    def apply(self, pop, group, t):
        if group.has_rel({ Site.AT: group.get_rel('home') }) and (not group.has_attr('did-attend-school-today') or group.has_attr({ 'did-attend-school-today': False })):
            return self.apply_at_home(group, t)

        if group.has_rel({ Site.AT:  group.get_rel('school') }):
            return self.apply_at_school(group, t)

    def apply_at_home(self, group, t):
        p = { 8:0.50, 9:0.50, 10:0.50, 11:0.50, 12:1.00 }.get(t, 0.00)  # TODO: Provide these as a CDF
            # prob of going to school = f(time of day)

        return [
            GroupSplitSpec(p=p, attr_set={ 'did-attend-school-today': True, 't-at-school': 0 }, rel_set={ Site.AT: group.get_rel('school') }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_at_school(self, group, t):
        t_at_school = group.get_attr('t-at-school')
        p = { 0: 0.00, 1:0.05, 2:0.05, 3:0.25, 4:0.50, 5:0.70, 6:0.80, 7:0.90, 8:1.00 }.get(t_at_school, 1.00) if t < self.t.t1 else 1.00
            # prob of going home = f(time spent at school)

        return [
            GroupSplitSpec(p=p, attr_set={ 't-at-school': (t_at_school + 1) }, rel_set={ Site.AT: group.get_rel('home') }),
            GroupSplitSpec(p=1 - p, attr_set={ 't-at-school': (t_at_school + 1) })
        ]

        # TODO: Give timer information to the rule so it can appropriate determine time passage (e.g., to add it to 't-at-school' above).

    def is_applicable(self, group, t):
        return (
            super().is_applicable(t) and
            group.has_attr({ 'is-student': True }) and
            group.has_rel(['home', 'school']))

    @staticmethod
    def setup(pop, group):
        return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]  # attr_del=['t-at-school'],


# ----------------------------------------------------------------------------------------------------------------------
sites = {
    'home'     : Site('home'),
    'school-a' : Site('school-a'),
    'school-b' : Site('school-b')
}

probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), memo='Mass distribution across sites')

(Simulation(7,1,10, rand_seed=rand_seed).
    new_group('0', 500).
        set_attr('is-student', True).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-a']).
        commit().
    new_group('1', 500).
        set_attr('is-student', True).
        set_rel(Site.AT,  sites['home']).
        set_rel('home',   sites['home']).
        set_rel('school', sites['school-b']).
        commit().
    add_rule(ResetDayRule(TimePoint(7))).
    add_rule(AttendSchoolRule()).
    add_probe(probe_grp_size_site).
    run()
    # .summary((False, True, False, False, False), (0,1))  # print groups at the end of simulation
)

# (Simulation(7,1,10, rand_seed=rand_seed).
#     new_group('0', 500).
#         set_attr('is-student', True).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-a']).
#         commit().
#     new_group('1', 500).
#         set_attr('is-student', True).
#         set_rel(Site.AT,  sites['home']).
#         set_rel('home',   sites['home']).
#         set_rel('school', sites['school-b']).
#         commit().
#     add_rule(ResetDayRule(TimePoint(7))).
#     add_rule(AttendSchoolRule()).
#     add_probe(probe_grp_size_site).
#            summary((False, True, False, False, False), (0,1)).
#     run(1).summary((False, True, False, False, False), (0,1)).
#     run(1).summary((False, True, False, False, False), (0,1)).
#     run(1).summary((False, True, False, False, False), (0,1)).
#     run(1).summary((False, True, False, False, False), (0,1)).
#     run(1).summary((False, True, False, False, False), (0,1))
# )
