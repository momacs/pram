'''
Segregation model.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import random

from scipy.stats import poisson

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation
from pram.util   import Time as TimeU


# ----------------------------------------------------------------------------------------------------------------------
class SegregateRule(Rule):
    '''
    Segregation model.
    '''

    def __init__(self, attr, attr_dom, p_migrate=0.05):
        super().__init__('segregation-rule', TimeAlways())

        self.attr = attr
        self.p_migrate = p_migrate           # proportion of the population that will migrate if repelled
        self.p_repel = 1.00 / len(attr_dom)  # population will be repelled (i.e., will move) if the site that population is at has a proportion of same self.attr lower than this

    def apply(self, pop, group, iter, t):
        attr   = group.ga(self.attr)
        site   = group.gr(Site.AT)
        n      = site.get_pop_size()
        n_team = site.get_pop_size(GroupQry(attr={ self.attr: attr }))

        if n == 0:
            return None

        p_team = n_team / n  # proportion of same self.attr

        if p_team < self.p_repel:
        # if max(self.p_repel - 0.05, 000) < p_team < min(self.p_repel + 0.05, 1.00):
            site_rnd = self.get_random_site(pop, site)
            return [
                GroupSplitSpec(p=    self.p_migrate, rel_set={ Site.AT: site_rnd }),
                GroupSplitSpec(p=1 - self.p_migrate)
            ]
        else:
            return None

    def get_random_site(self, pop, site):
        s = random.choice(list(pop.sites.values()))
        while s == site:
            s = random.choice(list(pop.sites.values()))
        return s

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha([self.attr]) and group.hr([Site.AT])


# ----------------------------------------------------------------------------------------------------------------------
# (1) Simulation (two locations)
#
# loc = [Site('a'), Site('b')]
#
# probe_loc = GroupSizeProbe.by_rel('site', Site.AT, loc, msg_mode=ProbeMsgMode.DISP)
#
# (Simulation().
#     set().
#         pragma_autocompact(True).
#         pragma_live_info(False).
#         done().
#     add([
#         SegregateRule('team', loc),
#         Group(n=200, attr={ 'team': 'blue' }, rel={ Site.AT: loc[0] }),
#         Group(n=300, attr={ 'team': 'blue' }, rel={ Site.AT: loc[1] }),
#         Group(n=100, attr={ 'team': 'red'  }, rel={ Site.AT: loc[0] }),
#         Group(n=400, attr={ 'team': 'red'  }, rel={ Site.AT: loc[1] }),
#         probe_loc
#     ]).
#     run(10)
# )


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation (arbitrary numer of locations)

# loc = [Site('a'), Site('b'), Site('c')]
#
# probe_loc = GroupSizeProbe.by_rel('site', Site.AT, loc, msg_mode=ProbeMsgMode.DISP)
#
# def grp_setup(pop, group):
#     p = [random.random() for _ in range(len(loc))]
#     p = [i / sum(p) for i in p]
#     return [GroupSplitSpec(p=p[i], rel_set={ Site.AT: loc[i] }) for i in range(len(loc))]
#
# (Simulation().
#     set().
#         # rand_seed(1928).
#         pragma_autocompact(True).
#         pragma_live_info(False).
#         fn_group_setup(grp_setup).
#         done().
#     add([
#         SegregateRule('team', loc),
#         Group(n=500, attr={ 'team': 'blue' }),
#         Group(n=500, attr={ 'team': 'red'  }),
#         probe_loc
#     ]).
#     run(10)
# )


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation (synthetic Allegheny County population)

fpath_db = os.path.join(os.path.dirname(__file__), 'db', 'allegheny-students.sqlite3')

# sites = [Site(450066968), Site(450086847), Site(450066968)]
# probe_loc = GroupSizeProbe.by_rel('site', Site.AT, sites, msg_mode=ProbeMsgMode.DISP)

sites = [Site(450066968)]
probe_loc = GroupSizeProbe(
    name='sex',
    queries=[
        GroupQry(attr={ 'sex': 'F' }, rel={ Site.AT: Site(450066968) }),
        GroupQry(attr={ 'sex': 'M' }, rel={ Site.AT: Site(450066968) })
    ],
    qry_tot=GroupQry(rel={ Site.AT: Site(450066968) }),
    msg_mode=ProbeMsgMode.DISP
)

(Simulation().
    set().
        # rand_seed(1928).
        pragma_autocompact(True).
        pragma_live_info(False).
        done().
    add().
        rule(SegregateRule('sex', ['F','M'])).
        # rule(SegregateRule('race', [1,2,3,5,6,7,8,9])).
        probe(probe_loc).
        done().
    db(fpath_db).
        gen_groups(
            tbl      = 'students',
            attr_db  = ['sex'],
            # attr_db  = ['race'],
            rel_db   = [GroupDBRelSpec(name=Site.AT, col='school_id')],
            attr_fix = {},
            rel_fix  = {}
        ).
        done().
    run(50)
    # summary(False, 1024,0,0,0, (1,0))
)
