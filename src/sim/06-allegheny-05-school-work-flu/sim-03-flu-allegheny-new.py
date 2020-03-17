import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import random

from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation

from util.probes03 import probe_flu_at


import signal
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ----------------------------------------------------------------------------------------------------------------------
class FluProgressRule(Rule):
    def __init__(self):
        super().__init__('flu-progress', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total   population at current location
            n_e = at.get_pop_size(GroupQry(attr={ 'flu': 'e' }))  # exposed population at current location

            p_infection = float(n_e) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Exposed:
        if group.has_attr({ 'flu': 'e' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.5, attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=0.3, attr_set={ 'flu': 'e' })
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' })
            ]

        raise ValueError('Unknown flu state')


# ----------------------------------------------------------------------------------------------------------------------
class FluLocationRule(Rule):
    def __init__(self):
        super().__init__('flu-location', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Exposed and low income:
        if group.has_attr({ 'flu': 'e', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Exposed and medium income:
        if group.has_attr({ 'flu': 'e', 'income': 'm' }):
            return [
                GroupSplitSpec(p=0.6, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.4)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.8, rel_set={ Site.AT: group.get_rel('school') }),
                GroupSplitSpec(p=0.2)
            ]

        return None


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

fpath_db  = os.path.join(os.path.dirname(__file__), 'db', 'allegheny-students.sqlite3')

pragma_live_info = True
pragma_live_info_ts = False


# ----------------------------------------------------------------------------------------------------------------------
# (1) Sites:

# sites = Simulation.gen_sites_from_db(
#     fpath_db,
#     lambda fpath_db: {
#         'school': Site.gen_from_db(fpath_db, tbl='schools', name_col='sp_id')
#     },
#     pragma_live_info=pragma_live_info,
#     pragma_live_info_ts=pragma_live_info_ts
# )

site_home = Site('home')


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

def setup_group(group):
    group.set_attr('flu', 'e' if random.random() <= 0.1 else 's')


s = (Simulation().
    set().
        rand_seed(1928).
        pragma_autocompact(True).
        pragma_live_info(pragma_live_info).
        pragma_live_info_ts(pragma_live_info_ts).
        commit().
    add().
        rule(FluProgressRule()).
        rule(FluLocationRule()).
        # probe(probe_flu_at(school_l, 'l.p')).
        # probe(probe_flu_at(school_m, 'm.p')).
        commit().
    # gen_sites_from_db(
    #     fpath_db = fpath_db,
    #     name     = 'school',
    #     tbl      = 'schools',
    #     name_col = 'sp_id'
    # ).
    gen_groups_from_db(
        fpath_db = fpath_db,
        tbl      = 'students',
        attr     = {},
        rel      = { 'home': site_home },
        attr_db  = [],
        # rel_db   = [GroupDBRelSpec(name='school', col='school_id', entities=sites['school'])],
        rel_db   = [GroupDBRelSpec(tbl='schools', col_from='school_id', col_to='sp_id', name='school')],
        rel_at   = 'school'  # TODO: move to group_setup()
    )
)

# print(s.pop.sites[450149323])
# print(s.pop.sites[450149323])
#
sys.exit(77)

(s.
    add().
        probe(probe_flu_at(s.pop.sites[450149323], 'l.p')).
        probe(probe_flu_at(s.pop.sites[450149323], 'm.p')).
        commit().
    setup_groups(setup_group).
    summary(False, 8,8,0,0)
)

# school_l = s.sites['school'][450149323]  # 88% low income
# school_m = s.sites['school'][450067740]  #  7% low income

s.run(15).summary(False, 8,8,0,0, (1,0))
# s.run(2).summary(True, 20,0,0,0, (1,0))


# ----------------------------------------------------------------------------------------------------------------------
# Key points
#     Static rule analysis  - Automatically form groups
#     Dynamic rule analysis - Alert the modeler they might have missed something
