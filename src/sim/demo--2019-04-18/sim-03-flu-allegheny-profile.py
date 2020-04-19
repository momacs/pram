from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation

from util.probes03 import probe_flu_at


import signal
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


fpath_db = os.path.join(os.path.dirname(__file__), 'db', 'allegheny-students.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# Rules are where machine reading will kick in over the summer.

class FluProgressRule(Rule):
    def __init__(self):
        super().__init__('flu-progress', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total    population at current location
            n_i = at.get_pop_size(GroupQry(attr={ 'flu': 'i' }))  # infected population at current location

            p_infection = float(n_i) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Infected:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r' }),  # group size after: 20% of before (recovered)
                GroupSplitSpec(p=0.8, attr_set={ 'flu': 'i' })   # group size after: 80% of before (still infected)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' })
            ]


# ----------------------------------------------------------------------------------------------------------------------
class FluLocationRule(Rule):
    def __init__(self):
        super().__init__('flu-location', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Infected and poor:
        if group.has_attr({ 'flu': 'i', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Infected and rich:
        if group.has_attr({ 'flu': 'i', 'income': 'm' }):
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
# (1) Sites:

site_home = Site('home')
school_l  = Site(450149323)  # 88% low income students
school_m  = Site(450067740)  #  7% low income students


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

def grp_setup(pop, group):
    return [
        GroupSplitSpec(p=0.9, attr_set={ 'flu': 's' }),
        GroupSplitSpec(p=0.1, attr_set={ 'flu': 'i' })
    ]


def run(iter):
    (Simulation().
        set().
            rand_seed(1928).
            pragma_autocompact(True).
            pragma_live_info(True).
            pragma_live_info_ts(False).
            fn_group_setup(grp_setup).
            done().
        add().
            rule(FluProgressRule()).
            rule(FluLocationRule()).
            probe(probe_flu_at(school_l, 'low-income')).  # the simulation output we care about and want monitored
            probe(probe_flu_at(school_m, 'med-income')).  # ^
            done().
        db(fpath_db).
            gen_groups(
                tbl      = 'students',
                attr_db  = [],
                rel_db   = [GroupDBRelSpec(name='school', col='school_id')],
                attr_fix = {},
                rel_fix  = { 'home': site_home },
                rel_at   = 'school'
            ).
            done().
        run(iter)
    )


# ----------------------------------------------------------------------------------------------------------------------
# (3) Profile:

import cProfile
import pstats

# cProfile.run('run(10)', os.path.join('restats', '10-iter-01')  # 190.718 s; the original implementation (2019.05.19)
# cProfile.run('run(10)', os.path.join('restats', '10-iter-02')  # 189.658 s; Site: self.hash = hash(self.name)
# cProfile.run('run(10)', os.path.join('restats', '10-iter-03')  # 180.142 s; no probes
# cProfile.run('run(10)', os.path.join('restats', '10-iter-04')  # 509.688 s; (Group.gen_hash(qry.attr, qry.rel) == g.__hash__()) in GroupPopulation.get_groups()
# cProfile.run('run(10)', os.path.join('restats', '10-iter-05')  # 302.901 s; previous implementation
# cProfile.run('run(10)', os.path.join('restats', '10-iter-06')  # 183.736 s; xxhash.xxh64(_).hexdigest() instead of hash(); JSON: return {'__Site__': o.__hash__()}
# cProfile.run('run(10)', os.path.join('restats', '10-iter-07')  # 180.579 s; xxhash.xxh64(_).hexdigest() instead of hash(); JSON: return {'__Site__': o.name}
# cProfile.run('run(10)', os.path.join('restats', '10-iter-08')  # 185.307 s; xxhash.xxh64(_).intdigest() instead of hash(); JSON: return {'__Site__': o.__hash__()}
# cProfile.run('run(10)', os.path.join('restats', '10-iter-09')  # 187.754 s; hashlib.sha1() instead of hash(); JSON: return {'__Site__': o.__hash__()}
# cProfile.run('run(10)', os.path.join('restats', '10-iter-10')  # 185.294 s; xxhash.xxh32(_).intdigest() instead of hash(); JSON: return {'__Site__': o.__hash__()}
# cProfile.run('run(10)', os.path.join('restats', '10-iter-11'))

# pstats.Stats(os.path.join('restats', '10-iter-11')).sort_stats('time', 'cumulative').print_stats(10)
