'''
A model of conflict-driven migration.

This simulation is an extenstion of 'sim-02.py' and adds the eventual settlement of the migrating population in one of
the counties cordering the conflict country.
'''

from pram.data   import Probe, ProbePersistenceMode, ProbePersistenceDB, ProbePersistenceMem, Var
from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import IterAlways, TimeAlways, Rule, Noop
from pram.sim    import Simulation

import random
import statistics


# ----------------------------------------------------------------------------------------------------------------------
site_sudan    = Site('Sudan')
site_ethiopia = Site('Ethiopia', attr={ 'travel-time':  8 })  # [months]
site_chad     = Site('Chad',     attr={ 'travel-time':  9 })
site_egypt    = Site('Egypt',    attr={ 'travel-time': 10 })
site_libya    = Site('Libya',    attr={ 'travel-time': 11 })

site_conflict = site_sudan
sites_dst = [site_egypt, site_ethiopia, site_chad, site_libya]  # migration destinations


# ----------------------------------------------------------------------------------------------------------------------
class ConflictRule(Rule):
    """
    Conflict causes death and migration.  The probability of death scales with the conflict's severity and scale
    while the probability of migration scales with the conflict's scale only.  Multipliers for both factors are exposed
    as parameters.

    Time of exposure to conflict also increases the probability of death and migration, but that influence isn't
    modeled directly.  Instead, a proportion of every group of non-migrating agents can die or migrate at every step of
    the simulation.

    Every time a proportion of population is beginning to migrate, the destination site is set and the distance to that
    site use used elewhere to control settlement.
    """

    def __init__(self, severity, scale, severity_death_mult=0.0001, scale_migration_mult=0.01, name='conflict', t=TimeAlways(), i=IterAlways()):
        super().__init__(name, t, i, group_qry=GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': False })]))

        self.severity = severity  # [0=benign .. 1=lethal]
        self.scale    = scale     # [0=contained .. 1=wide-spread]

        self.severity_death_mult  = severity_death_mult
        self.scale_migration_mult = scale_migration_mult

    def apply(self, pop, group, iter, t):
        p_death     = self.scale * self.severity * self.severity_death_mult
        p_migration = self.scale                 * self.scale_migration_mult

        site_dst = random.choice(sites_dst)

        return [
            GroupSplitSpec(p=p_death,     attr_set=Group.VOID),
            GroupSplitSpec(p=p_migration, attr_set={ 'is-migrating': True, 'migration-time': 0, 'travel-time-left': site_dst.get_attr('travel-time') }, rel_set={ 'site-dst': site_dst }),
            GroupSplitSpec(p=1 - p_death - p_migration)
        ]


# ----------------------------------------------------------------------------------------------------------------------
class MigrationRule(Rule):
    """
    Migrating population has a chance of dying and the probability of that happening is proportional to the harshness
    of the environment and the mass of already migrating population.  Multipliers for both factors are exposed as
    parameters.

    Time of exposure to the environment also increases the probability of death, but that influence isn't modeled
    directly.  Instead, a proportion of every group of migrating agents can die at every step of the simulation.

    Environmental harshness can be controled via another rule which conditions it on the time of year (e.g., winter
    can be harsher than summer or vice versa depending on region).

    A migrating population end to migrate by settling in its destination site.
    """

    def __init__(self, env_harshness, env_harshness_death_mult=0.001, migration_death_mult=0.05, name='migration', t=TimeAlways(), i=IterAlways()):
        super().__init__(name, t, i, group_qry=GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))

        self.env_harshness = env_harshness  # [0=benign .. 1=harsh]

        self.env_harshness_death_mult = env_harshness_death_mult
        self.migration_death_mult     = migration_death_mult

    def apply(self, pop, group, iter, t):
        if group.has_attr({ 'travel-time-left': 0} ):
            return self.apply_settle(pop, group, iter, t)
        else:
            return self.apply_keep_migrating(pop, group, iter, t)

    def apply_keep_migrating(self, pop, group, iter, t):
        migrating_groups = pop.get_groups(GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))
        if migrating_groups and len(migrating_groups) > 0:
            migrating_m = sum([g.m for g in migrating_groups])
            migrating_p = migrating_m / pop.m * 100
        else:
            migrating_p = 0

        p_death = min(self.env_harshness * self.env_harshness_death_mult + migrating_p * self.migration_death_mult, 1.00)

        return [
            GroupSplitSpec(p=p_death,     attr_set=Group.VOID),
            GroupSplitSpec(p=1 - p_death, attr_set={ 'migration-time': group.get_attr('migration-time') + 1, 'travel-time-left': group.get_attr('travel-time-left') - 1 })
        ]

    def apply_settle(self, pop, group, iter, t):
        return [
            GroupSplitSpec(p=1, attr_set={ 'migration-time': group.get_attr('migration-time') + 1, 'is-migrating': False, 'has-settled': True }, rel_set={ Site.AT: group.get_rel('site-dst') }, rel_del=['site-dst'])
        ]


# ----------------------------------------------------------------------------------------------------------------------
class PopProbe(Probe):
    """
    Prints a summary of the population at every iteration.  It also persists vital simulation characteristics for
    post-simulation plotting or data analysis.
    """

    def __init__(self, persistence=None):
        self.consts = []
        self.vars = [
            Var('pop_m',               'float'),
            Var('dead_m',              'float'),
            Var('migrating_m',         'float'),
            Var('migrating_p',         'float'),
            Var('migrating_time_mean', 'float'),
            Var('migrating_time_sd',   'float'),
            Var('settled_m',           'float'),
            Var('settled_p',           'float')
        ]

        super().__init__('pop', persistence)

    def run(self, iter, t):
        if iter is None:
            self.run_init()
        else:
            self.run_iter(iter, t)

    def run_init(self):
        self.pop_m_init = self.pop.m

    def run_iter(self, iter, t):
        # Migrating population:
        migrating_groups = self.pop.get_groups(GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))
        if migrating_groups and len(migrating_groups) > 0:
            migration_time_lst = [g.get_attr('migration-time') for g in migrating_groups]

            migrating_m         = sum([g.m for g in migrating_groups])
            migrating_p         = migrating_m / self.pop.m * 100
            migrating_time_mean = statistics.mean (migration_time_lst)
            migrating_time_sd   = statistics.stdev(migration_time_lst) if len(migration_time_lst) > 1 else 0
        else:
            migrating_m         = 0
            migrating_p         = 0
            migrating_time_mean = 0
            migrating_time_sd   = 0

        # Settled population:
        settled_groups = self.pop.get_groups(GroupQry(cond=[lambda g: g.has_attr({ 'has-settled': True })]))
        if settled_groups and len(settled_groups) > 0:
            settled_m = sum([g.m for g in settled_groups])
            settled_p = settled_m / self.pop.m * 100
        else:
            settled_m = 0
            settled_p = 0

        # Print and persist:
        print(
            f'{iter or 0:>4}  ' +
            f'pop: {self.pop.m:>9,.0f}    ' +
            f'dead: {self.pop.m_out:>9,.0f}|{self.pop.m_out / self.pop_m_init * 100:>3,.0f}%    ' +
            f'migrating: {migrating_m:>9,.0f}|{migrating_p:>3.0f}%    ' +
            f'migration-time: {migrating_time_mean:>6.2f} ({migrating_time_sd:>6.2f})    ' +
            f'settled: {settled_m:>9,.0f}|{settled_p:>3.0f}%'
        )

        if self.persistence:
            self.persistence.persist(self, [self.pop.m, self.pop.m_out, migrating_m, migrating_p, migrating_time_mean, migrating_time_sd, settled_m, settled_p], iter, t)


# ----------------------------------------------------------------------------------------------------------------------
persistence = None
# persistence = ProbePersistenceMem()
# persistence = ProbePersistenceDB(os.path.join(os.path.dirname(__file__), f'sim-03.sqlite3'), mode=ProbePersistenceMode.OVERWRITE)


# ----------------------------------------------------------------------------------------------------------------------
# Simulation:

sim = (Simulation().
    set_pragmas(analyze=False, autocompact=True).
    add([
        ConflictRule(severity=0.05, scale=0.2, i=[0,3*12]),
        MigrationRule(env_harshness=0.05),
        PopProbe(persistence),
        Group(m=1*1000*1000, attr={ 'is-migrating': False }, rel={ Site.AT: site_conflict })
    ]).
    run(4*12)  # months
)


# ----------------------------------------------------------------------------------------------------------------------
# Summary:

def print_settled_summary():
    print('')
    settled_m = 0
    for s in sites_dst:
        m = s.get_pop_size()
        settled_m += m
        print(f'{s.name:<12}: {m:>9,.0f}')
    print(f'{"TOTAL":<12}: {settled_m:>9,.0f}')

print_settled_summary()


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

if persistence:
    series = [
        { 'var': 'migrating_m', 'lw': 0.75, 'linestyle': '-',  'marker': 'o', 'color': 'blue',  'markersize': 0, 'lbl': 'Migrating' },
        { 'var': 'settled_m',   'lw': 0.75, 'linestyle': '--', 'marker': '+', 'color': 'green', 'markersize': 0, 'lbl': 'Settled'   },
        { 'var': 'dead_m',      'lw': 0.75, 'linestyle': ':',  'marker': 'x', 'color': 'red',   'markersize': 0, 'lbl': 'Dead'      }
    ]
    sim.probes[0].plot(series, ylabel='Population mass', xlabel='Iteration (month from start of conflict)', figsize=(12,4), subplot_b=0.15)
