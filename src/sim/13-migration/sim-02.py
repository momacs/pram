'''
A model of conflict-driven migration.

This simulation is identical to 'sim-01.py' but with persistance and the associated post-simulation plot.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.data   import Probe, ProbePersistanceMode, ProbePersistanceDB, ProbePersistanceMem, Var
from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import IterAlways, TimeAlways, Rule, Noop
from pram.sim    import Simulation

import statistics


# ----------------------------------------------------------------------------------------------------------------------
class ConflictRule(Rule):
    """
    Conflict causes death and migration.  The probability of death scales with the conflict's severity and scale
    while the probability of migration scales with the conflict's scale only.  Multipliers for both factors are exposed
    as parameters.

    Time of exposure to conflict also increases the probability of death and migration, but that influence isn't
    modeled directly.  Instead, a proportion of every group of non-migrating agents can die or migrate at every step of
    the simulation.
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

        return [
            GroupSplitSpec(p=p_death,     attr_set=Group.VOID),
            GroupSplitSpec(p=p_migration, attr_set={ 'is-migrating': True, 'migration-time': 0 }),
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
    """

    def __init__(self, env_harshness, env_harshness_death_mult=0.001, migration_death_mult=0.05, name='migration', t=TimeAlways(), i=IterAlways()):
        super().__init__(name, t, i, group_qry=GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))

        self.env_harshness = env_harshness  # [0=benign .. 1=harsh]

        self.env_harshness_death_mult = env_harshness_death_mult
        self.migration_death_mult     = migration_death_mult

    def apply(self, pop, group, iter, t):
        migrating_groups = pop.get_groups(GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))
        if migrating_groups and len(migrating_groups) > 0:
            migrating_m = sum([g.m for g in migrating_groups])
            migrating_p = migrating_m / pop.mass * 100
        else:
            migrating_p = 0

        p_death = min(self.env_harshness * self.env_harshness_death_mult + migrating_p * self.migration_death_mult, 1.00)

        return [
            GroupSplitSpec(p=p_death,     attr_set=Group.VOID),
            GroupSplitSpec(p=1 - p_death, attr_set={ 'migration-time': group.get_attr('migration-time') + 1 })
        ]


# ----------------------------------------------------------------------------------------------------------------------
class PopProbe(Probe):
    """
    Prints a summary of the population at every iteration.  It also persists vital simulation characteristics for
    post-simulation plotting or data analysis.
    """

    def __init__(self, persistance=None):
        self.consts = []
        self.vars = [
            Var('pop_m',               'float'),
            Var('dead_m',              'float'),
            Var('migrating_m',         'float'),
            Var('migrating_p',         'float'),
            Var('migrating_time_mean', 'float'),
            Var('migrating_time_sd',   'float')
        ]

        super().__init__('pop', persistance)

    def run(self, iter, t):
        if iter is None:
            self.run_init()
        else:
            self.run_iter(iter, t)

    def run_init(self):
        self.pop_m_init = self.pop.mass

    def run_iter(self, iter, t):
        migrating_groups = self.pop.get_groups(GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))
        if migrating_groups and len(migrating_groups) > 0:
            migration_time_lst = [g.get_attr('migration-time') for g in migrating_groups]

            migrating_m         = sum([g.m for g in migrating_groups])
            migrating_p         = migrating_m / self.pop.mass * 100
            migrating_time_mean = statistics.mean (migration_time_lst)
            migrating_time_sd   = statistics.stdev(migration_time_lst) if len(migration_time_lst) > 1 else 0
        else:
            migrating_m         = 0
            migrating_p         = 0
            migrating_time_mean = 0
            migrating_time_sd   = 0

        print(
            f'{iter or 0:>4}  ' +
            f'pop: {self.pop.mass:>9,.0f}    ' +
            f'dead: {self.pop.mass_out:>9,.0f}|{self.pop.mass_out / self.pop_m_init * 100:>3,.0f}%    ' +
            f'migrating: {migrating_m:>9,.0f}|{migrating_p:>3.0f}%    ' +
            f'migration-time: {migrating_time_mean:>6.2f} ({migrating_time_sd:>6.2f})'
        )

        if self.persistance:
            self.persistance.persist(self, [self.pop.mass, self.pop.mass_out, migrating_m, migrating_p, migrating_time_mean, migrating_time_sd], iter, t)


# ----------------------------------------------------------------------------------------------------------------------
persistance = None

# dpath_cwd = os.path.dirname(__file__)
# fpath_db  = os.path.join(dpath_cwd, f'sim.sqlite3')
# persistance = ProbePersistanceDB(fpath_db, mode=ProbePersistanceMode.OVERWRITE)

persistance = ProbePersistanceMem()


# ----------------------------------------------------------------------------------------------------------------------
# Simulation:

sim = (Simulation().
    set_pragmas(analyze=False, autocompact=True).
    add([
        ConflictRule(severity=0.05, scale=0.2),
        MigrationRule(env_harshness=0.05),
        PopProbe(persistance),
        Group(m=1*1000*1000, attr={ 'is-migrating': False }),
    ]).
    run(48)  # months
)


# ----------------------------------------------------------------------------------------------------------------------
# Plot:

if persistance:
    series = [
        { 'var': 'migrating_m', 'lw': 0.75, 'linestyle': '-',  'marker': 'o', 'color': 'blue', 'markersize': 0, 'lbl': 'Migrating' },
        { 'var': 'dead_m',      'lw': 0.75, 'linestyle': '--', 'marker': '+', 'color': 'red',  'markersize': 0, 'lbl': 'Dead'      }
    ]
    sim.probes[0].plot(series, ylabel='Population mass', xlabel='Iteration (month from start of conflict)', figsize=(12,4))
