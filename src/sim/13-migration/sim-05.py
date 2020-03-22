'''
A model of conflict-driven migration.

This simulation is an extenstion of 'sim-04.py' and wraps the simulation in a trajectory ensemble which enables running
many parametrizations of the simulation.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.data   import Probe, Var
from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import IterAlways, TimeAlways, Rule, Noop
from pram.sim    import Simulation
from pram.traj   import Trajectory, TrajectoryEnsemble

import random
import statistics

import numpy as np
import scipy

from collections.abc import Iterable


# ----------------------------------------------------------------------------------------------------------------------
site_sudan    = Site('Sudan')
site_ethiopia = Site('Ethiopia', attr={ 'travel-time':  8 })  # [months]
site_chad     = Site('Chad',     attr={ 'travel-time':  9 })
site_egypt    = Site('Egypt',    attr={ 'travel-time': 10 })
site_libya    = Site('Libya',    attr={ 'travel-time': 11 })

site_conflict = site_sudan
sites_dst = [site_egypt, site_ethiopia, site_chad, site_libya]  # migration destinations


# ----------------------------------------------------------------------------------------------------------------------
class MonthlyTemperature(object):
    """
    Args:
        ts (Iterable(Class)): Twelve subclasses of scipy.stats.rv_continuous or scipy.stats.rv_discrete.
        rv_cls (scipy.stats.rv_generic): A subclass of scipy.stats.rv_continuous or rv_discrete.
    """

    def __init__(self, ts, rv_cls=scipy.stats.norm):
        if len(ts) != 12:
            raise ValueError('Tweleve elements expected.')

        self.ts = []
        for i in ts:
            if isinstance(i, scipy.stats.rv_continuous) or isinstance(i, scipy.stats.rv_discrete):
                self.ts.append(i)
            elif not isinstance(i, str) and isinstance(i, Iterable) and len(i) == 2:
                self.ts.append(rv_cls(i[0], i[1]))
            else:
                raise ValueError()

    def mean(self, month):
        return self.ts[month].mean()

    def sample(self, month, n=1, bias=0.00):
        """
        Get one or more (n) samples of temperature for the given month.

        Args:
            month (int): [0..11]
            bias (float): The bias term to be added to all samples. Useful for simulating a warmer (or colder) world.

        Returns:
            int, float, list[int], or list[float]
        """

        return self.ts[month].rvs(n) + bias

    def stdev(self, month):
        return self.ts[month].stdev()


# ----------------------------------------------------------------------------------------------------------------------
class Environment(object):
    """
    Args:
        monthly_temp (MonthlyTemperature): Monthly temperatures.
    """

    TEMP_MIN =  5  # -15 -> hashness=1
    TEMP_MID = 20  #   0 -> hashness=0
    TEMP_MAX = 35  # +15 -> hashness=1

    def __init__(self, monthly_temp):
        self.monthly_temp = monthly_temp

    def get_harshness(self, month, do_sample=False):
        """
        Args:
            month(int): [0..1]

        Returns:
            float: [0..1], where 0=benign and 1=harsh

        Todo:
            Could memoize these values, but that won't help the future, more dynamic incarnations of this object.
        """

        if do_sample:
            t = self.monthly_temp.sample(month)[0]
        else:
            t = self.monthly_temp.mean(month)

        if t <= self.__class__.TEMP_MIN or t >= self.__class__.TEMP_MAX:
            return 1.00
        if t < self.__class__.TEMP_MID:
            return ( self.__class__.TEMP_MID - t) / ( self.__class__.TEMP_MID - self.__class__.TEMP_MIN)
        if t > self.__class__.TEMP_MID:
            return (-self.__class__.TEMP_MID + t) / (-self.__class__.TEMP_MID + self.__class__.TEMP_MAX)
        return 0.00


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

    def __init__(self, env, env_harshness_death_mult=0.001, migration_death_mult=0.05, name='migration', t=TimeAlways(), i=IterAlways()):
        super().__init__(name, t, i, group_qry=GroupQry(cond=[lambda g: g.has_attr({ 'is-migrating': True })]))

        self.env = env

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
            migrating_p = migrating_m / pop.mass * 100
        else:
            migrating_p = 0

        env_harshness = self.env.get_harshness(iter % 11, True)
        p_death = min(env_harshness * self.env_harshness_death_mult + migrating_p * self.migration_death_mult, 1.00)
        time_traveled = 1 if env_harshness <= 0.90 else 0  # no travel in very harsh climate

        return [
            GroupSplitSpec(p=p_death,     attr_set=Group.VOID),
            GroupSplitSpec(p=1 - p_death, attr_set={ 'migration-time': group.get_attr('migration-time') + 1, 'travel-time-left': group.get_attr('travel-time-left') - time_traveled })
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
            self.run_init(iter, t)
        else:
            self.run_iter(iter, t)

    def run_init(self, iter, t):
        self.pop_m_init = self.pop.mass

        if self.persistence:
            self.persistence.persist(self, [self.pop.mass, self.pop.mass_out, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], iter, t)

    def run_iter(self, iter, t):
        # Migrating population:
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

        # Settled population:
        settled_groups = self.pop.get_groups(GroupQry(cond=[lambda g: g.has_attr({ 'has-settled': True })]))
        if settled_groups and len(settled_groups) > 0:
            settled_m = sum([g.m for g in settled_groups])
            settled_p = settled_m / self.pop.mass * 100
        else:
            settled_m = 0
            settled_p = 0

        # Print and persist:
        # print(
        #     f'{iter or 0:>4}  ' +
        #     f'pop: {self.pop.mass:>9,.0f}    ' +
        #     f'dead: {self.pop.mass_out:>9,.0f}|{self.pop.mass_out / self.pop_m_init * 100:>3,.0f}%    ' +
        #     f'migrating: {migrating_m:>9,.0f}|{migrating_p:>3.0f}%    ' +
        #     f'migration-time: {migrating_time_mean:>6.2f} ({migrating_time_sd:>6.2f})    ' +
        #     f'settled: {settled_m:>9,.0f}|{settled_p:>3.0f}%    ' +
        #     f'groups: {len(self.pop.groups):>6,d}'
        # )

        if self.persistence:
            self.persistence.persist(self, [self.pop.mass, self.pop.mass_out, migrating_m, migrating_p, migrating_time_mean, migrating_time_sd, settled_m, settled_p], iter, t)


# ----------------------------------------------------------------------------------------------------------------------
conflict_dur = 1*12  # [months]

env = Environment(
    MonthlyTemperature([(24,5), (26,5), (29,5), (33,5), (35,5), (35,5), (33,5), (32,5), (33,5), (33,5), (29,5), (25,5)])  # Sudan
)


# ----------------------------------------------------------------------------------------------------------------------
# Simulations:

fpath_traj_db = os.path.join(os.path.dirname(__file__), f'sim-05-traj.sqlite3')

if os.path.isfile(fpath_traj_db): os.remove(fpath_traj_db)

te = TrajectoryEnsemble(fpath_traj_db)

if te.is_db_empty:
    te.set_pragma_memoize_group_ids(True)
    te.add_trajectories([
        Trajectory(
            (Simulation().
                add([
                    ConflictRule(severity=0.05, scale=scale, i=[0, conflict_dur]),
                    MigrationRule(env, env_harshness_death_mult=0.1, migration_death_mult=0.0001),  # most deaths due to environment (to show the seasonal effect)
                    PopProbe(),
                    Group(m=1*1000*1000, attr={ 'is-migrating': False }, rel={ Site.AT: site_conflict })
                ])
            )
        ) for scale in np.arange(0.10, 0.30, 0.10)
    ])
    te.run(conflict_dur + 12)  # [months]


# ----------------------------------------------------------------------------------------------------------------------
# Results analysis:

# te.plot_mass_locus_line     ((1200,300), os.path.join(os.path.dirname(__file__), 'sim-05-plot-line.png'), col_scheme='tableau10', opacity_min=0.35)
# te.plot_mass_locus_line_aggr((1200,300), os.path.join(os.path.dirname(__file__), 'sim-05-plot-ci.png'),   col_scheme='tableau10')
