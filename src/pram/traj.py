# -*- coding: utf-8 -*-
"""Contains trajectory and trajectory ensemble relevant code."""

# Altair
#     Docs
#         Customization: https://altair-viz.github.io/user_guide/customization.html
#         Error band: https://altair-viz.github.io/user_guide/generated/core/altair.ErrorBandDef.html#altair.ErrorBandDef
#     Misc
#         https://github.com/altair-viz/altair/issues/968
#     Timeout in selenium
#         /Volumes/d/pitt/sci/pram/lib/python3.6/site-packages/altair/utils/headless.py

import altair as alt
import altair_saver as alt_save
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
# import pickle
# import dill as pickle
import cloudpickle as pickle
import random
import ray
import socket
import sqlite3
import time
import tqdm

from dotmap              import DotMap
from pyrqa.neighbourhood import FixedRadius
from scipy.fftpack       import fft
from scipy               import signal
from sortedcontainers    import SortedDict

from .data   import ProbePersistenceDB
from .graph  import MassGraph
from .signal import Signal
from .sim    import Simulation
from .util   import DB, Size, Time

__all__ = ['ClusterInf', 'TrajectoryError', 'Trajectory', 'TrajectoryEnsemble']


# ----------------------------------------------------------------------------------------------------------------------
class ClusterInf(object):
    """Computational cluster information.

    This info is cluster-specific.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_args(self):
        return self.kwargs


# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
class TqdmUpdTo(tqdm.tqdm):
    """Progress bar that can be updated to a specified position."""

    def update_to(self, to, total=None):
        """Set the progess bar value.

        Args:
            to (int or float): Value to which the progres bar should move.
            total (int or float, optional): The maximum value.
        """

        if total is not None:
            self.total = total
        self.update(to - self.n)  # will also set self.n = blocks_so_far * block_size


# ----------------------------------------------------------------------------------------------------------------------
class Trajectory(object):
    """A time-ordered sequence of system configurations that occur as the system state evolves.

    Also called orbit.  Can also be thought of as a sequence of points in the system's phase space.

    This class delegates persistence management to the TrajectoryEnsemble class that also contains it.

    This class keeps a reference to a simulation object, but that reference is only needed when running the simulation
    is desired.  When working with an executed trajectory (i.e., the trace of past simulation run), 'self.sim' can be
    None.  For example, the mass graph created by the Trajectory class is not based on an instatiated Simulation object
    even if that object has been used to generate the substrate data; instead, the database content is the graph's
    basis.

    Args:
        sim (Simulation): The simulation.
        name (str): Trajectory name.
        memo (str): Trajectory memo.
        ensemble (TrajectoryEnsemble): The ensemble that contains this object.
        id (Any): The trajectory ensemble database ID of the trajectory.
    """

    def __init__(self, sim=None, name=None, memo=None, ensemble=None, id=None):
        self.sim  = sim
        self.name = name
        self.memo = memo
        self.ens  = ensemble  # TrajectoryEnsemble that contains this object

        self.set_id(id)

        self.mass_graph = None  # MassGraph object (instantiated when needed)

    def _check_ens(self):
        """Ensure the trajectory is a part of an ensemble."""

        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

    @staticmethod
    def comp_fft(y, T, N):
        """Compute Fast Fourier Transform (FFT).

        Args:
            y (numpy.ndarray): The signal.
            T (float): Nyquist sampling criterion.
            N (int): Sampling rate.

        Returns:
            float
        """

        f = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft = 2.0/N * np.abs(fft(y)[0:N//2])
        return (f, fft)

    def compact(self):
        """Compact the trajectory.

        Returns:
            ``self``
        """

        self.mass_graph = None
        return self

    def gen_agent(self, n_iter=-1):
        """See :meth:`TrajectoryEnsemble.gen_agent() <pram.traj.TrajectoryEnsemble.gen_agent>`."""

        self._check_ens()
        return self.ens.gen_agent(self, n_iter)

    def gen_agent_pop(self, n_agents=1, n_iter=-1):
        """See :meth:`TrajectoryEnsemble.gen_agent_pop() <pram.traj.TrajectoryEnsemble.gen_agent_pop>`."""

        self._check_ens()
        return self.ens.gen_agent_pop(self, n_agents, n_iter)

    def gen_mass_graph(self):
        """See :meth:`TrajectoryEnsemble.mass_graph() <pram.traj.TrajectoryEnsemble.mass_graph>`."""

        self._check_ens()
        if self.mass_graph is None:
            self.mass_graph = self.ens.gen_mass_graph(self)

        return self

    def get_signal(self, do_prob=False):
        """See :meth:`TrajectoryEnsemble.get_signal() <pram.traj.TrajectoryEnsemble.get_signal>`."""

        self._check_ens()
        return self.ens.get_signal(self, do_prob)

    def get_sim(self):
        """Get the simulation wrapped by this object.

        Returns:
            Simulation
        """

        return self.sim

    def get_time_series(self, group_hash):
        """See :meth:`TrajectoryEnsemble.get_time_series() <pram.traj.TrajectoryEnsemble.get_time_series>`."""

        self._check_ens()
        return self.ens.get_time_series(self, group_hash)

    def load_sim(self):
        """Loads settings from the trajectory ensemble DB.

        See :meth:`TrajectoryEnsemble.load_sim() <pram.traj.TrajectoryEnsemble.load_sim>`.

        Returns:
            ``self``
        """

        self._check_ens()
        self.ens.load_sim(self)
        return self

    def plot_heatmap(self, size, fpath):
        """Plots heatmap.

        Todo:
            Finish this method.

        Args:
            size (tuple[int,int]): The figure size.
            fpath (str): Destination filepath.

        Returns:
            ``self``
        """

        self._check_ens()

        # data = np.zeros((self.n_iter, self.n_group, self.n_group), dtype=float)

        iter = 1
        # data = np.array((len(self.group_hash_set), len(self.group_hash_set)))
        # data = {}
        data = []
        for h_src in self.group_hash_set:
            # data[h_src] = {}
            for h_dst in self.group_hash_set:
                if self.gg_flow[iter] is not None and self.gg_flow[iter].get(h_src) is not None: # and self.gg_flow[iter].get(h_src).get(h_dst) is not None:
                    # data[h_src][h_dst] = self.gg_flow[iter].get(h_src).get(h_dst)
                    data.append({ 'x': h_src, 'y': h_dst, 'z': self.gg_flow[iter].get(h_src).get(h_dst) })

        # print(data)
        # return self

        # c = alt.Chart(alt.Data(values=data)).mark_rect().encode(x='x:O', y='y:O', color='z:Q')
        c = alt.Chart(alt.Data(values=data)).mark_rect().encode(x='x:O', y='y:O', color=alt.Color('z:Q', scale=alt.Scale(type='linear', range=['#bfd3e6', '#6e016b'])))
        c.save(filepath, webdriver=self.__class__.WEBDRIVER)
        return self

    def plot_mass_flow_time_series(self, scale=(1.00, 1.00), filepath=None, iter_range=(-1, -1), v_prop=False, e_prop=False):
        """See :meth:`graph.MassGraph.plot_mass_flow_time_series() <pram.graph.MassGraph.plot_mass_flow_time_series>`."""

        self.gen_mass_graph()
        self.mass_graph.plot_mass_flow_time_series(scale, filepath, iter_range, v_prop, e_prop)
        return self

    def plot_mass_locus_fft(self, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        """See :meth:`TrajectoryEnsemble.plot_mass_locus_fft() <pram.traj.TrajectoryEnsemble.plot_mass_locus_fft>`."""

        self._check_ens()
        plot = self.ens.plot_mass_locus_fft(self, size, filepath, iter_range, sampling_rate, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_line(self, size, filepath, iter_range=(-1, -1), stroke_w=1, col_scheme='set1', do_ret_plot=False):
        """See :meth:`TrajectoryEnsemble.plot_mass_locus_line() <pram.traj.TrajectoryEnsemble.plot_mass_locus_line>`."""

        self._check_ens()
        plot = self.ens.plot_mass_locus_line(size, filepath, iter_range, self, 0, 1, stroke_w, col_scheme, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_recurrence(self, size, filepath, iter_range=(-1, -1), neighbourhood=FixedRadius(), embedding_dimension=1, time_delay=2, do_ret_plot=False):
        """See :meth:`TrajectoryEnsemble.plot_mass_locus_recurrence() <pram.traj.TrajectoryEnsemble.plot_mass_locus_recurrence>`."""

        self._check_ens()
        plot = self.ens.plot_mass_locus_recurrence(self, size, filepath, iter_range, neighbourhood, embedding_dimension, time_delay, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_scaleogram(self, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        """See :meth:`TrajectoryEnsemble.plot_mass_locus_scaleogram() <pram.traj.TrajectoryEnsemble.plot_mass_locus_scaleogram>`."""

        self._check_ens()
        plot = self.ens.plot_mass_locus_scaleogram(self, size, filepath, iter_range, sampling_rate, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_spectrogram(self, size, filepath, iter_range=(-1, -1), sampling_rate=None, win_len=None, noverlap=None, do_sort=False, do_ret_plot=False):
        """See :meth:`TrajectoryEnsemble.plot_mass_locus_spectrogram() <pram.traj.TrajectoryEnsemble.plot_mass_locus_spectrogram>`."""

        self._check_ens()
        plot = self.ens.plot_mass_locus_spectrogram(self, size, filepath, iter_range, sampling_rate, win_len, noverlap, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_streamgraph(self, size, filepath, iter_range=(-1, -1), do_sort=False, do_ret_plot=False):
        """See :meth:`TrajectoryEnsemble.plot_mass_locus_streamgraph() <pram.traj.TrajectoryEnsemble.plot_mass_locus_streamgraph>`."""

        self._check_ens()
        plot = self.ens.plot_mass_locus_streamgraph(self, size, filepath, iter_range, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def run(self, iter_or_dur=1):
        """Run the associated simulation.

        See :meth:`TrajectoryEnsemble.run() <pram.traj.TrajectoryEnsemble.run>`.

        Args:
            iter_or_dur (int or str): Number of iterations or a string representation of duration (see
                :meth:`util.Time.dur2ms() <pram.util.Time.dur2ms>`)

        Returns:
            ``self``
        """

        if self.sim is not None:
            self.sim.set_pragma_analyze(False)
            self.sim.run(iter_or_dur)
        return self

    def save_sim(self):
        """Saves the associated simulation in the trajectory ensemble DB.

        See :meth:`TrajectoryEnsemble.save_sim() <pram.traj.TrajectoryEnsemble.save_sim>`.

        Returns:
            ``self``
        """

        self._check_ens()
        self.ens.save_sim(self)
        return self

    def save_state(self, mass_flow_specs=None):
        """Save settings to the trajectory ensemble DB.

        See :meth:`TrajectoryEnsemble.save_state() <pram.traj.TrajectoryEnsemble.save_state>`.

        Returns:
            ``self``
        """

        self._check_ens()
        self.ens.save_state(self, mass_flow_specs)
        return self

    def set_id(self, id):
        """Loads trajectory ID from the trajectory ensemble DB.

        Returns:
            ``self``
        """

        self.id = id
        if self.sim is not None:
            self.sim.traj_id = id
        return self


# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryEnsemble(object):
    """A collection of trajectories.

    In mathematical physics, especially as introduced into statistical mechanics and thermodynamics by J. Willard Gibbs
    in 1902, an ensemble (also statistical ensemble) is an idealization consisting of a large number of virtual copies
    (sometimes infinitely many) of a system, considered all at once, each of which represents a possible state that the
    real system might be in. In other words, a statistical ensemble is a probability distribution for the state of the
    system.

    For portability, SQLite3 is currently used RDBMS for trajectory ensemble database.

    Database design notes:
        - While having a 'traj_id' field in the 'grp_name' table seems like a reasonable choice, a trajectory ensemble
          is assumed to hold only similar trajectories.  Therefore, the 'grp' and 'grp_name' tables can simply be
          joined on the 'hash' field.

    Args:
        fpath_db (str, optional): Database filepath.
        do_load_sims (bool): Load simulations?
        cluster_inf (ClusterInf, optional): Computational cluster information.
        flush_every (int): Data flushing frequency in iterations.
    """

    SQL_CREATE_SCHEMA = '''
        CREATE TABLE traj (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        ts   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        name TEXT,
        memo TEXT,
        sim  BLOB
        );

        CREATE TABLE iter (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        traj_id   INTEGER,
        i         INTEGER NOT NULL,
        host_name TEXT,
        host_ip   TEXT,
        UNIQUE (traj_id, i),
        CONSTRAINT fk__iter__traj FOREIGN KEY (traj_id) REFERENCES traj (id) ON UPDATE CASCADE ON DELETE CASCADE
        );

        CREATE TABLE mass_locus (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        iter_id INTEGER,
        grp_id  INTEGER,
        m       REAL NOT NULL,
        m_p     REAL NOT NULL,
        UNIQUE (iter_id, grp_id),
        CONSTRAINT fk__mass_locus__iter FOREIGN KEY (iter_id) REFERENCES iter (id) ON UPDATE CASCADE ON DELETE CASCADE,
        CONSTRAINT fk__mass_locus__grp  FOREIGN KEY (grp_id)  REFERENCES grp  (id) ON UPDATE CASCADE ON DELETE CASCADE
        );

        CREATE TABLE mass_flow (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        iter_id    INTEGER,
        grp_src_id INTEGER,
        grp_dst_id INTEGER,
        m          REAL NOT NULL,
        m_p        REAL NOT NULL,
        CONSTRAINT fk__mass_flow__iter    FOREIGN KEY (iter_id)    REFERENCES iter (id) ON UPDATE CASCADE ON DELETE CASCADE,
        CONSTRAINT fk__mass_flow__grp_src FOREIGN KEY (grp_src_id) REFERENCES grp  (id) ON UPDATE CASCADE ON DELETE CASCADE,
        CONSTRAINT fk__mass_flow__grp_dst FOREIGN KEY (grp_dst_id) REFERENCES grp  (id) ON UPDATE CASCADE ON DELETE CASCADE
        );

        CREATE TABLE grp (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        --hash    INTEGER NOT NULL UNIQUE,  -- SQLite3 doesn't support storing 64-bit integers
        hash    TEXT NOT NULL UNIQUE,
        attr    BLOB,
        rel     BLOB
        );

        CREATE TABLE grp_name (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        ord  INTEGER,
        hash TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL
        );
        '''

        # Add indices in case PostgreSQL was to be used (i.e., an actual production env)

        # CREATE TABLE grp (
        # id      INTEGER PRIMARY KEY AUTOINCREMENT,
        # iter_id INTEGER,
        # hash    TEXT NOT NULL,
        # m       REAL NOT NULL,
        # m_p     REAL NOT NULL,
        # attr    BLOB,
        # rel     BLOB,
        # UNIQUE (iter_id, hash),
        # CONSTRAINT fk__grp__iter FOREIGN KEY (iter_id) REFERENCES iter (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );

        # CREATE TABLE rule (
        # id      INTEGER PRIMARY KEY AUTOINCREMENT,
        # traj_id INTEGER,
        # ord     INTEGER NOT NULL,
        # name    TEXT NOT NULL,
        # src     TEXT NOT NULL,
        # UNIQUE (traj_id, ord),
        # CONSTRAINT fk__rule__traj FOREIGN KEY (traj_id) REFERENCES traj (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );

    FLUSH_EVERY = 16  # frequency of flushing data to the database
    WEBDRIVER = 'chrome'  # 'firefox'

    def __init__(self, fpath_db=None, do_load_sims=True, cluster_inf=None, flush_every=FLUSH_EVERY):
        self.cluster_inf = cluster_inf
        self.traj = {}  # index by DB ID
        self.conn = None

        self.pragma = DotMap(
            memoize_group_ids = False  # keep group hash-to-db-id map in memory (faster but increases memory utilization)
        )

        self.cache = DotMap(
            group_hash_to_id = {}
        )

        self.curr_iter_id = None  # ID of the last added row of the 'iter' table; keep for probe persistence to access

        self._db_conn_open(fpath_db, do_load_sims)

    def __del__(self):
        self._db_conn_close()

    def _db_conn_close(self):
        if self.conn is None: return

        self.conn.close()
        self.conn = None

    def _db_conn_open(self, fpath_db=None, do_load_sims=True):
        """Opens the DB connection and, if the file exists already, populates the trajectories dictionary with those from
        the DB.

        Args:
            fpath_db (str, optional): Database filepath.
            do_load_sims (bool): Load simulations?
        """

        if fpath_db is None:
            fpath_db = ':memory:'
            is_extant = False
        else:
            is_extant = os.path.isfile(fpath_db)

        self.fpath_db = fpath_db
        self.conn = sqlite3.connect(self.fpath_db, check_same_thread=False)
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.row_factory = sqlite3.Row

        # Database does not exist:
        if not is_extant:
            with self.conn as c:
                c.executescript(self.SQL_CREATE_SCHEMA)
            self.is_db_empty = True
            print('New database initialized')

        # Database exists:
        else:
            with self.conn as c:
                for r in c.execute('SELECT id, name, memo FROM traj', []):
                    self.traj[r['id']] = Trajectory(r['name'], r['memo'], ensemble=self, id=r['id'])

            if do_load_sims:
                self.load_sims()

            self.is_db_empty = False
            n_traj = self._db_get_one('SELECT COUNT(*) FROM traj', [])
            print(f'Using existing database (trajectories loaded: {n_traj})')

        self.probe_persistence = ProbePersistenceDB.with_traj(self, self.conn)

    def _db_get_id(self, tbl, where, col='rowid', conn=None):
        c = conn or self.conn
        row = c.execute('SELECT {} FROM {} WHERE {}'.format(col, tbl, where)).fetchone()
        return row[0] if row else None

    def _db_get_id_ins(self, tbl, where, qry, args, conn=None):
        c = conn or self.conn
        id = self._db_get_id(tbl, where, c)
        if id is None:
            id = self._db_ins(qry, args, c)
        return id

    def _db_get_one(self, qry, args, conn=None):
        c = conn or self.conn
        ret = c.execute(qry, args).fetchone()
        if ret is not None:
            ret = ret[0]
        return ret

    def _db_ins(self, qry, args, conn=None):
        if conn is not None:
            return conn.execute(qry, args).lastrowid

        with self.conn as c:
            return c.execute(qry, args).lastrowid

    def _db_upd(self, qry, args, conn=None):
        if conn is not None:
            conn.execute(qry, args)
        else:
            with self.conn as c:
                c.execute(qry, args)

    def add_trajectories(self, traj):
        """Add trajectories.

        Args:
            traj (Iterable[Trajectory]): The trajectories.

        Returns:
            ``self``
        """

        for t in traj:
            self.add_trajectory(t)
        return self

    def add_trajectory(self, t):
        """Add a trajectory.

        For convenience, ``t`` can be either a :class:`~pram.traj.Trajectory` or :class:`~pram.sim.Simulation` class
        instance.  In the latter case, a :class:`~pram.traj.Trajectory` object will automatically be created with the
        default values.

        Args:
            t (Trajectory or Simulation): The trajectory or the simulation that should wrapped by a trajectory.

        Returns:
            ``self``
        """

        if isinstance(t, Simulation):
            t = Trajectory(t)
        elif t.name is not None and self._db_get_one('SELECT COUNT(*) FROM traj WHERE name = ?', [t.name]) > 0:
            return print(f'A trajectory with the name specified already exists: {t.name}')

        with self.conn as c:
            t.set_id(c.execute('INSERT INTO traj (name, memo) VALUES (?,?)', [t.name, t.memo]).lastrowid)
            # for (i,r) in enumerate(t.sim.rules):
            #     c.execute('INSERT INTO rule (traj_id, ord, name, src) VALUES (?,?,?,?)', [t.id, i, r.__class__.__name__, inspect.getsource(r.__class__)])

            for p in t.sim.probes:
                p.set_persistence(self.probe_persistence)

        t.ens = self
        self.traj[t.id] = t

        return self

    def clear_group_names(self):
        """Removes group names from the trajectory ensemble database.

        Returns:
            ``self``
        """

        with self.conn as c:
            c.execute('DELETE FROM grp_name', [])
        return self

    def compact(self):
        """Compact the ensemble.

        Returns:
            ``self``
        """

        for t in self.traj:
            t.compact()
        return self

    def gen_agent(self, traj, n_iter=-1):
        """Generate a single agent's group transition path based on population-level mass dynamics that a PRAM
        simulation operates on.

        This is a two-step process:

        (1) Pick the agent's initial group respecting the initial mass distribution among the groups
        (2) Pick the next group respecting transition probabilities to all possible next groups

        Because step 1 always takes place, the resulting list of agent's states will be of size ``n_iter + 1``.

        Args:
            traj (Trajectory): The trajectory to use.
            n_iter (int): Number of iterations to generate (use -1 for as many as many iterations there are in the
                trajectory).

        Returns:
            Mapping[str, Mapping[str, Any]]: A dictionary with two keys, ``attr`` and ``rel`` which correspond to the
                attributes and relations of the PRAM group that agent would be a part of if it were in a PRAM model.
        """

        agent = { 'attr': {}, 'rel': {} }
        with self.conn as c:
            if n_iter <= -1:
                n_iter = self._db_get_one('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
            else:
                n_iter = max(0, min(n_iter, self._db_get_one('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])))

            for i in range(-1, n_iter):
                if i == -1:  # (1) setting the initial group
                    # groups = list(zip(*[[r[0], round(r[1],2)] for r in c.execute('SELECT g.id, g.m_p FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ?', [traj.id, -1])]))
                    groups = list(zip(*[[r[0], round(r[1],2)] for r in c.execute('SELECT grp_id, m_p FROM mass_locus ml INNER JOIN iter i ON ml.iter_id = i.id WHERE i.traj_id = ? AND i.i = ?', [traj.id, -1])]))
                else:  # (2) generating a sequence of group transitions
                    groups = list(zip(*[[r[0], round(r[1],2)] for r in c.execute('SELECT g_dst.id, mf.m_p FROM mass_flow mf INNER JOIN iter i ON i.id = mf.iter_id INNER JOIN grp g_src ON g_src.id = mf.grp_src_id INNER JOIN grp g_dst ON g_dst.id = mf.grp_dst_id WHERE i.traj_id = ? AND i.i = ? AND g_src.id = ?', [traj.id, i, grp_id])]))

                # print(groups)
                if sum(groups[1]) > 0:  # prevents errors when the sum is zero (should always be True for the 1st iter)
                    grp_id = random.choices(groups[0], groups[1])[0]

                for attr_rel in ['attr', 'rel']:
                    grp_attr_rel = DB.blob2obj(self._db_get_one(f'SELECT {attr_rel} FROM grp WHERE id = ?', [grp_id]))
                    for (k,v) in grp_attr_rel.items():
                        if k not in agent[attr_rel].keys():
                            agent[attr_rel][k] = [None] + [None] * n_iter
                        agent[attr_rel][k][i+1] = v

        return agent

    def gen_agent_pop(self, traj, n_agents=1, n_iter=-1):
        """Generate a agent population based on procedure described in :meth:`~pram.traj.TrajectoryEnsemble.gen_agent`.

        Args:
            traj (Trajectory): The trajectory to use.
            n_agents (int): Size of resulting agent population.
            n_iter (int): Number of iterations to generate (use -1 for as many as many iterations there are in the
                trajectory).

        Returns:
            Iterable[Mapping[str, Mapping[str, Any]]]: Each item is a dict with two keys, ``attr`` and ``rel`` which
                correspond to the attributes and relations of the PRAM group that agent would be a part of if it were
                in a PRAM model.
        """

        return [self.gen_agent(traj, n_iter) for _ in range(n_agents)]

    def gen_mass_graph(self, traj):
        """Generate a mass graph.

        Args:
            traj (Trajectory): The trajectory being the graph's basis.

        Returns:
            MassGraph
        """

        g = MassGraph()
        n_iter = self._db_get_one('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
        with self.conn as c:
            # Groups (vertices):
            for i in range(-1, n_iter + 1):
                for r in c.execute('''
                        SELECT g.hash, ml.m, ml.m_p FROM grp g
                        INNER JOIN mass_locus ml ON ml.grp_id = g.id
                        INNER JOIN iter i ON i.id = ml.iter_id
                        WHERE i.traj_id = ? AND i.i = ?
                        ORDER BY g.id''', [traj.id, i]):
                    g.add_group(i, r['hash'], r['m'], r['m_p'])

            # Mass flow (edges):
            for i in range(n_iter + 1):
                for r in c.execute('''
                        SELECT g1.hash AS src_hash, g2.hash AS dst_hash, mf.m AS m, mf.m_p AS m_p
                        FROM mass_flow mf
                        INNER JOIN iter i ON i.id = mf.iter_id
                        INNER JOIN grp g1 ON mf.grp_src_id = g1.id
                        INNER JOIN grp g2 ON mf.grp_dst_id = g2.id
                        WHERE i.traj_id = ? AND i.i = ?
                        ORDER BY mf.id''',
                        [traj.id, i]):
                    g.add_mass_flow(i, r['src_hash'], r['dst_hash'], r['m'], r['m_p'])

        return g

    def get_signal(self, traj, do_prob=False):
        """Get time series of masses (or proportions of total mass) of all groups.

        Args:
            traj (Trajectory): The trajectory.
            do_prob (bool): Do proportions of total mass?

        Returns:
            Signal
        """

        n_iter_max = self._db_get_one('''
            SELECT MAX(n_iter) AS n_iter FROM (
            SELECT g.hash, COUNT(*) AS n_iter FROM grp g
            INNER JOIN mass_locus ml ON ml.grp_id = g.id
            INNER JOIN iter i ON i.id = ml.iter_id
            INNER JOIN traj t ON t.id = i.traj_id
            WHERE t.id = ?
            GROUP BY g.hash
            )''', [traj.id]
        )

        y = 'm_p' if do_prob else 'm'
        signal = Signal()

        for g in self.conn.execute('SELECT DISTINCT g.hash, gn.name FROM grp g LEFT JOIN grp_name gn ON gn.hash = g.hash ORDER BY gn.ord, g.id').fetchall():
            s = np.full([1, n_iter_max], np.nan)  # account for missing values in the signal series
                    # SELECT i.i + 1 AS i, g.{y} AS y FROM grp g
            for iter in self.conn.execute(f'''
                    SELECT i.i + 1 AS i, ml.{y} AS y
                    FROM grp g
                    INNER JOIN mass_locus ml ON ml.grp_id = g.id
                    INNER JOIN iter i ON i.id = ml.iter_id
                    INNER JOIN traj t ON t.id = i.traj_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE t.id = ? AND g.hash = ?
                    ORDER BY gn.ord, g.hash, i.i''', [traj.id, g['hash']]).fetchall():
                s[0,iter['i']] = iter['y']
            signal.add_series(s, g['name'] or g['hash'])

        return signal

    def get_time_series(self, traj, group_hash):
        """Get a time series of group mass dynamics.

        Args:
            traj (Trajectory): The trajectory.
            group_hash (int or str): Group's hash obtained by calling
                :meth:`Group.get_hash() <pram.entity.Group.get_hash>`.

        Returns:
            sqlite3.Row
        """

        return self.conn.execute('''
            SELECT g.m, g.m_p, i.i FROM grp g
            INNER JOIN iter i ON i.id = g.iter_id
            INNER JOIN traj t ON t.id = i.traj_id
            WHERE t.id = ? AND g.hash = ?
            ORDER BY i.i
            ''', [traj.id, group_hash]).fetchall()

    def load_sim(self, traj):
        """Load simulation for the designated trajectory from the ensemble database.

        Args:
            traj (Trajectory): The trajectory.

        Returns:
            ``self``
        """

        traj.sim = DB.blob2obj(self.conn.execute('SELECT sim FROM traj WHERE id = ?', [traj.id]).fetchone()[0])
        if traj.sim:
            traj.sim.traj_id = traj.id  # restore the link severed at save time
        return self

    def load_sims(self):
        """Load simulations for all ensemble trajectories from the ensemble database.

        Args:
            traj (Trajectory): The trajectory.

        Returns:
            ``self``
        """

        gc.disable()
        for t in self.traj.values():
            self.load_sim(t)
        gc.enable()
        return self

    def normalize_iter_range(self, range=(-1, -1), qry_n_iter='SELECT MAX(i) FROM iter', qry_args=[]):
        """

        Args:
            range (tuple[int,int]): The range of values.
            qry_n_iter (str): SQL query for obtaining total number of iterations.
            qry_args (Iterable[str]): The SQL query arguments.

        Returns:
            tuple[int,int]
        """

        l = max(range[0], -1)

        n_iter = self._db_get_one(qry_n_iter, qry_args)
        if range[1] <= -1:
            u = n_iter
        else:
            u = min(range[1], n_iter)

        if l > u:
            raise ValueError('Iteration range error: Lower bound cannot be larger than upper bound.')

        return (l,u)

    def plot_mass_locus_bubble(self, size, filepath, iter_range=(-1, -1), do_ret_plot=False):
        """Generate a mass locus bubble plot.

        Todo:
            Finish implementing.
        """

        return self  # unfinished

        title = f'Trajectory Ensemble Mass Locus (Mean + Max; n={len(self.traj)})'

        # (1) Normalize iteration bounds:
        iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter', [])

        # (2) Plot:
        # (2.1) Construct data bundle:
        data = []
        for i in range(iter_range[0], iter_range[1] + 1, 1):
            with self.conn as c:
                for r in c.execute('SELECT g.m, g.hash FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.i = ?', [i]):
                    data.append({ 'i': i + 1, 'm': r['m'], 'grp': r['hash'], 'y': 10 })

        # (2.2) Plot iterations:
        plot = alt.Chart(alt.Data(values=data))
        plot.properties(title=title, width=size[0], height=size[1])
        plot.mark_point(strokeWidth=1, interpolate='basis')  # tension=1  # basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
        plot.configure_view()
        plot.configure_title(fontSize=20)
        plot.encode(
            alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
            alt.Y('y:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
            alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15)),
            alt.Size('mean(m):Q')
        )
        plot.save(filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER)

        return plot if do_ret_plot else self

    def plot_mass_locus_fft(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_ret_plot=False):
        """Generate a plot of mass locus Fast Fourier Transform.

        Args:
            traj (Trajectory): The trajectory.
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            sampling_rate (int): Sampling rate.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; altair chart object otherwise.
        """

        # (1) Data:
        data = { 'td': {}, 'fd': {} }  # time- and frequency-domain
        with self.conn as c:
            # (1.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
            n_iter = iter_range[1] - min(iter_range[0], 0)
            title = f'Trajectory Mass Locus Spectrum (FFT; Sampling Rate of {sampling_rate} on Iterations {iter_range[0]+1} to {iter_range[1]+1})'

            # (1.2) Construct time-domain data bundle:
            for r in c.execute('''
                    SELECT i.i, ml.m, COALESCE(gn.name, g.hash) AS name
                    FROM grp g
                    INNER JOIN mass_locus ml ON g.id = ml.grp_id
                    INNER JOIN iter i ON i.id = ml.iter_id
                    INNER JOIN traj t ON t.id = i.traj_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    ORDER BY gn.ord, g.id''', [traj.id, iter_range[0], iter_range[1]]):
                    # SELECT COALESCE(gn.name, g.hash) AS grp, g.m
                    # FROM grp g
                    # INNER JOIN iter i ON i.id = g.iter_id
                    # LEFT JOIN grp_name gn ON gn.hash = g.hash
                    # WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    # ORDER BY gn.ord, g.id''',
                    # [traj.id, iter_range[0], iter_range[1]]):
                if data['td'].get(r['grp']) is None:
                    data['td'][r['grp']] = []
                data['td'][r['grp']].append(r['m'])

            # (1.3) Move to frequency domain:
            N = sampling_rate
            T = 1 / sampling_rate  # Nyquist sampling criteria
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

            # import matplotlib.pyplot as plt
            for g in data['td'].keys():
                yf = fft(data['td'][g])        # positive and negative frequencies
                yf = 2/N * np.abs(yf[0:N//2])  # positive frequencies only
                # data['fd'][g] = [{ 'grp': g, 'x': z[0], 'y': z[1] / n_iter } for z in zip(xf,yf)]
                data['fd'][g] = [{ 'grp': g, 'x': z[0], 'y': z[1] } for z in zip(xf,yf)]

                # yf = fft(data['td'][g])        # positive and negative frequencies
                # yf = 2/N * np.abs(yf[0:N//2])  # positive frequencies only
                # data['fd'][g] = [{ 'grp': g, 'x': z[0], 'y': z[1] / n_iter } for z in zip(xf,yf)]

                # fig = plt.figure(figsize=size)
                # # plt.legend(['Susceptible', 'Infectious', 'Recovered'], loc='upper right')
                # plt.xlabel('Frequency')
                # plt.ylabel('Mass flow')
                # plt.grid(alpha=0.25, antialiased=True)
                # plt.plot(x, y, lw=1, linestyle='--', color='red', mfc='none', antialiased=True)
                # fig.savefig(f'__{g}', dpi=300)

            # (1.4) Group sorting (needs to be done here due to Altair's peculiarities):
            # ...

        # (2) Plot:
        plot = alt.Chart(alt.Data(values=[x for xs in data['fd'].values() for x in xs]))
        plot.properties(title=title, width=size[0], height=size[1])
        plot.mark_line(strokeWidth=1, opacity=0.75, interpolate='basis', tension=1)
        plot.encode(
            alt.X('x:Q', axis=alt.Axis(title='Frequency', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, sampling_rate // 2))),
            alt.Y('y:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
            alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))
            # alt.Order('year(data):O')
        )
        plot.configure_title(fontSize=20)
        plot.configure_view()
        plot.save(filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER)

        return plot if do_ret_plot else self

    def plot_mass_locus_scaleogram(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_ret_plot=False):
        """Generate a mass locus scalegram.

        Currently, Image Mark in not supported in Vega-Lite.  Consequently, raster images cannot be displayed via
        Altair.  The relevant issue is: https://github.com/vega/vega-lite/issues/3758

        Args:
            traj (Trajectory): The trajectory.
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            sampling_rate (int): Sampling rate.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; matplotlib figure object otherwise.
        """

        # https://docs.obspy.org/tutorial/code_snippets/continuous_wavelet_transform.html
        # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.cwt.html

        # (1) Data:
        data = { 'td': {}, 'fd': {} }  # time- and frequency-domain
        with self.conn as c:
            # (1.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
            n_iter = iter_range[1] - min(iter_range[0], 0)
            title = f'Trajectory Mass Locus Scalogram (Sampling Rate of {sampling_rate} on Iterations {iter_range[0]+1} to {iter_range[1]+1})'

            # (1.2) Construct time-domain data bundle:
            for r in c.execute('''
                    SELECT i.i, ml.m, COALESCE(gn.name, g.hash) AS name
                    FROM grp g
                    INNER JOIN mass_locus ml ON g.id = ml.grp_id
                    INNER JOIN iter i ON i.id = ml.iter_id
                    INNER JOIN traj t ON t.id = i.traj_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    ORDER BY gn.ord, g.id''', [traj.id, iter_range[0], iter_range[1]]):
                    # SELECT COALESCE(gn.name, g.hash) AS grp, g.m
                    # FROM grp g
                    # INNER JOIN iter i ON i.id = g.iter_id
                    # LEFT JOIN grp_name gn ON gn.hash = g.hash
                    # WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    # ORDER BY gn.ord, g.id''',
                    # [traj.id, iter_range[0], iter_range[1]]):
                if data['td'].get(r['grp']) is None:
                    data['td'][r['grp']] = []
                data['td'][r['grp']].append(r['m'])

        # (2) Move to frequency domain and plot:
        widths = np.arange(1, sampling_rate // 2 + 1)

        fig, ax = plt.subplots(len(data['td']), 1, figsize=size, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.suptitle(title, fontweight='bold')
        plt.xlabel('Time', fontweight='bold')
        fig.text(0.08, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontweight='bold')

        for (i,g) in enumerate(data['td'].keys()):
            data['fd'][g] = signal.cwt(data['td'][g], signal.ricker, widths)  # "Mexican hat wavelet"
            ax[i].imshow(data['fd'][g], extent=[-1, 1, 1, sampling_rate // 2 + 1], cmap='PRGn', aspect='auto', vmax=abs(data['fd'][g]).max(), vmin=-abs(data['fd'][g]).max())
            ax[i].set_ylabel(g, fontweight='bold')

        fig.savefig(filepath, dpi=300)

        return fig if do_ret_plot else self

    def plot_mass_locus_spectrogram(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=None, win_len=None, noverlap=None, do_ret_plot=False):
        """Generate a mass locus spectrogram (Short-Time Fourier Transform).

        Args:
            traj (Trajectory): The trajectory.
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            sampling_rate (int): Sampling rate.
            win_len (int): Length of the windowing segments.
            noverlap (int): Windowing segment overlap.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; matplotlib figure object otherwise.
        """

        # Docs
        #     https://kite.com/python/docs/matplotlib.mlab.specgram
        # Examples
        #     https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/specgram_demo.html#sphx-glr-gallery-images-contours-and-fields-specgram-demo-py
        #     https://stackoverflow.com/questions/35932145/plotting-with-matplotlib-specgram
        #     https://pythontic.com/visualization/signals/spectrogram
        #     http://www.toolsmiths.com/wavelet/wavbox
        # TODO
        #     http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
        #
        #     https://www.sciencedirect.com/topics/neuroscience/signal-processing
        #     https://www.google.com/search?client=firefox-b-1-d&q=Signal+Processing+for+Neuroscientists+pdf
        #
        #     https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.cwt.html
        #     https://www.semanticscholar.org/paper/A-wavelet-based-tool-for-studying-non-periodicity-Ben%C3%ADtez-Bol%C3%B3s/b7cb0789bd2d29222f2def7b70095f95eb72358c
        #     https://www.google.com/search?q=time-frequency+plane+decomposition&client=firefox-b-1-d&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiDxu3R9bXkAhWXqp4KHUSuBqYQ_AUIEigB&biw=1374&bih=829#imgrc=q2MCGaBIY3lrSM:
        #     https://www.mathworks.com/help/wavelet/examples/classify-time-series-using-wavelet-analysis-and-deep-learning.html;jsessionid=de786cc8324218efefc12d75c292

        # (1) Data:
        data = { 'td': {}, 'fd': {} }  # time- and frequency-domain
        with self.conn as c:
            # (1.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
            n_iter = iter_range[1] - min(iter_range[0], 0)

            # (1.2) Construct time-domain data bundle:
            for r in c.execute('''
                    SELECT i.i, ml.m, COALESCE(gn.name, g.hash) AS name
                    FROM grp g
                    INNER JOIN mass_locus ml ON g.id = ml.grp_id
                    INNER JOIN iter i ON i.id = ml.iter_id
                    INNER JOIN traj t ON t.id = i.traj_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    ORDER BY i.i, gn.ord''', [traj.id, iter_range[0], iter_range[1]]):
                    # SELECT COALESCE(gn.name, g.hash) AS grp, g.m
                    # FROM grp g
                    # INNER JOIN iter i ON i.id = g.iter_id
                    # LEFT JOIN grp_name gn ON gn.hash = g.hash
                    # WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    # ORDER BY gn.ord, g.id''',
                    # [traj.id, iter_range[0], iter_range[1]]):
                if data['td'].get(r['grp']) is None:
                    data['td'][r['grp']] = []
                data['td'][r['grp']].append(r['m'])

        # (2) Plot:
        sampling_rate = sampling_rate or self._db_get_one('SELECT MAX(i) + 1 FROM iter WHERE traj_id = ?', [traj.id])
        win_len = win_len or sampling_rate // 100

        NFFT = win_len           # the length of the windowing segments
        Fs = sampling_rate // 1  # the sampling frequency (same as sampling rate so we get 0..1 time range)
        noverlap = noverlap or NFFT // 2

        fig, ax = plt.subplots(len(data['td']), 1, figsize=size, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.suptitle(f'Trajectory Mass Locus Spectrogram (STFT; Sampling rate: {sampling_rate}, window size: {win_len}, window overlap: {noverlap}; Iterations: {iter_range[0]+1}-{iter_range[1]+1})', fontweight='bold')
        plt.xlabel('Time', fontweight='bold')
        fig.text(0.08, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontweight='bold')

        for (i,g) in enumerate(data['td'].keys()):
            ax[i].specgram(data['td'][g], NFFT=NFFT, Fs=Fs, noverlap=noverlap)  # cmap=plt.cm.gist_heat
            ax[i].set_ylabel(g, fontweight='bold')

        fig.savefig(filepath, dpi=300)

        return fig if do_ret_plot else self

    def plot_mass_locus_line(self, size, filepath, iter_range=(-1, -1), traj=None, n_traj=0, opacity_min=0.1, stroke_w=1, col_scheme='set1', do_ret_plot=False):
        """Generate a mass locus line plot (individual series).

        Args:
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            traj (Trajectory, optional): The trajectory.  If None, ``n_traj`` trajectories will be plotted.
            n_traj (int): Number of trajectories to sample from the ensemble.  All trajectories will be plotted if the
                value is non-positive or if it exceeds the total number of trajectories in the ensemble.
            opacity_min (float): Minimum line opacity.  Actual opacity value is scaled by the number of trajectories
                plotted; the more there are, the more transparent the lines will be.
            stroke_w (float): Line width.
            col_scheme (str): Color scheme name.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; altair chart object otherwise.
        """

        # (1) Sample trajectories (if necessary) + set title + set line alpha:
        if traj is not None:
            traj_sample = [traj]
            title = f'Trajectory Mass Locus'  #  (Iterations {iter_range[0]+1} to {iter_range[1]+1})
            opacity = 1.00
        else:
            traj_sample = []
            if n_traj <=0 or n_traj >= len(self.traj):
                traj_sample = self.traj.values()
                title = f'Trajectory Ensemble Mass Locus (n={len(self.traj)})'
            else:
                traj_sample = random.sample(list(self.traj.values()), n_traj)
                title = f'Trajectory Ensemble Mass Locus (Random Sample of {len(traj_sample)} from {len(self.traj)})'
            opacity = max(opacity_min, 1.00 / len(traj_sample))

        # iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [next(iter(traj_sample)).id])
        # title += f'Iterations {iter_range[0]+1} to {iter_range[1]+1})'

        # (2) Group sorting (needs to be done here due to Altair's peculiarities):
        with self.conn as c:
            sort = [r['name'] for r in c.execute('SELECT DISTINCT COALESCE(gn.name, g.hash) AS name FROM grp g LEFT JOIN grp_name gn ON gn.hash = g.hash ORDER BY gn.ord, g.id')]

        # (3) Plot trajectories:
        plots = []
        for (ti,t) in enumerate(traj_sample):
            # (3.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [t.id])

            # (3.2) Construct the trajectory data bundle:
            data = []
            with self.conn as c:
                for r in c.execute('''
                        SELECT i.i, ml.m, COALESCE(gn.name, g.hash) AS name
                        FROM grp g
                        INNER JOIN mass_locus ml ON g.id = ml.grp_id
                        INNER JOIN iter i ON i.id = ml.iter_id
                        LEFT JOIN grp_name gn ON gn.hash = g.hash
                        WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                        ORDER BY i.i, gn.ord''', [t.id, iter_range[0], iter_range[1]]):
                    data.append({ 'i': r['i'] + 1, 'm': r['m'], 'grp': r['name'] })

            # (3.3) Plot the trajectory:
            plots.append(
                alt.Chart(
                    alt.Data(values=data)
                ).mark_line(
                    strokeWidth=stroke_w, opacity=opacity, interpolate='basis', tension=1  # basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
                ).encode(
                    alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                    alt.Y('m:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                    color=(
                        alt.Color('grp:N', scale=alt.Scale(scheme=col_scheme), legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15), sort=sort)
                        if ti == 0 else
                        alt.Color('grp:N', scale=alt.Scale(scheme=col_scheme), sort=sort, legend=None)
                    )
                    # alt.Order('year(data):O')
                )
            )

        plot = alt.layer(*plots)
        plot.properties(title=title, width=size[0], height=size[1])
        plot.configure_view()  # strokeWidth=1
        plot.configure_title(fontSize=20)
        plot.resolve_scale(color='independent')
        # plot.save(filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER)
        alt_save.save(plot, filepath)

        return plot if do_ret_plot else self

    def plot_mass_locus_line_aggr(self, size, filepath, iter_range=(-1, -1), band_type='ci', stroke_w=1, col_scheme='set1', do_ret_plot=False):
        """Generate a mass locus line plot (aggregated).

        Args:
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            band_type (str): Band type.
            stroke_w (float): Line width.
            col_scheme (str): Color scheme name.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; altair chart object otherwise.
        """

        # Ordering the legend of a composite chart
        #     https://stackoverflow.com/questions/55783286/control-legend-color-and-order-when-joining-two-charts-in-altair
        #     https://github.com/altair-viz/altair/issues/820

        title = f'Trajectory Ensemble Mass Locus (Mean + {band_type.upper()}; n={len(self.traj)})'

        # (1) Normalize iteration bounds:
        iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter', [])

        # (2) Group sorting (needs to be done here due to Altair's peculiarities):
        with self.conn as c:
            sort = [r['name'] for r in c.execute('SELECT DISTINCT COALESCE(gn.name, g.hash) AS name FROM grp g LEFT JOIN grp_name gn ON gn.hash = g.hash ORDER BY gn.ord, g.id')]

        # (3) Plot:
        # (3.1) Construct data bundle:
        data = []
        with self.conn as c:
            for r in c.execute('''
                    SELECT i.i, ml.m, COALESCE(gn.name, g.hash) AS name
                    FROM grp g
                    INNER JOIN mass_locus ml ON g.id = ml.grp_id
                    INNER JOIN iter i ON i.id = ml.iter_id
                    INNER JOIN traj t ON t.id = i.traj_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.i BETWEEN ? AND ?
                    ORDER BY t.id, i.i, gn.ord''', [iter_range[0], iter_range[1]]):
                data.append({ 'i': r['i'] + 1, 'm': r['m'], 'grp': r['name'] })

        # (3.2) Plot iterations:
        plot_line = alt.Chart(
            ).mark_line(
                strokeWidth=stroke_w, interpolate='basis'#, tension=1  # basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
            ).encode(
                alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                alt.Y('mean(m):Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                alt.Color('grp:N', scale=alt.Scale(scheme=col_scheme), legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15), sort=sort)
            )

        plot_band = alt.Chart(  # https://altair-viz.github.io/user_guide/generated/core/altair.ErrorBandDef.html#altair.ErrorBandDef
            ).mark_errorband(
                extent=band_type, interpolate='basis'#, tension=1  # opacity, basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
            ).encode(
                alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                alt.Y('mean(m):Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                alt.Color('grp:N', scale=alt.Scale(scheme=col_scheme), legend=None, sort=sort)
            )

        plot = alt.layer(plot_band, plot_line, data=alt.Data(values=data))
        plot.properties(title=title, width=size[0], height=size[1])
        plot.configure_view()
        plot.configure_title(fontSize=20)
        plot.resolve_scale(color='independent')
        # plot.save(filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER)
        alt_save.save(plot, filepath)

        return plot if do_ret_plot else self

    def plot_mass_locus_line_probe(self, size, filepath, probe_name, series, iter_range=(-1, -1), traj=None, n_traj=0, opacity_min=0.1, stroke_w=1, col_scheme='set1', do_ret_plot=False):
        """Generate a mass locus line plot (probe).

        Args:
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            probe_name (str): The probe's name.
            series (Iterable[Mapping[str,str]]): Series to be plotted from the ones recorded by the probe.  Each series
                is a dict with two keys, ``var`` and ``lbl``.  The first selects the variable to be plotted while the
                seconds controls its name on the plot.
            iter_range (tuple[int,int]): Range of iterations.
            traj (Trajectory, optional): The trajectory.  If None, ``n_traj`` trajectories will be plotted.
            n_traj (int): Number of trajectories to sample from the ensemble.  All trajectories will be plotted if the
                value is non-positive or if it exceeds the total number of trajectories in the ensemble.
            opacity_min (float): Minimum line opacity.  Actual opacity value is scaled by the number of trajectories
                plotted; the more there are, the more transparent the lines will be.
            stroke_w (float): Line width.
            col_scheme (str): Color scheme name.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; altair chart object otherwise.
        """

        # (1) Sample trajectories (if necessary) + set title + set line alpha:
        if traj is not None:
            traj_sample = [traj]
            title = f'Trajectory Mass Locus'  #  (Iterations {iter_range[0]+1} to {iter_range[1]+1})
            opacity = 1.00
        else:
            traj_sample = []
            if n_traj <= 0 or n_traj >= len(self.traj):
                traj_sample = self.traj.values()
                title = f'Trajectory Ensemble Mass Locus (n={len(self.traj)})'
            else:
                traj_sample = random.sample(list(self.traj.values()), n_traj)
                title = f'Trajectory Ensemble Mass Locus (Random Sample of {len(traj_sample)} from {len(self.traj)})'
            opacity = max(opacity_min, 1.00 / len(traj_sample))

        # (2) Plot trajectories:
        plots = []
        for (ti,t) in enumerate(traj_sample):
            # (3.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [t.id])

            # (3.2) Construct the trajectory data bundle:
            data = []
            with self.conn as c:
                for s in series:
                    for r in c.execute(f'''
                            SELECT i.i, p.{s['var']} AS y
                            FROM {probe_name} p
                            INNER JOIN iter i ON i.id = p.iter_id
                            WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                            ORDER BY i.i''', [t.id, iter_range[0], iter_range[1]]):
                        data.append({ 'i': r['i'] + 1, 'y': r['y'], 'series': s['lbl'] })

            # (3.3) Plot the trajectory:
            plots.append(
                alt.Chart(
                    alt.Data(values=data)
                ).mark_line(
                    strokeWidth=stroke_w, opacity=opacity, interpolate='basis', tension=1  # basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
                ).encode(
                    alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                    alt.Y('y:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                    color=(
                        alt.Color('series:N', scale=alt.Scale(scheme=col_scheme), legend=alt.Legend(title='Series', labelFontSize=15, titleFontSize=15), sort=[s['lbl'] for s in series])
                        if ti == 0 else
                        alt.Color('series:N', scale=alt.Scale(scheme=col_scheme), sort=[s['lbl'] for s in series], legend=None)
                    )
                    # alt.Order('year(data):O')
                )
            )

        plot = alt.layer(*plots)
        plot.properties(title=title, width=size[0], height=size[1])
        plot.configure_view()  # strokeWidth=1
        plot.configure_title(fontSize=20)
        plot.resolve_scale(color='independent')
        # plot.save(filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER)
        alt_save.save(plot, filepath)

        return plot if do_ret_plot else self

    def plot_mass_locus_polar(self, size, filepath, iter_range=(-1, -1), n_traj=0, n_iter_per_rot=0, do_ret_plot=False):
        """Generate a mass locus polar plot.

        Note:
            Altair does not currently support projections, so we have to use matplotlib.

        Args:
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            n_traj (int): Number of trajectories to sample from the ensemble.  All trajectories will be plotted if the
                value is non-positive or if it exceeds the total number of trajectories in the ensemble.
                plotted; the more there are, the more transparent the lines will be.
            n_iter_per_rot (int): Number of iterations that one rotation should comprise.  If zero, it gets determined
                automatically based on ``iter_range``.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; matplotlib figure object otherwise.
        """

        # (1) Sample trajectories (if necessary) + set parameters and plot title:
        traj_sample = []
        if n_traj <=0 or n_traj >= len(self.traj):
            traj_sample = self.traj.values()
            title = f'Trajectory Ensemble Mass Locus (n={len(self.traj)}; '
        else:
            traj_sample = random.sample(list(self.traj.values()), n_traj)
            title = f'Trajectory Ensemble Mass Locus (Random Sample of {len(traj_sample)} from {len(self.traj)}; '

        iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [next(iter(traj_sample)).id])
        n_iter_per_rot = n_iter_per_rot if (n_iter_per_rot > 0) else iter_range[1] - iter_range[0]
        theta = np.arange(iter_range[0] + 1, iter_range[1] + 2, 1) * 2 * np.pi / n_iter_per_rot
        if n_iter_per_rot == iter_range[1] - iter_range[0]:
            title += f'Iterations {iter_range[0]+1} to {iter_range[1]+1})'
        else:
            title += f'Iterations {iter_range[0]+1} to {iter_range[1]+1} rotating every {n_iter_per_rot})'

        # (2) Plot trajectories:
        n_cmap = 10  # used to cycle through the colors later on
        cmap = plt.get_cmap(f'tab{n_cmap}')
        fig = plt.figure(figsize=size)
        plt.grid(alpha=0.20, antialiased=True)
        plt.suptitle(title, fontweight='bold')
        ax = plt.subplot(111, projection='polar')
        ax.set_rmax(1.00)
        ax.set_rticks([0.25, 0.50, 0.75])
        # ax.set_rlabel_position(0)

        for (i,t) in enumerate(traj_sample):
            # (2.2) Retrieve the mass dynamics signal:
            signal = self.get_signal(t, True)

            # (2.3) Plot the signal:
            for (j,s) in enumerate(signal.S):
                ax.plot(theta, s[iter_range[0] + 1:iter_range[1] + 2], lw=1, linestyle='-', alpha=0.1, color=cmap(j % n_cmap), mfc='none', antialiased=True)
            if i == 0:
                ax.legend(signal.names, loc='upper right')

        fig.tight_layout()
        plt.subplots_adjust(top=0.92)
        fig.savefig(filepath, dpi=300)

        return fig if do_ret_plot else self

    def plot_mass_locus_recurrence(self, traj, size, filepath, iter_range=(-1, -1), neighbourhood=FixedRadius(), embedding_dimension=1, time_delay=2, do_ret_plot=False):
        """Generate a mass locus recurrence plot.

        See `PyRQA <https://pypi.org/project/PyRQA>`_ for information on parameterizing the plot.

        Todo:
            Implement multivariate extensions of recurrence plots (including cross recurrence plots and joint
            recurrence plots).

        Args:
            traj (Trajectory): The trajectory.
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            neighbourhood (pyrqa.abstract_classes.AbstractNeighbourhood): Neighbourhood condition.
            embedding_dimension (int): Embedding dimension.
            time_delay (int): Time delay.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; pyrqa RPComputation object otherwise.
        """

        from pyrqa.time_series     import TimeSeries
        from pyrqa.settings        import Settings
        from pyrqa.computing_type  import ComputingType
        from pyrqa.metric          import EuclideanMetric
        from pyrqa.computation     import RQAComputation
        from pyrqa.computation     import RPComputation
        from pyrqa.image_generator import ImageGenerator

        iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
        signal = traj.get_signal()
        ts = TimeSeries(list(zip(*signal.S)), embedding_dimension=embedding_dimension, time_delay=time_delay)  # len(signal.S)

        # with self.conn as c:
            # ts = TimeSeries([r['m'] for r in c.execute('''
            #         SELECT g.m FROM grp g
            #         INNER JOIN iter i ON i.id = g.iter_id
            #         WHERE i.traj_id = ? AND i.i BETWEEN ? AND ? AND g.hash = ?
            #         ORDER BY i.i''',
            #         [traj.id, iter_range[0], iter_range[1], group_hash]
            #     )], embedding_dimension=1, time_delay=2)


        settings = Settings(ts, computing_type=ComputingType.Classic, neighbourhood=neighbourhood, similarity_measure=EuclideanMetric, theiler_corrector=1)

        # Debug:
        # computation = RQAComputation.create(settings, verbose=True)
        # result = computation.run()
        # result.min_diagonal_line_length = 2
        # result.min_vertical_line_length = 2
        # result.min_white_vertical_line_lelngth = 2
        # print(result)

        computation = RPComputation.create(settings)
        result = computation.run()
        ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse, filepath)

        return result if do_ret_plot else self

    def plot_mass_locus_streamgraph(self, traj, size, filepath, iter_range=(-1, -1), do_ret_plot=False):
        """Generate a mass locus steamgraph.

        Args:
            traj (Trajectory): The trajectory.
            size (tuple[int,int]): Figure size.
            filepath (str): Destination filepath.
            iter_range (tuple[int,int]): Range of iterations.
            do_ret_plot (bool): Return plot?  If False, ``self`` is returned.

        Returns:
            ``self`` if ``do_ret_plot`` is False; altair chart object otherwise.
        """

        # (1) Data:
        data = []
        with self.conn as c:
            # (1.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])

            # (1.2) Determine max mass sum:
            m_max = self._db_get_one('''
                SELECT ROUND(MAX(m_sum),4) FROM (
                SELECT SUM(m) AS m_sum FROM mass_locus ml INNER JOIN iter i on i.id = ml.iter_id WHERE i.traj_id = ? GROUP BY ml.iter_id
                )''', [traj.id]
            )  # without rounding, weird-ass max values can appear due to inexact floating-point arithmetic (four decimals is arbitrary though)

            # (1.3) Construct the data bundle:
            for r in c.execute('''
                    SELECT i.i, ml.m, COALESCE(gn.name, g.hash) AS name FROM grp g
                    INNER JOIN mass_locus ml ON g.id = ml.grp_id
                    INNER JOIN iter i ON i.id = ml.iter_id
                    INNER JOIN traj t ON t.id = i.traj_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    ORDER BY i.i, gn.ord, g.id''', [traj.id, iter_range[0], iter_range[1]]):
                data.append({ 'grp': r['name'], 'i': r['i'] + 1, 'm': r['m'] })

            # (1.4) Group sorting (needs to be done here due to Altair's peculiarities):
            sort = [r['name'] for r in c.execute('SELECT name FROM grp_name ORDER BY ord')]
            # sort = [r['name'] for r in c.execute('SELECT COALESCE(gn.name, g.hash) AS name FROM grp g LEFT JOIN grp_name gn ON gn.hash = g.hash ORDER BY gn.ord, g.id')]
            plot_color = alt.Color('grp:N', scale=alt.Scale(scheme='category20b'), legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15), sort=sort)

            # plot_color = alt.Color('grp:N', scale=alt.Scale(scheme='category20b'), legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))  # the do-not-sort version

        # (2) Plot:
        plot = alt.Chart(alt.Data(values=data))
        plot.properties(title='Trajectory Mass Locus', width=size[0], height=size[1])
        plot.mark_area().encode(
            alt.X('i:Q', axis=alt.Axis(domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
            alt.Y('sum(m):Q', axis=alt.Axis(domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), stack='center', scale=alt.Scale(domain=(0, m_max))),
            plot_color
            # alt.Order('year(data):O')
        )
        plot.configure_view(strokeWidth=0)
        # plot.save(filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER)
        alt_save.save(plot, filepath)

        return plot if do_ret_plot else self

    def run(self, iter_or_dur=1, is_quiet=False):
        """Run the ensemble.

        The ensemble will be executed on a computational cluster if the cluster info has been associated with it or
        sequentially otherwise.

        Args:
            iter_or_dur (int or str): Number of iterations or a string representation of duration (see
                :meth:`util.Time.dur2ms() <pram.util.Time.dur2ms>`)

        Returns:
            ``self``
        """

        if iter_or_dur < 1:
            return

        if not self.cluster_inf:
            return self.run__seq(iter_or_dur, is_quiet)
        else:
            return self.run__par(iter_or_dur, is_quiet)

    def run__seq(self, iter_or_dur=1, is_quiet=False):
        """Run the ensemble sequentially.

        Args:
            iter_or_dur (int or str): Number of iterations or a string representation of duration (see
                :meth:`util.Time.dur2ms() <pram.util.Time.dur2ms>`)

        Returns:
            ``self``
        """

        ts_sim_0 = Time.ts()
        self.unpersisted_probes = []  # added only for congruency with self.run__par()
        traj_col = len(str(len(self.traj)))
        for (i,t) in enumerate(self.traj.values()):
            if is_quiet:
                t.sim.set_cb_save_state(self.save_work)
                t.run(iter_or_dur)
                t.sim.set_cb_save_state(None)
            else:
                # print(f'Running trajectory {i+1} of {len(self.traj)} (iter count: {iter_or_dur}): {t.name or "unnamed simulation"}')
                with TqdmUpdTo(total=iter_or_dur, miniters=1, desc=f'traj: {i+1:>{traj_col}} of {len(self.traj):>{traj_col}},  iters:{Size.b2h(iter_or_dur, False)}', bar_format='{desc}  |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]', dynamic_ncols=True, ascii=' 123456789.') as pbar:
                    t.sim.set_cb_save_state(self.save_work)
                    t.sim.set_cb_upd_progress(lambda i,n: pbar.update_to(i+1))
                    t.run(iter_or_dur)
                    t.sim.set_cb_upd_progress(None)
                    t.sim.set_cb_save_state(None)
        print(f'Total time: {Time.tsdiff2human(Time.ts() - ts_sim_0)}')
        self.save_sims()
        self.is_db_empty = False
        del self.unpersisted_probes
        return self

    def run__par(self, iter_or_dur=1, is_quiet=False):
        """Run the ensemble on a computational cluster.

        Args:
            iter_or_dur (int or str): Number of iterations or a string representation of duration (see
                :meth:`util.Time.dur2ms() <pram.util.Time.dur2ms>`)

        Returns:
            ``self``
        """

        ts_sim_0 = Time.ts()
        try:
            ray.init(**self.cluster_inf.get_args())

            n_nodes = len(ray.nodes())
            n_cpu   = int(ray.cluster_resources()['CPU'])
            n_traj  = len(self.traj)
            n_iter  = n_traj * iter_or_dur

            work_collector = WorkCollector.remote(10)
            progress_mon = ProgressMonitor.remote()

            for t in self.traj.values():
                t.sim.remote_before()
            self.probe_persistence.remote_before(work_collector)

            self.unpersisted_probes = []  # probes which have not yet been persisted via self.save_work()

            workers = [Worker(i, t.id, t.sim, iter_or_dur, work_collector, progress_mon) for (i,t) in enumerate(self.traj.values())]

            wait_ids = [start_worker.remote(w) for w in workers]
            time.sleep(1)  # give workers time to start
            with TqdmUpdTo(total=n_iter, miniters=1, desc=f'nodes:{n_nodes}  cpus:{n_cpu}  trajs:{n_traj}  iters:{n_traj}×{iter_or_dur}={Size.b2h(n_iter, False)}', bar_format='{desc}  |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]', dynamic_ncols=True, ascii=' 123456789.') as pbar:
                while len(wait_ids) > 0:
                    done_id, wait_ids = ray.wait(wait_ids, timeout=0.1)

                    work = ray.get(work_collector.get.remote())
                    self.save_work(work)
                    del work

                    pbar.update_to(ray.get(progress_mon.get_i.remote()))

            # Code used previously instead of the progress bar:
            #     sys.stdout.write('\r')
            #     sys.stdout.write(ray.get(progress_mon.get_rep.remote()))
            #     sys.stdout.flush()
            # sys.stdout.write('\n')

            self.save_work(self.unpersisted_probes)  # save any remaining to-be-persisted probes

            progress_mon.rem_all_workers.remote()

            for t in self.traj.values():
                t.sim.remote_after()
            self.probe_persistence.remote_after(self, self.conn)
        finally:
            if self.cluster_inf.get_args().get('address') is None:
                ray.shutdown()
            if hasattr(self, 'unpersisted_probes'):
                del self.unpersisted_probes
            print(f'Total time: {Time.tsdiff2human(Time.ts() - ts_sim_0)}')

        self.save_sims()
        self.is_db_empty = False
        return self

    def save_sim(self, traj):
        """Persist the simulation associated with the designated trajectory in the trajectory ensemble database.

        Args:
            traj (Trajectory): The trajectory.

        Returns:
            ``self``
        """

        traj.sim.traj = None  # sever the link to avoid "pickling SQL connection" error (the link is restored at load time)
        # import dill
        # print(dill.detect.baditems(traj.sim))
        # print(dir(traj.sim))
        # print(traj.sim.db)
        # self._db_upd('UPDATE traj SET sim = ? WHERE id = ?', [DB.obj2blob(traj.sim), traj.id])
            # TODO: Uncomment the above line and fix for in Python 3.8 (works in 3.6)
        traj.sim.traj = traj  # restore the link

        return self

    def save_sims(self):
        """Persist simulations associated with all ensemble trajectories in the trajectory ensemble database.

        Returns:
            ``self``
        """

        for t in self.traj.values():
            self.save_sim(t)
        return self

    def save_iter(self, traj_id, iter, host_name, host_ip, conn):
        """Persist the simulation associated with the designated trajectory in the trajectory ensemble database.

        Args:
            traj_id (int or str): The trajectory's database ID.
            iter (int): Iteration.
            host_name (str): Name of host executing the iteration.
            host_ip (str): IP address of host executing the iteration.
            conn (sqlite3.Connection): The SQLite3 connection object.

        Returns:
            int: Iteration database ID.
        """

        return self._db_ins('INSERT INTO iter (traj_id, i, host_name, host_ip) VALUES (?,?,?,?)', [traj_id, iter, host_name, host_ip], conn)

    def save_groups(self, sim, iter_id, conn):
        """Persist all groups of the designated simulation and iteration in the trajectory ensemble database.

        Note:
            Currently unused.

        Args:
            traj_id (int or str): The trajectory's database ID.
            iter (int): Iteration.
            host_name (str): Name of host executing the iteration.
            host_ip (str): IP address of host executing the iteration.
            conn (sqlite.Connection): The SQLite3 connection object.

        Returns:
            ``self``
        """

        # https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field

        # m_pop = traj.sim.pop.get_mass()  # to get proportion of mass flow
        # for g in traj.sim.pop.groups.values():
        #     for s in g.rel.values():  # sever the 'pop.sim.traj.traj_ens._conn' link (or pickle error)
        #         s.pop = None
        #
        #     conn.execute(
        #         'INSERT INTO grp (iter_id, hash, m, m_p, attr, rel) VALUES (?,?,?,?,?,?)',
        #         [iter_id, str(g.get_hash()), g.m, g.m / m_pop, DB.obj2blob(g.attr), DB.obj2blob(g.rel)]
        #     )
        #
        #     for s in g.rel.values():  # restore the link
        #         s.pop = traj.sim.pop

        return self

    def save_mass_flow(self, iter_id, mass_flow_specs, conn):
        """Persist the mass flow in the designated simulation and iteration in the trajectory ensemble database.

        Mass flow is present for all but the initial state of a simulation.

        Note:
            This method has to be called *after* either :meth:`~pram.traj.TrajectoryEnsemble.save_mass_locus__seq()` or
            :meth:`~pram.traj.TrajectoryEnsemble.save_mass_locus__par()` which add all the groups to the ensemble
            database.

        Args:
            iter_id (int or str): Iteration database ID.
            mass_flow_specs (MassFlowSpec): Mass flow specs.
            conn (sqlite3.Connection): The SQLite3 connection object.

        Returns:
            ``self``
        """

        if mass_flow_specs is None:
            return self

        for mfs in mass_flow_specs:
            g_src_id = self._db_get_id('grp', f'hash = "{mfs.src.get_hash()}"')
            for g_dst in mfs.dst:
                g_dst_id = self._db_get_id('grp', f'hash = "{g_dst.get_hash()}"')

                self._db_ins(
                    'INSERT INTO mass_flow (iter_id, grp_src_id, grp_dst_id, m, m_p) VALUES (?,?,?,?,?)',
                    [iter_id, g_src_id, g_dst_id, g_dst.m, g_dst.m / mfs.m_pop]
                )

        return self

    def save_mass_locus__seq(self, pop, iter_id, conn):
        """Persist all new groups (and their attributes and relations) as well as masses of all groups participating
        in the designated iteration (sequential execution).

        Note:
            This method has to be called *before* :meth:`~pram.traj.TrajectoryEnsemble.save_mass_flow()` to ensure all
            groups are already present in the database.

        Args:
            pop (GroupPopulation): The group population.
            iter_id (int or str): Iteration database ID.
            conn (sqlite3.Connection): The SQLite3 connection object.

        Returns:
            ``self``
        """

        # https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field

        m_pop = pop.get_mass()  # to get proportion of mass flow
        for g in pop.groups.values():
            # New group -- persist:
            if self._db_get_one('SELECT COUNT(*) FROM grp WHERE hash = ?', [str(g.get_hash())], conn) == 0:
                # for s in g.rel.values():  # sever the 'pop.sim.traj.traj_ens._conn' link (or pickle error)
                #     s.pop = None

                group_id = conn.execute(
                    'INSERT INTO grp (hash, attr, rel) VALUES (?,?,?)',
                    [str(g.get_hash()), DB.obj2blob(g.attr), DB.obj2blob(g.rel)]
                ).lastrowid

                if self.pragma.memoize_group_ids:
                    self.cache.group_hash_to_id[g.get_hash()] = group_id

                # for s in g.rel.values():  # restore the link
                #     s.pop = pop

            # Extant group:
            else:
                if self.pragma.memoize_group_ids:
                    group_id = self.cache.group_hash_to_id.get(g.get_hash())
                    if group_id is None:  # just a precaution
                        group_id = self._db_get_id('grp', f'hash = "{g.get_hash()}"', conn=conn)
                        self.cache.group_hash_to_id[g.get_hash()] = group_id
                else:
                    group_id = self._db_get_id('grp', f'hash = "{g.get_hash()}"', conn=conn)

            # Persist the group's mass:
            conn.execute(
                'INSERT INTO mass_locus (iter_id, grp_id, m, m_p) VALUES (?,?,?,?)',
                [iter_id, group_id, g.m, g.m / m_pop]
            )

        return self

    def save_mass_locus__par(self, pop_m, groups, iter_id, conn):
        """Persist all new groups (and their attributes and relations) as well as masses of all groups participating
        in the designated iteration (parallelized execution).

        Note:
            This method has to be called *before* :meth:`~pram.traj.TrajectoryEnsemble.save_mass_flow()` to ensure all
            groups are already present in the database.

        Todo:
            Currently, group attributes and relations aren't added to the database.  This is to increase network
            bandwidth.  Should there be another actor responsible for collecting all group info and adding them to the
            database at the end of an ensemble execution?

        Args:
            pop_m (float): Total population mass.
            groups (Iterable[Mapping[str,Any]]): Each item is a dict with keys ``hash``, ``m``, ``attr``, and ``rel``.
            iter_id (int or str): Iteration database ID.
            conn (sqlite3.Connection): The SQLite3 connection object.

        Returns:
            ``self``
        """

        # https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field

        for g in groups:
            group_hash = g['hash']

            # New group -- persist:
            if self._db_get_one('SELECT COUNT(*) FROM grp WHERE hash = ?', [str(group_hash)], conn) == 0:
                # group_id = conn.execute('INSERT INTO grp (hash, attr, rel) VALUES (?,?,?)', [str(group_hash), None, None]).lastrowid
                group_id = conn.execute(
                    'INSERT INTO grp (hash, attr, rel) VALUES (?,?,?)',
                    [str(group_hash), DB.obj2blob(g['attr']), DB.obj2blob(g['rel'])]
                ).lastrowid

                if self.pragma.memoize_group_ids:
                    self.cache.group_hash_to_id[group_hash] = group_id

            # Extant group:
            else:
                if self.pragma.memoize_group_ids:
                    group_id = self.cache.group_hash_to_id.get(group_hash)
                    if group_id is None:  # just a precaution
                        group_id = self._db_get_id('grp', f'hash = "{group_hash}"', conn=conn)
                        self.cache.group_hash_to_id[group_hash] = group_id
                else:
                    group_id = self._db_get_id('grp', f'hash = "{group_hash}"', conn=conn)

            # Persist the group's mass:
            conn.execute(
                'INSERT INTO mass_locus (iter_id, grp_id, m, m_p) VALUES (?,?,?,?)',
                [iter_id, group_id, g['m'], g['m'] / pop_m]
            )

        return self

    # def save_state(self, traj, mass_flow_specs=None):
    #     ''' For saving both initial and regular states of simulations (i.e., ones involving mass flow). '''
    #
    #     with self.conn as c:
    #         self.curr_iter_id = self.save_iter(traj.id, traj.sim.get_iter(), None, None, c)  # remember curr_iter_id so that probe persistence can use it (yeah... nasty solution)
    #         # self.save_groups(traj, iter_id, c)
    #         self.save_mass_locus__seq(traj.sim.pop, self.curr_iter_id, c)
    #         self.save_mass_flow(self.curr_iter_id, mass_flow_specs, c)
    #
    #     return self

    # def save_state(self, traj_id, iter, pop_m, groups, mass_flow_specs=None):
    #     ''' For saving both initial and regular states of simulations (i.e., ones involving mass flow). '''
    #
    #     with self.conn as c:
    #         # self.curr_iter_id = self.save_iter(traj.id, traj.sim.get_iter(), None, None, c)  # remember curr_iter_id so that probe persistence can use it (yeah... nasty solution)
    #         # # self.save_groups(traj, iter_id, c)
    #         # self.save_mass_locus__seq(traj.sim.pop, self.curr_iter_id, c)
    #         # self.save_mass_flow(self.curr_iter_id, mass_flow_specs, c)
    #
    #         self.curr_iter_id = self.save_iter(traj_id, iter, None, None, c)  # remember curr_iter_id so that probe persistence can use it (yeah... nasty solution)
    #         # self.save_groups(traj, iter_id, c)
    #         self.save_mass_locus__par(pop_m, group, self.curr_iter_id, c)
    #         self.save_mass_flow(self.curr_iter_id, mass_flow_specs, c)
    #
    #     return self

    def save_work(self, work):
        """Persist payload delivered by a remote worker.

        Two types of payload are persisted: Simulation state or probe-recorded information.  Those payloads are
        delivered as dictionaries in the following formats::

            { 'type': 'state', 'host_name': '...', 'host_ip': '...', 'traj_id': 3, 'iter': 4, 'pop_m': 10, 'groups': [...], 'mass_flow_specs': MassFlowSpec(...) }
            { 'type': 'probe', 'qry': '...', 'vals': ['...', ...] }

        Args:
            work (Iterable[Mapping[str,Any]]): The payload.

        Returns:
            ``self``
        """

        with self.conn as c:
            for (i,p) in enumerate(self.unpersisted_probes):
                try:
                    c.execute(p['qry'], p['vals'])
                    del self.unpersisted_probes[i]
                except sqlite3.IntegrityError:
                    pass

            for w in work:
                if w['type'] == 'state':
                    host_name       = w['host_name']
                    host_ip         = w['host_ip']
                    traj_id         = w['traj_id']
                    iter            = w['iter']
                    pop_m           = w['pop_m']
                    groups          = w['groups']

                    if not w.get('mass_flow_specs') is None:
                        if isinstance(w.get('mass_flow_specs'), list):
                            mass_flow_specs = w['mass_flow_specs']
                        else:
                            mass_flow_specs = pickle.loads(w['mass_flow_specs'])
                    else:
                        mass_flow_specs = None

                    self.curr_iter_id = self.save_iter(traj_id, iter, host_name, host_ip, c)
                    self.save_mass_locus__par(pop_m, groups, self.curr_iter_id, c)
                    self.save_mass_flow(self.curr_iter_id, mass_flow_specs, c)
                elif w['type'] == 'probe':
                    try:
                        c.execute(w['qry'], w['vals'])
                    except sqlite3.IntegrityError:
                        self.unpersisted_probes.append(s)

        return self

    def set_group_name(self, ord, name, hash):
        """Set one group hash-to-name association.

        If names are present in the trajectory ensemble database, they will be used when plotting or exporting mass
        dynamics time series.

        Args:
            ord (int): Desired ordinal number of the group.
            name (str): Name.
            hash (int or str): Hash.  The best way to obtain a hash is by calling
                :meth:`Group.get_hash() <pram.entity.Group.get_hash()>`

        Returns:
            ``self``
        """

        with self.conn as c:
            id = self._db_get_id('grp_name', f'hash = "{hash}"', conn=c)
            if id is None:
                self._db_ins('INSERT INTO grp_name (ord, hash, name) VALUES (?,?,?)', [ord, str(hash), name], conn=c)
            else:
                self._db_upd('UPDATE grp_name SET ord = ? AND name = ? WHERE hash = ?', [ord, name, str(hash)], conn=c)

        return self

    def set_group_names(self, ord_name_hash):
        """Set multiple group hash-to-name associations.

        If names are present in the trajectory ensemble database, they will be used when plotting or exporting mass
        dynamics time series.

        Args:
            ord_name_hash (Iterable[tuple[int,str,int or str]]): Each item is a tuple corresponding to the arguments of
                the :meth:`~pram.traj.TrajectoryEnsemble.set_group_name>` method.

        Returns:
            ``self``
        """

        for (o,n,h) in ord_name_hash:
            self.set_group_name(o,n,h)
        return self

    def set_pragma_memoize_group_ids(self, value):
        """Set value of the *memoize_group_ids* pragma.

        Group databased IDs can be memoized (i.e., kept in memory).  That yields faster ensemble runs, especially that
        ensembles share group IDs because they are assumed to contain similar trajectories.  The downside is increased
        memory utilization.

        Args:
            value (bool): The value.

        Returns:
            ``self``
        """

        self.pragma.memoize_group_ids = value
        return self

    def show_stats(self):
        """Display ensemble statistics.

        Returns:
            ``self``
        """

        iter = [r['i_max'] for r in self.conn.execute('SELECT MAX(i.i) + 1 AS i_max FROM iter i GROUP BY traj_id', [])]

        print('Ensemble statistics')
        print(f'    Trajectories')
        print(f'        n: {len(self.traj)}')
        print(f'    Iterations')
        print(f'        mean:  {np.mean(iter)}')
        print(f'        stdev: {np.std(iter)}')

        return self


# ----------------------------------------------------------------------------------------------------------------------
@ray.remote
class WorkCollector(object):
    """Work collecting ray actor.

    This actor does not inspect or process payloads directly.  However, :meth:`~pram.traj.WorkCollector.get` returns
    the entirety of work collected and by default clears the payload storage (although this behavior can be changed).

    Currently, two types of payloads can be collected (per the actor's API): Simulation states
    (:meth:`~pram.traj.WorkCollector.save_state`) and simulation probe info
    (:meth:`~pram.traj.WorkCollector.save_probe`).

    Args:
        max_capacity (int): Maximum capacity of the collector.  Once reached, work collector will suggest workers to
            wait until the work collected has been processed by the head process.  Workers can check for go/wait
            suggestion by calling :meth:`~pram.traj.WorkCollector.do_wait`.
    """

    def __init__(self, max_capacity=0):
        self.work = []
        self.max_capacity = max_capacity

    def do_wait(self):
        """Indicate whether workers should keep doing work or wait.

        Returns:
            bool
        """

        return self.max_capacity > 0 and len(self.work) >= self.max_capacity

    def get(self, do_clear=True):
        """Retrieve all work collected so far.

        Args:
            do_clear (bool): Clear payload storage?

        Returns:
            Iterable[Any]
        """

        if do_clear:
            gc.collect()
            ret = self.work
            self.work = []
            return ret
        else:
            return self.work

    def save_probe(self, qry, vals=[]):
        """Save a simulation probe.

        Args:
            qry (str): Probe's SQL query.
            vals (Iterable[Any]): Values for the SQL query's parameters.
        """

        self.work.append({ 'type': 'probe', 'qry': qry, 'vals': vals })

    def save_state(self, state):
        """Save simulation state.

        Args:
            state (object): The state.
        """

        self.work.append(state)


# ----------------------------------------------------------------------------------------------------------------------
@ray.remote
class ProgressMonitor(object):
    """Progress monitoring ray actor.

    This actor monitors progress of workers.  Workers need to be added for the monitor to be aware of them and they
    should be removed after they're done.  This will ensure the monitor is aware of all active workers.  If that is so,
    it can provide aggregate work statistics (e.g., the total number of worker steps and the current progress towards
    that total goal).  Workers can be added and removed at any point as they are spawned or destroyed.
    """

    def __init__(self):
        self.workers = SortedDict()

    def add_worker(self, w_id, n, host_ip, host_name):
        """Add a worker.

        Args:
            w_id (int or str): Worker ID.
            n (int): Total number of work steps (i.e., simulation iterations).
            host_ip (str): Worker's host IP address.
            host_name (str): Worker's host name.
        """

        self.workers[w_id] = { 'n': n, 'i': 0, 'host_ip': host_ip, 'host_name': host_name }

    def get_i(self, w_id=None):
        """Get the number of steps completed for the designated worker or all workers.

        Args:
            w_id (int or str, optional): Worker ID.  If None, the sum of all workers' steps completed is returned.

        Returns:
            int
        """

        if w_id is not None:
            return self.workers[w_id]['i']
        else:
            return sum([w['i'] for w in self.workers.values()])

    def get_n(self, w_id=None):
        """Get the total number of steps for the designated worker or all workers.

        Args:
            w_id (int or str, optional): Worker ID.  If None, the sum of all workers' total steps is returned.

        Returns:
            int
        """

        if w_id is not None:
            return self.workers[w_id]['n']
        else:
            return sum([w['n'] for w in self.workers.values()])

    def get_rep(self):
        """Get progress report.

        Note:
            This method is no longer used; text report has been replaced by a progress bar.

        Returns:
            str
        """

        return '    '.join([f'{k:>3}: {v["i"]:>2} of {v["n"]:>2}' for (k,v) in self.workers.items()])

    def rem_all_workers(self):
        self.workers.clear()

    def rem_worker(self, w_id):
        """Remove worker.

        Args:
            w_id (int or str): Worker ID.
        """

        del self.workers[w_id]

    def upd_worker(self, w_id, i):
        """Update worker's progress.

        This method should be called by worker actors as they chew through their tasks (i.e., running simulations).

        Args:
            w_id (int or str): Worker ID.
            i (int): Number of steps completed.
        """

        self.workers[w_id]['i'] = i


# ----------------------------------------------------------------------------------------------------------------------
class Worker(object):
    """Working ray actor.

    Args:
        id (int or str): Worker ID.  This is an arbitrary value that must be hashable (because it's used as a dict
            key), and uniquely identify a worker.
        traj_id (int or str): Trajectory ensemble database ID of a trajectory associated with the worker.
        sim (Simulation): The simulation.
        work_collector (WorkCollector): Work collecting actor.
        progress_mon (ProgressMonitor): Progress monitoring actor.
    """

    def __init__(self, id, traj_id, sim, n, work_collector=None, progress_mon=None):
        self.id             = id
        self.traj_id        = traj_id
        self.sim            = sim
        self.n              = n
        self.work_collector = work_collector
        self.progress_mon   = progress_mon

        self.host_name = None  # set in self.run()
        self.host_ip   = None  # ^

    def do_wait_work(self):
        """Check if the worker should work or wait.

        Returns:
            bool
        """

        if not self.work_collector:
            return False
        return self.work_collector.do_wait.remote()

    def save_state(self, mass_flow_specs):
        """Save simulation state.

        Todo:
            Change method name to ``save_mass_flow_specs()``?

        Args:
            mass_flow_specs (MassFlowSpec): Mass flow specs.
        """

        # json = json.dumps(mass_flow_specs).encode('utf-8')  # str -> bytes

        # compressedFile = StringIO.StringIO()
        # compressedFile.write(response.read(json))

        if self.work_collector:
            # self.work_collector.submit.remote(json.dumps(mass_flow_specs).encode('utf-8'))
            self.work_collector.save_state.remote({
                'type'            : 'state',
                'host_name'       : self.host_name,
                'host_ip'         : self.host_ip,
                'traj_id'         : self.traj_id,
                'iter'            : self.sim.get_iter(),
                'pop_m'           : self.sim.pop.m,
                'groups'          : [{ 'hash': g.get_hash(), 'm': g.m } for g in self.sim.pop.groups.values()],
                'mass_flow_specs' : pickle.dumps(mass_flow_specs)
            })

    def upd_progress(self, i, n):
        """Update worker's progress towards the goal.

        Args:
            i (int): Numer of steps completed.
            n (int): Total number of steps.
        """

        if self.progress_mon:
            self.progress_mon.upd_worker.remote(self.id, i+1)

    def run(self):
        """Initialize and start the worker.

        Collect all necessary environment info (e.g., host IP address), set all :class:`~pram.sim.Simulation` object
        callbacks, and begin the simulation.  A short and random sleep time is exercised at the end of this method to
        lower the chance of simulations ending at the exact same time which is possible for highly similar models.
        """

        # (1) Initialize:
        # (1.1) Self:
        self.host_name = socket.gethostname()
        self.host_ip   = socket.gethostbyname(self.host_name)

        if self.progress_mon:
            self.progress_mon.add_worker.remote(self.id, self.n, self.host_ip, self.host_name)

        # (1.2) The simulation object:
        self.sim.set_cb_upd_progress(self.upd_progress)
        self.sim.set_cb_save_state(self.save_state)
        self.sim.set_cb_check_work(self.do_wait_work)

        # (2) Do work:
        self.sim.run(self.n)

        # (3) Finish up:
        self.sim.set_cb_save_state(None)
        self.sim.set_cb_upd_progress(None)
        self.sim.set_cb_check_work(None)

        # Normally, we'd remove the worker like below, but then the total number of workers goes down which messes up
        # the progress calculation.  Consequently, workers are removed all at once in TrajectoryEnsemble.run__par().
        # NBD either way.

        time.sleep(random.random() * 2)  # lower the chance of simulations ending at the exact same time (possible for highly similar models)


# ----------------------------------------------------------------------------------------------------------------------
@ray.remote(max_calls=1)  # ensure the worker is not reused to prevent Python and/or ray memory issues
def start_worker(w):
    """Start a worker.

    Returns:
        Worker
    """

    w.run()
    return w
