# Altair
#     Docs
#         Customization: https://altair-viz.github.io/user_guide/customization.html
#         Error band: https://altair-viz.github.io/user_guide/generated/core/altair.ErrorBandDef.html#altair.ErrorBandDef
#     Misc
#         https://github.com/altair-viz/altair/issues/968
#     Timeout in selenium
#         /Volumes/d/pitt/sci/pram/lib/python3.6/site-packages/altair/utils/headless.py

import altair as alt
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sqlite3

from dotmap              import DotMap
from pyrqa.neighbourhood import FixedRadius
from scipy.fftpack       import fft
from scipy               import signal

from .data   import ProbePersistenceDB
from .graph  import MassGraph
from .signal import Signal
from .sim    import Simulation
from .util   import DB


# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
class Trajectory(object):
    '''
    A time-ordered sequence of system configurations that occur as the system state evolves.

    Also called orbit.  Can also be thought of as a sequence of vectors in the state space (or a point in a phase
    space).

    This class delegates persistence management to the TrajectoryEnsemble class that contains it.

    This class keeps a reference to a simulation object, but that reference is only needed when running the simulation
    is desired.  When working with a historical trajectory (i.e., the trace of past simulation run), 'self.sim' can be
    None.  For example, the mass graph created by the Trajectory class is not based on an instatiated Simulation object
    even if that object has been used to generate the substrate data; instead, the database content is the graph's
    basis.
    '''

    def __init__(self, sim=None, name=None, memo=None, ensemble=None, id=None):
        self.sim  = sim
        self.name = name
        self.memo = memo
        self.id   = id
        self.ens  = ensemble  # TrajectoryEnsemble that contains this object

        if self.sim is not None:
            self.sim.traj = self

        self.mass_graph = None  # MassGraph object (instantiated when needed)

    def _check_ens(self):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

    @staticmethod
    def comp_fft(y, T, N):
        '''
        Computes Fast Fourier Transform (FFT).

        y - the signal
        T - Nyquist sampling criterion
        N - sampling rate
        '''

        f = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft = 2.0/N * np.abs(fft(y)[0:N//2])
        return (f, fft)

    def compact(self):
        self.rem_mass_graph()
        return self

    def gen_agent(self, n_iter=-1):
        self._check_ens()
        return self.ens.gen_agent(self, n_iter)

    def gen_agent_pop(self, n_agents=1, n_iter=-1):
        self._check_ens()
        return self.ens.gen_agent_pop(self, n_agents, n_iter)

    def gen_mass_graph(self):
        self._check_ens()
        if self.mass_graph is None:
            self.mass_graph = self.ens.gen_mass_graph(self)

        return self

    def get_signal(self, do_prob=False):
        self._check_ens()
        return self.ens.get_signal(self, do_prob)

    def get_time_series(self, group_hash):
        self._check_ens()
        return self.ens.get_time_series(self, group_hash)

    def load_sim(self):
        self._check_ens()
        self.ens.load_sim(self)
        return self

    def plot_heatmap(self, size, filepath):
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

    def plot_mass_flow_time_series(self, scale=(1.00, 1.00), filepath=None, iter_range=(-1, -1), v_prop=False, e_prop=False):
        self.gen_mass_graph()
        self.mass_graph.plot_mass_flow_time_series(scale, filepath, iter_range, v_prop, e_prop)
        return self

    def plot_mass_locus_fft(self, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        self._check_ens()
        plot = self.ens.plot_mass_locus_fft(self, size, filepath, iter_range, sampling_rate, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_line(self, size, filepath, iter_range=(-1, -1), stroke_w=1, col_scheme='set1', do_ret_plot=False):
        self._check_ens()
        plot = self.ens.plot_mass_locus_line(size, filepath, iter_range, self, 0, 1, stroke_w, col_scheme, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_recurrence(self, size, filepath, iter_range=(-1, -1), neighbourhood=FixedRadius(), embedding_dimension=1, time_delay=2, do_ret_plot=False):
        self._check_ens()
        plot = self.ens.plot_mass_locus_recurrence(self, size, filepath, iter_range, neighbourhood, embedding_dimension, time_delay, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_scaleogram(self, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        self._check_ens()
        plot = self.ens.plot_mass_locus_scaleogram(self, size, filepath, iter_range, sampling_rate, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_spectrogram(self, size, filepath, iter_range=(-1, -1), sampling_rate=None, win_len=None, noverlap=None, do_sort=False, do_ret_plot=False):
        self._check_ens()
        plot = self.ens.plot_mass_locus_spectrogram(self, size, filepath, iter_range, sampling_rate, win_len, noverlap, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_streamgraph(self, size, filepath, iter_range=(-1, -1), do_sort=False, do_ret_plot=False):
        self._check_ens()
        plot = self.ens.plot_mass_locus_streamgraph(self, size, filepath, iter_range, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def rem_mass_graph(self):
        self.mass_graph = None
        return self

    def run(self, iter_or_dur=1):
        if self.sim is not None:
            self.sim.set_pragma_analyze(False)
            self.sim.run(iter_or_dur)
        return self

    def save_sim(self):
        self._check_ens()
        self.ens.save_sim(self)
        return self

    def save_state(self, mass_flow_specs):
        self._check_ens()
        self.ens.save_state(self, mass_flow_specs)
        return self


# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryEnsemble(object):
    '''
    A collection of trajectories.

    All database-related logic is implemented in this class, even if it might as well belong to the Trajectory class.
    This provides an important benefit of keeping that logic from being spread all over the class hierarchy.

    --------------------------------------------------------------------------------------------------------------------

    Database design notes
        - While having a 'traj_id' field in the 'grp_name' table seems like a reasonable choice, a trajectory ensemble
          is assumed to hold only similar trajectories.  Therefore, the 'grp' and 'grp_name' tables can simply be
          joined on the 'hash' field.

    --------------------------------------------------------------------------------------------------------------------

    In mathematical physics, especially as introduced into statistical mechanics and thermodynamics by J. Willard Gibbs
    in 1902, an ensemble (also statistical ensemble) is an idealization consisting of a large number of virtual copies
    (sometimes infinitely many) of a system, considered all at once, each of which represents a possible state that the
    real system might be in. In other words, a statistical ensemble is a probability distribution for the state of the
    system.
    '''

    DDL = '''
        CREATE TABLE traj (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        ts   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        name TEXT,
        memo TEXT,
        sim  BLOB
        );

        CREATE TABLE iter (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        traj_id INTEGER,
        i       INTEGER NOT NULL,
        host    TEXT NOT NULL,
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

    def __init__(self, fpath_db=None, do_load_sims=True, flush_every=FLUSH_EVERY):
        self.traj = {}  # index by DB ID
        self.conn = None
        self.hosts = []  # hostnames of machines that will run trajectories

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
        '''
        Opens the DB connection and, if the file exists already, populates the trajectories dictionary with those from
        the DB.
        '''

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
                c.executescript(self.DDL)
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
        for t in traj:
            self.add_trajectory(t)
        return self

    def add_trajectory(self, t):
        '''
        For convenience, 't' can be either a Trajectory class instance of a Simulation class instance.  In the latter
        case, a Trajectory object will automatically be created with the default values.
        '''

        if isinstance(t, Simulation):
            t = Trajectory(t)
        elif t.name is not None and self._db_get_one('SELECT COUNT(*) FROM traj WHERE name = ?', [t.name]) > 0:
            return print(f'A trajectory with the name specified already exists: {t.name}')

        with self.conn as c:
            t.id = c.execute('INSERT INTO traj (name, memo) VALUES (?,?)', [t.name, t.memo]).lastrowid
            # for (i,r) in enumerate(t.sim.rules):
            #     c.execute('INSERT INTO rule (traj_id, ord, name, src) VALUES (?,?,?,?)', [t.id, i, r.__class__.__name__, inspect.getsource(r.__class__)])

            for p in t.sim.probes:
                p.set_persistence(self.probe_persistence)

        t.ens = self
        self.traj[t.id] = t

        return self

    def clear_group_names(self):
        with self.conn as c:
            c.execute('DELETE FROM grp_name', [])
        return self

    def compact(self):
        for t in self.traj:
            t.compact()
        return self

    def gen_agent(self, traj, n_iter=-1):
        '''
        Generate a single agent's group transition path based on population-level mass dynamics that a PRAM simulation
        operates on.

        This is a two-step process:

        (1)  Pick the agent's initial group taking into account the initial mass distribution among the groups
        (2+) Pick the next group taking into account transition probabilities to all possible next groups

        Because step 1 always takes place, the resulting list of agent's states will be of size 'n_iter + 1'.
        '''

        agent = { 'attr': {}, 'rel': {} }
        with self.conn as c:
            if n_iter <= -1:
                n_iter = self._db_get_one('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
            else:
                n_iter = max(0, min(n_iter, self._db_get_one('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])))

            for i in range(-1, n_iter):
                if i == -1:  # (1) setting the initial group
                    groups = list(zip(*[[r[0], round(r[1],2)] for r in c.execute('SELECT g.id, g.m_p FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ?', [traj.id, -1])]))
                else:  # (2) generating a sequence of group transitions
                    groups = list(zip(*[[r[0], round(r[1],2)] for r in c.execute('SELECT g_dst.id, mf.m_p FROM mass_flow mf INNER JOIN iter i ON i.id = mf.iter_id INNER JOIN grp g_src ON g_src.id = mf.grp_src_id INNER JOIN grp g_dst ON g_dst.id = mf.grp_dst_id WHERE i.traj_id = ? AND i.i = ? AND g_src.id = ?', [traj.id, i, grp_id])]))

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
        return [self.gen_agent(traj, n_iter) for _ in range(n_agents)]

    def gen_mass_graph(self, traj):
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
        '''
        Returns time series of masses of all groups.  Proportions of the total mass can be requested as well.
        '''

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
            for iter in self.conn.execute(f'''
                    SELECT i.i + 1 AS i, g.{y} AS y FROM grp g
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
        return self.conn.execute('''
            SELECT g.m, g.m_p, i.i FROM grp g
            INNER JOIN iter i ON i.id = g.iter_id
            INNER JOIN traj t ON t.id = i.traj_id
            WHERE t.id = ? AND g.hash = ?
            ORDER BY i.i
            ''', [traj.id, group_hash]).fetchall()

    def load_sim(self, traj):
        traj.sim = DB.blob2obj(self.conn.execute('SELECT sim FROM traj WHERE id = ?', [traj.id]).fetchone()[0])
        if traj.sim:
            traj.sim.traj = traj  # restore the link severed at save time
        return self

    def load_sims(self):
        gc.disable()
        for t in self.traj.values():
            self.load_sim(t)
        gc.enable()
        return self

    def normalize_iter_range(self, range=(-1, -1), qry='SELECT MAX(i) FROM iter', qry_args=[]):
        '''
        The SQL query should return the maximum number of iterations, as desired.  That is overall within the ensemble
        or for a given trajectory or a set thereof.
        '''

        l = max(range[0], -1)

        n_iter = self._db_get_one(qry, qry_args)
        if range[1] <= -1:
            u = n_iter
        else:
            u = min(range[1], n_iter)

        if l > u:
            raise ValueError('Iteration range error: Lower bound cannot be larger than upper bound')

        return (l,u)

    def plot_mass_locus_bubble(self, size, filepath, iter_range=(-1, -1), do_sort=False, do_ret_plot=False):
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

    def plot_mass_locus_fft(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        '''
        Fast Fourier Transform (FFT)

        The Fourier Transform will work very well when the frequency spectrum is stationary.
        '''

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
        alt.Chart(alt.Data(values=[x for xs in data['fd'].values() for x in xs])).properties(
            title=title, width=size[0], height=size[1]
        ).mark_line(
            strokeWidth=1, opacity=0.75, interpolate='basis', tension=1
        ).encode(
            alt.X('x:Q', axis=alt.Axis(title='Frequency', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, sampling_rate // 2))),
            alt.Y('y:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
            alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))
            # alt.Order('year(data):O')
        ).configure_title(
            fontSize=20
        ).configure_view(
        ).save(
            filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER
        )

        return plot if do_ret_plot else self

    def plot_mass_locus_scaleogram(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        '''
        Currently, Image Mark in not supported in Vega-Lite.  Consequently, raster images cannot be displayed via
        Altair.  The relevant issue is: https://github.com/vega/vega-lite/issues/3758
        '''

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

    def plot_mass_locus_spectrogram(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=None, win_len=None, noverlap=None, do_sort=False, do_ret_plot=False):
        '''
        Docs
            https://kite.com/python/docs/matplotlib.mlab.specgram
        Examples
            https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/specgram_demo.html#sphx-glr-gallery-images-contours-and-fields-specgram-demo-py
            https://stackoverflow.com/questions/35932145/plotting-with-matplotlib-specgram
            https://pythontic.com/visualization/signals/spectrogram
            http://www.toolsmiths.com/wavelet/wavbox
        TODO
            http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

            https://www.sciencedirect.com/topics/neuroscience/signal-processing
            https://www.google.com/search?client=firefox-b-1-d&q=Signal+Processing+for+Neuroscientists+pdf

            https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.cwt.html
            https://www.semanticscholar.org/paper/A-wavelet-based-tool-for-studying-non-periodicity-Ben%C3%ADtez-Bol%C3%B3s/b7cb0789bd2d29222f2def7b70095f95eb72358c
            https://www.google.com/search?q=time-frequency+plane+decomposition&client=firefox-b-1-d&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiDxu3R9bXkAhWXqp4KHUSuBqYQ_AUIEigB&biw=1374&bih=829#imgrc=q2MCGaBIY3lrSM:
            https://www.mathworks.com/help/wavelet/examples/classify-time-series-using-wavelet-analysis-and-deep-learning.html;jsessionid=de786cc8324218efefc12d75c292
        '''

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

    def plot_mass_locus_line(self, size, filepath, iter_range=(-1, -1), traj=None, nsamples=0, opacity_min=0.1, stroke_w=1, col_scheme='set1', do_ret_plot=False):
        '''
        If 'traj' is not None, only that trajectory is plotted.  Otherwise, 'nsample' determines the number of
        trajectories plotted.  Specifically, if smaller than or equal to zero, all trajectories are plotted; otherwise,
        the given number of trajectories is selected randomly.  If the number provided exceeds the number of
        trajectories present in the ensamble, all of them are plotted.
        '''

        # (1) Sample trajectories (if necessary) + set title + set line alpha:
        if traj is not None:
            traj_sample = [traj]
            title = f'Trajectory Mass Locus'  #  (Iterations {iter_range[0]+1} to {iter_range[1]+1})
            opacity = 1.00
        else:
            traj_sample = []
            if nsamples <=0 or nsamples >= len(self.traj):
                traj_sample = self.traj.values()
                title = f'Trajectory Ensemble Mass Locus (n={len(self.traj)})'
            else:
                traj_sample = random.sample(list(self.traj.values()), nsamples)
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

        plot = alt.layer(
            *plots
        ).properties(
            title=title, width=size[0], height=size[1]
        ).configure_view(
            # strokeWidth=1
        ).configure_title(
            fontSize=20
        ).resolve_scale(
            color='independent'
        ).save(
            filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER
        )

        return plot if do_ret_plot else self

    def plot_mass_locus_line_aggr(self, size, filepath, iter_range=(-1, -1), band_type='ci', stroke_w=1, col_scheme='set1', do_ret_plot=False):
        '''
        Ordering the legend of a composite chart
            https://stackoverflow.com/questions/55783286/control-legend-color-and-order-when-joining-two-charts-in-altair
            https://github.com/altair-viz/altair/issues/820
        '''

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

        plot = alt.layer(
            plot_band, plot_line, data=alt.Data(values=data)
        ).properties(
            title=title, width=size[0], height=size[1]
        ).configure_view(
        ).configure_title(
            fontSize=20
        ).resolve_scale(
            color='independent'
        ).save(
            filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER
        )

        return plot if do_ret_plot else self

    def plot_mass_locus_polar(self, size, filepath, iter_range=(-1, -1), nsamples=0, n_iter_per_rot=0, do_sort=False, do_ret_plot=False):
        '''
        Altair does not currently support projections, so we must revert to good old matplotlib.
        '''

        # (1) Sample trajectories (if necessary) + set parameters and plot title:
        traj_sample = []
        if nsamples <=0 or nsamples >= len(self.traj):
            traj_sample = self.traj.values()
            title = f'Trajectory Ensemble Mass Locus (n={len(self.traj)}; '
        else:
            traj_sample = random.sample(list(self.traj.values()), nsamples)
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
        '''
        https://en.wikipedia.org/wiki/Recurrence_plot

        TODO: Multivariate extensions of recurrence plots include cross recurrence plots and joint recurrence plots.
        '''

        from pyrqa.time_series     import TimeSeries
        from pyrqa.settings        import Settings
        from pyrqa.computing_type  import ComputingType
        # from pyrqa.neighbourhood   import FixedRadius
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

        computation = RQAComputation.create(settings, verbose=True)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        result.min_white_vertical_line_lelngth = 2
        print(result)

        computation = RPComputation.create(settings)
        result = computation.run()
        ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse, filepath)

        return result if do_ret_plot else self

    def plot_mass_locus_streamgraph(self, traj, size, filepath, iter_range=(-1, -1), do_sort=False, do_ret_plot=False):
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
            if do_sort:
                sort = [r['name'] for r in c.execute('SELECT name FROM grp_name ORDER BY ord')]
                # sort = [r['name'] for r in c.execute('SELECT COALESCE(gn.name, g.hash) AS name FROM grp g LEFT JOIN grp_name gn ON gn.hash = g.hash ORDER BY gn.ord, g.id')]
                plot_color = alt.Color('grp:N', scale=alt.Scale(scheme='category20b'), legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15), sort=sort)
            else:
                plot_color = alt.Color('grp:N', scale=alt.Scale(scheme='category20b'), legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))

        # (2) Plot:
        alt.Chart(alt.Data(values=data)).properties(
            title='Trajectory Mass Locus', width=size[0], height=size[1]
        ).mark_area().encode(
            alt.X('i:Q', axis=alt.Axis(domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
            alt.Y('sum(m):Q', axis=alt.Axis(domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), stack='center', scale=alt.Scale(domain=(0, m_max))),
            plot_color
            # alt.Order('year(data):O')
        ).configure_view(
            strokeWidth=0
        ).save(
            filepath, scale_factor=2.0, webdriver=self.__class__.WEBDRIVER
        )

        return plot if do_ret_plot else self

    def plot_matplotlib(self, size, filepath, iter_range=(-1, -1), do_sort=False):
        ''' TODO: Remove (a more general version has been implemented). '''

        import matplotlib.pyplot as plt
        from .entity  import Group

        # Plot:
        cmap = plt.get_cmap('tab20')
        fig = plt.figure(figsize=size)
        # plt.legend(['Susceptible', 'Infectious', 'Recovered'], loc='upper right')
        plt.xlabel('Iteration')
        plt.ylabel('Mass')
        plt.grid(alpha=0.25, antialiased=True)

        # Series:
        with self.conn as c:
            for t in self.traj.values():
                data = []

                iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [t.id])

                data_iter = []
                data_s    = []
                data_i    = []
                data_r    = []

                hash_s = Group.gen_hash(attr={ 'flu': 's' })
                hash_i = Group.gen_hash(attr={ 'flu': 'i' })
                hash_r = Group.gen_hash(attr={ 'flu': 'r' })

                for i in range(iter_range[0], iter_range[1]):
                    m_s = self._db_get_one('SELECT g.m FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ? AND g.hash = ?', [t.id, i, hash_s])
                    m_i = self._db_get_one('SELECT g.m FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ? AND g.hash = ?', [t.id, i, hash_i])
                    m_r = self._db_get_one('SELECT g.m FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ? AND g.hash = ?', [t.id, i, hash_r])

                    data_iter.append(i + 1)

                    data_s.append(m_s)
                    data_i.append(m_i)
                    data_r.append(m_r)

                plt.plot(data_iter, data_s, lw=1, linestyle='--', color=cmap(0), alpha=0.1, mfc='none', antialiased=True)
                plt.plot(data_iter, data_i, lw=1, linestyle='-',  color=cmap(4), alpha=0.1, mfc='none', antialiased=True)
                plt.plot(data_iter, data_r, lw=1, linestyle=':',  color=cmap(6), alpha=0.1, mfc='none', antialiased=True)

        # Save:
        fig.savefig(filepath, dpi=300)

        return self

    def run(self, iter_or_dur=1, do_memoize_group_ids=False):
        '''
        Parallelization
            https://docs.python.org/2/library/multiprocessing.html
            https://pyfora.readthedocs.io
        '''

        if iter_or_dur < 1:
            return

        for i,t in enumerate(self.traj.values()):
            print(f'Running trajectory {i+1} of {len(self.traj)} (iter count: {iter_or_dur}): {t.name or "unnamed simulation"}')
            t.run(iter_or_dur)

        self.save_sims()
        self.is_db_empty = False
        return self

    def save_sim(self, traj):
        '''
        To pickle the Simulation object, we need to temporarily disconnect it from its Trajectory object container.
        This is because the Trajectory object is connected to the TrajectoryEnsemble object which holds a database
        connection object and those objects cannot be pickled.  Besides, there is no point in saving that anyway.
        '''

        traj.sim.traj = None  # sever the link to avoid "pickling SQL connection" error (the link is restored at load time)
        # import dill
        # print(dill.detect.baditems(traj.sim))
        # print(dir(traj.sim))
        # print(traj.sim.db)
        # self._db_upd('UPDATE traj SET sim = ? WHERE id = ?', [DB.obj2blob(traj.sim), traj.id])
            # TODO: Uncomment the above line and fix (if at all possible) in Python 37 (works in 36...)
        traj.sim.traj = traj  # restore the link

        return self

    def save_sims(self):
        for t in self.traj.values():
            self.save_sim(t)
        return self

    def save_state(self, traj, mass_flow_specs):
        ''' For saving both initial and regular states of simulations (i.e., ones involving mass flow). '''

        with self.conn as c:
            self.curr_iter_id = self.save_iter(traj, c)  # remember so that probe persistence can use it (yeah... nasty solution)
            # self.save_groups(traj, iter_id, c)
            self.save_mass_locus(traj, self.curr_iter_id, c)
            self.save_mass_flow(traj, self.curr_iter_id, mass_flow_specs, c)

        return self

    def save_iter(self, traj, conn):
        if traj.sim.timer.is_running:
            iter = traj.sim.timer.i  # regular iteration
        else:
            iter = -1                # initial condition

        return self._db_ins('INSERT INTO iter (traj_id, i, host) VALUES (?,?,?)', [traj.id, iter, 'localhost'], conn)

    def save_groups(self, traj, iter_id, conn):
        '''
        Inserts all groups for the given iteration and trajectory.  This captures the current simulation state (at
        least to the degree that we care about for the time being).

        https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field
        '''

        m_pop = traj.sim.pop.get_mass()  # to get proportion of mass flow
        for g in traj.sim.pop.groups.values():
            for s in g.rel.values():  # sever the 'pop.sim.traj.traj_ens._conn' link (or pickle error)
                s.pop = None

            conn.execute(
                'INSERT INTO grp (iter_id, hash, m, m_p, attr, rel) VALUES (?,?,?,?,?,?)',
                [iter_id, g.get_hash(), g.m, g.m / m_pop, DB.obj2blob(g.attr), DB.obj2blob(g.rel)]
            )

            for s in g.rel.values():  # restore the link
                s.pop = traj.sim.pop

        return self

    def save_mass_flow(self, traj, iter_id, mass_flow_specs, conn):
        '''
        Inserts the mass flow among all groups for the given iteration and trajectory.  Mass flow is present for all
        but the initial state of a simulation.
        '''

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

    def save_mass_locus(self, traj, iter_id, conn):
        '''
        Persists all new groups and masses of all groups participating in the current iteration and trajectory.  The
        attributes and relations of all groups are saved in the database which enables restoring the state of the
        simulation at any point in time.

        https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field
        '''

        m_pop = traj.sim.pop.get_mass()  # to get proportion of mass flow
        for g in traj.sim.pop.groups.values():
            # Persist the group if new:
            if self._db_get_one('SELECT COUNT(*) FROM grp WHERE hash = ?', [g.get_hash()], conn) == 0:
                for s in g.rel.values():  # sever the 'pop.sim.traj.traj_ens._conn' link (or pickle error)
                    s.pop = None

                group_id = conn.execute(
                    'INSERT INTO grp (hash, attr, rel) VALUES (?,?,?)',
                    [g.get_hash(), DB.obj2blob(g.attr), DB.obj2blob(g.rel)]
                ).lastrowid

                if self.pragma.memoize_group_ids:
                    self.cache.group_hash_to_id[g.get_hash()] = group_id

                for s in g.rel.values():  # restore the link
                    s.pop = traj.sim.pop
            else:
                if self.pragma.memoize_group_ids:
                    group_id = self.cache.group_hash_to_id.get(g.get_hash())
                    if group_id is None:  # precaution
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

    def set_group_name(self, ord, name, hash):
        with self.conn as c:
            id = self._db_get_id('grp_name', f'hash = "{hash}"', conn=c)
            if id is None:
                self._db_ins('INSERT INTO grp_name (ord, hash, name) VALUES (?,?,?)', [ord, hash, name], conn=c)
            else:
                self._db_upd('UPDATE grp_name SET ord = ? AND name = ? WHERE hash = ?', [ord, name, hash], conn=c)

        return self

    def set_group_names(self, ord_name_hash):
        for (o,n,h) in ord_name_hash:
            self.set_group_name(o,n,h)
        return self

    def set_hosts(self, hosts):
        self.hosts = hosts
        return self

    def set_pragma_memoize_group_ids(self, value):
        self.pragma.memoize_group_ids = value
        return self

    def stats(self):
        iter = [r['i_max'] for r in self.conn.execute('SELECT MAX(i.i) + 1 AS i_max FROM iter i GROUP BY traj_id', [])]

        print('Ensemble statistics')
        print(f'    Trajectories')
        print(f'        n: {len(self.traj)}')
        print(f'    Iterations')
        print(f'        mean:  {np.mean(iter)}')
        print(f'        stdev: {np.std(iter)}')

        return self
