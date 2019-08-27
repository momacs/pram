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
import numpy as np
import os
import random
import sqlite3

from .graph import MassGraph
from .util  import DB


# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
class Trajectory(object):
    '''
    A time-ordered sequence of system configurations that occur as the system state evolves.

    Also called orbit.  Can also be thought of as a sequence of vectors in the state space (or a point in a phase
    space).

    This class delegates persistance management to the TrajectoryEnsemble class that contains it.

    This class keeps a reference to a simulation object, but that reference is only needed when running the simulation
    is desired.  When working with a historical trajectory (i.e., the trace of past simulation run), 'self.sim' can be
    None.  For example, the mass graph created by the Trajectory class is not based on an instatiated Simulation object
    even if that object has been used to generate the substrate data; instead, the database content is the graph's
    basis.
    '''

    def __init__(self, name=None, memo=None, sim=None, ensemble=None, id=None):
        self.name = name
        self.memo = memo
        self.sim  = sim
        self.id   = id
        self.ens  = ensemble  # TrajectoryEnsemble that contains this object

        if self.sim is not None:
            self.sim.traj = self

        self.mass_graph = None  # MassGraph object (instantiated when needed)

    def compact(self):
        self.rem_mass_graph()
        return self

    def gen_mass_graph(self):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        if self.mass_graph is None:
            self.mass_graph = self.ens.gen_mass_graph(self)

        return self

    def load_sim(self):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        self.ens.load_sim(self)
        return self

    def plot_heatmap(self, size, filepath):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

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
        c.save(filepath, webdriver='firefox')

    def plot_mass_flow_time_series(self, scale=(1.00, 1.00), filepath=None, iter_range=(-1, -1), v_prop=False, e_prop=False):
        self.gen_mass_graph()
        self.mass_graph.plot_mass_flow_time_series(scale, filepath, iter_range, v_prop, e_prop)
        return self

    def plot_mass_locus_freq(self, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        plot = self.ens.plot_mass_locus_freq(self, size, filepath, iter_range, sampling_rate, do_sort, do_ret_plot)
        return plot if do_ret_plot else self

    def plot_mass_locus_streamgraph(self, size, filepath, iter_range=(-1, -1), do_sort=False, do_ret_plot=False):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

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
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        self.ens.save_sim(self)
        return self

    def save_state(self, mass_flow_specs):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        self.ens.save_state(self, mass_flow_specs)
        return self


# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryEnsemble(object):
    '''
    A collection of trajectories.

    All database-related logic is implemented in this class, even if it might as well belong to the Trajectory class.
    This provides an important benefit of keeping that logic from being spread all over the class hierarchy.

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

        CREATE TABLE grp (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        iter_id INTEGER,
        hash    TEXT NOT NULL,
        m       REAL NOT NULL,
        m_p     REAL NOT NULL,
        attr    BLOB,
        rel     BLOB,
        UNIQUE (iter_id, hash),
        CONSTRAINT fk__grp__iter FOREIGN KEY (iter_id) REFERENCES iter (id) ON UPDATE CASCADE ON DELETE CASCADE
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

        CREATE TABLE grp_name (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        ord  INTEGER,
        hash TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL
        );
        '''

        # CREATE TABLE rule (
        # id      INTEGER PRIMARY KEY AUTOINCREMENT,
        # traj_id INTEGER,
        # ord     INTEGER NOT NULL,
        # name    TEXT NOT NULL,
        # src     TEXT NOT NULL,
        # UNIQUE (traj_id, ord),
        # CONSTRAINT fk__rule__traj FOREIGN KEY (traj_id) REFERENCES traj (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );

    def __init__(self, fpath_db=None, do_load_sims=True):
        self.traj = {}  # index by DB ID
        self.conn = None
        self.hosts = []  # hostnames of machines that will run trajectories

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
            print('New database initialized')

        # Database exists:
        else:
            with self.conn as c:
                for r in c.execute('SELECT id, name, memo FROM traj', []):
                    self.traj[r['id']] = Trajectory(r['name'], r['memo'], ensemble=self, id=r['id'])

            if do_load_sims:
                self.load_sims()

            n_traj = self._db_get_one('SELECT COUNT(*) FROM traj', [])
            print(f'Using existing database (trajectories loaded: {n_traj})')

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
        if self._db_get_one('SELECT COUNT(*) FROM traj WHERE name = ?', [t.name]) > 0:
            return print(f'A trajectory with the name specified already exists: {t.name}')

        with self.conn as c:
            t.id = c.execute('INSERT INTO traj (name, memo) VALUES (?,?)', [t.name, t.memo]).lastrowid
            # for (i,r) in enumerate(t.sim.rules):
            #     c.execute('INSERT INTO rule (traj_id, ord, name, src) VALUES (?,?,?,?)', [t.id, i, r.__class__.__name__, inspect.getsource(r.__class__)])

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

    def gen_mass_graph(self, traj):
        g = MassGraph()
        n_iter = self._db_get_one('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
        with self.conn as c:
            # Groups (vertices):
            for i in range(-1, n_iter + 1):
                for r in c.execute('SELECT g.hash, g.m, g.m_p FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ? ORDER BY g.id', [traj.id, i]):
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

    def load_sim(self, traj):
        traj.sim = DB.blob2obj(self.conn.execute('SELECT sim FROM traj WHERE id = ?', [traj.id]).fetchone()[0])
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
        plot.save(filepath, scale_factor=2.0, webdriver='firefox')

        return plot if do_ret_plot else self

    def plot_mass_locus_freq(self, traj, size, filepath, iter_range=(-1, -1), sampling_rate=1, do_sort=False, do_ret_plot=False):
        from scipy.fftpack import fft

        # (1) Data:
        data = { 'td': {}, 'fd': {} }  # time- and frequency-domain
        with self.conn as c:
            # (1.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
            n_iter = iter_range[1] - min(iter_range[0], 0)
            title = f'Trajectory Ensemble Mass Locus Oscillations (Sampling Rate of {sampling_rate} on Iterations {iter_range[0]+1} to {iter_range[1]+1})'

            # (1.2) Construct time-domain data bundle:
            for r in c.execute('''
                    SELECT COALESCE(gn.name, g.hash) AS grp, g.m
                    FROM grp g
                    INNER JOIN iter i ON i.id = g.iter_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    ORDER BY gn.ord, g.id''',
                    [traj.id, iter_range[0], iter_range[1]]):
                if data['td'].get(r['grp']) is None:
                    data['td'][r['grp']] = []
                data['td'][r['grp']].append(r['m'])

            # (1.3) Move to frequency domain:
            t = 1 / sampling_rate  # Nyquist sampling criteria
            x = np.linspace(0.0, 1.0 / (2.0 * t), int(sampling_rate / 2))  # ^

            # import matplotlib.pyplot as plt
            for g in data['td'].keys():
                y = fft(data['td'][g])                                          # positive and negative frequencies
                y = 2 / sampling_rate * np.abs(y[0:np.int(sampling_rate / 2)])  # positive frequencies only
                data['fd'][g] = [{ 'grp': g, 'x': z[0], 'y': z[1] / n_iter } for z in zip(x,y)]

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
            alt.X('x:Q', axis=alt.Axis(title='Frequency', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
            alt.Y('y:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
            alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))
            # alt.Order('year(data):O')
        ).configure_title(
            fontSize=20
        ).configure_view(
        ).save(
            filepath, scale_factor=2.0, webdriver='firefox'
        )

        return plot if do_ret_plot else self

    def plot_mass_locus_line(self, size, filepath, iter_range=(-1, -1), nsamples=0, do_sort=False, do_ret_plot=False):
        # (1) Sample trajectories (if necessary) + set title:
        traj_sample = []
        if nsamples <=0 or nsamples >= len(self.traj):
            traj_sample = self.traj.values()
            title = f'Trajectory Ensemble Mass Locus (n={len(self.traj)})'
        else:
            traj_sample = random.sample(list(self.traj.values()), nsamples)
            title = f'Trajectory Ensemble Mass Locus (Random Sample of {len(traj_sample)} from {len(self.traj)})'

        # (2) Plot trajectories:
        plots = []
        for t in traj_sample:
            # (2.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [t.id])

            # (2.2) Construct the trajectory data bundle:
            data = []
            with self.conn as c:
                for r in c.execute('SELECT i.i, g.m, g.hash FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i BETWEEN ? AND ? ORDER BY i.i', [t.id, iter_range[0], iter_range[1]]):
                    data.append({ 'i': r['i'] + 1, 'm': r['m'], 'grp': r['hash'] })

            # (2.3) Plot the trajectory:
            plots.append(
                alt.Chart(
                    alt.Data(values=data)
                ).properties(
                    title=title, width=size[0], height=size[1]
                ).mark_line(
                    strokeWidth=1, opacity=0.25, interpolate='basis', tension=1  # basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
                ).encode(
                    alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                    alt.Y('m:Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                    alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))  # scale=alt.Scale(scheme='category20b')
                    # alt.Order('year(data):O')
                )
            )

        plot = alt.layer(*plots)
        plot.configure_view(
            # strokeWidth=1
        ).configure_title(
            fontSize=20
        ).save(
            filepath, scale_factor=2.0, webdriver='firefox'
        )

        return plot if do_ret_plot else self

    def plot_mass_locus_line_aggr(self, size, filepath, iter_range=(-1, -1), band_type='iqr', do_sort=False, do_ret_plot=False):
        title = f'Trajectory Ensemble Mass Locus (Mean + {band_type.upper()}; n={len(self.traj)})'

        # (1) Normalize iteration bounds:
        iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter', [])

        # (2) Plot:
        # (2.1) Construct data bundle:
        data = []
        for i in range(iter_range[0], iter_range[1] + 1, 1):
            with self.conn as c:
                for r in c.execute('SELECT g.m, g.hash FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.i = ?', [i]):
                    data.append({ 'i': i + 1, 'm': r['m'], 'grp': r['hash'] })

        # (2.2) Plot iterations:
        plot_line = alt.Chart(
                alt.Data(values=data)
            ).properties(
                title=title, width=size[0], height=size[1]
            ).mark_line(
                strokeWidth=1, interpolate='basis'#, tension=1  # basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
            ).encode(
                alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                alt.Y('mean(m):Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))
            )

        plot_band = alt.Chart(  # https://altair-viz.github.io/user_guide/generated/core/altair.ErrorBandDef.html#altair.ErrorBandDef
                alt.Data(values=data)
            ).properties(
                title=title, width=size[0], height=size[1]
            ).mark_errorband(
                extent=band_type, interpolate='basis'#, tension=1  # opacity, basis, basis-closed, cardinal, cardinal-closed, bundle(tension)
            ).encode(
                alt.X('i:Q', axis=alt.Axis(title='Iteration', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=(0, iter_range[1]))),
                alt.Y('mean(m):Q', axis=alt.Axis(title='Mass', domain=False, tickSize=0, grid=False, labelFontSize=15, titleFontSize=15)),
                alt.Color('grp:N', legend=alt.Legend(title='Group', labelFontSize=15, titleFontSize=15))
            )

        plot = plot_band + plot_line
        plot.configure_view(
        ).configure_title(
            fontSize=20
        ).save(
            filepath, scale_factor=2.0, webdriver='firefox'
        )

        return plot if do_ret_plot else self

    def plot_mass_locus_streamgraph(self, traj, size, filepath, iter_range=(-1, -1), do_sort=False, do_ret_plot=False):
        # (1) Data:
        data = []
        with self.conn as c:
            # (1.1) Normalize iteration bounds:
            iter_range = self.normalize_iter_range(iter_range, 'SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])

            # (1.2) Determine max mass sum:
            m_max = self._db_get_one('''
                SELECT ROUND(MAX(m_sum),4) FROM (
                SELECT SUM(m) AS m_sum FROM grp g INNER JOIN iter i on i.id = g.iter_id WHERE i.traj_id = ? GROUP BY g.iter_id
                )''',
                [traj.id]
            )  # without rounding, weird-ass max values can appear due to inexact floating-point arithmetic (four decimals is arbitrary)

            # (1.3) Construct the data bundle:
            for r in c.execute('''
                    SELECT COALESCE(gn.name, g.hash) AS grp, g.m, i.i
                    FROM grp g
                    INNER JOIN iter i ON i.id = g.iter_id
                    LEFT JOIN grp_name gn ON gn.hash = g.hash
                    WHERE i.traj_id = ? AND i.i BETWEEN ? AND ?
                    ORDER BY i.i, gn.ord, g.id''',
                    [traj.id, iter_range[0], iter_range[1]]):
                data.append({ 'grp': r['grp'], 'i': r['i'] + 1, 'm': r['m'] })

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
            filepath, scale_factor=2.0, webdriver='firefox'
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

    def run(self, iter_or_dur=1):
        for i,t in enumerate(self.traj.values()):
            print(f'Running trajectory {i+1} of {len(self.traj)} (iter count: {iter_or_dur}): {t.name or "unnamed simulation"}')
            t.run(iter_or_dur)

        self.save_sims()
        return self

    def save_sim(self, traj):
        '''
        To pickle the Simulation object, we need to temporarily disconnect it from its Trajectory object container.
        This is because the Trajectory object is connected to the TrajectoryEnsemble object which holds a database
        connection object and those objects cannot be pickled.  Besides, there is no point in saving the entire class
        hierarchy anyway.
        '''

        s = traj.sim
        s.traj = None  # sever the link to avoid "pickling SQL connection" error (the link is restored at load time)
        self._db_upd('UPDATE traj SET sim = ? WHERE id = ?', [DB.obj2blob(s), traj.id])
        s.traj = traj

        return self

    def save_sims(self):
        for t in self.traj.values():
            self.save_sim(t)
        return self

    def save_state(self, traj, mass_flow_specs):
        ''' For saving both initial and regular states of simulations (i.e., ones involving mass flow). '''

        with self.conn as c:
            iter_id = self.save_iter(traj, c)
            self.save_groups(traj, iter_id, c)
            self.save_mass_flow(traj, iter_id, mass_flow_specs, c)

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
        Inserts the mass flow among all groups for the given iteration and trajectory.  No mass flow is present for the
        initial state of a simulation.
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

    def stats(self):
        iter = [r['i_max'] for r in self.conn.execute('SELECT MAX(i.i) + 1 AS i_max FROM iter i GROUP BY traj_id', [])]

        print('Ensemble statistics')
        print(f'    Trajectories')
        print(f'        n: {len(self.traj)}')
        print(f'    Iterations')
        print(f'        mean:  {np.mean(iter)}')
        print(f'        stdev: {np.std(iter)}')

        return self
