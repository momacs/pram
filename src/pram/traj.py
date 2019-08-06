import altair as alt
import graph_tool.all as gt
import os
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

    def __init__(self, name, memo=None, sim=None, ensemble=None, id=None):
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

    def gen_mass_graph(self):
        if self.ens is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        if self.mass_graph is None:
            self.mass_graph = self.ens.gen_mass_graph(self)
        return self

    def load_sim(self):
        if self.mass_graph is None:
            raise TrajectoryError('A trajectory can only perform this action if it is a part of an ensemble.')

        self.ens.load_sim(self)
        return self

    def plot_heatmap(self, size, filepath):
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

    def plot_states(self, size, filepath):
        self.gen_mass_graph()
        self.mass_graph.plot_states(size, filepath)
        return self

    def plot_streamgraph_group(self, size, filepath):
        data = []

        # for v in self.g.vertices():
        #     data.append({ "group": self.vp.hash[v], "iter": i + 1, "mass": self.vp.mass[v] })

        for i in range(-1, self.n_iter):
            for v in gt.find_vertex(self.g, self.vp.iter, i):
                data.append({ "group": self.vp.hash[v], "iter": i + 1, "mass": self.vp.mass[v] })

        c = alt.Chart(alt.Data(values=data), width=size[0], height=size[1]).mark_area().encode(
            alt.X('iter:Q', axis=alt.Axis(domain=False, tickSize=0), scale=alt.Scale(domain=(0, self.n_iter))),
            alt.Y('sum(mass):Q', stack='center', scale=alt.Scale(domain=(0, self.mass_max))),
            alt.Color('group:N', scale=alt.Scale(scheme='category20b'))
        )
        c.configure_axis(grid=False)
        c.configure_view(strokeWidth=0)
        c.save(filepath, scale_factor=2.0, webdriver='firefox')

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
        name TEXT NOT NULL UNIQUE,
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
        UNIQUE (iter_id, grp_src_id, grp_dst_id),
        CONSTRAINT fk__mass_flow__iter    FOREIGN KEY (iter_id)    REFERENCES iter (id) ON UPDATE CASCADE ON DELETE CASCADE,
        CONSTRAINT fk__mass_flow__grp_src FOREIGN KEY (grp_src_id) REFERENCES grp  (id) ON UPDATE CASCADE ON DELETE CASCADE,
        CONSTRAINT fk__mass_flow__grp_dst FOREIGN KEY (grp_dst_id) REFERENCES grp  (id) ON UPDATE CASCADE ON DELETE CASCADE
        );
        '''

        # CREATE TABLE rule (
        # id      INTEGER PRIMARY KEY AUTOINCREMENT,
        # traj_id INTEGER,
        # name    TEXT NOT NULL,
        # src     TEXT NOT NULL,
        # CONSTRAINT fk__rule__traj FOREIGN KEY (traj_id) REFERENCES traj (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );

    def __init__(self, fpath_db, do_load_sims=True):
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

    def _db_conn_open(self, fpath_db, do_load_sims=True):
        '''
        Opens the DB connection and, if the file exists already, populates the trajectories dictionary with those from
        the DB.
        '''

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

            n_traj = self._db_get_int('SELECT COUNT(*) FROM traj', [])
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

    def _db_get_int(self, qry, args, conn=None):
        c = conn or self.conn
        return c.execute(qry, args).fetchone()[0]

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
        if self._db_get_int('SELECT COUNT(*) FROM traj WHERE name = ?', [t.name]) > 0:
            return print(f'A trajectory with the name specified already exists: {t.name}')

        with self.conn as c:
            t.id = c.execute('INSERT INTO traj (name, memo) VALUES (?,?)', [t.name, t.memo]).lastrowid
            # for r in t.sim.rules:
            #     c.execute('INSERT INTO rule (traj_id, name, src) VALUES (?,?,?)', [t.id, r.__class__.__name__, inspect.getsource(r.__class__)])

        t.ens = self
        self.traj[t.id] = t

        return self

    def compact(self):
        for t in self.traj:
            t.compact()

    def gen_mass_graph(self, traj):
        g = MassGraph()
        n_iter = self._db_get_int('SELECT MAX(i) FROM iter WHERE traj_id = ?', [traj.id])
        with self.conn as c:
            # Groups:
            for i in range(-1, n_iter + 1):
                for r in c.execute('SELECT g.hash, g.m FROM grp g INNER JOIN iter i ON i.id = g.iter_id WHERE i.traj_id = ? AND i.i = ? ORDER BY g.id', [traj.id, -1]):
                    g.add_group(i, r[0], r[1])

            # Mass flow:
            # for i in range(n_iter + 1):
            #     for r in c.execute('''
            #             SELECT g1.hash AS hash_src, g2.hash AS hash_dst, mf.m mf.m_p FROM mass_flow mf
            #             INNER JOIN iter i ON i.id = mf.iter_id
            #             INNER JOIN grp g1 ON mf.grp_src_id = g1.id
            #             INNER JOIN grp g2 ON mf.grp_src_id = g2.id
            #             WHERE i.traj_id = ? AND i.i = ?
            #             ORDER BY mf.id''',
            #             [traj.id, iter]):
            #         mg.add_mass_flow(i, r['hash_src'], r['hash_dst'], r['m'], r['m_p'])

        return g

    def load_sim(self, traj):
        traj.sim = DB.blob2obj(self.conn.execute('SELECT sim FROM traj WHERE id = ?', [traj.id]).fetchone()[0])
        traj.sim.traj = traj
        return self

    def load_sims(self):
        for t in self.traj.values():
            self.load_sim(t)
        return self

    def save_sim(self, traj):
        '''
        To pickle the Simulation object, we need to temporarily disconnect it from its Trajectory object container.
        This is because the Trajectory object is connected to the TrajectoryEnsemble object which holds a database
        connection object and those objects cannot be pickled.  Besides, there is no point in saving the entire class
        hierarchy anyway.
        '''

        s = traj.sim
        s.traj = None
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

        for g in traj.sim.pop.groups.values():
            conn.execute(
                'INSERT INTO grp (iter_id, hash, m, attr, rel) VALUES (?,?,?,?,?)',
                [iter_id, g.get_hash(), g.m, DB.obj2blob(g.attr), DB.obj2blob(g.rel)]
            )

        return self

    def save_mass_flow(self, traj, iter_id, mass_flow_specs, conn):
        '''
        Inserts the mass flow among all groups for the given iteration and trajectory.  No mass flow is present for the
        initial state of a simulation.
        '''

        if mass_flow_specs is None:
            return self

        m_pop = traj.sim.pop.get_mass()  # to get proportion of mass flow

        for mfs in mass_flow_specs:
            g_src_id = self._db_get_id('grp', f'hash = "{mfs.src.get_hash()}"')
            for g_dst in mfs.dst:
                g_dst_id = self._db_get_id('grp', f'hash = "{g_dst.get_hash()}"')

                self._db_ins(
                    'INSERT INTO mass_flow (iter_id, grp_src_id, grp_dst_id, m, m_p) VALUES (?,?,?,?,?)',
                    [iter_id, g_src_id, g_dst_id, g_dst.m, g_dst.m / m_pop]
                )

        return self

    def run(self, iter_or_dur=1):
        for i,t in enumerate(self.traj.values()):
            print(f'Running trajectory {i+1} of {len(self.traj)} (iter: {iter_or_dur}): {t.name}')
            t.run(iter_or_dur)

        self.save_sims()

        return self

    def set_hosts(self, hosts):
        self.hosts = hosts
        return self
