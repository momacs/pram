# -*- coding: utf-8 -*-
"""Contains data-capturing code.

One purpose of simulating a model is to elucidate the dynamics captured by that model.  In the case of PRAMs, an
organic example of such dynamics is agent mass dynamics, i.e., the ways in which agent mass moves between groups as a
result of the application of rules.  PyPRAM's *probes* are the facility the user can deploy to capture the interesting
aspects of the model's dynamics.  For example, a probe could monitor and persist the sizes of all groups at every step
of the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3

from abc         import abstractmethod, ABC
from attr        import attrs, attrib, converters, validators
from collections import namedtuple
from enum        import IntEnum, Flag, auto

from .entity import GroupQry
from .util   import DB


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistenceMode(IntEnum):
    """Probe persistence mode enum.

    A probe can either append new data or overwrite the existing data.
    """

    APPEND    = 1
    OVERWRITE = 2


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class ProbePersistenceDBItem(object):
    """A description of how a probe will interact with a database for the purpose of data persistence.

    Args:
        name (str): The item's name.
        ins_qry (str): The SQL INSERT query that is used for persisting data in the database.  This query should be
            parameterized to expect the value to be persisted.
        sel_qry (str): The SQL SELECT query that is used for retrieving data from the database (e.g., to generate
            plots).
        ins_val (list): The values to be inserted into the database using the INSERT query.
    """

    name    : str  = attrib()
    ins_qry : str  = attrib()  # insert query (used for persisting data in the DB)
    sel_qry : str  = attrib()  # select query (used for retrieving data from the DB, e.g., to generate plots)
    ins_val : list = attrib(factory=list, converter=converters.default_if_none(factory=list))


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistence(ABC):
    """Probe persistence base class.

    After being instantiated, probes are registered with an ProbePersistace object.  This design allows a single probe
    to persist values to multiple databases, files, etc.
    """

    VAR_NAME_KEYWORD = ['id', 'ts', 'i', 't']

    @abstractmethod
    def flush(self):
        """Flushes all buffered values to the persistence layer."""

        pass

    @abstractmethod
    def plot(self, probe, series, figpath=None, figsize=(12,4), legend_loc='upper right', dpi=150):
        """Plots data associated with a probe.

        Args:
            probe (Probe): The probe.
            series (dict): Series specification (see examples below).
            figpath (str, optional): Filepath to save the figure.
            figsize ((int,int)): Figure size in (w,h) format.
            legend_loc (str): Legend location (e.g., 'upper right').
            dpi (int): Resolution.

        Examples::

            p = GroupSizeProbe.by_attr('flu', 'flu', ['s', 'i', 'r'], persistence=ProbePersistenceDB())

            # create a simulation

            series = [
                { 'var': 'p0', 'lw': 0.75, 'linestyle': '-',  'marker': 'o', 'color': 'red',   'markersize': 0, 'lbl': 'S' },
                { 'var': 'p1', 'lw': 0.75, 'linestyle': '--', 'marker': '+', 'color': 'blue',  'markersize': 0, 'lbl': 'I' },
                { 'var': 'p2', 'lw': 0.75, 'linestyle': ':',  'marker': 'x', 'color': 'green', 'markersize': 0, 'lbl': 'R' }
            ]
            p.plot(series, figsize=(16,3))
                # while the probe's plot() method is being called, that method simply comes here
        """

        pass

    @abstractmethod
    def get_data(self, probe):
        """Retrieves data associated with a probe.

        Args:
            probe (Probe): The probe.

        Returns:
            A dictionary based on the SQL SELECT query of the probe
        """

        pass

    @abstractmethod
    def persist(self):
        """Stores values to be persisted and persists them in accordance with the flushing frequency set."""

        pass

    @abstractmethod
    def reg_probe(self, probe):
        """Registeres a probe.

        Args:
            probe (Probe): The probe.
        """

        pass


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistenceFS(ProbePersistence):
    """Filesystem-based probe persistence.

    Args:
        fpath (str): Path to the file.
        mode (int): Persistence mode (see :class:`~pram.data.ProbePersistenceMode` enum).

    Note:
        This class is a stub and will be fully implemented when it is needed.
    """

    def __init__(self, fpath, mode=ProbePersistenceMode.APPEND):
        self.path = path

        # ...

    def __del__(self):
        pass

    def persist(self):
        pass

    def reg_probe(self, probe, do_overwrite=False):
        """Registers a probe.

        Args:
            probe (Probe): The probe.
            do_overwrite (bool): Flag: Should an already registered probe with the same name be overwriten?
        """

        pass


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistenceDB(ProbePersistence):
    """Relational database based probe persistence.

    At this point, the SQLite3 database is used.  More serious RDBMSs will be added later (PostgreSQL being the first
    of them).

    Database inserts are cached and flushed when the buffer fills up.  The buffer size is under the user's control.
    The default size of 16 decreases the time spent persisting data to the database about 15 times.  Higher values
    can be used to achieve even better results, but memory utilization cost needs to be considered too.

    [ Probes are unaware of the kind of persistence (i.e., standalone or trajectory ensemble) they are connected to. ]

    Args:
        fpath (str): Path to the database file (``':memory:'`` for in-memory database which is good for testing).
        mode (int): Persistence mode (see :class:`~pram.data.ProbePersistenceMode` enum).
        flush_every (int): Memory-to-database flush frequency.
    """

    FLUSH_EVERY = 16

    def __init__(self, fpath=':memory:', mode=ProbePersistenceMode.APPEND, flush_every=FLUSH_EVERY, traj_ens=None, conn=None):
        self.probes = {}  # objects of the ProbePersistenceDBItem class hashed by the name of the probe
        self.conn = None
        self.fpath = fpath
        self.mode = mode
        self.flush_every = flush_every
        self.traj_ens = traj_ens

        if conn is None:
            if os.path.isfile(self.fpath) and mode == ProbePersistenceMode.OVERWRITE:
                os.remove(self.fpath)
            self.conn_open()
            self.do_close_conn = True
        else:
            self.conn = conn
            self.do_close_conn = False

    @classmethod
    def with_traj(cls, traj_ens, conn):
        return cls(fpath=None, traj_ens=traj_ens, conn=conn)

    def __del__(self):
        self.conn_close()

    def conn_close(self):
        """Closes the database connection."""

        if self.conn is None or not self.do_close_conn:
            return

        self.flush()
        self.conn.close()
        self.conn = None

    def conn_open(self):
        """Opens the database connection."""

        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.fpath, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def flush(self):
        """Flushes all buffered values to the database."""

        with self.conn as c:
            for p in self.probes.values():
                if len(p.ins_val) > 0:
                    c.executemany(p.ins_qry, p.ins_val)
                    p.ins_val = []

    def get_data(self, probe):
        """Retrieves data associated with a probe from the database.

        Args:
            probe (Probe): The probe.

        Returns:
            A dictionary based on the SQL SELECT query of the probe
        """

        probe_item = self.probes[DB.str_to_name(probe.name)]
        return [dict(r) for r in self.conn.execute(probe_item.sel_qry).fetchall()]

    def persist(self, probe, vals, iter, t):
        """Dispatch.

        Args:
            probe (Probe): The probe.
            vals (Iterable[Any]): An iterable with the values to be stored.
            iter (int): Simulation iteration (only expected for standalone persistence).
            t (int): Simulation time (only expected for standalone persistence).
        """

        if self.traj_ens:
            self.persist_traj_ens(probe, vals)
        else:
            self.persist_standalone(probe, vals, iter, t)

    def persist_standalone(self, probe, vals, iter, t):
        """Stores values to be persisted and persists them in accordance with the flushing frequency.

        Args:
            probe (Probe): The probe.
            vals (Iterable[Any]): An iterable with the values to be stored.
            iter (int): Simulation iteration.
            t (int): Simulation time.
        """

        probe_item = self.probes[DB.str_to_name(probe.name)]

        if self.flush_every <= 1:
            with self.conn as c:
                c.execute(probe_item.ins_qry, [iter,t] + [c.val for c in probe.consts] + vals)
        else:
            probe_item.ins_val.append([iter,t] + [c.val for c in probe.consts] + vals)
            if len(probe_item.ins_val) >= self.flush_every:
                with self.conn as c:
                    c.executemany(probe_item.ins_qry, probe_item.ins_val)
                probe_item.ins_val = []

    def persist_traj_ens(self, probe, vals):
        """Persists scheduled items in accordance with the flushing frequency.  The value for the 'iter_id' field gets
        added to every item being persisted.

        Note:
            Flushing frequency is currently ignored to make the implementation simpler.  Specifically, the flush()
            method doesn't need to do anything.

        Args:
            iter_id (int): ID of the iteration from the trajectory ensemble database.
        """

        # probe_item = self.probes[DB.str_to_name('probe_' + probe.name)]
        # with self.conn as c:
        #     c.execute(probe_item.ins_qry, [self.traj_ens.curr_iter_id] + [c.val for c in probe.consts] + vals)

        probe_item = self.probes[DB.str_to_name('probe_' + probe.name)]

        if self.flush_every <= 1:
            with self.conn as c:
                c.execute(probe_item.ins_qry, [self.traj_ens.curr_iter_id] + [c.val for c in probe.consts] + vals)
        else:
            probe_item.ins_val.append([self.traj_ens.curr_iter_id] + [c.val for c in probe.consts] + vals)
            if len(probe_item.ins_val) >= self.flush_every:
                with self.conn as c:
                    c.executemany(probe_item.ins_qry, probe_item.ins_val)
                probe_item.ins_val = []

    def persist_traj_ens_exec__(self, iter_id):
        # Pending removal

        """Persists scheduled items in accordance with the flushing frequency.  The value for the 'iter_id' field gets
        added to every item being persisted.

        Args:
            iter_id (int): ID of the iteration from the trajectory ensemble database.
        """

        for p in self.probes.values():
            if len(p.ins_val) == 0:
                continue

            for iv in p.ins_val:
                iv.insert(0, iter_id)
            with self.conn as c:
                c.executemany(p.ins_qry, p.ins_val)
            p.ins_val = []

    def persist_traj_ens_schedule__(self, probe, vals):
        # Pending removal

        """Schedules values to be persisted.  The trajectory ensemble object which wrapps the present object executes
        all scheduled persists at an opportune time (when 'iter_id' is known).  The value for the 'iter_id' field gets
        added at execution time.

        Args:
            probe (Probe): The probe.
            vals (Iterable[Any]): An iterable with the values to be stored.
        """

        probe_item = self.probes[DB.str_to_name(probe.name)]
        probe_item.ins_val.append([c.val for c in probe.consts] + vals)

    def plot(self, probe, series, ylabel, xlabel='Iteration', figpath=None, figsize=(12,4), legend_loc='upper right', dpi=150, subplot_l=0.08, subplot_r=0.98, subplot_t=0.95, subplot_b=0.25):
        """Plots data associated with a probe.

        Args:
            probe (Probe): The probe.
            series (dict): Series specification (see examples below).
            ylabel (str): Label of the Y axis.
            figpath (str, optional): Filepath to save the figure.
            figsize ((int,int)): Figure size in (w,h) format.
            legend_loc (str): Legend location (e.g., 'upper right').
            dpi (int): Resolution.

        Examples::

            p = GroupSizeProbe.by_attr('flu', 'flu', ['s', 'i', 'r'], persistence=ProbePersistenceDB())

            # define a simulation

            series = [
                { 'var': 'p0', 'lw': 0.75, 'linestyle': '-',  'marker': 'o', 'color': 'red',   'markersize': 0, 'lbl': 'S' },
                { 'var': 'p1', 'lw': 0.75, 'linestyle': '--', 'marker': '+', 'color': 'blue',  'markersize': 0, 'lbl': 'I' },
                { 'var': 'p2', 'lw': 0.75, 'linestyle': ':',  'marker': 'x', 'color': 'green', 'markersize': 0, 'lbl': 'R' }
            ]
            p.plot(series, figsize=(16,3))

        """

        probe_item = self.probes[DB.str_to_name(probe.name)]

        data = { s['var']:[] for s in series }
        data['i'] = []
        with self.conn as c:
            for r in c.execute(probe_item.sel_qry).fetchall():
                data['i'].append(r['i'])
                for s in series:
                    data[s['var']].append(r[s['var']])

        # Plot:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # plt.title('SIR Model')
        for s in series:
            plt.plot(data['i'], data[s['var']], lw=s.get('lw'), ls=s.get('ls'), dashes=s.get('dashes', []), marker=s.get('marker'), color=s.get('color'), ms=s.get('ms'), mfc='none', antialiased=True)
        plt.legend([s['lbl'] for s in series], loc=legend_loc)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25, antialiased=True)
        plt.subplots_adjust(left=subplot_l, right=subplot_r, top=subplot_t, bottom=subplot_b)

        if figpath is None:
            mng = plt.get_current_fig_manager()
            # mng.frame.Maximize(True)    # TODO: [low priority] The below should work on different OSs
            # mng.window.showMaximized()
            # mng.full_screen_toggle()
            # mng.window.state('zoomed')

            plt.show()
        else:
            fig.savefig(figpath, dpi=dpi)

        return fig

    def reg_probe(self, probe, do_overwrite=False):
        """Dispatcher.

        Args:
            probe (Probe): The probe.
            do_overwrite (bool): Flag: Should an already registered probe with the same name be overwriten?
        """

        if self.traj_ens:
            self.reg_probe_traj_ens(probe)
        else:
            self.reg_probe_standalone(probe, do_overwrite)

    def reg_probe_standalone(self, probe, do_overwrite=False):
        """Registers a probe (for standalone persitance).

        Args:
            probe (Probe): The probe.
            do_overwrite (bool): Flag: Should an already registered probe with the same name be overwriten?

        Throws:
            ValueError
        """

        name_db = DB.str_to_name(probe.name)

        if name_db in self.probes and not do_overwrite:
            raise ValueError(f"The probe '{probe.name}' has already been registered.")

        # Create the table:
        cols = [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL',
            'i INTEGER NOT NULL',
            't REAL NOT NULL'
        ] + \
        [ f'{c.name} {c.type}' for c in probe.consts ] + \
        [ f'{v.name} {v.type}' for v in probe.vars ]

        with self.conn as c:
            c.execute(f'CREATE TABLE IF NOT EXISTS {name_db} (' + ','.join(cols) + ');')

        # Store probe:
        ins_qry = \
            f"INSERT INTO {name_db} " + \
            f"({','.join(['i','t'] + [c.name for c in probe.consts] + [v.name for v in probe.vars])}) " + \
            f"VALUES ({','.join(['?'] * (len(cols) - 2))})"  # -2 for the 'id' and 'ts' columns

        sel_qry = f"SELECT {','.join(['i','t'] + [c.name for c in probe.consts] + [v.name for v in probe.vars])} FROM {name_db}"

        self.probes[name_db] = ProbePersistenceDBItem(name_db, ins_qry, sel_qry)

    def reg_probe_traj_ens(self, probe):
        """Registers a probe (for trajectory ensemble persistence).

        Args:
            probe (Probe): The probe.
            do_overwrite (bool): Flag: Should an already registered probe with the same name be overwriten?

        Throws:
            ValueError
        """

        name_db = DB.str_to_name('probe_' + probe.name)

        if name_db in self.probes:
            return

        # Create the table:
        cols = [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'iter_id INTEGER NOT NULL'
        ] + \
        [ f'{c.name} {c.type}' for c in probe.consts ] + \
        [ f'{v.name} {v.type}' for v in probe.vars ]

        with self.conn as c:
            c.execute(f'CREATE TABLE IF NOT EXISTS {name_db} (' + ','.join(cols) + ');')

        # Store probe:
        ins_qry = \
            f"INSERT INTO {name_db} " + \
            f"({','.join(['iter_id'] + [c.name for c in probe.consts] + [v.name for v in probe.vars])}) " + \
            f"VALUES ({','.join(['?'] * (len(cols) - 1))})"  # -1 for the 'id' column

        sel_qry = f"SELECT {','.join(['i.i'] + [c.name for c in probe.consts] + [v.name for v in probe.vars])} FROM {name_db} a INNER JOIN iter i ON i.id = a.iter_id"

        self.probes[name_db] = ProbePersistenceDBItem(name_db, ins_qry, sel_qry)


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistenceMem(ProbePersistenceDB):
    """Relational in-memory database based probe persistence.

    This class is a very light and elegant specialization of the ProbePersistenceDB class.

    Args:
        mode (int): Persistence mode (i.e., append or overwrite; use the ProbePersistenceMode enum type).
        flush_every (int): Memory-to-database flush frequency.
    """

    def __init__(self, mode=ProbePersistenceMode.APPEND, flush_every=ProbePersistenceDB.FLUSH_EVERY):
        super().__init__(':memory:', mode, flush_every)


# ----------------------------------------------------------------------------------------------------------------------
class ProbeMsgMode(Flag):
    """Probe's message mode enum.

    A probe can be designated to send messages to stdout which is useful for testing and debugging.  This class
    determines how those messages are handled.  The three options are:

        - Ignore (do not cumulate nor display any messages)
        - Display (Display messages right away)
        - Cumulate (Store messages in an internal list for future retrieval)

    When testing or debugging a simulation with only one probe, the display mode seems most useful (and it is in fact
    the default).  However, the simulation output can quickly become unreasable if multiple probes are sending messages
    to stdout.  The cumulate mode addresses this scenario; messages are stored and can be retrieved at any point with
    the end of the simulation being one reasonable choice.

    Probe messages are a simpler and chronologically older alternative to probe persistence.
    """

    NONE  = auto()  # ignore the value (i.e., do not display and do not store, unless persistence layer is defined)
    DISP  = auto()  # display messages
    CUMUL = auto()  # hold messages in the buffer


# ----------------------------------------------------------------------------------------------------------------------
# Having the following two namedtuple declarations inside of the Probe class as subclasses where they belong breaks
# Flask/Celery.  No matter what I tried I couldn't get this to work so it seems it's a bug... (@#!%$#).  Moreover, the
# namedtuple type needs to be wrapped in a new class declaration in order for Sphinx (the docs generator) to properly
# display docstrings.
#
#     https://stackoverflow.com/questions/1606436/adding-docstrings-to-namedtuples

class Var(namedtuple('Var', ['name', 'type'])):
    """Probe's variable.

    A probe can be in charge of storing multiple variables that relate to the state of the simulation as it evolves.  This
    class shows how such a variable can be defined.

    Args:
        name (str): The variable's name.  If relational database persistence is used, this name will become the name of a
            table column.  While PyPRAM escapes special characters, problematic names should be avoided (e.g., those
            containing spaces or weird characters).
        type (str): The variable's type.  If relational database persistence is used, this needs to be a valid data types
            of the RDBMS of choice.
    """

    pass

class Const(namedtuple('Const', ['name', 'type', 'val'])):
    """Probe's constant.

    A probe can be in charge of storing multiple constants that relate to the simulation context at large.  Naturally,
    being constants their values will not change, but it may beneficial to have them as database table columns for later
    data processing and analysis (e.g., forming a UNION of two or more tables with congruent schemas).  This class shows
    how such a constant can be defined.

    Args:
        name (str): The constants's name.  If relational database persistence is used, this name will become the name of a
            table column.  While PyPRAM escapes special characters, problematic names should be avoided (e.g., those
            containing spaces or weird characters).
        type (str): The constants's type.  If relational database persistence is used, this needs to be a valid data types
            of the RDBMS of choice.
        val (Any): The variable's value (must match the ``type``, although in case of database persistence the driver
            or the RDBMS itself may attempt to cast).
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
class Probe(ABC):
    """Probe base class."""

    def __init__(self, name, persistence=None, pop=None, memo=None):
        self.name = name
        self.consts = []
        self.persistence = None
        self.pop = pop  # pointer to the population (can be set elsewhere too)
        self.memo = memo

        self.set_persistence(persistence)

    def plot(self, series, ylabel, xlabel='Iteration', fig_fpath=None, figsize=(8,8), legend_loc='upper right', dpi=150, subplot_l=0.08, subplot_r=0.98, subplot_t=0.95, subplot_b=0.25):
        """Plots data associated with a probe.

        This method calls :meth:`~pram.data.ProbePersistence.plot`.
        """

        if not self.persistence:
            return print('Plotting error: The probe is not associated with a persistence backend')

        return self.persistence.plot(self, series, ylabel, xlabel, fig_fpath, figsize, legend_loc, dpi, subplot_l, subplot_r, subplot_t, subplot_b)

    @abstractmethod
    def run(self, iter, t):
        """Runs the probe.

        A probe is run by the :meth:`~pram.sim.Simulation.run` method of the
        :class:`~pram.sim.Simulation` class as it steps through the simulation.  Probes are
        run as the last order of business before the simulation advances to the next iteration which is congruent with
        probe capturing the state of the simulation after it has settled at every iteration.

        Setting both the 'iter' and 't' to 'None' will prevent persistence from being invoked and message cumulation to
        occur.  It will still allow printint to stdout however.  In fact, this mechanism is used by the
        :class:`~pram.sim.Simulation` class to print the intial state of the system (as seen by those probes that
        actually print), that is before the simulation run begins.

        Args:
            iter (int): The simulation iteration.
            t (int): The simulation time.
        """

        pass

    def set_persistence(self, persistence):
        """Associates the probe with a ProbePersistence object.

        Args:
            persistence (ProbePersistence): The ProbePersistence object.
        """

        if persistence is None or self.persistence == persistence:
            return

        self.persistence = persistence
        if self.persistence is not None:
            self.persistence.reg_probe(self)

    def set_pop(self, pop):
        """Sets the group population the probe should interact with.

        Currently, a PyPRAM simulation operates on a single instance of the GroupPopulation object.  This may change in
        the future and this method will associate the probe with one of the group populations.

        Args:
            pop (GroupPopulation): Group population.
        """

        self.pop = pop


# ----------------------------------------------------------------------------------------------------------------------
class GroupProbe(Probe, ABC):
    """A probe that monitors any aspect of a PRAM group.

    Args:
        name (str): Probe's name; probes are identified by names.
        queries (Iterable[GroupQry]): A GroupQry object selects one of more PRAM groups based on their attributes and
            relations.  For example, ``GroupQry(attr={ 'flu': 's' }, rel={ Site.AT: Site('home') })`` selects all
            groupsof agents that are susceptible to flu that are at the particular site called ``home``.
        qry_tot (GroupQry, optional): Apart from capturing the absolute numbers of agents residing in a group or being
            moved between groups, a probe also calculates those numbers as proportions; naturally, both of these
            happen every iteration of the simulation.  To calculate proportions, the total agent population mass needs
            to be known.  ``qry_tot`` provides that total mass.  It can be a ``GroupQry`` object that selects groups to
            have their masses summed to form the total mass (e.g., all agents that have the flu irrespective of where
            they currently are).  Alternatively, if left at the default value of None, the entire population mass will
            be used as the normalizing factor.
        consts (Iterable[Const], optional): Constants (see :class:`pram.data.Const` class for details).
        persistence (ProbePersistence, optional): A ProbePersistence object reference.
        msg_mode (int): Probe's message mode (see :class:`~pram.data.ProbeMsgMode` enum).
        pop (GroupPopulation, optional): The group population in question.
        memo (str, optional): Probe's description.

    Note:
        All classes extending this class must populate the ``self.vars`` list before calling the constructor of this
        class.

    Todo:
        Figure out the relation between the 'self.persistence' and registering probes with multiple ProbePersistence
        objects.  At this point it looks like due to the interaction of multiple designs the probe can only be
        associated with one such objects.
    """

    def __init__(self, name, queries, qry_tot=None, consts=None, persistence=None, msg_mode=ProbeMsgMode.DISP, pop=None, memo=None):
        super().__init__(name, persistence, pop, memo)

        self.queries = queries
        self.qry_tot = qry_tot
        self.consts = []
        self.msg_mode = msg_mode
        self.msg = []  # used to cumulate messages (only when 'msg_mode & ProbeMsgMode == True')

        self.set_consts(consts)
        self.set_persistence(persistence)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return 'GroupProbe: {:16}  query-cnt: {:>3}'.format(self.name, len(self.queries))

    @classmethod
    def by_attr(cls, probe_name, attr_name, attr_values, qry_tot=None, var_names=None, consts=None, persistence=None, msg_mode=0, pop=None, memo=None):
        """Instantiates the probe by attributes.

        This constructor generates QueryGrp objects automatically for the attribute name and values specified.  It is a
        convenience method for probes that only use only a single attribute of PRAM groups and do not use relations.

        Args:
            probe_name (str): Probe's name; probes are identified by names.
            attr_name (str): The name of the attribute.
            attr_values (Iterable[str]): The values of the attribute to be monitored.  Sizes of PRAM groups with
                different values of the attribute in question will not be monitored.
            qry_tot (GroupQry, optional): See :meth:`~pram.data.GroupProbe.__init__`
            var_names (Iterable[str], optional): Names which should be assigned to the attribute values.  Those names
                must correspond to ``attr_values`` in that every element of ``attr_values`` must resolve to **two**
                elements in ``var_names``.  That is because both the proportion and the absolute number of the agents
                are recorded by the probe.  In case of database persistence, the variable names will be column names.
                If left None, default names will be used (i.e., ``p0``, ``p1``, ... for proportions and ``m0``, ``m1``,
                ... for mass).
            consts (Iterable[Const], optional): Constants (see :class:`pram.data.Const` class for details).
            persistence (ProbePersistence, optional): A ProbePersistence object reference.
            msg_mode (int): Probe's message mode (see :class:`~pram.data.ProbeMsgMode` enum).
            pop (GroupPopulation, optional): The group population in question.
            memo (str, optional): Probe's description.
        """

        return cls(probe_name, [GroupQry(attr={ attr_name: v }) for v in attr_values], qry_tot, var_names, consts, persistence, msg_mode, pop, memo)

    @classmethod
    def by_rel(cls, probe_name, rel_name, rel_values, qry_tot=None, var_names=None, consts=None, persistence=None, msg_mode=0, pop=None, memo=None):
        """Instantiates the probe by relations.

        This constructor generates QueryGrp objects automatically for the relation name and values specified.  It is a
        convenience method for probes that only use only a single relation of PRAM groups and do not use attributes.

        Args:
            probe_name (str): Probe's name; probes are identified by names.
            rel_name (str): The name of the relation.
            rel_values (Iterable[str]): The values of the relation to be monitored.  Sizes of PRAM groups with
                different values of the relation in question will not be monitored.
            qry_tot (GroupQry, optional): See :meth:`~pram.data.GroupProbe.__init__`
            var_names (Iterable[str], optional): Names which should be assigned to the relation values.  Those names
                must correspond to ``rel_values`` in that every element of ``rel_values`` must resolve to **two**
                elements in ``var_names``.  That is because both the proportion and the absolute number of the agents
                are recorded by the probe.  In case of database persistence, the variable names will be column names.
                If left None, default names will be used (i.e., ``p0``, ``p1``, ... for proportions and ``m0``, ``m1``,
                ... for mass).
            consts (Iterable[Const], optional): Constants (see :class:`pram.data.Const` class for details).
            persistence (ProbePersistence, optional): A ProbePersistence object reference.
            msg_mode (int): Probe's message mode (see :class:`~pram.data.ProbeMsgMode` enum).
            pop (GroupPopulation, optional): The group population in question.
            memo (str, optional): Probe's description.
        """

        return cls(probe_name, [GroupQry(rel={ rel_name: v }) for v in rel_values], qry_tot, var_names, consts, persistence, msg_mode, pop, memo)

    def clear_msg(self):
        """Clear any cumulated messages."""

        self.msg.clear()

    def get_data(self):
        """Retrieves data associated with the probe.

        For this to work the probe needs to be associated with a ProbePersistence object.
        """

        if not self.persistence:
            return print('Data access error: The probe is not associated with a persistence backend')

        return self.persistence.get_data(self)

    def get_msg(self, do_join=True):
        """Retrieve all cumulated messages.

        Args:
            do_join (bool): Flag: Join the messages with the new-line character?

        Returns:
            str: The message.
        """

        return '\n'.join(self.msg)

    def plot(self, series, fig_fpath=None, figsize=(8,8), legend_loc='upper right', dpi=300):
        """Plots data associated with a probe.

        This method calls :meth:`~pram.data.ProbePersistence.plot`.
        """

        if not self.persistence:
            return print('Plotting error: The probe is not associated with a persistence backend')

        return self.persistence.plot(self, series, 'Probability', fig_fpath, figsize, legend_loc, dpi)

    def set_consts(self, consts=None):
        """Sets the probe's constants.

        Args:
            consts (Iterable[Const], optional): The constants (see :class:`pram.data.Const` class for details).
        """

        consts = consts or []

        cn_db_used = set()  # to identify duplicates

        for c in consts:
            if c.name in ProbePersistence.VAR_NAME_KEYWORD:
                raise ValueError(f"The following constant names are restricted: {ProbePersistence.VAR_NAME_KEYWORD}")

            cn_db = DB.str_to_name(c.name)
            if cn_db in cn_db_used:
                raise ValueError(f"Variable name error: Name '{c.name}' translates into a database name '{cn_db}' which already exists.")

            cn_db_used.add(cn_db)

        self.consts = consts or []


# ----------------------------------------------------------------------------------------------------------------------
class GroupAttrProbe(GroupProbe):
    """A probe that monitors a PRAM group's attribute values.

    See :meth:`~pram.data.GroupProbe.__init__` for more details.
    """

    def __init__(self, name, queries, qry_tot=None, var_names=None, consts=None, persistence=None, msg_mode=0, pop=None, memo=None):
        raise Error('Implementation pending completion.')

    def run(self, iter, t):
        """Runs the probe.

        More details in the :meth:`abstract method <pram.data.GroupProbe.run>`.

        Args:
            iter (int): The simulation iteration.
            t (int): The simulation time.
        """

        pass


# ----------------------------------------------------------------------------------------------------------------------
class GroupSizeProbe(GroupProbe):
    """A probe that monitors a PRAM group's size.

    See :class:`~pram.data.GroupProbe` for more details.
    """

    def __init__(self, name, queries, qry_tot=None, var_names=None, consts=None, persistence=None, msg_mode=0, pop=None, memo=None):
        if var_names is None:
            self.vars = \
                [Var(f'p{i}', 'float') for i in range(len(queries))] + \
                [Var(f'm{i}', 'float') for i in range(len(queries))]
                # proportions and numbers
            # self.vars = [GroupProbe.Var(f'v{i}', 'float') for i in range(len(queries))]
        else:
            self.vars = []
            if len(var_names) != (len(queries) * 2):
                raise ValueError(f'Incorrect number of variable names: {len(var_names)} supplied, {len(queries) * 2} expected (i.e., {len(queries)} for proportions and numbers each).')
            # if len(var_names) != len(queries):
            #     raise ValueError(f'Incorrect number of variable names: {len(var_names)} supplied, {len(queries)} expected.')

            vn_db_used = set()  # to identify duplicates
            for vn in var_names:
                if vn in ProbePersistence.VAR_NAME_KEYWORD:
                    raise ValueError(f"The following variable names are restrictxxed: {ProbePersistence.VAR_NAME_KEYWORD}")

                # vn_db = DB.str_to_name(vn)  # commented out because plotting method was expecting quoted values
                vn_db = vn
                if vn_db in vn_db_used:
                    raise ValueError(f"Variable name error: Name '{vn}' translates into a database name '{vn_db}' which already exists.")

                vn_db_used.add(vn_db)
                self.vars.append(Var(vn_db, 'float'))

        super().__init__(name, queries, qry_tot, consts, persistence, msg_mode, pop, memo)

    def run(self, iter, t):
        """Runs the probe.

        More details in the :meth:`abstract method <pram.data.GroupProbe.run>`.

        Args:
            iter (int): The simulation iteration.
            t (int): The simulation time.
        """

        if self.msg_mode != 0 or self.persistence:
            n_tot = sum([g.m for g in self.pop.get_groups(self.qry_tot)])  # TODO: If the total mass never changed, we could memoize this (either here or in GroupPopulation).
            n_qry = [sum([g.m for g in self.pop.get_groups(q)]) for q in self.queries]

        # Message:
        if self.msg_mode != 0:
            msg = []
            if n_tot > 0:
                msg.append('{:2}  {}: ('.format(t if not t is None else '.', self.name))
                for n in n_qry:
                    msg.append('{:.2f} '.format(abs(round(n / n_tot, 2))))  # abs solves -0.00, likely due to rounding and string conversion
                msg.append(')   (')
                for n in n_qry:
                    msg.append('{:>7} '.format(abs(round(n, 1))))  # abs solves -0.00, likely due to rounding and string conversion
                msg.append(')   [{}]'.format(round(n_tot, 1)))
            else:
                msg.append('{:2}  {}: ---'.format(t if not t is None else '.', self.name))

            if self.msg_mode & ProbeMsgMode.DISP:
                print(''.join(msg))
            if self.msg_mode & ProbeMsgMode.CUMUL:
                self.msg.append(''.join(msg))

        # Persistence:
        if self.persistence and not iter is None:
            vals_p = []
            vals_n = []
            for n in n_qry:
                vals_p.append(n / n_tot)
                vals_n.append(n)

            self.persistence.persist(self, vals_p + vals_n, iter, t)
