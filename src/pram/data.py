import os
import sqlite3

from abc         import abstractmethod, ABC
from collections import namedtuple
from enum        import Flag, auto

from .entity import GroupQry
from .util   import DB


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistance(ABC):
    @abstractmethod
    def persist(self):
        pass

    @abstractmethod
    def reg_probe(self, probe):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistanceFS(ProbePersistance):
    def __init__(self, path, do_overwrite=False):
        self.path = path

        # ...

    def __del__(self):
        pass

    def persist(self):
        pass

    def reg_probe(self, probe):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistanceDB(ProbePersistance):
    def __init__(self, fpath, do_id=True, do_ts=True, do_overwrite=False):
        self.probes = set()  # to identify duplicates
        self.conn   = None
        self.fpath  = fpath
        self.do_id  = do_id  # flag: use id column?
        self.do_ts  = do_ts  # flag: use timestamp column?

        if os.path.isfile(self.fpath):
            if do_overwrite:
                os.remove(self.fpath)
            else:
                raise ValueError(f'The database already exists: {self.fpath}')

        self.conn_open()

    def __del__(self):
        self.conn_close()

    def conn_close(self):
        if self.conn is None:
            return

        self.conn.close()
        self.conn = None

    def conn_open(self):
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.fpath, check_same_thread=False)

    def persist(self, probe, vals, iter ,t):
        name_db = DB.str_to_name(probe.name)
        with self.conn as c:
            c.execute(f"INSERT INTO {name_db} ({','.join(['i','t'] + [v.name for v in probe.vars])}) VALUES ({','.join(['?','?'] + ['?' for _ in vals])})", [iter,t] + vals)

    def reg_probe(self, probe):
        name_db = DB.str_to_name(probe.name)

        if name_db in self.probes:
            raise ValueError(f"The probe '{probe.name}' has already been registered.")

        self.probes.add(name_db)

        cols = []
        if self.do_id: cols.append('id INTEGER PRIMARY KEY AUTOINCREMENT')
        if self.do_ts: cols.append('ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL')  # (datetime('now'))
        cols.append('i INTEGER NOT NULL')
        cols.append('t REAL NOT NULL')

        for v in probe.vars:
            cols.append(f'{v.name} {v.type}')  # name and type need to be DB-compliant at this point

        with self.conn as c:
            c.execute(f'CREATE TABLE {name_db} (' + ','.join(cols) + ');')


# ----------------------------------------------------------------------------------------------------------------------
class ProbeMsgMode(Flag):
    DISP  = auto()  # display messages
    CUMUL = auto()  # hold messages in the buffer


class Probe(ABC):
    __slots__ = ('name', 'persist', 'msg_mode','pop', 'memo', 'msg')

    Var = namedtuple('Var', ['name', 'type'])

    def __init__(self, name, persist=None, msg_mode=0, pop=None, memo=None):
        self.name = name
        self.persist = persist
        self.msg_mode = msg_mode
        self.pop = pop  # pointer to the population (can be set elsewhere too)
        self.memo = memo
        self.msg = []  # used to cumulate messages (only when 'msg_mode & ProbeMsgMode == True')

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return 'Probe'

    def clear(self):
        self.msg.clear()

    def get_msg(self, do_join=True):
        return '\n'.join(self.msg)

    @abstractmethod
    def run(self, i,t):
        pass

    def set_pop(self, pop):
        self.pop = pop


# ----------------------------------------------------------------------------------------------------------------------
class GroupSizeProbe(Probe):
    __slots__ = ('queries', 'qry_tot', 'vars')

    def __init__(self, name, queries, var_names=None, qry_tot=None, persist=None, msg_mode=0, pop=None, memo=None):
        '''
        queries: iterable of GroupQry
        '''

        super().__init__(name, persist, msg_mode, pop, memo)

        self.queries = queries
        self.qry_tot = qry_tot
        self.vars = []

        if var_names is None:
            self.vars = \
                [Probe.Var(f'p{i}', 'float') for i in range(len(self.queries))] + \
                [Probe.Var(f'n{i}', 'float') for i in range(len(self.queries))]
                # proportions and numbers
        else:
            if len(var_names) != (len(self.queries) * 2):
                raise ValueError(f'Incorrect number of variable names: {len(var_names)} supplied, {len(self.queries) * 2} expected (i.e., {len(self.queries)} for proportions and numbers each).')

            vn_db_used = set()  # to identify duplicates
            for vn in var_names:
                if vn == 'id' or vn == 'ts':
                    raise ValueError(f"Variable name error: Names 'id', and 'ts' are restricted.")

                vn_db = DB.str_to_name(vn)
                if vn_db in vn_db_used:
                    raise ValueError(f"Variable name error: Name '{vn}' translates into a database name '{vn_db}' which already exists.")

                vn_db_used.add(vn_db)
                self.vars.append(Probe.Var(vn_db, DB.str_to_type('float')))

        if self.persist is not None:
            self.persist.reg_probe(self)

    @classmethod
    def by_attr(cls, probe_name, attr_name, attr_values, var_names=None, qry_tot=None, persist=None, msg_mode=0, pop=None, memo=None):
        ''' Generates QueryGrp objects automatically for the attribute name and values specified. '''

        return cls(probe_name, [GroupQry(attr={ attr_name: v }) for v in attr_values], var_names, qry_tot, persist, msg_mode, pop, memo)

    @classmethod
    def by_rel(cls, probe_name, rel_name, rel_values, var_names=None, qry_tot=None, persist=None, msg_mode=0, pop=None, memo=None):
        ''' Generates QueryGrp objects automatically for the relation name and values specified. '''

        return cls(probe_name, [GroupQry(rel={ rel_name: v }) for v in rel_values], var_names, qry_tot, persist, msg_mode, pop, memo)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return 'Probe  name: {:16}  query-cnt: {:>3}'.format(self.name, len(self.queries))

    def run(self, iter, t):
        if self.msg_mode != 0 or self.persist is not None:
            n_tot = sum([g.n for g in self.pop.get_groups(self.qry_tot)])  # TODO: If the total mass never changed, we could memoize this (likely in GroupPopulation).
            n_qry = [sum([g.n for g in self.pop.get_groups(q)]) for q in self.queries]

        # Message:
        if self.msg_mode != 0:
            msg = []
            if n_tot > 0:
                msg.append('{:2}  {}: ('.format(t, self.name))
                for n in n_qry:
                    msg.append('{:.2f} '.format(abs(round(n / n_tot, 2))))  # abs solves -0.00, likely due to rounding and string conversion
                msg.append(')   (')
                for n in n_qry:
                    msg.append('{:>7} '.format(abs(round(n, 1))))  # abs solves -0.00, likely due to rounding and string conversion
                msg.append(')   [{}]'.format(round(n_tot, 1)))
            else:
                msg.append('{:2}  {}: ---'.format(t, self.name))

            if self.msg_mode & ProbeMsgMode.DISP:
                print(''.join(msg))
            if self.msg_mode & ProbeMsgMode.CUMUL:
                self.msg.append(''.join(msg))

        # Persistance:
        if self.persist is not None:
            vals_p = []
            vals_n = []
            for n in n_qry:
                vals_p.append(n / n_tot)
                vals_n.append(n)

            self.persist.persist(self, vals_p + vals_n, iter, t)
