import os
import sqlite3

from abc         import abstractmethod, ABC
from attr        import attrs, attrib, converters, validators
from collections import namedtuple
from enum        import IntEnum, Flag, auto

from .entity import GroupQry
from .util   import DB


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistanceMode(IntEnum):
    APPEND    = 1
    OVERWRITE = 2


@attrs(slots=True)
class ProbePersistanceDBItem(object):
    name    : str  = attrib()
    ins_qry : str  = attrib()
    ins_val : list = attrib(factory=list, converter=converters.default_if_none(factory=list))


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistance(ABC):
    VAR_NAME_KEYWORD = ['id', 'ts', 'i', 't']

    @abstractmethod
    def persist(self):
        pass

    @abstractmethod
    def reg_probe(self, probe):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class ProbePersistanceFS(ProbePersistance):
    '''
    Persists probe results to a filesystem.
    '''

    def __init__(self, path, mode=ProbePersistanceMode.APPEND):
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
    '''
    Persists probe results to a database.

    Database inserts are cached and flushed when the buffer fills up.  The buffer size is under the user's control.
    The default size of 16 decreases the time spent persisting data to the database about 15 times.  Higher values
    can be used to achieve even better results.  The cost of the memory used needs to be considered as well.
    '''

    FLUSH_EVERY = 16

    def __init__(self, fpath, mode=ProbePersistanceMode.APPEND, flush_every=FLUSH_EVERY):
        self.probes = {}
        self.conn = None
        self.fpath = fpath
        self.mode = mode
        self.flush_every = flush_every

        if os.path.isfile(self.fpath) and mode == ProbePersistanceMode.OVERWRITE:
            os.remove(self.fpath)

        self.conn_open()

    def __del__(self):
        self.conn_close()

    def conn_close(self):
        if self.conn is None:
            return

        self.flush()
        self.conn.close()
        self.conn = None

    def conn_open(self):
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(self.fpath, check_same_thread=False)

    def flush(self):
        with self.conn as c:
            for p in self.probes.values():
                if len(p.ins_val) > 0:
                    c.executemany(p.ins_qry, p.ins_val)
                    p.ins_val = []

    def persist(self, probe, vals, iter, t):
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

    def reg_probe(self, probe, do_overwrite=False):
        name_db = DB.str_to_name(probe.name)

        if name_db in self.probes and not do_overwrite:
            raise ValueError(f"The probe '{probe.name}' has already been registered.")

        # Create the table:
        cols = [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL',  # (datetime('now'))
            'i INTEGER NOT NULL',
            't REAL NOT NULL'
        ] + \
        [ f'{c.name} {DB.str_to_type(c.type)}' for c in probe.consts ] + \
        [ f'{v.name} {DB.str_to_type(v.type)}' for v in probe.vars ]

        with self.conn as c:
            c.execute(f'CREATE TABLE IF NOT EXISTS {name_db} (' + ','.join(cols) + ');')

        # Store probe:
        ins_qry = \
            f"INSERT INTO {name_db} " + \
            f"({','.join(['i','t'] + [c.name for c in probe.consts] + [v.name for v in probe.vars])}) " + \
            f"VALUES ({','.join(['?'] * (len(cols) - 2))})"

        self.probes[name_db] = ProbePersistanceDBItem(name_db, ins_qry)


# ----------------------------------------------------------------------------------------------------------------------
class ProbeMsgMode(Flag):
    DISP  = auto()  # display messages
    CUMUL = auto()  # hold messages in the buffer


class Probe(ABC):
    Var   = namedtuple('Var', ['name', 'type'])
    Const = namedtuple('Const', ['name', 'type', 'val'])

    def __init__(self, name, msg_mode=0, pop=None, memo=None):
        self.name = name
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
    def __init__(self, name, queries, qry_tot=None, var_names=None, consts=None, persistance=None, msg_mode=0, pop=None, memo=None):
        '''
        queries: iterable of GroupQry
        '''

        super().__init__(name, msg_mode, pop, memo)

        self.queries = queries
        self.qry_tot = qry_tot
        self.consts = None
        self.vars = []
        self.persistance = None

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
                if vn in ProbePersistance.VAR_NAME_KEYWORD:
                    raise ValueError(f"The following variable names are restricted: {ProbePersistance.VAR_NAME_KEYWORD}")

                vn_db = DB.str_to_name(vn)
                if vn_db in vn_db_used:
                    raise ValueError(f"Variable name error: Name '{vn}' translates into a database name '{vn_db}' which already exists.")

                vn_db_used.add(vn_db)
                self.vars.append(Probe.Var(vn_db, 'float'))

        self.set_consts(consts)
        self.set_persistance(persistance)

    @classmethod
    def by_attr(cls, probe_name, attr_name, attr_values, qry_tot=None, var_names=None, consts=None, persistance=None, msg_mode=0, pop=None, memo=None):
        ''' Generates QueryGrp objects automatically for the attribute name and values specified. '''

        return cls(probe_name, [GroupQry(attr={ attr_name: v }) for v in attr_values], qry_tot, var_names, consts, persistance, msg_mode, pop, memo)

    @classmethod
    def by_rel(cls, probe_name, rel_name, rel_values, qry_tot=None, var_names=None, consts=None, persistance=None, msg_mode=0, pop=None, memo=None):
        ''' Generates QueryGrp objects automatically for the relation name and values specified. '''

        return cls(probe_name, [GroupQry(rel={ rel_name: v }) for v in rel_values], qry_tot, var_names, consts, persistance, msg_mode, pop, memo)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return 'Probe  name: {:16}  query-cnt: {:>3}'.format(self.name, len(self.queries))

    def run(self, iter, t):
        if self.msg_mode != 0 or self.persistance is not None:
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
        if self.persistance is not None:
            vals_p = []
            vals_n = []
            for n in n_qry:
                vals_p.append(n / n_tot)
                vals_n.append(n)

            self.persistance.persist(self, vals_p + vals_n, iter, t)

    def set_consts(self, consts):
        consts = consts or []

        cn_db_used = set()  # to identify duplicates

        for c in consts:
            if c.name in ProbePersistance.VAR_NAME_KEYWORD:
                raise ValueError(f"The following constant names are restricted: {ProbePersistance.VAR_NAME_KEYWORD}")

            cn_db = DB.str_to_name(c.name)
            if cn_db in cn_db_used:
                raise ValueError(f"Variable name error: Name '{c.name}' translates into a database name '{cn_db}' which already exists.")

            cn_db_used.add(cn_db)

        self.consts = consts or []

    def set_persistance(self, persistance):
        if self.persistance == persistance:
            return

        self.persistance = persistance
        if self.persistance is not None:
            self.persistance.reg_probe(self)
