# -*- coding: utf-8 -*-
"""Contains utilities code."""

import bz2
import copy
import datetime
import gc
import gzip
import math
import multiprocessing
import os
# import pickle
# import dill as pickle
import cloudpickle as pickle
import random
import re
import platform
import psycopg2
import psycopg2.extras
import shutil
import sqlite3
import string
import sys
import time

from abc         import abstractmethod, ABC
from collections import namedtuple
from dotmap      import DotMap
from pathlib     import Path


# ----------------------------------------------------------------------------------------------------------------------
class Data(object):
    """Data utilities.
    """

    @staticmethod
    def rand_bin_lst(n):
        return [random.randint(0, 1) for b in range(n)]

    @staticmethod
    def rand_float_lst(l, u, n):
        return [random.uniform(l, u) for _ in range(n)]


# ----------------------------------------------------------------------------------------------------------------------
DB_FK = namedtuple('DB_FK', ['schema_from', 'schema_to'])

class DB(ABC):
    """Database interface base class.
    """

    VALID_CHARS = f'_{string.ascii_letters}{string.digits}'

    PATT_VALID_NAME = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')

    def __init__(self):
        self.conn = None

    def __del__(self):
        if self.conn:
            self.conn.close()

    def open_conn(self):
        self.conn = self.get_new_conn()

    @abstractmethod
    def get_new_conn(self):
        pass

    @staticmethod
    def blob2obj(b, do_decompress=True):
        if b is None:
            return None

        if do_decompress:
            return pickle.loads(gzip.decompress(b))
        else:
            return pickle.loads(str(b))

    @staticmethod
    def obj2blob(o, do_compress=True):
        if do_compress:
            return gzip.compress(pickle.dumps(o))  #, pickle.HIGHEST_PROTOCOL
        else:
            return pickle.dumps(o)  #, pickle.HIGHEST_PROTOCOL

    def exec(self, qry):
        with self.conn.cursor() as c:
            c.execute(qry)

    def exec_get(self, qry):
        with self.conn.cursor() as c:
            c.execute(qry)
            return c.fetchall()

    @abstractmethod
    def get_cols(self, schema, tbl):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_num(self, tbl, col, where=None):
        pass

    def get_row_cnt(self, tbl, where=None):
        return self.get_num(tbl, 'COUNT(*)', where)

    @abstractmethod
    def get_id(self, tbl, col='rowid', where=None):
        pass

    @abstractmethod
    def get_fk(self, schema, tbl, col_from):
        ''' Get the foreign key constraint for the specified table and column. '''

        pass

    @staticmethod
    def str_to_name(s, do_raise_on_empty=True):
        s = s.strip().lower()
        s = re.sub(r'( |-|,|[.])', '_', s)                # preserve select punctuation and white spaces as underscores
        s = ''.join(c for c in s if c in DB.VALID_CHARS)  # remove invalid characters
        s = re.sub(r'_+', '_', s)                         # compress multiple sequential underscores to one
        s = re.sub(r'^\d+', '', s)                        # leading digits
        s = re.sub(r'^_+', '', s)                         # leading underscores
        s = re.sub(r'_+$', '', s)                         # trailing underscores

        if len(s) == 0 and do_raise_on_empty:
            raise TypeError(f'The string provided translates into an empty database-compliant name: {s}')

        return f"'{s}'"  # quote in case it's a keyword

    @abstractmethod
    def str_to_type(s):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class PgDB(DB):
    """PostgreSQL database interface.
    """

    # def __init__(self, host, port, usr, pwd, db, cursor_factory=psycopg2.extras.NamedTupleCursor):
    def __init__(self, host, port, usr, pwd, db, cursor_factory=psycopg2.extras.DictCursor):
        super().__init__()

        self.host = host
        self.port = port
        self.usr  = usr
        self.pwd  = pwd
        self.db   = db

        self.cursor_factory = cursor_factory

    def get_num(self, tbl, col, where=None):
        where = '' if where is None else f' WHERE {where}'
        with self.conn.cursor() as c:
            c.execute(f'SELECT {col} FROM {tbl}{where}')
            row = c.fetchone()
            return row[0] if row else None

    def get_cols(self, schema, tbl):
        with self.conn.cursor() as c:
            # c.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{schema}' AND table_name = '{tbl}'")
            # c.execute(f"SELECT attname AS col FROM pg_attribute WHERE attrelid = '{schema}.{tbl}'::regclass AND attnum > 0 ORDER BY attnum;")
        #     c.execute(f"""
        #        select t.table_schema as schema_name,
        #        t.table_name as view_name,
        #        c.column_name,
        #        c.data_type
        #        from information_schema.tables t
        #     left join information_schema.columns c
        #               on t.table_schema = c.table_schema
        #               and t.table_name = c.table_name
        # where table_type = 'VIEW' AND t.table_schema = '{schema}' AND t.table_name = '{tbl}';
        #     """)
            # return [i[1] for i in c.fetchall()]

            c.execute(f"SELECT * FROM {schema+'.' if schema is not None else ''}{tbl} LIMIT 0")
            return [d[0] for d in c.description]

    def get_id(self, tbl, col='rowid', where=None):
        return self.get_num(conn, tbl, 'rowid', where)

    def get_fk(self, schema, tbl, col_from):
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT
                	tc.table_schema AS schema_from,
                	tc.table_name AS tbl_from,
                	kcu.column_name AS col_from,
                	ccu.table_schema AS schema_to,
                	ccu.table_name AS tbl_to,
                	ccu.column_name AS col_to,
                	tc.constraint_name AS fk_name
                FROM
                	information_schema.table_constraints AS tc
                	INNER JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name AND
                		tc.table_schema = kcu.table_schema
                	INNER JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name AND
                		ccu.table_schema = tc.table_schema
                WHERE
                	tc.constraint_type = 'FOREIGN KEY' AND
                	tc.table_schema = '{schema}' AND
                	tc.table_name = '{tbl}' AND
                	kcu.column_name = '{col_from}';
            """)
            r = c.fetchone()
            if r:
                return DB_FK(r.schema_from, r.tbl_from, r.col_from, r.schema_to, r.tbl_to, r.col_to, r.fk_name)
            return None

    def get_name(self):
        return self.db

    def get_new_conn(self):
        return psycopg2.connect(host=self.host, port=self.port, user=self.usr, password=self.pwd, database=self.db, cursor_factory=self.cursor_factory)

    @staticmethod
    def str_to_type(s):
        return {
            'int'   : 'integer',
            'float' : 'double precision',
            'str'   : 'text',
            'obj'   : 'blob'
        }.get(s, None)


# ----------------------------------------------------------------------------------------------------------------------
class SQLiteDB(DB):
    """SQLite3 database interface.
    """

    def __init__(self, fpath):
        super().__init__()

        self.fpath = fpath

    def get_num(self, tbl, col, where=None):
        where = '' if where is None else f' WHERE {where}'
        row = self.conn.execute(f'SELECT {col} FROM {tbl}{where}').fetchone()
        return row[0] if row else None

    def get_cols(self, schema, tbl):
        return [i[1] for i in self.conn.execute(f'PRAGMA table_info({tbl})')]

    def get_id(conn, tbl, col='rowid', where=None):
        return self.get_num(self, tbl, 'rowid', where)

    def get_fk(self, schema, tbl, col_from):
        with self.conn as c:
            for r in c.execute(f'PRAGMA foreign_key_list({tbl})').fetchall():
                if r['from'] == col_from:
                    # return DotMap(tbl_to=row['table'], col_to=row['to'])
                    return DB_FK(None, tbl, col_from, None, r['table'], r['to'], None)
            return None

    def get_name(self):
        return self.fpath

    def get_new_conn(self):
        FS.req_file(self.fpath, f'The database does not exist: {self.fpath}')

        conn = sqlite3.connect(self.fpath, check_same_thread=False)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode=WAL')  # PRAGMA journal_mode = DELETE
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def str_to_type(s):
        return {
            'int'   : 'INTEGER',
            'float' : 'REAL',
            'str'   : 'TEXT',
            'obj'   : 'BLOB'
        }.get(s, None)


# ----------------------------------------------------------------------------------------------------------------------
class Err(object):
    @staticmethod
    def type(arg, arg_name, type, can_be_none=False):
        '''
        Raises a TypeError exception if the argument isn't of the required type.  Returns 'True' which enables it to be
        used as a logical condition.
        '''

        if not can_be_none:
            if arg is None:
                raise TypeError("The argument '{}' cannot be None and has to be of type '{}'.".format(arg_name, type.__name__))
        else:
            if arg is not None and not isinstance(arg, type):
                raise TypeError("The argument '{}' has to be of type '{}'{}.".format(arg_name, type.__name__, ' (can be None too)'))

        return True


# ----------------------------------------------------------------------------------------------------------------------
class FS(object):
    """Filesystem utilities.
    """

    @staticmethod
    def bz2(fpath_src, fpath_dst=None, compress_lvl=9, do_del=False):
        # TODO: Add support for directories.

        if fpath_src.endswith('.bz2'): return  # already a BZ2 file

        # Compress to the same directory:
        if fpath_dst is None:
            fpath_dst = "{}.bz2".format(fpath_src)

        # Append the 'bz2' extension:
        if not fpath_dst.endswith('.bz2'):
            fpath_dst = '{}.bz2'.format(fpath_dst)

        # Compress:
        with open(fpath_src, 'rb') as fi, bz2.open(fpath_dst, 'wb', compresslevel=compress_lvl) as fo:
            shutil.copyfileobj(fi, fo)

        # Remove the source file:
        if do_del:
            os.remove(fpath_src)

    @staticmethod
    def bz2_decomp(fpath_src, fpath_dst=None, do_del=False):
        # TODO: Add support for directories.

        if not fpath_src.endswith('.bz2'): return  # not a BZ2 file

        # Decompress to the same directory:
        if fpath_dst is None:
            fpath_dst = "{}.bz2".format(fpath_src)

        if Path(fpath_dst).exists(): return  # destination name taken

        # Create the destination directory:
        dir_dst = os.path.dirname(fpath_dst)
        if not os.path.exists(dir_dst): os.makedirs(dir_dst)

        # Decompress:
        with bz2.open(fpath_src, 'rb') as fi, open(fpath_dst, 'wb') as fo:
            shutil.copyfileobj(fi, fo)

        # Remove the source file:
        if do_del:
            os.remove(fpath_src)

    @staticmethod
    def dir_mk(path):
        dpath = os.path.join(*path)
        if not os.path.exists(dpath): os.makedirs(dpath)
        return dpath

    @staticmethod
    def gz(fpath_src, fpath_dst=None, compress_lvl=9, do_del=False):
        # TODO: Add support for directories.

        if fpath_src.endswith('.gz'): return  # already a GZ file

        # Compress to the same directory:
        if fpath_dst is None:
            fpath_dst = "{}.gz".format(fpath_src)

        # Append the 'gz' extension:
        if not fpath_dst.endswith('.gz'):
            fpath_dst = '{}.gz'.format(fpath_dst)

        # Compress:
        with open(fpath_src, 'rb') as fi, gzip.open(fpath_dst, 'wb', compresslevel=compress_lvl) as fo:
            shutil.copyfileobj(fi, fo)

        # Remove the source file:
        if do_del:
            os.remove(fpath_src)

    @staticmethod
    def gz_decomp(fpath_src, fpath_dst=None, do_del=False):
        # TODO: Add support for directories.

        if not fpath_src.endswith('.gz'): return  # not a GZ file

        # Decompress to the same directory:
        if fpath_dst is None:
            fpath_dst = "{}.gz".format(fpath_src)

        if Path(fpath_dst).exists(): return  # destination name taken

        # Create the destination directory:
        dir_dst = os.path.dirname(fpath_dst)
        if not os.path.exists(dir_dst): os.makedirs(dir_dst)

        # Decompress:
        with gzip.open(fpath_src, 'rb') as fi, open(fpath_dst, 'wb') as fo:
            shutil.copyfileobj(fi, fo)

        # Remove the source file:
        if do_del:
            os.remove(fpath_src)

    @staticmethod
    def load_or_gen(fpath, fn_gen, name='data', is_verbose=False, hostname_gen=set()):
        '''
        If the 'fpath' file exists, it is ungzipped and unpickled.  If it does not exist, 'fn_gen()' is called and the
        result is pickled and gzipped in 'fpath' and returned.  Name can be redefined for customized progress messages.
        No filesystem interactions are made if 'fpath' is None.

        Generation of the data can be restricted to a set of machines designated by hostname.
        '''

        # Load:
        if fpath is not None and os.path.isfile(fpath):
            if is_verbose: print(f'Loading {name}... ', end='', flush=True)
            with gzip.GzipFile(fpath, 'rb') as f:
                gc.disable()
                data = pickle.load(f)
                gc.enable()
            if is_verbose: print('done.', flush=True)

        # Generate:
        else:
            if len(hostname_gen) > 0 and not platform.node() in hostname_gen:
                return None

            if is_verbose: print(f'Generating {name}... ', end='', flush=True)
            data = fn_gen()
            if is_verbose: print('done.', flush=True)

            if fpath is not None:
                if is_verbose: print(f'Saving {name}... ', end='')
                with gzip.GzipFile(fpath, 'wb') as f:
                    pickle.dump(data, f)
                if is_verbose: print('done.', flush=True)

        return data

    @staticmethod
    def load(fpath, mode='r'):
        # # TODO: Detect implied compression through file extension.
        # with open(fpath, mode) as f:
        #     return f.read()

        pass

    @staticmethod
    def load_bz2(fpath):
        with bz2.open(fpath, 'rb') as f:
            return f.read()

    @staticmethod
    def load_gz(fpath):
        with gzip.open(fpath, 'rb') as f:
            return f.read()

    @staticmethod
    def req_file(fpath, msg):
        if not os.path.isfile(fpath):
            raise ValueError(msg)

    @staticmethod
    def save(fpath, data, mode='w', compress_lvl=9):
        # # TODO: Detect implied compression through extension.
        # with open(fpath, mode) as f:
        #     f.write(data)

        pass

    @staticmethod
    def save_bz2(fpath, data, compress_lvl=9):
        if not fpath.endswith('.bz2'):
            fpath = '{}.bz2'.format(fpath)

        with bz2.open(fpath, 'wb', compresslevel=compress_lvl) as f:
            f.write(str.encode(data) if isinstance(data, str) else data)

    @staticmethod
    def save_gz(fpath, data, compress_lvl=9):
        if not fpath.endswith('.gz'):
            fpath = '{}.gz'.format(fpath)

        with gzip.open(fpath, 'wb', compresslevel=compress_lvl) as f:
            f.write(str.encode(data) if isinstance(data, str) else data)


# ----------------------------------------------------------------------------------------------------------------------
class MPCounter(object):
    """Fixes issues with locking of multiprocessing's Value object.

    Source: eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
    """

    def __init__(self, val_init=0):
        self.val  = multiprocessing.RawValue('i', val_init)
        self.lock = multiprocessing.Lock()

    def dec(self):
        with self.lock:
            self.val.value -= 1

    def inc(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


# ----------------------------------------------------------------------------------------------------------------------
class Size(object):
    """File size utilities.
    """

    PREFIX_BIN = { 'K': 1024, 'M': 1048576, 'G': 1073741824, 'T': 1099511627776, 'P': 1125899906842624, 'E': 1152921504606846976, 'Z': 1180591620717411303424, 'Y': 1208925819614629174706176 }
    PREFIX_DEC = { 'K': 1000, 'M': 1000000, 'G': 1000000000, 'T': 1000000000000, 'P': 1000000000000000, 'E': 1000000000000000000, 'Z': 1000000000000000000000, 'Y': 1000000000000000000000000 }

    @staticmethod
    def get_size(obj0):
        ''' https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python/30316760#30316760 '''

        from numbers import Number
        from collections import Set, Mapping, deque

        _seen_ids = set()

        def inner(obj):
            obj_id = id(obj)
            if obj_id in _seen_ids:
                return 0
            _seen_ids.add(obj_id)
            size = sys.getsizeof(obj)
            if isinstance(obj, (str, bytes, Number, range, bytearray)):
                pass # bypass remaining control flow and return
            elif isinstance(obj, (tuple, list, Set, deque)):
                size += sum(inner(i) for i in obj)
            elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
                size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
            # Check for custom object instances - may subclass above too
            if hasattr(obj, '__dict__'):
                size += inner(vars(obj))
            if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
                size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
            return size

        return inner(obj0)

    @staticmethod
    def b2h(b, do_bin=True, dec_places=1, do_ret_tuple=False):
        return __class__.bytes2human(b, do_bin, dec_places, do_ret_tuple)

    @staticmethod
    def bytes2human(b, do_bin=True, dec_places=1, do_ret_tuple=False):
        if do_bin:
            prefix = __class__.PREFIX_BIN
            unit   = 'B'
        else:
            prefix = __class__.PREFIX_DEC
            unit   = ''

        for p in reversed(prefix.keys()):
            if b >= prefix[p]:
                value = float(b) / prefix[p]
                if do_ret_tuple:
                    return (value, p)
                else:
                    return f'{value:.{dec_places}f}{p}'

        if do_ret_tuple:
            return (b, unit)
        else:
            return f'{b}{unit}'

    # @staticmethod
    # def b2h(b, do_ret_tuple=False):
    #     return __class__.bytes2human(b, do_ret_tuple)
    #
    # @staticmethod
    # def bytes2human(b, base=do_ret_tuple=False):
    #     symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    #     prefix = {}
    #
    #     for (i,s) in enumerate(symbols):
    #         prefix[s] = 1 << (i + 1) * 10
    #
    #     for s in reversed(symbols):
    #         if b >= prefix[s]:
    #             value = float(b) / prefix[s]
    #             if do_ret_tuple:
    #                 return (value, s)
    #             else:
    #                 return '{:.1f}{}'.format(value, s)
    #
    #     if do_ret_tuple:
    #         return (b, 'B')
    #     else:
    #         return '{}B'.format(b)


# ----------------------------------------------------------------------------------------------------------------------
class Str(object):
    """String utilities.
    """

    @staticmethod
    def float(s):
        '''
        Formats a string containing a floating point number to suppress scientific notation and strip trailing zeros.
        '''

        return '{:f}'.format(s).rstrip('0').rstrip('.')


# ----------------------------------------------------------------------------------------------------------------------
class Tee(object):
    """Script output forker.

    When instantiated, this class writes the output of the script to stdout and to a file at the same time (much like
    the UNIX command line utility by the same name).
    """

    def __init__(self, fname, fmode='a'):
        self.file = open(fname, fmode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        self.end()

    def end(self):
        sys.stdout = self.stdout

        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def flush(self):
        self.file.flush()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)


# ----------------------------------------------------------------------------------------------------------------------
class Time(object):
    """Time utilities.

    POSIX (or UNIX) time (i.e., Jan 1, 1970) is the point of reference for this class.

    All timestamps and datetime differences default to milliseconds (as opposed to seconds, which is what the
    ``datetime`` module seems to prefer).
    """

    DOTW_NUM2STR = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']  # day-of-the-week number-to-string list

    POSIX_TS = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=0)  # POSIX timestamp; datetime.datetime.utcfromtimestamp(0)
    POSIX_DT = datetime.datetime(1970, 1, 1)                                  # POSIX datetime

    PATT_STR_DUR = re.compile('^(\d+)\s*(\w+)$')

    MS = DotMap({
        'ys'  : 10 ** -21,
        'zs'  : 10 ** -18,
        'as'  : 10 ** -15,
        'fs'  : 10 ** -12,
        'ps'  : 10 **  -9,
        'ns'  : 10 **  -6,
        'mus' : 10 **  -3,
        'ms'  : 1,
        's'   : 1000,
        'm'   : 1000 * 60,
        'h'   : 1000 * 60 * 60,
        'd'   : 1000 * 60 * 60 * 24,
        'w'   : 1000 * 60 * 60 * 24 * 7,
        'M'   : 1000 * 60 * 60 * 24 * 365 / 12,
        'y'   : 1000 * 60 * 60 * 24 * 365
    })

    @staticmethod
    def day_of_year_to_dt(year, day, format=None):
        dt = (datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1))
        if format is None:
            return dt
        else:
            return dt.strftime(format)

    @staticmethod
    def dt():
        ''' Datetime now. '''

        return datetime.datetime.now()

    @staticmethod
    def dur2ms(s):
        m = __class__.PATT_STR_DUR.match(s)

        # Extract the parts:
        if len(m.groups()) != 2:
            raise ValueError(f'Incorrect duration specification: {s}')

        # The number part:
        try:
            n = int(m[1])
        except:
            raise ValueError(f'Incorrect duration number: {s}')

        # The unit part:
        if m[2] not in __class__.MS:
            raise ValueError(f'Incorrect duration unit: {m[2]}')
        ms = __class__.MS[m[2]]

        return n * ms

    @staticmethod
    def ts(do_ms=True):
        """Timestamp now."""

        if do_ms: return (datetime.datetime.now() - Time.POSIX_DT).total_seconds() * 1000
        else:     return (datetime.datetime.now() - Time.POSIX_DT).total_seconds()

    @staticmethod
    def ts_sec(sec, do_ms=True):
        """Timestamp at the specified number of seconds."""

        if do_ms: return datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec) * 1000
        else:     return datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec)

    @staticmethod
    def dt2ts(dt, do_ms=True):
        """Datetime to timestamp."""

        if do_ms: return (dt - Time.POSIX_TS).total_seconds() * 1000
        else:     return (dt - Time.POSIX_TS).total_seconds()

    @staticmethod
    def ts2dt(dt, is_ts_ms=True):
        """Timestamp to datetime."""

        if is_ts_ms: return datetime.datetime.utcfromtimestamp(dt / 1e3)
        else:        return datetime.datetime.utcfromtimestamp(dt)

    @staticmethod
    def diff(dt_0, dt_1, do_ms=True):
        """Difference between two datetime objects ('dt_0' being the earlier one). There is no point in providing a 'diff'
        method for timestamps because being floats they can simply be subtracted."""

        if do_ms: return round((dt_1 - dt_0).total_seconds() * 1000, 3)
        else:     return       (dt_1 - dt_0).total_seconds()

    @staticmethod
    def diffs(dt_0, dt_1, do_ms=True):
        """Casts the return of :meth:`~pram.util.Time.diff` to string.

        This method is useful because it guarantees a proper formatting of the resulting float (i.e., no scientific
        notation and trailing zeros stripped) making it a good choice for the purpose of display, log, etc."""

        return str(Time.diff(dt_0, dt_1, do_ms))  # TODO: Use Str.float() instead of str() if this doesn't work.

    @staticmethod
    def dotw(dt):
        """String short name of day-of-the-week."""

        return Time.DOTW_NUM2STR[int(dt.strftime('%w'))]

    @staticmethod
    def tsdiff2human(ts_diff, do_print_ms=True):
        """Convert timestamp difference to 'D days, HH:MM:SS.FFF'."""

        ms, s = math.modf(round(ts_diff) / 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        ms = ms * 1000

        if d == 0:
            if do_print_ms:
                return f'{int(h):02}:{int(m):02}:{int(s):02}.{int(ms):03}'
            else:
                return f'{int(h):02}:{int(m):02}:{int(s):02}'
        else:
            if do_print_ms:
                return f'{int(d):d}d {int(h):02}:{int(m):02}:{int(s):02}.{int(ms):03}'
            else:
                return f'{int(d):d}d {int(h):02}:{int(m):02}:{int(s):02}'

    @staticmethod
    def sec2time(sec, n_msec=0):
        """Convert seconds to 'D days, HH:MM:SS.FFF'

        References:
            - stackoverflow.com/questions/775049/python-time-seconds-to-hms
            - humanfriendly.readthedocs.io/en/latest/#humanfriendly.format_timespan
            - stackoverflow.com/questions/41635547/convert-python-datetime-to-timestamp-in-milliseconds/41635888
            - datetime.datetime.fromtimestamp(self.pop.sim.last_iter.t).strftime('%H:%M:%S.%f')
        """

        if hasattr(sec,'__len__'): return [sec2time(s) for s in sec]

        m, s = divmod(sec, 60)
        h, m = divmod(m,   60)
        d, h = divmod(h,   24)

        if n_msec > 0:
            pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
        else:
            pattern = r'%02d:%02d:%02d'

        if d == 0:
            return pattern % (h, m, s)

        return ('%d days, ' + pattern) % (d, h, m, s)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('--- Size: Binary ---')
    print(Size.b2h(154))
    print(Size.b2h(1540))
    print(Size.b2h(15400))
    print(Size.b2h(154000))
    print(Size.b2h(1540000))
    print(Size.b2h(15400000))
    print(Size.b2h(154000000))
    print(Size.b2h(1540000000))
    print(Size.b2h(15400000000))
    print(Size.b2h(154000000000))
    print(Size.b2h(1540000000000))
    print(Size.b2h(1540000000000, dec_places=0))
    print(Size.b2h(1540000000000, dec_places=2))

    print(*Size.b2h(154, True))
    print('{:.1f}{}'.format(*Size.b2h(154, do_ret_tuple=True)))

    print('--- Size: Decimal (SI) ---')
    print(Size.b2h(154, False))
    print(Size.b2h(1540, False))
    print(Size.b2h(15400, False))
    print(Size.b2h(154000, False))
    print(Size.b2h(1540000, False))
    print(Size.b2h(15400000, False))
    print(Size.b2h(154000000, False))
    print(Size.b2h(1540000000, False))
    print(Size.b2h(15400000000, False))
    print(Size.b2h(154000000000, False))
    print(Size.b2h(1540000000000, False))
    print(Size.b2h(1540000000000, False, dec_places=0))
    print(Size.b2h(1540000000000, False, dec_places=2))

    print('\n--- DB ---')
    print(DB.str_to_name('one fine column name'))
    print(DB.str_to_name('one fine column name 3'))
    print(DB.str_to_name('one fine column name?,.-'))
    print(DB.str_to_name('1   fine column name'))
    print(DB.str_to_name('123 fine column name'))
    print(DB.str_to_name('123 fine column ... name'))
    print(DB.str_to_name('123 /a77&<>.,":[)(]'))

    print('\n--- Time ---')
    print(Time.dur2ms('7ms'))
    print(Time.dur2ms('1s'))
    print(Time.dur2ms('1d'))
    print(Time.dur2ms('25h'))
    print(Time.dur2ms('2d'))
    print(Time.dur2ms('10m'))
    print(Time.dur2ms('10M'))
