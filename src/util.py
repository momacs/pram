#
# TODO
#     Python 3.3
#         datetime.timestamp()  # the number of seconds from 1970-01-01 UTC
#
# ----------------------------------------------------------------------------------------------------------------------

import bz2
import copy
import datetime
import gzip
import multiprocessing
import os
import random
import shutil
import sys
import time

from pathlib import Path


# ----------------------------------------------------------------------------------------------------------------------
class Data(object):
    @staticmethod
    def rand_bin_lst(n):
        return [random.randint(0, 1) for b in range(n)]

    @staticmethod
    def rand_float_lst(l, u, n):
        return [random.uniform(l, u) for _ in range(n)]


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

    # TODO: Add support for directories.
    @staticmethod
    def bz2(fpath_src, fpath_dst=None, compress_lvl=9, do_del=False):
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

    # TODO: Add support for directories.
    @staticmethod
    def bz2_decomp(fpath_src, fpath_dst=None, do_del=False):
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

    # TODO: Add support for directories.
    @staticmethod
    def gz(fpath_src, fpath_dst=None, compress_lvl=9, do_del=False):
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

    # TODO: Add support for directories.
    @staticmethod
    def gz_decomp(fpath_src, fpath_dst=None, do_del=False):
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

    # ------------------------------------------------------------------------------------------------------------------
    # @staticmethod
    # def load(fpath, mode='r'):
    #     # TODO: Detect implied compression through file extension.
    #     with open(fpath, mode) as f:
    #         return f.read()

    @staticmethod
    def load_bz2(fpath):
        with bz2.open(fpath, 'rb') as f:
            return f.read()

    @staticmethod
    def load_gz(fpath):
        with gzip.open(fpath, 'rb') as f:
            return f.read()

    # ------------------------------------------------------------------------------------------------------------------
    # @staticmethod
    # def save(fpath, data, mode='w', compress_lvl=9):
    #     # TODO: Detect implied compression through extension.
    #     with open(fpath, mode) as f:
    #         f.write(data)

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
class Hash(object):
    # TODO: Implement. (https://stackoverflow.com/questions/5884066/hashing-a-dictionary)
	pass


# ----------------------------------------------------------------------------------------------------------------------
class MPCounter(object):
    '''
    Fixes issues with locking of multiprocessing's Value object.

    Source: eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
    '''

    def __init__(self, val_init=0):
        self.val  = multiprocessing.RawValue('i', val_init)
        self.lock = multiprocessing.Lock()

    def dec(self):
        with self.lock: self.val.value -= 1

    def inc(self):
        with self.lock: self.val.value += 1

    def value(self):
        with self.lock: return self.val.value


# ----------------------------------------------------------------------------------------------------------------------
class Size(object):

    @staticmethod
    def bytes2human(b, do_ret_tuple=False):
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        prefix = {}

        for i, s in enumerate(symbols):
            prefix[s] = 1 << (i + 1) * 10

        for s in reversed(symbols):
            if b >= prefix[s]:
                value = float(b) / prefix[s]
                if do_ret_tuple:
                    return (value, s)
                else:
                    return '{:.1f}{}'.format(value, s)

        if do_ret_tuple:
            return (b, 'B')
        else:
            return '{}B'.format(b)


# ----------------------------------------------------------------------------------------------------------------------
class Str(object):

    @staticmethod
    def float(s):
        '''
        Formats a string containing a floating point number to suppress scientific notation and strip trailing zeros.
        '''

        return '{:f}'.format(s).rstrip('0').rstrip('.')


# ----------------------------------------------------------------------------------------------------------------------
class Tee(object):
    '''
    Script output forker.

    When instantiated, this class writes the output of the script to stdout and to a file at the same time (much like
    the UNIX command line utility by the same name).
    '''

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
    '''
    POSIX (or UNIX) time (i.e., Jan 1, 1970) is the point of reference for this class.

    All timestamps and datetime differences default to milliseconds (as opposed to seconds, which is what the
    'datetime' module seems to prefer).
    '''

    DOTW_NUM2STR = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']  # day-of-the-week number-to-string list

    POSIX_TS = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=0)  # POSIX timestamp; datetime.datetime.utcfromtimestamp(0)
    POSIX_DT = datetime.datetime(1970, 1, 1)                                  # POSIX datetime

    @staticmethod
    def dt():
        ''' Datetime now. '''

        return datetime.datetime.now()

    @staticmethod
    def ts(do_ms=True):
        ''' Timestamp now. '''

        if do_ms: return (datetime.datetime.now() - Time.POSIX_DT).total_seconds() * 1000
        else:     return (datetime.datetime.now() - Time.POSIX_DT).total_seconds()

    @staticmethod
    def ts_sec(sec, do_ms=True):
        ''' Timestamp at the specified number of seconds. '''

        if do_ms: return datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec) * 1000
        else:     return datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec)

    @staticmethod
    def dt2ts(dt, do_ms=True):
        ''' Datetime to timestamp. '''

        if do_ms: return (dt - Time.POSIX_TS).total_seconds() * 1000
        else:     return (dt - Time.POSIX_TS).total_seconds()

    @staticmethod
    def ts2dt(dt, is_ts_ms=True):
        ''' Timestamp to datetime. '''

        if is_ts_ms: return datetime.datetime.utcfromtimestamp(dt / 1e3)
        else:        return datetime.datetime.utcfromtimestamp(dt)

    @staticmethod
    def diff(dt_0, dt_1, do_ms=True):
        '''
        Difference between two datetime objects ('dt_0' being the earlier one). There is no point in providing a 'diff'
        method for timestamps because being floats they can simply be subtracted.
        '''

        if do_ms: return round((dt_1 - dt_0).total_seconds() * 1000, 3)
        else:     return       (dt_1 - dt_0).total_seconds()

    @staticmethod
    def diffs(dt_0, dt_1, do_ms=True):
        '''
        Casts the return of 'diff()' to string. This method is useful because it guarantees a proper formatting of the
        resulting float (i.e., no scientific notation and trailing zeros stripped) making it a good choice for the
        purpose of display, log, etc.
        '''

        return str(Time.diff(dt_0, dt_1, do_ms))  # TODO: Use Str.float() instead of str() if this doesn't work.

    @staticmethod
    def dotw(dt):
        ''' String short name of day-of-the-week. '''

        return Time.DOTW_NUM2STR[int(dt.strftime('%w'))]

    @staticmethod
    def sec2time(sec, n_msec=0):
        '''
        Convert seconds to 'D days, HH:MM:SS.FFF'

        References
            stackoverflow.com/questions/775049/python-time-seconds-to-hms
            humanfriendly.readthedocs.io/en/latest/#humanfriendly.format_timespan
        '''

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
    print(Size.bytes2human(154))
    print(Size.bytes2human(1540))
    print(Size.bytes2human(15400))
    print(Size.bytes2human(154000))
    print(Size.bytes2human(1540000))
    print(Size.bytes2human(15400000))
    print(Size.bytes2human(154000000))
    print(Size.bytes2human(1540000000))
    print(Size.bytes2human(15400000000))
    print(Size.bytes2human(154000000000))
    print(Size.bytes2human(1540000000000))

    print(*Size.bytes2human(154, True))
    print('{:.1f}{}'.format(*Size.bytes2human(154, True)))
