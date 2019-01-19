import os
import sys
from inspect import getsourcefile

__file__ = os.path.abspath(getsourcefile(lambda: None))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from util import Str, Time


class Log(object):
    '''
    A single log manages opening and closing of the destination file, adding headers, and adding records. It also
    handles timestamping of all records.  Note that a record timestamp denotes the time of adding that record to the
    log file, not the time related to the message itself.
    '''

    DEBUG_LVL = 1

    DELIM = "\t"

    def __init__(self, dt_0, path, do_append=True, do_ts=True, delim=DELIM):
        self.dt_0 = dt_0
        self.ts_0 = Time.dt2ts(dt_0)

        self.path = path
        self.do_ts = do_ts
        self.delim = delim
        self.col_n = None  # inferred from header and used for warning when logging

        self.f = open(path, 'a' if do_append else 'w')

    def __del__(self):
        if self.f is None: return

        self.f.close()

    def close(self):
        if self.f is None: return

        self.f.close()

    def put(self, msg):
        if self.f is None: return

        if self.do_ts:
            ts_1 = Time.ts()
            dt_1 = Time.ts2dt(ts_1)

            res = [Str.float(round(ts_1)), str(round(ts_1 - self.ts_0, 3)), str(dt_1)]
        else:
            res = []

        if isinstance(msg, str):  # noqa  # Python 2.7: isinstance(msg, basestring)
            res.append(msg)
        else:
            res.extend(msg)

        self.f.write(self.delim.join(res))
        self.f.write("\n")

        if (self.DEBUG_LVL >= 1) and (self.col_n is not None) and (len(res) != self.col_n):
            print("Log warning: Incorrect number of columns (log: '%s'; expecting %d; recieved %d)" % (self.path, self.col_n, len(res)))

    def put_head(self, cols):
        if self.do_ts:
            cols.insert(0, "rec_ts")      # timestamp
            cols.insert(1, "rec_ts_exp")  # timestamp (within the experiment)
            cols.insert(2, "rec_dt")      # datetime

        self.col_n = len(cols)

        self.f.write(self.delim.join(cols))
        self.f.write("\n")
