import os
import log


class LogMan(object):
    """Log manager.

    Log manager cummulates logs used throughout the application.  Thanks to that, objects in the application don't need
    to keep track of logs by themselves and can instead use this central repository to record events and information in
    approprioate files.
    """

    FILE_EXT = "txt"
    DELIM = "\t"

    def __init__(self, dt_0, dir_root, ext=FILE_EXT, do_ts=True, delim=DELIM):
        self.dt_0 = dt_0
        self.dir_root = dir_root
        self.ext = ext
        self.do_ts = do_ts
        self.delim = delim

        self.logs = {}

    def __del__(self):
        for (k, l) in self.logs.items():
            l.close()

    def close(self, name):
        if name not in self.logs: return

        self.logs[name].close()

    def open(self, fdir, fname, name=None, do_append=True, do_ts=None, delim=None):
        if name in self.logs: return

        if name  is None: name = fname
        if do_ts is None: do_ts = self.do_ts
        if delim is None: delim = self.delim

        l = log.Log(self.dt_0, os.path.join(self.dir_root, fdir, "%s.%s" % (fname, self.ext)), do_append, do_ts, self.delim)
        self.logs[name] = l

        return l

    def put(self, name, msg):
        if name not in self.logs is None: return

        self.logs[name].put(msg)

    def put_head(self, name, cols):
        if name not in self.logs is None: return

        self.logs[name].put_head(cols)
