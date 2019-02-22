import csv
import os
import pandas as pd
import re
import sqlite3

from collections import namedtuple


class FilesToDB(object):
    '''
    Generates a SQLite database based on a set of CSV files.

    The CSV separator character is infered.  Missing values strings need to be provided.  Data is assumed to have been
    clean beforehand.
    '''

    PATT_VALID_DB_NAME = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')
    TBL_TMP_SUFF = '__tmp'

    File = namedtuple('File', ['path', 'missing_values', 'tbl', 'refs'])
    Ref  = namedtuple('Ref', ['src_col', 'dst_tbl', 'dst_col'])

    def __init__(self, fpath_db, dpath_files):
        self.conn = None
        self.fpath_db = fpath_db
        self.dpath_files = dpath_files

        self.files = []

        if not os.path.isdir(self.dpath_files):
            print('The specified files path is not a directory: {}'.format(self.dpath_files))
            raise ValueError()

    def __del__(self):
        self.conn_close()

    # ------------------------------------------------------------------------------------------------------------------
    def add_file(self, fname, na_values=None, refs=[]):
        # Validate table name:
        tbl = os.path.splitext(fname)[0]
        if FilesToDB.PATT_VALID_DB_NAME.fullmatch(os.path.basename(tbl)) is None:
            print('The filename does not translate into a valid SQL table name: {}'.format(fname))
            raise ValueError()

        # Validate file path:
        fpath = os.path.join(self.dpath_files, fname)
        if not os.path.isfile(fpath):
            print('The specified file does not exist: {}'.format(fpath))
            raise ValueError()

        # Add:
        self.files.append(self.File(fpath, na_values, tbl, refs))

        return self

    # ------------------------------------------------------------------------------------------------------------------
    def conn_close(self):
        if self.conn is None:
            return

        self.conn.close()
        self.conn = None

        return self

    def conn_open(self, fpath_db=None):
        ''' Connects to the designated database.  The existing connection to a database is closed beforehand. '''

        if fpath_db is None:
            return

        if (self.conn is not None):
            self.conn_close()

        self.fpath_db = fpath_db

        self.conn = sqlite3.connect(self.fpath_db, check_same_thread=False)
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.conn.execute('PRAGMA journal_mode=WAL')  # PRAGMA journal_mode = DELETE

    # ------------------------------------------------------------------------------------------------------------------
    def proc_file_data(self, c, file):
        '''
        To save ourselves some coding, we use 'pandas'.  The price to pay is elsewhere, when adding references due to
        the necessity of copying data to a temporary table.
        '''

        print('    File: {}'.format(os.path.basename(file.path)))

        with open(file.path, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024), delimiters='\t,')
            print('        Sep   : \'{}\''.format(dialect.delimiter))
            print('        NA    : {}'.format(file.missing_values))

        df = pd.read_csv(file.path, sep=dialect.delimiter, na_values=file.missing_values)
        df.to_sql(file.tbl, c, if_exists='fail', index=False)

        print('        Table : {}'.format(file.tbl))
        print('        Rows  : {}'.format(c.execute('SELECT COUNT(*) FROM {}'.format(file.tbl)).fetchone()[0]))

    # ------------------------------------------------------------------------------------------------------------------
    def proc_file_refs(self, c, file):
        '''
        SQLite doesn't support adding constraints to existing tables so we need to:

            1. Create a temporary table with the constraints.
            2. Copy the data from the original table to the temporary table.
            3. Drop the original.
            4. Rnaming the temporary.

        References
            https://stackoverflow.com/questions/1884818/how-do-i-add-a-foreign-key-to-an-existing-sqlite-table
            https://www.sqlite.org/omitted.html
        '''

        if len(file.refs) == 0:
            return

        print('    Table: {}'.format(os.path.basename(file.tbl)))
        print('        Count: {}'.format(len(file.refs)))

        tbl_tmp = file.tbl + self.TBL_TMP_SUFF

        # Construct the temporary table DDL:
        sql = c.execute('SELECT sql FROM sqlite_master WHERE type="table" AND name = "{}"'.format(file.tbl)).fetchone()[0].split('\n')[:-1]  # -1 removes the trailing ')'
        sql[0] = sql[0].replace(file.tbl, tbl_tmp)
        sql[1] = '  ' + sql[1]
        sql[-1] = sql[-1] + ','
        for (i,r) in enumerate(file.refs):
            sql.append('  CONSTRAINT fk__{0}__{2} FOREIGN KEY ({1}) REFERENCES {2} ({3}) ON UPDATE CASCADE ON DELETE CASCADE{4}'.format(file.tbl, r.src_col, r.dst_tbl, r.dst_col, ',' if i < (len(file.refs) - 1) else ''))
        sql.append(')')

        # Do the work:
        c.execute(''.join(sql))
        c.execute('INSERT INTO {} SELECT * FROM {}'.format(tbl_tmp, file.tbl))
        c.execute('DROP TABLE {}'.format(file.tbl))
        c.execute('ALTER TABLE {} RENAME TO {}'.format(tbl_tmp, file.tbl))

    # ------------------------------------------------------------------------------------------------------------------
    def run(self, do_del=False):
        if os.path.isfile(self.fpath_db):
            if do_del:
                os.remove(self.fpath_db)
                print('Removing existing database file: {}'.format(self.fpath_db))
            else:
                print('The database already exists: {}'.format(self.fpath_db))
                raise ValueError()

        self.conn_open(self.fpath_db)

        with self.conn as c:
            print('Importing data')
            for f in self.files:
                self.proc_file_data(c,f)

            print('Adding references')
            c.execute('PRAGMA foreign_keys=OFF')
            for f in self.files:
                self.proc_file_refs(c,f)
            c.execute('PRAGMA foreign_keys=ON')

        self.conn.execute('VACUUM')
        self.conn_close()

        print('Done.')

        return self


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    dpath_res   = os.path.join(os.sep, 'Volumes', 'd', 'pitt', 'sci', 'pram', 'res', 'fred')
    dpath_files = os.path.join(dpath_res, '2019 02 05 - Flu Transmission Model', '42003')
    fpath_db    = os.path.join(dpath_res, 'flu.sqlite3')

    na = ['X']  # missing values strings

    try:
        (FilesToDB(fpath_db, dpath_files).
            add_file('gq_people.txt',  na, refs=[
                FilesToDB.Ref('sp_gq_id', 'gq', 'sp_id')
            ]).
            add_file('gq.txt',         na).
            add_file('hospitals.txt',  na).
            add_file('households.txt', na).
            add_file('people.txt',     na, refs=[
                FilesToDB.Ref('sp_hh_id',  'households', 'sp_id'),
                FilesToDB.Ref('school_id', 'school',     'sp_id'),
                FilesToDB.Ref('work_id',   'workplaces', 'sp_id')
            ]).
            add_file('schools.txt',    na).
            add_file('workplaces.txt', na).
            run(True)
        )
    except ValueError:
        pass
