''' Read the schema of a SQLite DB. '''

import os
import sqlite3

fpath_db = os.path.join(os.path.dirname(__file__), '..', 'db', 'allegheny-students.sqlite3')

schema = {}

conn = sqlite3.connect(fpath_db, check_same_thread=False)
conn.execute('PRAGMA foreign_keys = ON')
conn.execute('PRAGMA journal_mode=WAL')
conn.row_factory = sqlite3.Row

with conn as c:
    schema = {r[0]: {} for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}  # sql

    for tbl in schema.keys():
        schema[tbl]['cols'] = { r['name']: { 'type': r['type'] } for r in c.execute(f"PRAGMA table_info('{tbl}')").fetchall()}  # cid,name,type,notnull,dflt_value,pk

        for row in c.execute(f'PRAGMA foreign_key_list({tbl})').fetchall():  # id,seq,tbl,from,to,on_update,on_delete,match
            schema[tbl]['cols'][row['from']]['fk'] = { 'tbl': row['table'], 'col': row['to'] }

print(schema)
