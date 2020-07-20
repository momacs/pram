# -*- coding: utf-8 -*-

import psycopg2
import psycopg2.extras

from pram.util import PgDB


__all__ = ['LocaleDB']


# ----------------------------------------------------------------------------------------------------------------------
class LocaleDB(object):
    def __init__(self, db_host, db_port, db_usr, db_pwd, db_name):
        self.db = PgDB(db_host, db_port, db_usr, db_pwd, db_name)
        self.db.open_conn()

    def get_pop(self):
        return self.db.get_row_cnt('person_view')

    def ls_geo_co(self, statefp10):
        return self.db.exec_get(f"SELECT gid, statefp10, countyfp10, geoid10, name10, namelsad10 FROM geo.co WHERE statefp10 = '{statefp10}' ORDER BY geoid10;")

    def ls_geo_st(self):
        return self.db.exec_get('SELECT gid, statefp10, geoid10, stusps10, name10 FROM geo.st ORDER BY geoid10;')

    def set_pop_view_household(self, stcotrbg):
        self.db.exec(f"""
            DROP VIEW IF EXISTS person_view;
            CREATE OR REPLACE TEMP VIEW person_view AS
            SELECT p.*
            FROM pop.person AS p
            INNER JOIN pop.household AS h ON p.household_id = h.id
            WHERE h.stcotrbg LIKE '{stcotrbg}%';
        """)

    def set_pop_view_household_geo(self, stcotrbg, geo_tbl):
        self.db.exec(f"""
            DROP VIEW IF EXISTS person_view;
            CREATE OR REPLACE TEMP VIEW person_view AS
            SELECT p.*, g.gid AS household_geo_id
            FROM pop.person AS p
            INNER JOIN pop.household AS h ON p.household_id = h.id
            INNER JOIN geo.{geo_tbl} AS g ON ST_Contains(g.geom, h.coords)
            WHERE h.stcotrbg LIKE '{stcotrbg}%';
        """)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    db = LocaleDB('localhost', 5432, 'tomek', os.environ.get('POSTGRESQL_PWD'), 'c19')  # export POSTGRESQL_PWD='...'

    print(db.ls_geo_st())
    print(db.ls_geo_co('42'))

    db.set_pop_view_household('01')
    print(db.get_pop())

    db.set_pop_view_household('02')
    print(db.get_pop())

    db.set_pop_view_household('01001')
    print(db.get_pop())
    print(db.db.exec_get('SELECT COUNT(*) FROM person_view p INNER JOIN pop.school s ON p.school_id = s.id'))

    db.set_pop_view_household('01003')
    print(db.get_pop())
    print(db.db.exec_get('SELECT COUNT(*) FROM person_view p INNER JOIN pop.school s ON p.school_id = s.id'))

    db.set_pop_view_household('42003')
    print(db.get_pop())
    print(db.db.exec_get('SELECT COUNT(*) FROM person_view p INNER JOIN pop.school s ON p.school_id = s.id'))
