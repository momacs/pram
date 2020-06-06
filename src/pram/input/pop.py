import os
import sqlite3

from pram.util import Time

from abc import abstractmethod, ABC


# ----------------------------------------------------------------------------------------------------------------------
class Geography(ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class EUGeography(Geography):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class USGeography(Geography):
    UNITS = [
        { 'name': 'block',       'fips-substr-idx': ( 0,  2) },  # e.g., ____________003
        { 'name': 'block group', 'fips-substr-idx': ( 3,  3) },  # e.g., ___________1___
        { 'name': 'tract',       'fips-substr-idx': ( 4,  6) },  # e.g., _____110600____
        { 'name': 'county',      'fips-substr-idx': ( 7,  9) },  # e.g., __003__________
        { 'name': 'state',       'fips-substr-idx': (10, 11) }   # e.g., 42_____________
    ]

    pass


# ----------------------------------------------------------------------------------------------------------------------
class PopulationLocation(object):
    def __init__(self, fpath_db, tbl_contacts='contacts', tbl_mobility='mobility', date_format='%Y-%m-%d'):
        if not os.path.isfile(fpath_db):
            print(f'The database specified does not exist: {fpath_db}')
            raise ValueError()

        self.data = {
            'contacts': {}, 'contacts-date-mean': {}, 'contacts-bg-mean': {},
            'mobility': {}, 'mobility-date-mean': {}, 'mobility-bg-mean': {}
        }

        self.load_data(fpath_db, tbl_contacts, tbl_mobility)

        self.settings = {
            'contacts': { 'day-of-year-0': 1 },
            'mobility': { 'day-of-year-0': 1 }
        }

        self.date_format = date_format

    def get(self, var, census_block_grp, date, do_get_avg=True):
        if not census_block_grp in self.data[var].keys():
            return None

        if not date in self.data[var][census_block_grp].keys():
            if not do_get_avg:
                return None
            return self.get_date_mean(var, date)

        return self.data[var][census_block_grp][date]

    def get_bg_mean(self, var, census_block_grp):
        return self.data[f'{var}-bg-mean'].get(census_block_grp)

    def get_date_mean(self, var, date):
        return self.data[f'{var}-date-mean'].get(date)

    def get_contacts(self, census_block_grp, date, do_get_avg=False):
        return self.get('contacts', census_block_grp, date, do_get_avg)

    def get_contacts_by_day_of_year(self, census_block_grp, year, day, do_get_avg=False):
        return self.get('contacts', census_block_grp, Time.day_of_year_to_dt(year, self.settings['contacts']['day-of-year-0'] + day, self.date_format), do_get_avg)

    def get_contacts_bg_mean(self, census_block_grp):
        return self.get_bg_mean('contacts', census_block_grp)

    def get_contacts_date_mean(self, date):
        return self.get_date_mean('contacts', date)

    def get_contacts_date_mean_by_day_of_year(self, year, day):
        return self.get_date_mean('contacts', Time.day_of_year_to_dt(year, self.settings['contacts']['day-of-year-0'] + day, self.date_format))

    def get_mobility(self, census_block_grp, date, do_get_avg=False):
        return self.get('mobility', census_block_grp, date, do_get_avg)

    def get_mobility_by_day_of_year(self, census_block_grp, year, day, do_get_avg=False):
        return self.get('mobility', census_block_grp, Time.day_of_year_to_dt(year, self.settings['mobility']['day-of-year-0'] + day, self.date_format), do_get_avg)

    def get_mobility_bg_mean(self, census_block_grp):
        return self.get_bg_mean('mobility', census_block_grp)

    def get_mobility_date_mean(self, date):
        return self.get_date_mean('mobility', date)

    def get_mobility_date_mean_by_day_of_year(self, year, day):
        return self.get_date_mean('mobility', Time.day_of_year_to_dt(year, self.settings['mobility']['day-of-year-0'] + day, self.date_format))

    def load_data(self, fpath_db, tbl_contacts, tbl_mobility):
        conn = None

        try:
            conn = sqlite3.connect(fpath_db, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode=WAL')  # PRAGMA journal_mode = DELETE

            self.load_data_tbl('contacts', tbl_contacts, conn)
            self.load_data_tbl('mobility', tbl_mobility, conn)
        except Exception as e:
            print(e)
            # raise e
        finally:
            if conn:
                conn.close()

    def load_data_tbl(self, var, tbl, conn):
        # Full resolution (i.e., per census block group and date):
        d = {}
        with conn as c:
            for r in c.execute(f'SELECT stcotrbg_id AS bg, date, {var} AS {var} FROM {tbl}').fetchall():
                if r['bg'] not in d.keys():
                    d[r['bg']] = {}
                d[r['bg']][r['date']] = r[var]
        self.data[var] = d

        # Aggregated by date (mean):
        d = {}
        with conn as c:
            for r in c.execute(f'SELECT date, ROUND(AVG({var}),2) AS {var} FROM {tbl} GROUP BY date').fetchall():
                d[r['date']] = r[var]
        self.data[f'{var}-date-mean'] = d

        # Aggregated by census block group (mean):
        d = {}
        with conn as c:
            for r in c.execute(f'SELECT stcotrbg_id, ROUND(AVG({var}),2) AS {var} FROM {tbl} GROUP BY stcotrbg_id').fetchall():
                d[r['stcotrbg_id']] = r[var]
        self.data[f'{var}-bg-mean'] = d

    def set_setting(self, var, name, val):
        self.settings[var][name] = val
        return self

    def set_contacts_first_day_of_year(self, day=1):
        return self.set_setting('contacts', 'day-of-year-0', day)

    def set_mobility_first_day_of_year(self, day=1):
        return self.set_setting('mobility', 'day-of-year-0', day)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pl = PopulationLocation(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'allegheny-county', 'allegheny-geo-01.sqlite3'))

    print(f'mobility: {len(pl.data["mobility"])}')                          # 1087
    print(f'contacts: {len(pl.data["contacts"])}')                          # 1089

    print(f'01  {pl.get_mobility(420031106001, "2020-01-01")}')             # 0.27
    print(f'02  {pl.get_mobility(420031106001, "1900-01-01")}')             # None
    print(f'03  {pl.get_mobility(000000000000, "2020-01-01")}')             # None
    print(f'04  {pl.get_mobility_by_day_of_year(420031106001, 2020, 0)}')   # 0.27
    print(f'05  {pl.get_mobility_by_day_of_year(420031106001, 2020, 1)}')   # 0.32

    print(f'06  {pl.get_contacts(420034772002, "2020-03-01")}')             # 31.25
    print(f'07  {pl.get_contacts(420034772002, "1900-03-01")}')             # None
    print(f'08  {pl.get_contacts(000000000000, "2020-03-01")}')             # None
    print(f'09  {pl.get_contacts_by_day_of_year(420034772002, 2020, 60)}')  # 31.25
    print(f'10  {pl.get_contacts_by_day_of_year(420034772002, 2020, 61)}')  # 84.87

    print('----')

    pl.set_mobility_first_day_of_year(61)                                   # offset all day-of-year queries by 61 days (i.e., March 1 for 2020)
    print(f'11  {pl.get_mobility_by_day_of_year(420031106001, 2020, 1)}')   # 1.0

    pl.set_contacts_first_day_of_year(61)                                   # offset all day-of-year queries by 61 days (i.e., March 1 for 2020)
    print(f'12  {pl.get_contacts_by_day_of_year(420034772002, 2020, 1)}')   # 84.87

    print('----')

    print(f'13  {pl.get_mobility_bg_mean(420031106001)}')    #  0.43
    print(f'14  {pl.get_contacts_bg_mean(420034772002)}')    # 16.28

    print(f'15  {pl.get_mobility_date_mean("2020-01-01")}')  #  0.12
    print(f'16  {pl.get_contacts_date_mean("2020-03-01")}')  # 30.69

    print('----')

    print(f'17  {pl.get_mobility_date_mean_by_day_of_year(2020, 1)}')  # 0.28
    print(f'28  {pl.get_contacts_date_mean_by_day_of_year(2020, 1)}')  # 70.9
