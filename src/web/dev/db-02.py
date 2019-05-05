''' Generate an ER diagram from a SQLite DB. '''

from eralchemy import render_er

fpath_db = os.path.join(os.path.dirname(__file__), '..', 'db', 'allegheny-students.sqlite3')

render_er('sqlite:///{fpath_db}', 'erd.png')
