''' List SQLite DBs present. '''

import os

path = os.path.join(os.path.dirname(__file__), '..', 'db')

# for _, _, files in os.walk(path):
#     ls = [f for f in files if f.endswith('.sqlite3')]

ls = [f[:-len('.sqlite3')] for f in os.listdir(path) if f.endswith('.sqlite3')]

print(ls)
