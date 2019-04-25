'''
Visualize the simulation results.
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3


MARKER_LOW = 'o'  # https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
MARKER_MED = '+'

fpath_db = os.path.join(os.path.dirname(__file__), 'out', '03-flu-allegheny-100-iter.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
def plot_one_school(tbl, title, marker, fpath):
    ''' Plot one of the schools (i.e., a low- or medium-income one). '''

    # Data:
    data = { 'i':[], 'ps':[], 'pe':[], 'pr':[] }
    with sqlite3.connect(fpath_db, check_same_thread=False) as c:
        for r in c.execute(f'SELECT i, ps,pe,pr FROM {tbl}').fetchall():
            data['i'].append(r[0])
            data['ps'].append(r[1])
            data['pe'].append(r[2])
            data['pr'].append(r[3])

    # Plot:
    cmap = plt.get_cmap('tab10')  # https://matplotlib.org/tutorials/colors/colormaps.html
    fig = plt.figure(figsize=(20,8))
    plt.title(title)
    plt.plot(data['i'], data['ps'], lw=1, linestyle='--', marker=marker, color=cmap(0), markersize=5, mfc='none', antialiased=True)
    plt.plot(data['i'], data['pe'], lw=1, linestyle='-',  marker=marker, color=cmap(2), markersize=5, mfc='none', antialiased=True)
    plt.plot(data['i'], data['pr'], lw=1, linestyle=':',  marker=marker, color=cmap(3), markersize=5, mfc='none', antialiased=True)
    plt.legend(['Susceptible', 'Exposed', 'Recovered'], loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.grid(alpha=0.25, antialiased=True)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'out', fpath), dpi=300)

plot_one_school('low_income', 'School with 88% of Low-Income Students', MARKER_LOW, '03-a-low-income-school.png')
plot_one_school('med_income', 'School with 7% of Low-Income Students',  MARKER_MED, '03-b-med-income-school.png')


# ----------------------------------------------------------------------------------------------------------------------
def plot_compare_schools(col, name, fpath):
    ''' Compare the low- and medium-income school based on the specified column (e.g., 'pe'). '''

    # Data:
    data = { 'i':[], 'p.l':[], 'p.m':[] }
    with sqlite3.connect(fpath_db, check_same_thread=False) as c:
        for r in c.execute('SELECT l.i, l.{0} AS pl, m.{0} AS pm FROM low_income l INNER JOIN med_income m ON l.i = m.i'.format(col)).fetchall():
            data['i'].append(r[0])
            data['p.l'].append(r[1])
            data['p.m'].append(r[2])

    # Plot:
    cmap = plt.get_cmap('tab10')  # https://matplotlib.org/tutorials/colors/colormaps.html
    fig = plt.figure(figsize=(20,8))
    plt.title(f'Probability of {name} at Low- and Medium-Income Schools')
    plt.plot(data['i'], data['p.l'], lw=1, linestyle='-', marker=MARKER_LOW, color=cmap(6), markersize=4, mfc='none', antialiased=True)
    plt.plot(data['i'], data['p.m'], lw=1, linestyle='-', marker=MARKER_MED, color=cmap(9), markersize=4, mfc='none', antialiased=True)
    plt.legend(['Low-income', 'Medium-income'], loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.grid(alpha=0.25, antialiased=True)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'out', fpath), dpi=300)

plot_compare_schools('pe', 'Exposed',     '03-c-low-vs-med-income-school-pe.png')
plot_compare_schools('ps', 'Susceptible', '03-d-low-vs-med-income-school-ps.png')
plot_compare_schools('pr', 'Recovered',   '03-e-low-vs-med-income-school-pr.png')
