'''
Visualize the simulation results.
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3


MARKER_S = '+'  # https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
MARKER_I = 'o'
MARKER_R = 'x'


# ----------------------------------------------------------------------------------------------------------------------
def plot(fpath_db, fpath_fig, tbl):  # tab10: 0,2,3
    # Data:
    data = { 'i':[], 'ps':[], 'pi':[], 'pr':[] }
    with sqlite3.connect(os.path.join(os.path.dirname(__file__), fpath_db), check_same_thread=False) as c:
        for r in c.execute(f'SELECT i+1, ps,pi,pr FROM {tbl}').fetchall():
            data['i'] .append(r[0])
            data['ps'].append(r[1])
            data['pi'].append(r[2])
            data['pr'].append(r[3])

    # Plot:
    cmap = plt.get_cmap('tab20')  # https://matplotlib.org/tutorials/colors/colormaps.html
    fig = plt.figure(figsize=(8,8))
    # plt.title('SIR Model')
    plt.plot(data['i'], data['ps'], lw=1, linestyle='--', marker=MARKER_S, color=cmap(0), markersize=5, mfc='none', antialiased=True)
    plt.plot(data['i'], data['pi'], lw=1, linestyle='-',  marker=MARKER_I, color=cmap(4), markersize=5, mfc='none', antialiased=True)
    plt.plot(data['i'], data['pr'], lw=1, linestyle=':',  marker=MARKER_R, color=cmap(6), markersize=5, mfc='none', antialiased=True)
    plt.legend(['Susceptible', 'Infectious', 'Recovered'], loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.grid(alpha=0.25, antialiased=True)
    fig.savefig(os.path.join(os.path.dirname(__file__), fpath_fig), dpi=300)

plot('sim.sqlite3', 'fig.png', 'flu')
