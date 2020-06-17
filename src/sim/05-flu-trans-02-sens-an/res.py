import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3

from collections import namedtuple
from mpl_toolkits import mplot3d

from sim import school_specs


# ----------------------------------------------------------------------------------------------------------------------
# (0) init:

sim_dur_days_data = 1
sim_dur_days_plot = 1

var_sa = 'p_inf_min'  # variable for sensitivity analysis

dir = os.path.dirname(__file__)
fpath_db = os.path.join(dir, f'probes-{sim_dur_days_data}d--{var_sa}.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Data:

def get_data(var, val):
    if not os.path.isfile(fpath_db):
        raise ValueError(f'Database does not exist: {fpath_db}')

    qry01 = [f't{school_specs[0].name}.{var}'] + [f't{s.name} AS p_inf_{s.name}' for s in school_specs]
    qry02 = \
        [f'FROM flu_{school_specs[0].name} t{school_specs[0].name}'] + \
        [f'INNER JOIN flu_{s.name} t{s.name} on (t{s.name}.{var} = t{school_specs[0].name}.{var} AND t{s.name}.i = t{school_specs[0].name}.i)' for s in school_specs[1:]]
    qry = \
        f'SELECT t{school_specs[0].name}.i, ' + ', '.join(qry01) + '\n' + '\n'.join(qry02) + \
        f' WHERE t{school_specs[0].name}.{var} = {val} AND t{school_specs[0].name}.i <= {24 * sim_dur_days_plot}'

    with sqlite3.connect(fpath_db, check_same_thread=False) as c:
        df = pd.read_sql(qry, c)  # [:24 * sim_dur_days_plot]

    return df


# ----------------------------------------------------------------------------------------------------------------------
# (2) Plot:

dpi = 300
fpath_plot = os.path.join(dir, f'plot-{sim_dur_days_data}d-{sim_dur_days_plot}d--{var_sa}.png')

p_inf_lst = np.arange(0.01, 0.1, 0.025).tolist()

fig = plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')

for x in p_inf_lst:
    df = get_data(var_sa, x)
    for s in school_specs:
        ax.plot3D(df['i'] / 24, df[f'p_inf_{s.name}'], x, lw=1, antialiased=True)

# print(f'{ax.azim}:{ax.elev}')  # default: -60:30

# ax.view_init(60, -90)
ax.view_init(135, -80)

legend = set()
for s in school_specs:
    legend.add(f'n={s.m}')
plt.legend(legend, loc='lower right')

ax.set_xlabel('simulation duration [days]')
ax.set_ylabel('proportion of infected agents')
ax.set_zlabel('min p(infection)')
ax.set_xticks(range(sim_dur_days_plot + 1))
plt.grid(alpha=0.25, antialiased=True)
fig.savefig(fpath_plot)
