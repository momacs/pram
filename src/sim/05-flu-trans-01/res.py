import matplotlib.pyplot as plt
import os
import pandas as pd
import sqlite3

from collections import namedtuple

from sim import school_specs


# ----------------------------------------------------------------------------------------------------------------------
# (0) init:

sim_dur_days_data = 3
sim_dur_days_plot = 3


# ----------------------------------------------------------------------------------------------------------------------
# (1) Data:

dir = os.path.dirname(__file__)
fpath_db = os.path.join(dir, f'probes-{sim_dur_days_data}d.sqlite3')

if not os.path.isfile(fpath_db):
    raise ValueError(f'Database does not exist: {fpath_db}')

qry01 = [f't{s.name}.pi AS p_inf_{s.name}' for s in school_specs]
qry02 = [f'FROM flu_{school_specs[0].name} t{school_specs[0].name}'] + [f'INNER JOIN flu_{s.name} t{s.name} on (t{s.name}.i = t{school_specs[0].name}.i)' for s in school_specs[1:]]
qry = f'SELECT t{school_specs[0].name}.i, ' + ', '.join(qry01) + '\n' + '\n'.join(qry02) + f' WHERE t{school_specs[0].name}.i <= {24 * sim_dur_days_plot}'

with sqlite3.connect(fpath_db, check_same_thread=False) as c:
    df = pd.read_sql(qry, c)  # [:24 * sim_dur_days_plot]


# ----------------------------------------------------------------------------------------------------------------------
# (2) Plot:

dpi = 300
fpath_plot = os.path.join(dir, f'plot-{sim_dur_days_data}d-{sim_dur_days_plot}d.png')

fig = plt.figure(figsize=(20,10))
legend = []
for s in school_specs:
    plt.plot(df['i'] / 24, df[f'p_inf_{s.name}'], lw=1, antialiased=True)
    legend.append(f'n={s.m}')
plt.legend(legend, loc='lower right')
plt.xlabel('days')
plt.ylabel('proportion of infected students')
plt.xticks(range(sim_dur_days_plot + 1))
plt.grid(alpha=0.25, antialiased=True)
fig.savefig(fpath_plot, dpi=dpi)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Conclusions:

'''
Given the specific model progression and transmission models implemented above, roughly 50% of the populations is in
the symptomatic state once a certain school size is reached.  For example, the symptomatic proportion of a 50-student
school is 13% while it is the aforementioned 50% for any size of 250 students and above.  The 50% ratio hold even for
a school of size 50,000.  This pattern isn't built-in nor evident from the mechanisms impemented in this simulation.
'''
