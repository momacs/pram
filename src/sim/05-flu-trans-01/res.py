import matplotlib.pyplot as plt
import os
import pandas as pd
import sqlite3

from collections import namedtuple


# ----------------------------------------------------------------------------------------------------------------------
# (0) init:

Spec = namedtuple('Spec', ('name', 'n'))
specs = [
    Spec('0',    50),
    Spec('1',   100),
    Spec('2',   200),
    Spec('3',   300),
    Spec('4',   500),
    Spec('5',  5000),
    Spec('6', 50000)
]

sim_dur_days_data = 2
sim_dur_days_plot = 2


# ----------------------------------------------------------------------------------------------------------------------
# (1) Data:

dir = os.path.dirname(__file__)
fpath_db = os.path.join(dir, f'probes-{sim_dur_days_data}d.sqlite3')

if not os.path.isfile(fpath_db):
    raise ValueError(f'Database does not exist: {fpath_db}')

qry01 = [f'(t{s.name}.pa + t{s.name}.ps) AS p_inf_{s.name}' for s in specs]
qry02 = [f'FROM flu_{specs[0].name} t{specs[0].name}'] + [f'INNER JOIN flu_{s.name} t{s.name} on (t{s.name}.i = t{specs[0].name}.i)' for s in specs[1:]]
qry = f'SELECT t{specs[0].name}.i, ' + ', '.join(qry01) + '\n' + '\n'.join(qry02) + f' WHERE t{specs[0].name}.i <= {24 * sim_dur_days_plot}'

with sqlite3.connect(fpath_db, check_same_thread=False) as c:
    df = pd.read_sql(qry, c)  # [:24 * sim_dur_days_plot]


# ----------------------------------------------------------------------------------------------------------------------
# (2) Plot:

dpi = 300
fpath_plot = os.path.join(dir, f'plot-{sim_dur_days_data}d-{sim_dur_days_plot}d.png')

fig = plt.figure(figsize=(20,10))
legend = []
for s in specs:
    plt.plot(df['i'] / 24, df[f'p_inf_{s.name}'], lw=1, antialiased=True)
    legend.append(f'n={s.n}')
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
