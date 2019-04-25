import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3


# ----------------------------------------------------------------------------------------------------------------------
# (0) init:

dir = os.path.dirname(__file__)
fpath_db = os.path.join(dir, f'school-pop-size-1-home.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Data:

def get_col_names_school_order(do_num=False, do_by_size=False):
    if not os.path.isfile(fpath_db):
        raise ValueError(f'Database does not exist: {fpath_db}')

    # Column names:
    # qry_cols = "SELECT p.name FROM pragma_table_info('school_pop_size') WHERE name LIKE '{}%'".format('n' if do_num else 'p')  # works in SQLite3 borwser; ugh...
    with sqlite3.connect(fpath_db, check_same_thread=False) as c:
        cols = pd.read_sql('PRAGMA table_info("school_pop_size")', c)
        cols = cols.loc[cols['name'].str.startswith('n' if do_num else 'p')]['name'].tolist()

    if not do_by_size:
        return cols

    # The order:
    qry = f'SELECT {",".join(cols)} FROM school_pop_size WHERE i = 5'
    with sqlite3.connect(fpath_db, check_same_thread=False) as c:
        data = pd.read_sql(qry, c).transpose().sort_values(by=0, ascending=False)

    return data.index.tolist()

def get_data(cols, iter=-1, do_num=False):
    if not os.path.isfile(fpath_db):
        raise ValueError(f'Database does not exist: {fpath_db}')

    # Data:
    qry = 'SELECT {} FROM school_pop_size{}'.format(','.join(cols), '' if iter == -1 else f' WHERE i = {iter}')
    with sqlite3.connect(fpath_db, check_same_thread=False) as c:
        data = pd.read_sql(qry, c)
        # data = pd.wide_to_long(data, ['p'], i='id', j='x')
        data = data.values.tolist()[0]

    return data


# ----------------------------------------------------------------------------------------------------------------------
# (2) Plot - Density plots:

dpi = 300
fpath_plot = os.path.join(dir, f'plot-school-pop-size-1-home.png')

do_num = True
do_order_by_size = True
do_log10 = True

cols = get_col_names_school_order(do_num, do_order_by_size)

# print(cols)
# import sys
# sys.exit(99)


# (2.) ...
# data = get_data(cols, 3, do_num)
# fig = plt.figure(figsize=(20,5))
# # plt.plot(data, lw=1, antialiased=True)
# plt.bar(range(len(data)), data, lw=1, antialiased=True)
# plt.xlabel('schools')
# plt.ylabel(f'population size ({"number of agents" if do_num else "proportion of all at-school agents"})')
# plt.grid(alpha=0.25, antialiased=True)
# fig.savefig(fpath_plot, dpi=dpi)


# (2.) Vertically-stacked box plots:
# iter = range(1,3,1)
# fig, axes = plt.subplots(len(iter), 1, sharex=True, figsize=(20,20))
# for i in iter:
#     data = get_data(cols, i, do_num)
#     ax = axes[i - 1]
#     ax.bar(range(len(data)), data, lw=1, antialiased=True)
#     ax.text(0.025, 0.975, f'iter: {i}', fontsize=14, ha='center', va='center', transform=ax.transAxes)
#     # ax_i.grid(alpha=0.25, antialiased=True)
# # plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# plt.subplots_adjust(hspace=0)
# plt.xlabel('school')
# plt.ylabel(f'population size ({"number of agents" if do_num else "proportion of all at-school agents"})')
# fig.savefig(fpath_plot, dpi=dpi)


# (2.) A single line plot:
iter = range(1,6,1)
print(iter)
# import sys
# sys.exit(99)

fig = plt.figure(figsize=(20,10))
legend = []
for i in iter:
    t = i + 7
    data = get_data(cols, i, do_num)
    # plt.scatter(range(len(data)), data, s=i*7, c='#000000', alpha=0.20, antialiased=True)
    if do_log10:
        plt.plot(range(len(data)), np.log10(data), lw=1, alpha=0.75 if do_order_by_size else 0.50, antialiased=True)
    else:
        plt.plot(range(len(data)), data, lw=1, alpha=0.75 if do_order_by_size else 0.50, antialiased=True)
    legend.append('Time: {}{}'.format(t, 'am' if t < 12 else 'pm'))
plt.legend(legend, loc='upper right')
plt.title('Agent Population Size per School over Time')
plt.xlabel(f'Schools {" (ordered by size)" if do_order_by_size else " (arbitrary order)"}')
if do_log10:
    plt.ylabel(f'Population size ({"number of agents; log10" if do_num else "proportion of all at-school agents"})')
else:
    plt.ylabel(f'Population size ({"number of agents"        if do_num else "proportion of all at-school agents"})')
plt.grid(alpha=0.25, antialiased=True)
fig.savefig(fpath_plot, dpi=dpi)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Plot - Line plot for select schools:

# dpi = 300
# fpath_plot = os.path.join(dir, f'plot-school-pop-size-1-home.png')
#
# fig = plt.figure(figsize=(20,10))
# plt.plot(df['i'] / 24, df[f'p_inf_{s.name}'], lw=1, antialiased=True)
# plt.xlabel('school')
# plt.ylabel('proportion of infected students')
# plt.xticks(range(sim_dur_days_plot + 1))
# plt.grid(alpha=0.25, antialiased=True)
# fig.savefig(fpath_plot, dpi=dpi)
