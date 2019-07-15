import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.entity import Group, GroupQry
from pram.rule   import Noop
from pram.sim    import Simulation


def ls_grp(name, sim, qry):
    print(f'----[ {name} ]----')
    G = sim.pop.get_groups(qry)
    for g in G:
        print(g)
    print()


s = (
    Simulation().
    add([
        Noop(),
        Group(n=1, attr={ 'x': 100, 'y':  200 }),
        Group(n=2, attr={ 'x': 150, 'y':  200 }),
        Group(n=3, attr={ 'x': 100, 'y': -200 })
    ])
)

ls_grp('1', s, None)
ls_grp('2', s, GroupQry(attr={ 'x': 100 }))
ls_grp('3', s, GroupQry(attr={ 'y': 200 }))
ls_grp('4', s, GroupQry(attr={ 'z':   0 }))
ls_grp('5', s, GroupQry(cond=[lambda g: g.attr['x'] > 100]))
ls_grp('6', s, GroupQry(cond=[lambda g: g.attr['x'] > 100, lambda g: g.attr['y'] == 200]))
ls_grp('7', s, GroupQry(cond=[lambda g: g.attr['x'] > 100 and g.attr['y'] ==  200]))
ls_grp('8', s, GroupQry(cond=[lambda g: g.attr['x'] > 100 or  g.attr['y'] == -200]))
