from abc import abstractmethod


class Probe(object):
    __slots__ = ('pop')

    def __init__(self, pop=None):
        self.pop = pop  # pointer to the population (can be set elsewhere too)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    @abstractmethod
    def run(self, t):
        pass

    def set_pop(self, pop):
        self.pop = pop


class GroupSizeProbe(Probe):
    __slots__ = ('name', 'qry')

    def __init__(self, name, qry, pop=None):
        super().__init__(pop)

        self.name = name
        self.qry = qry

    def run(self, t):
        n_tot = sum([g.n for g in self.pop.get_groups()])
        n_qry = [sum([g.n for g in self.pop.get_groups(q.attr, q.rel)]) for q in self.qry]

        if n_tot > 0:
            print('{:2}  {}: ('.format(t, self.name), end='')
            for n in n_qry:
                print('{:.2f} '.format(round(n / n_tot, 2)), end='')
            print(')   (', end='')
            for n in n_qry:
                print('{:>7} '.format(round(n, 1)), end='')
            print(')   [{}]'.format(round(n_tot, 1)))
        else:
            print('{:2}  {}: ---'.format(t, self.name))


# ======================================================================================================================
if __name__ == '__main__':
    from entity import AttrFluStatus, GroupQry

    p = GroupSizeProbe('flu', [GroupQry({ 'flu-status': x }, None) for x in AttrFluStatus])
