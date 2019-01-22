from abc import abstractmethod
from entity import GroupQry


class Probe(object):
    __slots__ = ('do_cumul', 'pop', 'memo', 'msg')

    def __init__(self, do_cumul=False, pop=None, memo=None):
        self.do_cumul = do_cumul
        self.pop = pop  # pointer to the population (can be set elsewhere too)
        self.memo = memo
        self.msg = []  # used to cumulate messages (only when 'do_cumul=True')

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return 'Probe'

    def clear(self):
        self.msg.clear()

    def get_msg(self, do_join=True):
        return '\n'.join(self.msg)

    @abstractmethod
    def run(self, t):
        pass

    def set_pop(self, pop):
        self.pop = pop


class GroupSizeProbe(Probe):
    __slots__ = ('name', 'queries')

    def __init__(self, name, queries, do_cumul=False, pop=None, memo=None):
        '''
        queries: iterable of GroupQry
        '''

        super().__init__(do_cumul, pop, memo)

        self.name = name
        self.queries = queries

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return 'Probe  name: {:16}  query-cnt: {:>3}'.format(self.name, len(self.queries))

    def run(self, t):
        n_tot = sum([g.n for g in self.pop.get_groups()])  # TODO: If the total mass never changes, we could memoize this.
        n_qry = [sum([g.n for g in self.pop.get_groups(q)]) for q in self.queries]

        msg = []
        if n_tot > 0:
            msg.append('{:2}  {}: ('.format(t, self.name))
            for n in n_qry:
                msg.append('{:.2f} '.format(round(n / n_tot, 2)))
            msg.append(')   (')
            for n in n_qry:
                msg.append('{:>7} '.format(round(n, 1)))
            msg.append(')   [{}]'.format(round(n_tot, 1)))
        else:
            msg.append('{:2}  {}: ---'.format(t, self.name))

        if self.do_cumul:
            self.msg.append(''.join(msg))
        else:
            print(''.join(msg))


# ======================================================================================================================
if __name__ == '__main__':
    from entity import AttrFluStatus, GroupQry

    p = GroupSizeProbe('flu', [GroupQry({ 'flu-status': x }, None) for x in AttrFluStatus])
