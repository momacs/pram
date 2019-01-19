from entity import Group


class Population(object):
    def __init__(self):
        self.agents = AgentPopulation()
        self.groups = GroupPopulation()


class AgentPopulation(object):
    def gen_groups(self):
        ''' Generates a population of groups based on the current agents population. '''

        pass


class GroupPopulation(object):
    DEBUG_LVL = 0  # 0=none, 1=normal, 2=full

    def __init__(self):
        self.groups = {}  # key is the group's hash
        self.sites  = {}

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1: print(msg)

    def add_group(self, group):
        self.groups[group.get_hash()] = group

    def add_groups(self, groups):
        for g in groups:
            self.add_group(g)

    def add_site(self, site):
        self.sites[site.get_hash()] = site
        site.set_pop(self)

    def add_sites(self, sites):
        for s in sites:
            self.add_site(s)

    def apply_rules(self, rules, t):
        '''
        Iterates through groups and for each applies all rules (which is handles intelligently by the Group class).
        The result of (possible) rules applications is a list of new groups the original group should be split into.
        When all the groups have been processed in this way, and all new groups have been defined, they are
        subsequently used for actual mass redistribution (which updates existing groups and creates new ones).
        '''

        new_groups = []
        upd_group_hashes = set()  # hashes of groups to be updated (to safeguard against resetting unaffected groups)
        for g in self.groups.values():
            new_groups_g = g.apply_rules(rules, t)
            if new_groups_g is not None:
                new_groups.extend(new_groups_g)
                upd_group_hashes.add(g.get_hash())

        if len(new_groups) == 0:  # no mass to distribute
            return

        self.distribute_mass(upd_group_hashes, new_groups)

    def create_group(self, n, attr, rel):
        g = Group('g.{}'.format(len(self.groups)), n, attr, rel)
        self.add_group(g)
        return g

    def distribute_mass(self, upd_group_hashes, new_groups):
        '''
        Distributes the mass as described by the list of new groups.  Because not all groups may participate in mass
        distribution, sizes of only those that actually do are updated.
        '''

        # Reset the population mass (but only for the groups that are being updated):
        for h in upd_group_hashes:
            self.groups[h].n = 0

        # Distribute the mass based on new groups:
        for g01 in new_groups:
            g02 = self.get_group(g01.attr, g01.rel)

            # The group already exists:
            if g02 is not None:
                self._debug('group found    : {:16}  {:8} --> {:8}'.format(g02.name, g02.n, (g02.n + g01.n)))
                g02.n += g01.n

            # The group does not exist:
            else:
                self._debug('group not found: {:16}  {:8}'.format(g01.name, g01.n))
                self.add_group(g01)

    def get_group(self, attr=None, rel=None):
        '''
        Returns the group with the all attributes and relations as specified; or None if such a group does not exist.
        '''

        return self.groups.get(Group.gen_hash(attr, rel))

    def get_groups(self, attr=None, rel=None):
        '''
        Returns a list of groups that contain the specified attributes and relation.  Both those dictionaries could be
        empty, in which case all groups would be returned.
        '''

        attr = attr or {}
        rel  = rel  or {}

        attr_set = set(attr.items())
        rel_set  = set(rel.items())

        ret = []
        for g in self.groups.values():
            if (set(g.attr.items()) & attr_set == attr_set) and (set(g.rel.items()) & rel_set == rel_set):
                ret.append(g)
        return ret

    def gen_agent_pop(self):
        ''' Generates a population of agents based on the current groups population. '''

        pass

    def split_group(self):
        '''
        Splits the designated group into one or more new groups.  No intermediate group objects are instantiated unless
        new groups need to be created.  Consequently, population mass redistributions happens automatically too.
        '''

        pass


# ======================================================================================================================
if __name__ == '__main__':
    # Set operations:
    assert(set({'a':1, 'b':2, 'c':3}.items()) & set({'b':2,'c':3}.items()) == set({'c':3,'b':2}.items()))
    assert(set({'a':1, 'b':1, 'c':3}.items()) & set({'b':2,'c':3}.items()) != set({'c':3,'b':2}.items()))
    assert(set({'a':1,        'c':3}.items()) & set({'b':2,'c':3}.items()) != set({'c':3,'b':2}.items()))
