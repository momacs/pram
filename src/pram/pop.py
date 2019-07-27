from .entity import Group, GroupQry, Resource, Site


class Population(object):
    def __init__(self):
        self.agents = AgentPopulation()
        self.groups = GroupPopulation()


class AgentPopulation(object):
    def gen_groups(self):
        ''' Generates a population of groups based on the current agents population. '''

        pass


class GroupPopulation(object):
    def __init__(self):
        self.groups = {}
        self.sites = {}
        self.resources = {}

        self.size = 0
        self.n_distrib_last = 0  # number of agents distributed during the most recent mass distribution

        self.is_frozen = False  # the simulation freezes the population on first run

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def add_group(self, group):
        '''
        Add a group if it doesn't exist and update the size if it does. This method also adds all Site objects from
        the group's relations so there is no need for the user to do this manually.

        All groups added to the population become frozen to prevent the user from changing their attribute and relations
        directly; doing it via group splitting is the proper way.
        '''

        if not self.is_frozen:
            self.size += group.n

        g = self.groups.get(group.get_hash())
        if g is not None:
            g.n += group.n
        else:
            self.groups[group.get_hash()] = group
            self.add_sites    ([v for (_,v) in group.get_rel().items() if isinstance(v, Site)])
            self.add_resources([v for (_,v) in group.get_rel().items() if isinstance(v, Resource)])
            group.freeze()

        return self

    def add_groups(self, groups):
        for g in groups:
            self.add_group(g)
        return self

    def add_resource(self, resource):
        h = resource.get_hash()
        if h in self.resources.keys():
            return

        self.resources[h] = resource
        # resource.set_pop(self)

        return self

    def add_resources(self, resources):
        for r in resources:
            self.add_resource(r)
        return self

    def add_site(self, site):
        h = site.get_hash()
        if h in self.sites.keys():
            return

        self.sites[h] = site
        site.set_pop(self)

        return self

    def add_sites(self, sites):
        for s in sites:
            self.add_site(s)
        return self

    def apply_rules(self, rules, iter, t, traj=None, is_rule_setup=False, is_rule_cleanup=False, is_sim_setup=False):
        '''
        Iterates through groups and for each applies all rules (which is handled intelligently by the Group class).
        The result of (possible) rules applications is a list of new groups the original group should be split into.
        When all the groups have been processed in this way, and all new groups have been defined, they are
        subsequently used for actual mass redistribution (which updates existing groups and creates new ones).

        Returns the total number of agents displaced.
        '''

        new_groups = []
        upd_group_hashes = set()  # hashes of groups to be updated (to safeguard against resetting unaffected groups)
        for g in self.groups.values():
            new_groups_g = g.apply_rules(self, rules, iter, t, is_rule_setup, is_rule_cleanup, is_sim_setup)
            if new_groups_g is not None:
                new_groups.extend(new_groups_g)
                upd_group_hashes.add(g.get_hash())

        if len(new_groups) == 0:  # no mass to distribute
            return self

        return self.distribute_mass(upd_group_hashes, new_groups)

    def compact(self):
        # Remove empty groups:
        self.groups = { k:v for k,v in self.groups.items() if v.n > 0 }

        return self

    def distribute_mass(self, upd_group_hashes, new_groups):
        '''
        Distributes the mass as described by the list of new groups.  Because not all groups may participate in mass
        distribution, sizes of only those that actually do are updated.

        Returns the total number of agents displaced.
        '''

        n = 0  # number of agents redistributed

        # Reset the population mass (but only for the groups that are being updated):
        for h in upd_group_hashes:
            self.groups[h].n = 0

        # Distribute the mass based on new groups:
        for g01 in new_groups:
            g02 = self.get_group(GroupQry(g01.attr, g01.rel))

            # The group already exists:
            if g02 is not None:
                g02.n += g01.n

            # The group does not exist:
            else:
                self.add_group(g01)

            n += g01.n

        # Notify sites of mass redistribution:
        # for s in self.sites.values():
        #     s.invalidate_pop()  # TODO: Develop this further (AFAIR, unused ATM).

        self.n_distrib_last = n

        return self

    def freeze(self):
        # [g.freeze() for g in self.groups.values()]
        # self.groups = { g.get_hash(): g for g in self.groups.values() }
        self.is_frozen = True
        return self

    def gen_agent_pop(self):
        ''' Generates a population of agents based on the current groups population. '''

        pass

    def get_group(self, qry=None):
        '''
        Returns the group with the all attributes and relations as specified; or None if such a group does not
        exist.

        qry: GroupQry
        '''

        qry = qry or GroupQry()
        return self.groups.get(Group.gen_hash(qry.attr, qry.rel))

    def get_group_cnt(self, only_non_empty=False):
        if only_non_empty:
            return len([g for g in self.groups.values() if g.n > 0])
        else:
            return len(self.groups)

    def get_groups(self, qry=None):
        '''
        Returns a list of groups that contain the attributes and relations specified in the query.  If the query is
        None, all groups are returned.

        qry: GroupQry
        '''

        # qry = qry or GroupQry()
        # attr_set = set(qry.attr.items())
        # rel_set  = set(qry.rel.items())
        #
        # ret = []
        # for g in self.groups.values():
        #     if (set(g.attr.items()) & attr_set == attr_set) and (set(g.rel.items()) & rel_set == rel_set):
        #         ret.append(g)
        #
        # return ret

        if qry is None:
            return self.groups.values()

        # return [g for g in self.groups.values() if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items())]
        return [g for g in self.groups.values() if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items()) and all([fn(g) for fn in qry.cond])]

    def get_next_group_name(self):
        return f'g.{len(self.groups)}'

    def get_site_cnt(self):
        return len(self.sites)

    def get_size(self):
        # return sum([g.n for g in self.groups.values()])
        return self.size
