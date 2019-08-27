from attr   import attrs, attrib
from dotmap import DotMap

from .entity import Group, GroupQry, Resource, Site


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class MassFlowSpec(object):
    '''
    A specification of mass flow from one source group to a (possibly single-element) list of destination groups.

    A list of objects of this class for one simulation iteration encodes the full picture of mass flow in th system.
    '''

    m_pop : float = attrib(converter=float)  # total population mass at the time of mass flow
    src   : Group = attrib()
    dst   : list  = attrib(factory=list)


# ----------------------------------------------------------------------------------------------------------------------
class Population(object):
    def __init__(self):
        self.agents = AgentPopulation()
        self.groups = GroupPopulation()


# ----------------------------------------------------------------------------------------------------------------------
class AgentPopulation(object):
    def gen_group_pop(self):
        '''
        Generates a population of groups based on the current agents population.

        This method provides a general interface between popular agent-based modeling packages (e.g., NetLogo) and
        PramPy.
        '''

        pass


# ----------------------------------------------------------------------------------------------------------------------
class GroupPopulation(object):
    def __init__(self, sim, do_keep_mass_flow_specs=False):
        self.sim = sim

        self.groups = {}
        self.sites = {}
        self.resources = {}

        self.mass = 0

        self.is_frozen = False  # the simulation freezes the population on first run

        self.last_iter = DotMap(    # the most recent iteration
            mass_flow_tot = 0,      # total mass transfered
            mass_flow_specs = None  # a list of MassFlowSpec objects (i.e., the full picture of mass flow)
        )

        self.do_keep_mass_flow_specs = do_keep_mass_flow_specs

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
            self.mass += group.m

        g = self.groups.get(group.get_hash())
        if g is not None:
            g.m += group.m
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
            return self

        self.sites[h] = site
        site.set_pop(self)

        return self

    def add_sites(self, sites):
        for s in sites:
            self.add_site(s)
        return self

    def apply_rules(self, rules, iter, t, is_rule_setup=False, is_rule_cleanup=False, is_sim_setup=False):
        '''
        Iterates through groups and for each applies all rules (which is handled by the Group class).  The result of
        (possible) rules applications is a list of new groups the original group should be split into.  When all the
        groups have been processed in this way, and all resulting groups have been defined, those resulting groups are
        subsequently used for mass transfer (which updates existing groups creates new ones).  Note that "resulting" is
        different from "new" because a group might have been split into to resulting groups one or more already exists
        in the group population.  Therefore, not all resulting groups (local scope) need to be new (global scope).
        '''

        mass_flow_specs = []
        src_group_hashes = set()  # hashes of groups to be updated (a safeguard against resetting mass of unaffected groups)
        for g in self.groups.values():
            dst_groups_g = g.apply_rules(self, rules, iter, t, is_rule_setup, is_rule_cleanup, is_sim_setup)
            if dst_groups_g is not None:
                mass_flow_specs.append(MassFlowSpec(self.get_mass(), g, dst_groups_g))
                src_group_hashes.add(g.get_hash())

        if len(mass_flow_specs) == 0:  # no mass to transfer
            return self

        return self.transfer_mass(src_group_hashes, mass_flow_specs, iter, t)

    def compact(self):
        # Remove empty groups:
        self.groups = { k:v for k,v in self.groups.items() if v.m > 0 }

        return self

    def freeze(self):
        # [g.freeze() for g in self.groups.values()]
        # self.groups = { g.get_hash(): g for g in self.groups.values() }

        self.is_frozen = True
        if self.sim.traj is not None and not self.sim.timer.i > 0:  # we check timer not to save initial state of a simulation that's been run already
            self.sim.traj.save_state(None)

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
            return len([g for g in self.groups.values() if g.m > 0])
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

    def get_mass(self):
        # return sum([g.m for g in self.groups.values()])
        return self.mass

    def transfer_mass(self, src_group_hashes, mass_flow_specs, iter, t):
        '''
        Transfers the mass as described by the list of "destination" groups.  "Source" groups (i.e., those that
        participate in mass transfer) have their masses reset before the most-transfer mass is tallied up.
        '''

        m_flow_tot = 0  # total mass transfered

        # Reset the mass of the groups being updated:
        for h in src_group_hashes:
            self.groups[h].m = 0

        for mts in mass_flow_specs:
            for g01 in mts.dst:
                g02 = self.get_group(GroupQry(g01.attr, g01.rel))

                if g02 is not None:  # group already exists
                    g02.m += g01.m
                else:                # group not found
                    self.add_group(g01)

                m_flow_tot += g01.m

        # Notify sites of mass transfer:
        # for s in self.sites.values():
        #     s.invalidate_pop()  # TODO: Develop this further (AFAIR, unused ATM).

        # Save last iteration info:
        self.last_iter.m_flow_tot = m_flow_tot
        if self.do_keep_mass_flow_specs:
            self.last_iter.mass_flow_specs = mass_flow_specs

        # Save the trajectory state:
        if self.sim.traj is not None:
            self.sim.traj.save_state(mass_flow_specs)

        return self
