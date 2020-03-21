# -*- coding: utf-8 -*-
"""Contains PRAM group and agent populations code."""

import math

from attr   import attrs, attrib
from dotmap import DotMap

from .entity import Group, GroupQry, Resource, Site


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class MassFlowSpec(object):
    """A specification of agent population mass flow.

    A list of objects of this class for one simulation iteration encodes the full picture of agent population mass flow
    in the model.  This class encodes mass flow from one source group to one or more destination groups.

    Args:
        m_pop (float): Total population mass at the time of mass flow.
        src (Group): The source group (i.e., the mass donor).
        dst (Iterable[Group]): Destination groups (i.e., mass acceptors).
    """

    m_pop : float = attrib(converter=float)  # total population mass at the time of mass flow
    src   : Group = attrib()
    dst   : list  = attrib(factory=list)


# ----------------------------------------------------------------------------------------------------------------------
class Population(object):
    """A population of both groups and agents.

    This class is outward looking and as much is not currently implemented.
    """

    def __init__(self):
        self.agents = AgentPopulation()
        self.groups = GroupPopulation()


# ----------------------------------------------------------------------------------------------------------------------
class AgentPopulation(object):
    """Population of individual agents.

    While PRAM extends agent-based models and therefore it seems logical to invoke the concept of agent population (i.e.,
    a population of individual agents, not groups thereof), that isn't how PRAM works.  Consequently, this class is a
    stub at this point but may rise to prominence at some later point.
    """

    def gen_group_pop(self):
        """Generates a population of groups based on the current agents population.

        This method provides a general interface between popular agent-based modeling packages (e.g., NetLogo) and
        PyPRAM.
        """

        pass


# ----------------------------------------------------------------------------------------------------------------------
class GroupPopulation(object):
    """Population of groups of agents.

    PRAM models functionally equivalent agents jointly as groups.  The formalism does not invoke any "hard" notion of a
    group population.  However, the PyPRAM package does use that concept to elegantly compartmentalize groups along
    with operations on them.

    Because groups can be associated with sites via group relations, those sites are also stored inside of this
    class' instance.

    VITA groups, or groups that are the source of new population mass are stored separately from the actual group
    population.  At the end of every iteration, masses of all VITA groups are transfered back to the population.  It is
    also at that time that all VOID groups are removed.  VOID groups contain mass that should be removed from the\
    simulation.

    Args:
        sim (Simulation): The simulation.
        do_keep_mass_flow_specs (bool): Flag: Store the last iteration mass flow specs?  This is False by default for
            memory usage sake.  If set to True, ``self.last_iter.mass_flow_specs`` will hold the specs until they are
            overwriten at the next iteration of the simulation.
    """

    def __init__(self, sim, do_keep_mass_flow_specs=False):
        self.sim = sim

        self.groups = {}
        self.sites = {}
        self.resources = {}

        self.vita_groups = {}  # all VITA groups for the current iteration

        self.mass     = 0  # total population mass
        self.mass_in  = 0  # total population mass added (e.g., via the birth process)
        self.mass_out = 0  # total population mass removed (e.g., via the death process)

        self.is_frozen = False  # the simulation freezes the population on first run

        self.last_iter = DotMap(    # the most recent iteration info
            mass_flow_tot = 0,      # total mass transfered
            mass_flow_specs = None  # a list of MassFlowSpec objects (i.e., the full picture of mass flow)
        )

        self.do_keep_mass_flow_specs = do_keep_mass_flow_specs

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def add_group(self, group):
        """Adds a group to the population.

        If the groups doesn't exist in the population, it will be added.  Otherwise, the group's size will be added to
        the already existing group's size.  This behavior ensures the group population only contains exactly one
        instance of a group.  As a reminder, groups are identical if their attributes and relations are equal.

        This method also adds all Site objects from the group's relations so there is no need for the user to do this
        manually.

        All groups added to the population become frozen to prevent the user from changing their attribute and
        relations via their API; modifying groups via group splitting mechamism is the only proper way.

        Args:
            group (Group): The group being added.

        Returns:
            self: For method call chaining.
        """

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
        """Adds multiple groups to the population.

        See :meth:`~pram.pop.GroupPopulation.add_group` method for details.

        Args:
            groups (Iterable[Group]): Groups to be added.

        Returns:
            self: For method call chaining.
        """

        for g in groups:
            self.add_group(g)
        return self

    def add_resource(self, resource):
        """Adds a resource to the population.

        Only adds if the resource doesn't exist yet.

        Args:
            resource (Resource): The resource to be added.

        Returns:
            self: For method call chaining.
        """

        h = resource.get_hash()
        if h in self.resources.keys():
            return

        self.resources[h] = resource
        # resource.set_pop(self)

        return self

    def add_resources(self, resources):
        """Adds multiple resources to the population.

        See :meth:`~pram.pop.GroupPopulation.add_resource` method for details.

        Args:
            resource (Resource): The resource to be added.

        Returns:
            self: For method call chaining.
        """

        for r in resources:
            self.add_resource(r)
        return self

    def add_site(self, site):
        """Adds a site to the population.

        Only adds if the site doesn't exist yet.

        Args:
            site (Site): The site to be added.

        Returns:
            self: For method call chaining.
        """

        h = site.get_hash()
        if h in self.sites.keys():
            return self

        self.sites[h] = site
        site.set_pop(self)

        return self

    def add_sites(self, sites):
        """Adds multiple sites to the population.

        See :meth:`~pram.pop.GroupPopulation.add_site` method for details.

        Args:
            sites (Site): The sites to be added.

        Returns:
            self: For method call chaining.
        """

        for s in sites:
            self.add_site(s)
        return self

    def add_vita_group(self, group):
        """Adds a VITA group.

        If a VITA group with the same hash already exists, the masses are combined (i.e., no VITA group duplication
        occurs, much like is the case with regular groups).

        Args:
            group (Group): The group being added.

        Returns:
            self: For method call chaining.
        """

        g = self.vita_groups.get(group.get_hash())
        if g is not None:
            g.m += group.m
        else:
            self.vita_groups[group.get_hash()] = group

        return self

    def apply_rules(self, rules, iter, t, is_rule_setup=False, is_rule_cleanup=False, is_sim_setup=False):
        """Applies all rules in the simulation to all groups in the population.

        This method iterates through groups and for each applies all rules (that step is handled by the Group class
        itself in the :meth:`~pram.entity.Group.apply_rules` method).  The result of those rules applications is a list
        of new groups the original group should be split into.  When all the groups have been processed in this way,
        and consequently all resulting groups have been defined, those resulting groups are used for mass transfer
        (which updates existing groups and creates new ones).  Note that "resulting" is different from "new" because a
        group might have been split into resulting groups of which one or more already exists in the group population.
        In other words, not all resulting groups (local scope) need to be new (global scope).

        Args:
            rules (Iterable[Rule]): The rules.
            iter (int): Simulation interation.
            t (int): Simulation time.
            is_rule_setup (bool): Flag: Is this invocation of this method during rule setup stage of the simulation?
            is_rule_cleanup (bool): Flag: Is this invocation of this method during rule cleanup stage of the
                simulation?
            is_sim_setup (bool): Flag: Is this invocation of this method during simulation setup stage?

        Returns:
            self: For method call chaining.
        """


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
        """Compacts the population by removing empty groups.

        Whether this is what the user needs is up to them.  However, for large simulations with a high new group
        turnover compacting the group population will make the simulation run faster.  Autocompacting can be turned on
        on the simulation level via the `autocompact` pragma (e.g., via :meth:`~pram.sim.Simulation.set_pragma_autocompact`
        method).

        Returns:
            self: For method call chaining.
        """

        self.groups = { k:v for k,v in self.groups.items() if v.m > 0 }
        return self

    def do_post_iter(self):
        """Performs all post-iteration routines (i.e., current iter cleanup and next iter prep).

        Routines performed:

        (1) Remove VOID groups.
        (2) Move mass from VITA groups to their corresponding groups.

        Returns:
            self: For method call chaining.
        """

        # (1) Remove VOID groups (no dict comprehension because we want to edit the existing dict in-place):
        del_keys = []
        for (k,v) in self.groups.items():
            if v.is_void():
                # print(-v.m)
                self.mass     -= v.m
                self.mass_out += v.m
                del_keys.append(k)
                # print(f'{k}: {v}')
        # for _ in del_keys:
        #     if k in self.groups.keys():
        #         del self.groups[k]
        self.groups = { k:v for k,v in self.groups.items() if not v.is_void() }

        # (2) Move mass from VITA groups to their corresponding groups:
        for (k,v) in self.vita_groups.items():
            # print(v.m)
            self.mass    += v.m
            self.mass_in += v.m
            self.groups[k].m += v.m
        self.vita_groups = {}

        return self

    def freeze(self):
        """Freeze the population.

        The :class:`~pram.sim.Simulation` object freezes the population on first run.  Freezing a population is used
        only used to determine the total population size.

        Returns:
            self: For method call chaining.
        """

        # [g.freeze() for g in self.groups.values()]
        # self.groups = { g.get_hash(): g for g in self.groups.values() }

        self.is_frozen = True
        if self.sim.traj is not None and not self.sim.timer.i > 0:  # we check timer not to save initial state of a simulation that's been run already
            self.sim.traj.save_state(None)

        return self

    def gen_agent_pop(self):
        """
        Args:


        Returns:
            self: For method call chaining.
        """

        ''' Generates a population of agents based on the current groups population. '''

        pass

    def get_group(self, qry=None):
        """
        Returns the group with the all attributes and relations as specified; or None if such a group does not exist.

        Args:
            qry (GroupQry): The query.

        Returns:
            Group if a group is found; None otherwise
        """

        qry = qry or GroupQry()
        return self.groups.get(Group.gen_hash(qry.attr, qry.rel))

    def get_group_cnt(self, only_non_empty=False):
        """
        Args:


        Returns:
            self: For method call chaining.
        """

        if only_non_empty:
            return len([g for g in self.groups.values() if g.m > 0])
        else:
            return len(self.groups)

    def get_groups(self, qry=None):
        """
        Args:


        Returns:
            self: For method call chaining.
        """

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

    def get_groups_mass(self, qry=None):
        """Get the mass of groups that match the query specified.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            float: Mass
        """

        return math.fsum([g.m for g in self.get_groups(qry)])

    def get_groups_mass_prop(self, qry=None):
        """Get the proportion of the total mass accounted for the groups that match the query specified.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            float: Mass proportion [0..1]
        """

        return self.get_groups_mass(qry) / self.mass


    def get_next_group_name(self):
        """
        Args:


        Returns:
            self: For method call chaining.
        """

        return f'g.{len(self.groups)}'

    def get_site_cnt(self):
        """
        Args:


        Returns:
            self: For method call chaining.
        """

        return len(self.sites)

    def get_mass(self):
        # return sum([g.m for g in self.groups.values()])
        return self.mass

    def transfer_mass(self, src_group_hashes, mass_flow_specs, iter, t):
        """
        Args:


        Returns:
            self: For method call chaining.
        """

        '''
        Transfers the mass as described by the list of "destination" groups.  "Source" groups (i.e., those that
        participate in mass transfer) have their masses reset before the most-transfer mass is tallied up.

        Because this method is called only once per simulation iteration, it is a good place to put
        simulation-wide computations that should happen after the iteration-specific computations have
        been completed (e.g., the entirety of mass transfer has been figured out).
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
        for s in self.sites.values():
            s.invalidate_pop()  # TODO: Develop this further (AFAIR, unused ATM).

        # Save last iteration info:
        self.last_iter.m_flow_tot = m_flow_tot
        if self.do_keep_mass_flow_specs:
            self.last_iter.mass_flow_specs = mass_flow_specs

        # Save the trajectory state:
        if self.sim.traj is not None:
            self.sim.traj.save_state(mass_flow_specs)

        return self
