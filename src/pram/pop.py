# -*- coding: utf-8 -*-
"""Contains PRAM group and agent populations code."""

import json
import math
import xxhash

from attr            import attrs, attrib
from collections     import deque
from collections.abc import Iterable
from dotmap          import DotMap
from functools       import lru_cache

from .data   import GroupProbe
from .entity import Entity, Group, GroupQry, Resource, Site, EntityJSONEncoder

__all__ = ['MassFlowSpec', 'GroupPopulation', 'GroupPopulationHistory']


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
class AttrRelEncoder(object):
    """Encoder and decoder of Group and GroupQry attributes and relations.

    Before encoding:

        group.attr = { 'flu': 's', 'age-group': '10-19' }

    After encoding:

        group.attr = ((0,0), (1,1))  # a set of tuples (hashable as opposed to a dictionary and much smaller)

        self.attr_k2i = {'flu': 0, 'age-group': 1}
        self.attr_v2i = [{'s': 0, 'i': 1, 'r': 2}, {'10-19': 0, '20-29': 1}]
        self.attr_i2k = ['flu', 'age-group']
        self.attr_i2v = [['s', 'i', 'r'], ['10-19', '20-29']]

    Possible future extension invonling bit-wise operations:

        group.attr_keys = [0,1]
        group.attr_keys_bin = 0b00000011  # for simplicity, assuming 8-bit architecture

    Currently, decoding isn't necessary and therefore it is not implemented.  Consequently, the ``i2k`` and ``i2v``
    iterables (i.e., the decoding ones) aren't used, but are still populated.
    """

    def __init__(self):
        self.attr_k2i = {}  # encoding
        self.attr_v2i = []  # ^
        self.attr_i2k = []  # decoding
        self.attr_i2v = []  # ^

        self.rel_k2i  = {}  # encoding
        self.rel_v2i  = []  # ^
        self.rel_i2k  = []  # decoding
        self.rel_i2v  = []  # ^

    @staticmethod
    def _prep_obj(obj, is_enc=True):
        """Prepare the object to be encoded/decoded.

        If a single object is passed, it is turned into a list.  Only elements that are congruent with encoding/decoding
        are left in the list.  The list of conditions is:

        - Object is not None.
        - Object has ``attr`` and ``rel`` instance variables.
        - Object is (for decoding) or is not (for encoding) already encoded.

        Args:
            obj (object): The object (currently, an instance of Group or GroupQry).
            is_enc (bool, optional): Should the object be already encoded?

        Returns:
            Iterable[object]: The list of objects that meet the criteria specified above.
        """

        if not isinstance(obj, Iterable):
            obj = [obj]
        return [o for o in obj if o is not None and hasattr(o, 'attr') and hasattr(o, 'rel') and o.is_enc == is_enc]

    def decode(self, obj):
        """Decodes an object passed or a list of objects.

        Args:
            obj (object): The object (currently, an instance of Group or GroupQry).

        Returns:
            ``self``

        Todo:
            Currently unused and unneeded.  Implement if that changes.
        """

        return self

    def encode(self, obj):
        """Encodes an object passed or a list of objects.

        The object encoded will have two new instance variables set: ``attr_enc`` and ``rel_enc``.

        Args:
            obj (object): The object (currently, an instance of Group or GroupQry).

        Returns:
            ``self``
        """

        for o in self.__class__._prep_obj(obj, False):
            o.attr_enc = self.encode_attr(o.attr)
            o.rel_enc  = self.encode_rel(o.rel)
            o.is_enc   = True
        return self

    def encode_attr(self, attr):
        """Encodes an attributes dictionary.

        Args:
            attr (dict): Attributes to be encoded.

        Returns:
            set(tuple(int, int)): A set of tuples with two indices per tuple: The key and the value of the attribute.
        """

        return self.encode_dict(attr, self.attr_k2i, self.attr_v2i, self.attr_i2k, self.attr_i2v)

    def encode_dict(self, d, k2i, v2i, i2k, i2v):
        """Encodes a dictionary.

        Currently used only internally by the class to encode both attributes and relations dictionaries.

        Args:
            d (dict): Dictionary to be encoded.
            k2i (dict): Key-to-index.
            v2i (Iterable): Value-to-index.
            i2k (Iterable): Index-to-key.
            i2v (Iterable): Index-to-value.

        Returns:
            set(tuple(int, int)): A set of tuples with two indices per tuple: The key and the value of the dict.
        """

        enc = []
        for (k,v) in sorted(d.items()):  # sorting not necessary for sets, necessary for tuples
            if isinstance(v, Entity):
                v = v.get_hash()

            ki = k2i.get(k)
            if ki is None:
                ki = len(k2i)
                k2i[k] = ki
                v2i.append({})
                i2k.append(k)
                i2v.append([])

            v2i_k = v2i[ki]
            vi = v2i_k.get(v)
            if vi is None:
                vi = len(v2i_k)
                v2i_k[v] = vi
                i2v[ki].append(v)

            enc.append((ki,vi))
        return set(enc)

    def encode_probe(self, p):
        """Encodes GroupQry objects from the Probe instance.

        Args:
            p (Probe): Probe to be encoded.

        Returns:
            ``self``
        """

        if isinstance(p, GroupProbe):
            self.encode(p.queries)
            self.encode(p.qry_tot)
        return self

    def encode_rel(self, rel):
        """Encodes a relations dictionary.

        Args:
            rel (dict): Relations to be encoded.

        Returns:
            set(tuple(int, int)): A set of tuples with two indices per tuple: The key and the value of the relation.
        """

        return self.encode_dict(rel, self.rel_k2i, self.rel_v2i, self.rel_i2k, self.rel_i2v)


# ----------------------------------------------------------------------------------------------------------------------
class GroupPopulation(object):
    """Population of groups of agents.

    PRAM models functionally equivalent agents jointly as groups.  The formalism does not invoke any "hard" notion of a
    group population.  However, the PyPRAM package does use that concept to elegantly compartmentalize groups along
    with operations on them.

    Because groups can be associated with sites via group relations, those sites are also stored inside of this
    class' instance.

    VITA groups, or groups that are the source of new population mass are stored separately from the actual group
    population.  At the end of every iteration, masses of all VITA groups are transferred back to the population.  It is
    also at that time that all VOID groups are removed.  VOID groups contain mass that should be removed from the\
    simulation.

    Args:
        sim (Simulation): The simulation.
        do_keep_mass_flow_specs (bool, optional): Store the last iteration mass flow specs?  This is False by default
            for memory usage sake.  If set to True, ``self.last_iter.mass_flow_specs`` will hold the specs until they
            are overwriten at the next iteration of the simulation.
    """

    def __init__(self, sim, hist_len=0, do_keep_mass_flow_specs=False):
        self.sim = sim

        self.groups = {}
        self.sites = {}
        self.resources = {}  # TODO: doesn't seem to be used
        self.ar_enc = AttrRelEncoder()

        self.vita_groups = {}  # all VITA groups for the current iteration

        self.m     = 0  # total population mass
        self.m_in  = 0  # total population mass added (e.g., via the birth process)
        self.m_out = 0  # total population mass removed (e.g., via the death process)

        self.hist_len = hist_len
        self.hist = deque(maxlen=hist_len) if hist_len > 0 else None

        self.is_frozen = False  # the simulation freezes the population on first run

        self.last_iter = DotMap(    # the most recent iteration info
            mass_flow_tot = 0,      # total mass transferred
            mass_flow_specs = None  # a list of MassFlowSpec objects (i.e., the full picture of mass flow)
        )

        self.do_keep_mass_flow_specs = do_keep_mass_flow_specs

        # self.cache = DotMap(
        #     qry_to_groups = {},   # cache for get_groups(qry) calls
        #     qry_to_m      = {}    # cache for get_groups_mass(qry) calls
        # )

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
            ``self``
        """

        if not self.is_frozen:
            self.m += group.m

        rels = {}
        for (k,v) in group.rel.items():
            if isinstance(v, Entity):
                self.add_site(v)
                rels[k] = v.get_hash()
        group.set_rels(rels, True)

        group_hash = group.get_hash()
        if group_hash in self.groups.keys():
            self.groups.get(group_hash).m += group.m
        else:
            self.ar_enc.encode(group)
            group_hash = group.get_hash()
            group.pop = self
            group.link_to_site_at()
            self.groups[group_hash] = group

        return self

    def add_groups(self, groups):
        """Adds multiple groups to the population.

        See :meth:`~pram.pop.GroupPopulation.add_group` method for details.

        Args:
            groups (Iterable[Group]): Groups to be added.

        Returns:
            ``self``
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
            ``self``
        """

        h = resource.get_hash()
        if h not in self.resources.keys():
            self.resources[h] = resource
            # resource.set_pop(self)
        # return self
        return self.resources[h]

    def add_resources(self, resources):
        """Adds multiple resources to the population.

        See :meth:`~pram.pop.GroupPopulation.add_resource` method for details.

        Args:
            resource (Resource): The resource to be added.

        Returns:
            ``self``
        """

        for r in resources:
            self.add_resource(r)
        return self

    def add_site(self, site):
        """Adds a site to the population if it doesn't exist.

        Args:
            site (Site): The site to be added.

        Returns:
            ``self``
        """

        h = site.get_hash()
        if h not in self.sites.keys():
            self.sites[h] = site
            site.set_pop(self)
        return site

    def add_sites(self, sites):
        """Adds multiple sites to the population.

        See :meth:`~pram.pop.GroupPopulation.add_site` method for details.

        Args:
            sites (Site): The sites to be added.

        Returns:
            ``self``
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
            ``self``
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
            is_rule_setup (bool): Is this invocation of this method during rule setup stage of the simulation?
            is_rule_cleanup (bool): Is this invocation of this method during rule cleanup stage of the simulation?
            is_sim_setup (bool): Is this invocation of this method during simulation setup stage?

        Returns:
            ``self``

        Todo:
            Optimize by using hashes if groups already exist and Group objects if they don't.
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

        return self.transfer_mass(src_group_hashes, mass_flow_specs, iter, t, is_sim_setup)

    def archive(self):
        if self.hist_len == 0:
            return
        self.hist.append(GroupPopulationHistory(self))

    def compact(self):
        """Compacts the population by removing empty groups.

        Whether this is what the user needs is up to them.  However, for large simulations with a high new group
        turnover compacting the group population will make the simulation run faster.  Autocompacting can be turned on
        on the simulation level via the `autocompact` pragma (e.g., via :meth:`~pram.sim.Simulation.set_pragma_autocompact`
        method).

        Returns:
            ``self``
        """

        self.groups = { k:v for k,v in self.groups.items() if v.m > 0 }
        return self

    def do_post_iter(self):
        """Performs all post-iteration routines (i.e., current iter cleanup and next iter prep).

        Routines performed:

        (1) Remove VOID groups.
        (2) Move mass from VITA groups to their corresponding groups.

        Returns:
            ``self``
        """

        # (1) Remove VOID groups (no dict comprehension because we want to edit the existing dict in-place):
        del_keys = []
        for (k,v) in self.groups.items():
            if v.is_void():
                # print(-v.m)
                self.m     -= v.m
                self.m_out += v.m
                del_keys.append(k)
                # print(f'{k}: {v}')
        # for _ in del_keys:
        #     if k in self.groups.keys():
        #         del self.groups[k]
        self.groups = { k:v for k,v in self.groups.items() if not v.is_void() }

        # (2) Move mass from VITA groups to their corresponding groups:
        for (k,v) in self.vita_groups.items():
            # print(v.m)
            self.m    += v.m
            self.m_in += v.m
            self.groups[k].m += v.m
        self.vita_groups = {}

        return self

    def freeze(self):
        """Freeze the population.

        The :class:`~pram.sim.Simulation` object freezes the population on first run.  Freezing a population is used
        only used to determine the total population size.

        Returns:
            ``self``
        """

        # [g.freeze() for g in self.groups.values()]
        # self.groups = { g.get_hash(): g for g in self.groups.values() }

        self.is_frozen = True
        # if self.sim.traj is not None and not self.sim.timer.i > 0:  # we check timer not to save initial state of a simulation that's been run before
        #     self.sim.traj.save_state(None)
        self.sim.save_state(None)

        return self

    def gen_agent_pop(self):
        """Generates a population of agents based on the current groups population.

        Todo:
            Implement.

        Returns
            AgentPopulation
        """

        pass

    def get_group(self, attr, rel={}):
        """
        Returns the group with the all attributes and relations as specified; or None if such a group does not exist.

        Args:
            attr (Mapping[str, Any]): Group's attributes.
            rel (Mapping[str, Site], optional): Group's relations.

        Returns:
            Group if a group is found; None otherwise
        """

        return self.groups.get(Group.gen_hash(attr, rel))

    def get_group_cnt(self, only_non_empty=False):
        """Get number of groups.

        Args:
            only_non_empty (bool, optional): Count only non-empty group?

        Returns:
            int
        """

        if only_non_empty:
            return len([g for g in self.groups.values() if g.m > 0])
        else:
            return len(self.groups)

    @lru_cache(maxsize=None)
    def get_groups(self, qry=None):
        """Get groups that match the group query specified, or all groups if no query is specified.

        This method should only be used for population-wide queries.  The
        :meth:`Site.get_groups() pram.entity.Site.get_groups` should be used instead for querying groups located at a
        :class:`~pram.entity.Site`.

        Args:
            qry (GroupQry, optional): The group query.

        Returns:
            [Group]: List of groups (can be empty)
        """

        if qry is None:
            return self.groups.values()

        # groups = self.cache.qry_to_groups.get(qry)
        # if groups is None:
        #     groups = [g for g in self.groups.values() if g.matches_qry(qry)]
        #     self.cache.qry_to_groups[qry] = groups
        # return groups

        return [g for g in self.groups.values() if g.matches_qry(qry)]

    @lru_cache(maxsize=None)
    def get_groups_mass(self, qry=None, hist_delta=0):
        """Get the mass of groups that match the query specified.

        This method should only be used for population-wide queries.  The
        :meth:`Site.get_groups_mass() pram.entity.Site.get_groups_mass` should be used instead for querying groups
        located at a :class:`~pram.entity.Site`.

        Args:
            qry (GroupQry, optional): Group condition.
            hist_delta (int, optional): Number of steps back in the history to base the delta off of.  For instance, a
                value of 2 will yield the change of mass compared to two iterations back.  The group's history must be
                long enough (controlled by ``hist_len`` attribute).

        Returns:
            float: Mass

        Raises:
            ValueError
        """

        if hist_delta == 0:
            return math.fsum([g.m for g in self.get_groups(qry)])
        else:
            if hist_delta > self.hist_len:
                raise ValueError('History delta provided (hist_delta) is larger than the history depth (hist_len).')

            m = 0.0
            if len(self.hist) >= hist_delta:
                hist_groups_m = self.hist[-hist_delta].groups_m
                for g in self.get_groups(qry):
                    m += g.m - hist_groups_m.get(g, 0.0)
            return m

    def get_groups_mass_prop(self, qry=None):
        """Get the proportion of the total mass accounted for the groups that match the query specified.

        This method should only be used for population-wide queries.  The
        :meth:`Site.get_groups_prop() pram.entity.Site.get_groups_prop` should be used instead for querying groups
        located at a :class:`~pram.entity.Site`.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            float: Mass proportion
        """

        return self.get_groups_mass(qry) / self.m if self.m > 0 else 0

    def get_groups_mass_and_prop(self, qry=None):
        """Get the total mass and its proportion that corresponds to the groups that match the query specified.

        This method should only be used for population-wide queries.  The
        :meth:`Site.get_groups_mass_prop() pram.entity.Site.get_groups_mass_prop` should be used instead for querying
        groups located at a :class:`~pram.entity.Site`.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            tuple(float, float): (Mass, Mass proportion)
        """

        m = self.get_groups_mass(qry)
        return (m, m / self.m if self.m > 0 else 0)

    def get_mass(self):
        # return sum([g.m for g in self.groups.values()])
        return self.m

    def get_next_group_name(self):
        """Get the next group name.

        To be used by sequential group adding if naming groups is desired.

        Returns:
            str
        """

        return f'g.{len(self.groups)}'

    def get_site_cnt(self):
        """Get number of sites.

        Returns:
            int
        """

        return len(self.sites)

    def transfer_mass(self, src_group_hashes, mass_flow_specs, iter, t, is_sim_setup):
        """Transfers population mass.

        Transfers the mass as described by the list of "destination" groups.  "Source" groups (i.e., those that
        participate in mass transfer) have their masses reset before the most-transfer mass is tallied up.

        Because this method is called only once per simulation iteration, it is a good place to put simulation-
        wide computations that should happen after the iteration-specific computations have concluded.

        Returns:
            ``self``
        """

        m_flow_tot = 0  # total mass transferred

        # Reset the mass of the groups being updated:
        for h in src_group_hashes:
            self.groups[h].m = 0.0
            # if not is_sim_setup:
            #     # self.groups[h].m_delta = -self.groups[h].m
            #     self.groups[h].archive()

        # if not is_sim_setup:
        #     for mfs in mass_flow_specs:
        #         for g01 in mfs.dst:
        #             g01.m_delta = -g01.m

        for mfs in mass_flow_specs:
            for g01 in mfs.dst:
                g02 = self.groups.get(g01)

                if g02 is not None:  # group already exists
                    g02.m       += g01.m
                    # g02.m_delta += g01.m
                    # print(g02.m_delta)
                else:                # group not found
                    self.add_group(g01)
                    # g01.m_delta = -g01.m

                m_flow_tot += g01.m

        # Save last iteration info:
        self.last_iter.m_flow_tot = m_flow_tot
        if self.do_keep_mass_flow_specs:
            self.last_iter.mass_flow_specs = mass_flow_specs

        # Save the trajectory state:
        # if self.sim.traj is not None:
        #     self.sim.traj.save_state(mass_flow_specs)
        if not is_sim_setup:
            self.sim.save_state(mass_flow_specs)
        # self.sim.save_state([mfs.m_pop for mfs in mass_flow_specs])

        # Relink groups to the sites they are currently at:
        for s in self.sites.values():
            s.reset_group_links()
        for g in self.groups.values():
            g.link_to_site_at()

        # Finish up:
        self.get_groups.cache_clear()
        self.get_groups_mass.cache_clear()
        self.archive()

        return self


# ----------------------------------------------------------------------------------------------------------------------
class GroupPopulationHistory(object):
    """History of the GroupPopulation object's states.

    For efficiency, this class operates on a group's hash (i.e., a integer) and a group's mass (i.e., a float) and does
    not store any actual Group objects.  GroupPopulation object's groups dict is necessary for recovery of full group
    definition.

    Args:
        pop (GroupPopulation): Group population to be archived.
    """

    def __init__(self, pop):
        self.m     = pop.m
        self.m_in  = pop.m_in
        self.m_out = pop.m_out

        self.groups_m = {}  # group masses
        for g in pop.groups.values():
            self.groups_m[g] = g.m

    def get_group_m(self, group_hash, ret_if_not_found=0.0):
        return self.groups_m.get(group_hash, ret_if_not_found)
