# -*- coding: utf-8 -*-
"""Contains PRAM entity code.

The three types of entities which can comprise a PRAM model are *groups*, *sites*, and *resources*.
"""

import copy
import gc
import hashlib
import inspect
import itertools
import json
import jsonpickle
import math
import numpy as np
import os
import xxhash

from dotmap import DotMap

from abc             import ABC
from attr            import attrs, attrib, converters, validators
from collections.abc import Iterable
from enum            import auto, unique, IntEnum
from iteround        import saferound
from scipy.stats     import rv_continuous

from .util import DB, Err, FS


# ----------------------------------------------------------------------------------------------------------------------
class GroupFrozenError(Exception):
    """Raised when an attempt at modifying a frozen group is made.

    Once instantiated, PRAM groups can only be subject to arbitraty modifications when they are not part of a PRAM
    simulation.  That is to prevent the user from inadvertantly interfering with the group splitting mechanics of PRAM.
    That mechanics dictates that group masses, attributes, and relations (i.e., the very definition of PRAM groups) are
    never changed; instead, a group at iteration *n* is split into multiple groups and all new groups that result from
    all group splittings are combined to form new groups at iteration *n+1*.
    """

    pass


# ----------------------------------------------------------------------------------------------------------------------
@unique
class AttrSex(IntEnum):
    F = 1
    M = 2


# ----------------------------------------------------------------------------------------------------------------------
class DistributionAgeSchool(rv_continuous):
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)


class DistributionAgeWork(rv_continuous):
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------------------------------------------------------------
class EntityType(IntEnum):
    """Entity type enum.

    Note:
        The ``AGENT`` type is not currently used and is included here or potential future extensions.
    """

    AGENT    = 1
    GROUP    = 2
    SITE     = 3  # e.g., home, school, etc.
    RESOURCE = 4  # e.g., a public bus


class Entity(ABC):
    """Entity base class.

    Args:
        type (int): Entity type (see :class:`~pram.entity.EntityType` enum).
        id (str): Entity identifier string (currently not used and pending removal).

    Todo:
        Remove the ``id`` argument
    """

    __slots__ = ('type', 'id')

    def __init__(self, type, id):
        self.type = type
        self.id   = id

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return '{}  type: {}   id: {}'.format(self.__class__, self.type.name, self.id)


# ----------------------------------------------------------------------------------------------------------------------
class Resource(Entity):
    """A resource entity.

    A resource is shared by multiple agents (e.g., a public bus).

    This is a basic implementation of a shared resource and can safely be used only within a single simulation.  A more
    elaborate implementation based on synchronization mechanisms will be provided later and will accommodate multiple
    concurrent simulations of different and interacting systems with the agent population moving seamlessly between
    them.

    The terminology used in the API of this class is consistent with that of concurent computing.  For example, a
    resource is said to *accommodate* agents and is *released* after an agent (or agents) are done using it.  A
    resource's capacity dicates how many agents can be accommodated at the same time.

    Args:
        name (str): A resource's name.
        capacity_max (int): The maximum number of agents that can be using the resouce concurrently.
    """

    __slots__ = ('name', 'capacity', 'capacity_max')

    def __init__(self, name, capacity_max=1):
        super().__init__(EntityType.RESOURCE, '')

        self.name = name
        self.capacity = 0
        self.capacity_max = capacity_max

    def __eq__(self, other):
        '''
        We will make two resources identical if their keys are equal (i.e., object identity is not necessary).  This
        will let us recognize resources even if they are instantiated multiple times.
        '''

        return isinstance(self, type(other)) and (self.__key() == other.__key())

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return '{}({} {} {})'.format(self.__class__.__name__, self.name, self.capacity, self.capacity_max)

    def __str__(self):
        return '{}  name: {:16}  cap: {}/{}  hash: {}'.format(self.__class__.__name__, self.name, self.capacity, self.capacity_max, self.__hash__())

    def __key(self):
        return (self.name)

    def allocate(self, n, do_all=False):
        """Allocate the resource to ``n`` agents.

        Args:
            n (int): The number of agents.
            do_all (bool): Flag: Accomodate all-or-nothing or any that the resourcs can accomodate.

        Returns:
            int: Number of not accomodated agents (i.e., those over the resource's max capacity).
        """

        if do_all:
            return self.accommodate_all(n)
        else:
            return self.accommodate_any(n)

    def allocate_any(self, n):
        """Allocates as many out of ``n`` agents as possible.

        Args:
            n (int): The number of agents.
        Returns:
            int: Number of not accommodated agents (i.e., those over the resource's max capacity).
        """

        n_accommodated = self.capacity_max - n
        self.capacity += n_accommodated
        return n - n_accommodated

    def allocate_all(self, n):
        """Attempts to allocate all ``n`` agents.

        Args:
            n (int): The number of agents.

        Returns:
            bool: True if all agents can be accommodated and False otherwise.
        """

        if self.capacity + n <= self.capacity_max:
            self.capacity += n
            return True
        else:
            return False

    def can_accommodate_all(self, n):
        """Checks if all ``n`` agents be accomodated.

        Args:
            n (int): The number of agents.

        Returns:
            bool: True if all ``n`` agents can be accomodated and False otherwise.
        """

        return self.capacity + n <= self.capacity_max

    def can_accommodate_any(self, n):
        """Checks if at least one agent can be accomodated.

        Args:
            n (int): The number of agents.

        Returns:
            bool: True if at least one agent can be accomodated and False otherwise.
        """

        return self.capacity < self.capacity_max

    def can_accommodate_one(self):
        """Checks if at least one agent can be accomodated.

        Returns:
            bool: True if at least one agent can be accomodated and False otherwise.
        """

        return self.capacity < self.capacity_max

    def get_capacity(self):
        return self.capacity

    def get_capacity_left(self):
        return self.capacity_max - self.capacity

    def get_capacity_max(self):
        return self.capacity_max

    def get_hash(self):
        return self.__hash__()

    def release(self, n):
        """Releases ``n`` agent spots.

        Because releasing is always allowed, no return value needs to be checked.  Naturally, a resource cannot release
        more spots than it's max capacity.

        Args:
            n (int): The number of agents.
        """

        if self.capacity == 0:
            return

        self.capacity = max(0, self.capacity - n)

    def toJson(self):
        # return json.dumps(self, default=lambda o: o.__dict__)
        return json.dumps(self, default=lambda o: o.__dict__)


# ----------------------------------------------------------------------------------------------------------------------
class EntityJSONEncoder(json.JSONEncoder):
    """JSON encoder used by the :meth:`~pram.entity.Group.gen_hash` method.

    JSON encoding is used when hashing :class:`~pram.enity.Group` objects.  Because those objects may hold references
    to :class:`~pram.enity.Site` and :class:`~pram.enity.Resource` objects, we accomodate them here.
    """

    def default(self, o):
        """The method to be implemented when extending json.JSONEncoder.

        Args:
            o (object): The object to be encoded.

        Returns:
            The JSON-encoded representation of the object.
        """

        if isinstance(o, Site):
            return { '__Site__': o.get_hash() }  # return {'__Site__': o.__hash__()}
        if isinstance(o, Resource):
            return { '__Resource__': o.get_hash() }  # return { '__Resource__': o.name }
        if callable(o):
            return { '__function__': xxhash.xxh64(str(inspect.getsource(o))).hexdigest() }
        return json.JSONEncoder.default(self, o)


# ----------------------------------------------------------------------------------------------------------------------
class Site(Resource):
    """A physical site (e.g., a school or a store) agents can reside at.

    A site has a sensible interface which makes it useful.  For example, it makes sense to ask about the size and
    composition of population (e.g., groups) that are at that location.  However, because this information (and other,
    similar pieces of information) may be desired at arbitrary times, it makes most sense to compute it lazily and
    memoize it.  For that reason, a site stores a link to the population it is associated with; it queries that
    population to compute quantities of interested when they are needed.  An added benefit of this design is fostering
    proper composition; that is, updating the state of a site should be done by a site, not the population.

    A Site is a Resource which means a Site may choose to utilize the capacity property with all its related methods.
    A good example of when this would be useful is a hospital with a limited patient capacity that may be reached
    during an epidemic outbreak.

    Args:
        name (str): Name of the site.
        attr (Mapping[], optinoal): Attributes describing the site.
        rel_name (str): The name of the relation that should be used when associating a group of agents with this site.
        pop (GroupPopulation, optional): The GroupPopulation object.
        capacity_max (int): The maximum capacity of the Site when considered a Resource.
    """

    AT = '@'  # relation name for the group's current site

    __slots__ = ('name', 'attr', 'rel_name', 'pop', 'm', 'groups', 'hash', 'cache_qry_to_groups', 'cache_qry_to_m')

    def __init__(self, name, attr=None, rel_name=AT, pop=None, capacity_max=1):
        super().__init__(name, capacity_max)  # previously called as: (EntityType.SITE, '')

        self.name     = name
        self.rel_name = rel_name    # name of the relation the site is the object of
        self.attr     = attr or {}
        self.pop      = pop         # pointer to the population (can be set elsewhere too)
        self.m        = 0.0
        self.groups   = set()

        self.hash = None  # computed lazily

        # self.cache = DotMap(      # reset by the GroupPopulation object (that manages sites and groups) after mass transfer that crowns each iteration
        #     qry_to_groups = {},   # groups currently at this site
        #     qry_to_m      = {}    # mass of population at this site
        # )
        self.cache_qry_to_groups = {}   # groups currently at this site
        self.cache_qry_to_m      = {}   # mass of population at this site

    def __eq__(self, other):
        return isinstance(self, type(other)) and (self.name == other.name) and (self.rel_name == other.rel_name) and (self.attr == other.attr)

    def __hash__(self):
        if self.hash is None:
            self.hash = Site.gen_hash(self.name, self.rel_name, self.attr)
        return self.hash

    @staticmethod
    def gen_hash(name, rel_name, attr={}):
        return xxhash.xxh64(json.dumps((name, rel_name, attr), sort_keys=True)).intdigest()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__hash__())

    def __str__(self):
        return '{}  name: {:16}  hash: {}  attr: {}'.format(self.__class__.__name__, self.name, self.__hash__(), self.attr)

    def __key(self):
        return (self.name)

    def add_group_link(self, group):
        """Adds a link to a group.  This is to speed up lookup from sites to groups.

        Args:
            group (Group): The group.

        Returns:
            self: For method call chaining.
        """

        self.groups.add(group)
        # self.groups.add(group.get_hash())
        self.m += group.m
        return self

    def freeze(self):
        """Freezes the site thus disallowing any direct changes to it."""

        self.is_frozen = True

    def ga(self, name=None):
        """See :meth:`~pram.entity.Site.get_attr` method."""

        return self.get_attr(name)

    @classmethod
    def gen_from_db(cls, db_fpath, tbl, name_col, rel_name=AT, attr=[], limit=0):
        """Generates sites from a relational database.

        Args:
            db_fpath (str): Path to the database file (SQLite3).
            tbl (str): Table name.
            name_col (str): Table column storing names of sites.
            rel_name (str): Name of the relation to be associated with each of the sites generated.  For example, if
                hospital sites are being generated, the ``rel_name`` could be set to ``hospital``.
            attr (Iterable[str]): Names of table columns storing attributes to be internalized by the site objects
                being generated.
            limit (int): The maximum number of sites to be generated.  Ordinarily, this is not changed from its default
                value of zero.  It is however useful for testing, especially with very large databases.

        Returns:
            dict(str, Site): A dictonary of sites with the Site object's database IDs as keys.  Previously, Site hashes
                were used as keys, but a recent change in the internal design did away with that.
        """

        FS.req_file(db_fpath, f'The database does not exist: {db_fpath}')

        sites = {}
        with DB.open_conn(db_fpath) as c:
            for row in c.execute('SELECT {} FROM {}{}'.format(','.join(attr + [name_col]), tbl, '' if limit <= 0 else f' LIMIT {limit}')).fetchall():
                s = cls(row[name_col], { a: row[a] for a in attr }, rel_name=rel_name)
                # sites[s.get_hash()] = s  # old ways of handling sites
                sites[row[name_col]] = s

        return sites

    @classmethod
    def gen_from_db_tmp1(cls, sim, db_fpath, tbl, name_col, rel_name=AT, attr=[], limit=0):
        """A legacy method pending non-immediate removal.

        Goes together with Group.gen_from_db_tmp1().  See the comment :meth:`therein pram.entity.Group.gen_from_db_tmp1`.
        """

        if not os.path.isfile(db_fpath):
            raise ValueError(f'The database does not exist: {db_fpath}')

        # sites = {}
        with DB.open_conn(db_fpath) as c:
            for row in c.execute('SELECT {} FROM {}{}'.format(','.join(attr + [name_col]), tbl, '' if limit <= 0 else f' LIMIT {limit}')).fetchall():
                s = cls(row[name_col], { a: row[a] for a in attr }, rel_name=rel_name)
                # sites[s.get_hash()] = s
                sim.pop.add_site(s)

        # return sites

    def get_attr(self, name=None):
        """Retrieves attribute's value.

        Args:
            name (str): Attribute's name.

        Returns:
            Any: Attribute's value.
        """

        return self.attr.get(name) if name is not None else self.attr

    def get_groups(self, qry=None, non_empty_only=False):
        """Returns groups which currently are at this site.

        The word "currently" is key here because as a PRAM simulation evolved, agents move between groups and different
        compositions of groups will in general reside at any given site.

        Args:
            qry (GroupQry, optional): Further restrictions imposed on attributes and relations of groups currently at
                this site.  For example, the user may be interested in retrieving only groups of agents infected with
                the flu (which is a restriction on the group's attribute) or only groups of agents who attend a
                specific school (which is a restriction on the group's relation).
            non_empty_only (bool): Flag: Return only groups with non-zero agent population mass?

        Returns:
            list[Group]: List of groups currently at this site.

        Todo:
            Implement memoization (probably of only all the groups, i.e., without accounting for the ``qry``).
        """

        # No optimization:
        # qry = qry or GroupQry()
        # # qry.rel.update({ self.rel_name: self.get_hash() })
        # qry.rel.update({ self.rel_name: self })
        # groups = self.pop.get_groups(qry)
        #
        # if non_empty_only:
        #     return [g for g in groups if g.m > 0]
        # else:
        #     return groups

        # Cache results by GroupQry (currently doesn't work with attrs and looses groups with regular class due to a bug somewhere):
        # qry = qry or GroupQry()
        # qry.rel.update({ self.rel_name: self })
        #
        # # if not qry in self.groups.keys():
        # #     self.groups[qry] = self.pop.get_groups(qry)
        # groups = self.cache_qry_to_groups.get(qry)
        # if groups is None:
        #     groups = self.pop.get_groups(qry)
        #     self.cache_qry_to_groups[qry] = groups
        #
        # if non_empty_only:
        #     return [g for g in groups if g.m > 0]
        # else:
        #     return groups

        # Current attempt:
        # groups = [g for g in self.groups if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items()) and all([fn(g) for fn in qry.cond])]

        if not qry:
            groups = self.groups
        else:
            groups = self.cache_qry_to_groups.get(qry.__hash__())
            if groups is None:
                groups = [g for g in self.groups if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items()) and all([fn(g) for fn in qry.cond])]
                # groups = []
                # for gh in self.groups:
                #     g = self.pop.groups[gh]
                #     if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items()) and all([fn(g) for fn in qry.cond]):
                #         groups.append(g)
                self.cache_qry_to_groups[qry.__hash__()] = groups

        if non_empty_only:
            return [g for g in groups if g.m > 0]
        else:
            return groups

    def get_mass(self, qry=None):
        """Get the mass of groups that match the query specified.  Only groups currently residing at the site are
        searched.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            float: Mass
        """

        if not qry:
            return self.m
        else:
            m = self.cache_qry_to_m.get(qry)
            if m is None:
                m = math.fsum(g.m for g in self.get_groups(qry))
                self.cache_qry_to_m[qry] = m
            # else:
            #     print('hit')
            return m

    def get_mass_prop(self, qry=None):
        """Get the proportion of the total mass accounted for the groups that match the query specified.  Only groups
        currently residing at the site are searched.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            float: Mass proportion
        """

        return self.get_mass(qry) / self.m if self.m > 0 else 0

    def get_mass_and_prop(self, qry=None):
        """Get the total mass and its proportion that corresponds to the groups that match the query specified.  Only
        groups currently residing at the site are searched.

        Args:
            qry (GroupQry, optional): Group condition.

        Returns:
            tuple(float, float): (Mass, Mass proportion)
        """

        m = self.get_mass(qry)
        return (m, m / self.m if self.m > 0 else 0)

    def set_pop(self, pop):
        """Sets the group population.

        Args:
            pop (GroupPopulation): The group population.

        Returns:
            self: For method call chaining.
        """

        self.pop = pop
        return self

    def reset_group_links(self):
        """Resets the groups located at the site and other cache and memoization data structures.

        Returns:
            self: For method call chaining.
        """

        self.groups = set()
        self.m = 0.0
        # self.cache.qry_to_groups = {}
        # self.cache.qry_to_m = {}
        self.cache_qry_to_groups = {}
        self.cache_qry_to_m = {}
        return self

    # def toJson(self):
    #     # return json.dumps(self, default=lambda o: o.__dict__)
    #     return json.dumps(self.hash)


# ----------------------------------------------------------------------------------------------------------------------
class Agent(Entity):
    """A singular agent.

    A legacy class pending non-immediate removal or a proper integration into PRAM framework.  As such, it makes no
    sense to document it well at this point.
    """

    __slots__ = ('name', 'sex', 'age', 'school', 'work', 'location')

    AGE_MIN =   0
    AGE_MAX = 120
    AGE_M   =  40
    AGE_SD  =  20

    P_STUDENT = 0.25  # unconditional prob. of being a student
    P_WORKER  = 0.60  # unconditional prob. of being a worker

    def __init__(self, name=None, sex=AttrSex.F, age=AGE_M, school=None, work=None, location='home'):
        super().__init__(EntityType.AGENT, '')

        self.name     = name or '.'
        self.sex      = sex
        self.age      = age
        self.school   = school
        self.work     = work
        self.location = location

    def __repr__(self):
        return '{}(name={}, sex={}, age={}, school={}, work={}, location={})'.format(self.__class__.__name__, self.name, self.sex.name, round(self.age, 2), self.school, self.work, self.location)

    def __str__(self):
        return '{}  name: {:12}  sex:{}  age: {:3}  school: {:16}  work: {:16}  location: {:12}'.format(self.__class__.__name__, self.name, self.sex.name, round(self.age), self.school or '.', self.work or '.', self.location or '.')

    @classmethod
    def gen(cls, name=None):
        """Generates a singular agent."""

        name     = name or '.'
        sex      = Agent.random_sex()
        age      = Agent.random_age()
        school   = None
        work     = None
        location = 'home'

        # Student:
        if (np.random.random() > Agent.P_STUDENT):
            school = np.random.choice(['school-01', 'school-02', 'school-03'], p=[0.6, 0.2, 0.2])
            if (np.random.random() > 0.3):
                location = school

        # Worker:
        if (np.random.random() > Agent.P_WORKER):
            work = np.random.choice(['work-01', 'work-02'], p=[0.5, 0.5])
            if (np.random.random() > 0.4):
                location = work

        return cls(name, sex, age, school, work, location)

    @classmethod
    def gen_lst(cls, n):
        """Generates a list of agents (with auto-incrementing names)."""

        if n <= 0:
            return []
        return [cls.gen('a.{}'.format(i)) for i in range(n)]

    @staticmethod
    def random_age():
        return min(Agent.AGE_MAX, max(Agent.AGE_MIN, np.random.normal(Agent.AGE_M, Agent.AGE_SD)))

    @staticmethod
    def random_sex():
        return AttrSex(np.random.choice(AttrSex))


# ----------------------------------------------------------------------------------------------------------------------
# Classes based on attrs are not hashable.  This definitoin is kept here for later investigation, but moving to regular
# Python classes as of Mar 27, 2020.

# @attrs(slots=True)
# class GroupQry(object):
#     """A group query.
#
#     Objects of this class are used to select groups from a group population using attribute- and relation-based search
#     criteria.
#
#     Typical usage example for selecting groups of agents that meet certain criteria:
#
#         GroupQry(attr={ 'flu': 's' })                 # susceptible to the flu
#         GroupQry(rel={ Site.AT: Site('school-83') })  # currently located at site 'school-83'
#
#         GroupQry(cond=[lambda g: g.get_attr('x') > 100]))                                   # with attribute 'x' > 100
#         GroupQry(cond=[lambda g: g.get_attr('x') > 100, lambda g: get_attr('y') == 200]))   # with attribute 'x' > 100 and 'y' == 200
#         GroupQry(cond=[lambda g: g.get_attr('x') > 100 and g.get_attr('y') ==  200]))       # explicit AND condition between attributes
#         GroupQry(cond=[lambda g: g.get_attr('x') > 100 or  g.get_attr('y') == -200]))       # explicit OR  condition between attributes
#
#     It would make sense to declare this class frozen (i.e., 'frozen=True'), but as is revealed by the following two
#     measurements, performance suffers slightly when slotted classes get frozen.
#
#     python -m timeit -s "import attr; C = attr.make_class('C', ['x', 'y', 'z'], slots=True)"             "C(1,2,3)"
#     python -m timeit -s "import attr; C = attr.make_class('C', ['x', 'y', 'z'], slots=True,frozen=True)" "C(1,2,3)"
#
#     Args:
#         attr (Mapping[str, Any], optional): Group's attributes.
#         rel (Mapping[str, Any], optional): Group's relations.
#         cond (list(Callable), optional): Conditions on group's attributes and relations.  These conditions are given as
#             callables which take one argument, the group.  Assuming the group argument is ``g``, the callables can then
#             access the group's attributes and relations respectively as ``g.attr`` and ``g.rel``.  See the typical
#             usage examples above.
#         full (bool): Flag: Does the match need to be full?  To satisfy a full match, a group's attributes and relations
#             need to fully match the query.  Because PRAM cannot have two groups with the same attributes and relations,
#             it follows that a full match can either return one group on no groups (if no match exists).  A partial
#             match requires that a group's attributes _contain_ the query's attributes (and same for relations).
#     """
#
#     attr : dict = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
#     rel  : dict = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
#     cond : list = attrib(factory=list, converter=converters.default_if_none(factory=list))
#     full : bool = attrib(default=False)


# ----------------------------------------------------------------------------------------------------------------------
class GroupQry(object):
    """A group query.

    Objects of this class are used to select groups from a group population using attribute- and relation-based search
    criteria.

    Typical usage example for selecting groups of agents that meet certain criteria:

        GroupQry(attr={ 'flu': 's' })                 # susceptible to the flu
        GroupQry(rel={ Site.AT: Site('school-83') })  # currently located at site 'school-83'

        GroupQry(cond=[lambda g: g.get_attr('x') > 100]))                                   # with attribute 'x' > 100
        GroupQry(cond=[lambda g: g.get_attr('x') > 100, lambda g: get_attr('y') == 200]))   # with attribute 'x' > 100 and 'y' == 200
        GroupQry(cond=[lambda g: g.get_attr('x') > 100 and g.get_attr('y') ==  200]))       # explicit AND condition between attributes
        GroupQry(cond=[lambda g: g.get_attr('x') > 100 or  g.get_attr('y') == -200]))       # explicit OR  condition between attributes

    Group query objects do not have any utility outside of a simulation context (which implies population bound groups)
    and consequently won't play well with standalong groups because all Site references are turned into their hashes
    (which is what a GroupPopulation object operates on internally).

    Args:
        attr (Mapping[str, Any], optional): Group's attributes.
        rel (Mapping[str, Any], optional): Group's relations.
        cond (list(Callable), optional): Conditions on group's attributes and relations.  These conditions are given as
            callables which take one argument, the group.  Assuming the group argument is ``g``, the callables can then
            access the group's attributes and relations respectively as ``g.attr`` and ``g.rel``.  See the typical
            usage examples above.
        full (bool): Flag: Does the match need to be full?  To satisfy a full match, a group's attributes and relations
            need to fully match the query.  Because PRAM cannot have two groups with the same attributes and relations,
            it follows that a full match can either return one group on no groups (if no match exists).  A partial
            match requires that a group's attributes _contain_ the query's attributes (and same for relations).
    """

    __slots__ = ('attr', 'rel', 'cond', 'full', 'hash')

    def __init__(self, attr={}, rel={}, cond=[], full=False):
        self.attr = attr
        self.rel  = rel
        self.cond = cond
        self.full = full

        self.hash = None  # computed lazily

        for (k,v) in self.rel.items():
            if isinstance(v, Site):
                self.rel[k] = v.__hash__()
            # TODO: If we need to distinguish between Sites and Resources, do it here and also look in
            #       GroupPopulation.add_group().

    def __eq__(self, other):
        return isinstance(self, type(other)) and (self.attr == other.attr) and (self.rel == other.rel) and (self.cond == other.cond) and (self.full == other.full)

    def __hash__(self):
        if self.hash is None:
            self.hash = GroupQry.gen_hash(self.attr, self.rel, self.cond, self.full)
        return self.hash

    @staticmethod
    def gen_hash(attr={}, rel={}, cond=[], full=False):
        # hash = xxhash.xxh64(json.dumps((attr, rel, cond, full), sort_keys=True, cls=EntityJSONEncoder)).intdigest()
        # hash = xxhash.xxh64(jsonpickle.encode((attr, rel, str([inspect.getsource(i) for i in cond], full))).intdigest()
        return xxhash.xxh64(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True)).intdigest()

    # def toJson(self):
    #     # return json.dumps(self, default=lambda o: o.__dict__)
    #     return json.dumps(self.hash)


# ----------------------------------------------------------------------------------------------------------------------
@attrs(kw_only=True, slots=True)
class GroupSplitSpec(object):
    """A single group-split specification.

    These specifications are oridinarily provided in a list to indicate new groups that one other group is being split
    into at the end of a rule's application.

    Args:
        p (float): The probability of the agents mass being trasnferred to the new group.
        attr_set (Mapping[str, Any]): Attributes to be set in the new group.
        attr_del (Iterable(str)): Attributes to be removed from the new group (with respect to the current group).
        rel_set (Mapping[str, Any]): Relations to be set in the new group.
        rel_del (Iterable(str)): Relations to be removed from the new group (with respect to the current group).

    Todo:
        At this point, attributes and relations to be removed are assumed to be identified by their names only and not
        not their values (i.e., we use a set to hold the keys that should be removed from the dictionaries for
        attributes and relations).  Perhaps this is not the way to go and we should instead be using both names and
        values.
    """

    p        : float = attrib(default=0.0,  converter=float)  # validator=attr.validators.instance_of(float))
    attr_set : dict  = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
    attr_del : set   = attrib(factory=set,  converter=converters.default_if_none(factory=set))
    rel_set  : dict  = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
    rel_del  : set   = attrib(factory=set,  converter=converters.default_if_none(factory=set))

    @p.validator
    def is_prob(self, attribute, value):
        if not isinstance(value, float):
            raise TypeError(Err.type('p', 'float'))
        if not (0 <= value <= 1):
            raise ValueError("The probability 'p' must be in [0,1] range.")


# ----------------------------------------------------------------------------------------------------------------------
@attrs()
class GroupDBRelSpec(object):
    """Specification for group relation to be retrieved from a relational database.

    Args:
        name (str): Table name.
        col (str): Column name.
        sites (Mapping[str, Site]):
    """

    name  : str  = attrib()
    col   : str  = attrib()
    sites : dict = attrib(default=None)  # if None it will be generated from the DB


# ----------------------------------------------------------------------------------------------------------------------
class Group(Entity):
    """A group of agents.

    Grouping of functionally equivalent agents is at the heart of the idea behind PRAMs.  To that end, a group is the
    primary type of entity in the framework.  Groups are considered identical if they have ideantical attributes and
    relations (i.e., the keys and values of those dictrionaries must be identical).  Consequently, agents belonging to
    identical groups are considered (by definition) functionally equivalent.  It is important to note that only one
    group with a particular combination of attributes and relations can exist in PRAM.

    A Group object can be either standalone or group population bound.  All Group objects are created standalone, but
    when added to a simulation (more specifically, an instance of the GroupPopulation class) it changes its mode.  This
    is to minimize memory utilization by storing Site instances only once in a simulation.

    Here is an example of creating and setting up a group in a chain of method calls::

        (Simulation().
            add().
                rule(...).   # add a rule that encodes the dynamics of the simulation
                probe(...).  # add a probe to monitor/save the results
                done().
            new_group(1000).                      # a group of a 1000 agents
                set_attr('income', 'medium').     # with medium income
                set_rel(Site.AT, Site('home'])).  # who are all currently at a site called 'home'
                done().                           # go back to the Simulation object
            run(12)
        )

    Groups can also be created in a regular fashion like so::

        (Simulation().
            add([
               ...,  # a rule
               ...,  # a probe
               Group(m=1000, attr={ 'income': 'medium' }, rel={ Site.AT: Site('home'] })
            ]).
            run(12)
        )

    PyPRAM supports this sort of expressive plurarism because doing stuff in only one way doesn't always yield the
    cleanest and most readible code.

    Args:
        name (str, optional): The name of the group.  This argument is inconsequential to PyPRAM's engine operation.
        m (float): Agent population mass.  While it is not entirely logical to think about agent mass in terms of
            fractional numbers, PRAM distributes population mass probabilistically and it doesn't guarantee
            intra-group movement of "entire" agents.  What it does guarantee, however, is that the total mass being
            moved adds up to one.  In large populations, treating agents as continuous mass rather than individuals is
            inconsequential.
        attr (Mapping[str, Any]): The group's attributes.
        rel (Mapping[str, Site]): The group's relations.
        callee (object, optional): The object invoking the contructor.  This argument is used only throughout the
            process of creating a group.  The reference to the invoking object can is returned by
            :meth:`Group.done() pram.entity.Group.done`.  See usage examples above.
    """

    __slots__ = ('name', 'm', 'attr', 'rel', 'pop', 'is_frozen', 'hash', 'callee')

    VOID = { '__void__': True }  # all groups with this attribute are removed at the end of every iteration

    attr_used = None  # a set of attribute that has been conditioned on by at least one rule
    rel_used  = None  # ^ for relations
        # both of the above should be kept None unless a simulation is running and the dynamic rule analysis
        # should be on-going

    def __init__(self, name=None, m=0.0, attr={}, rel={}, pop=None, callee=None):
        super().__init__(EntityType.GROUP, '')

        self.name     = name
        self.m        = float(m)
        self.attr     = attr or {}
        self.rel      = rel  or {}
        self.pop      = pop

        self.is_frozen = False
        self.hash = None      # computed lazily
        self.callee = callee  # used only throughout the process of creating group; unset by done()

        self.link_to_site_at()

    def __eq__(self, other):
        """Compare this group to another.

        When comparing groups, only attributes and relations matter; name and size are irrelevant.  Note that we need
        to implement this method regardless, because the one inherited from the 'object' class works by object identity
        only which is largely useless for us.
        """

        # TODO: Should we just compared the two objects using their '_hash' property?  Check how Python interpreter
        #       works.
        #
        #       https://hynek.me/articles/hashes-and-equality

        return isinstance(self, type(other)) and (self.attr == other.attr) and (self.rel == other.rel)

    def __hash__(self):
        if self.hash is None:
            self.hash = Group.gen_hash(self.attr, self.rel)
        return self.hash

    def __repr__(self):
        return '{}(name={}, m={}, attr={}, rel={})'.format(__class__.__name__, self.name or '.', self.m, self.attr, self.rel)

    def __str__(self):
        return '{}  name: {:16}  m: {:8}  attr: {}  rel: {}'.format(self.__class__.__name__, self.name or '.', round(self.m, 2), self.attr, self.rel)

    # @staticmethod
    def _has(self, d, qry, used_set):
        """Checks if a dictionary has keys or key-value pairs specified.

        This method compares the dictionary ``d`` against ``qry`` which can be a mapping, an iterable, and a string.
        Depending on the type of ``qry``, the method returns True only if (and False otherwise):

        - string: ``qry`` must be a key in ``d``
        - iterable: all items in ``qry`` must be keys in ``d``
        - mapping: all items in ``qry`` must exist in ``d``

        Args:
            d (Mapping[str, Any]): The original mapping.
            qry (Union[str, Iterable[str], Mapping[str, Any]]): The required content that ``d`` will be queried
                against.
            used_set (Set[Any]): A set of attributes or relations that stores the ones that have been conditioned upon
                by the simulation rules.  This is used internally by PyPRAM.

        Returns:
            bool: True if (False otherwise):
                - ``qry`` is a string and is a key in ``d``
                - ``qry`` is an iterable and all items in ``qry`` are keys in ``d``
                - ``qry`` is a mapping and all items in ``qry`` exist in ``d``
        """

        if isinstance(qry, dict):
            if used_set is not None:
                used_set.update(qry.keys())
            if self.pop:
                qry = { k:v.__hash__() if isinstance(v, Site) else v for (k,v) in qry.items() }  # TODO: Double-check this line
            return qry.items() <= d.items()

        if isinstance(qry, str):  # needs to be above the Iterable check because a string itself is an Iterable
            if used_set is not None:
                used_set.add(qry)
            return qry in d.keys()

        if isinstance(qry, Iterable):
            if used_set is not None:
                used_set.update(qry)
            if self.pop:
                qry = [q.__hash__() if isinstance(q, Site) else q for q in qry]
            return all(i in d.keys() for i in qry)

        raise TypeError(Err.type('qry', 'dictionary, Iterable, or string'))

    def apply_rules(self, pop, rules, iter, t, is_rule_setup=False, is_rule_cleanup=False, is_sim_setup=False):
        """Applies all the simulation rules to the group.

        Applies the list of rules, each of which may split the group into (possibly already extant) subgroups.  A
        sequential rule application scheme is (by the definition of sequentiality) bound to produce order effects which
        are undesirable.  To mitigate that problem, a Cartesian product of all the rule outcomes (i.e., split
        specifications; GroupSplitSpec class) is computed and the resulting cross-product spit specs are used to do the
        actual splitting.

        When creating the product of split specs created by the individual rules, the probabilities associated with
        those individual split specs are multiplied because the rules are assumed to be independent.  Any dependencies
        are assumed to have been handles inside the rules themselves.

        The two special rule modes this method can be called in are: setup and cleanup.  Neither of these modes
        checks for applicability; that should be performed inside the setup() and cleanup() method of a rule.  An
        additional special mode is that of the simulation group setup (i.e., is_sim_setup = True).  In that mode,
        the 'rules' argument is assumed to be a function to be called for the group (and not an iterable of Rule
        classes as is usual in normal operation).

        Args:
            pop (GroupPopulation): The group population.
            rules (Iterable[Rule]): Rules to be applied.  Only rules compatible with the group will actually be applied
                as determined by the :meth:`~pram.entity.Rule.is_applicable` method.
            iter (int): Simulation iterations.
            t (int): Simulation time.
            is_rule_setup (bool): Flag: Is this invocation of this method during rule setup stage of the simulation?
            is_rule_cleanup (bool): Flag: Is this invocation of this method during rule cleanup stage of the
                simulation?
            is_sim_setup (bool): Flag: Is this invocation of this method during simulation setup stage?

        Todo:
            Think if the dependencies between rules could (or perhaps even should) be read from some sort of a graph.
            Perhaps then multiplying the probabilities would not be appropriate.


        ----[ Possible future extension ]----

        Superposition principle (or superposition property), states that, for all linear systems, the net response
        caused by two or more stimuli is the sum of the responses that would have been caused by each stimulus
        individually. So that if input A produces response X and input B produces response Y then input (A + B)
        produces response (X + Y).

        A function F(x) that satisfies the superposition principle is called a linear function. Superposition can be
        defined by two simpler properties, additivity and homogeneity::

            F(x1 + x2) = F(x1) + F(x2)   # additivity
            F(ax) = aF(x)                # homogeneity

        for scalar ``a``.

        (...)

        The superposition principle applies to any linear system, including algebraic equations, linear differential
        equations, and systems of equations of those forms. The stimuli and responses could be numbers, functions,
        vectors, vector fields, time-varying signals, or any other object that satisfies certain axioms. Note that when
        vectors or vector fields are involved, a superposition is interpreted as a vector sum.

        SRC: https://en.wikipedia.org/wiki/Superposition_principle
        """

        # (1) Apply all the rules and get their respective split specs (ss):
        if is_rule_setup:
            ss_rules = [r.setup(pop, self) for r in rules]
        elif is_rule_cleanup:
            ss_rules = [r.cleanup(pop, self) for r in rules]
        elif is_sim_setup:
            ss_rules = [rules(pop, self)]
        else:
            ss_rules = [r.apply(pop, self, iter, t) for r in rules if r.is_applicable(self, iter, t)]

        ss_rules = [i for i in ss_rules if i is not None]
        if len(ss_rules) == 0:
            return None

        # (2) Create a Cartesian product of the split specs (ss):
        ss_prod = []
        for ss_lst in itertools.product(*ss_rules):
            ss_comb = GroupSplitSpec(p=1.0, attr_set={}, attr_del=set(), rel_set={}, rel_del=set())  # the combined split spec
            for i in ss_lst:
                ss_comb.p *= i.p  # this assumes rule independence

                # for k,v_new in i.attr_set.items():
                #     v_curr = ss_comb.attr_set.get(k)
                #     if v_curr is None:
                #         ss_comb.attr_set[k] = v_new
                #     else:
                #         if isinstance(v_curr, int) or isinstance(v_curr, float):
                #             ss_comb.attr_set[k] = ss_comb.attr_set[k] + v_new
                #         elif isinstance(v_curr, bool):
                #             ss_comb.attr_set[k] = ss_comb.attr_set[k] and v_curr  # TODO: allow 'or'
                #         else:
                #             if v_new != v_curr:
                #                 raise ValueError(f'The result of rule application results in an attribute update conflict:\n    Name: {k}\n    Type: {type(v_curr)}\n    Current value: {v_curr}\n    New value: {v_new}')

                ss_comb.attr_set.update(i.attr_set)  # this update has been subsistuted by the logic above (currently commented out)
                ss_comb.attr_del.update(i.attr_del)
                ss_comb.rel_set.update(i.rel_set)
                ss_comb.rel_del.update(i.rel_del)
            ss_prod.append(ss_comb)

        # (3) Split the group:
        return self.split(ss_prod)

    def copy(self, is_deep=False):
        """Generates the group's hash.

        Returns a shallow or deep copy of self.

        Args:
            is_deep (bool): Flag: Is deep copy?

        Returns:
            object (Group): A copy of self.
        """

        return copy.copy(self) if is_deep is False else copy.deepcopy(self)

    def done(self):
        """Ends creating the group by notifing the callee that has begun the group creation.

        Returns:
            object: Reference to the object that initiated the group creation (can be None).
        """

        if self.callee is None:
            return None

        c = self.callee
        self.callee.commit_group(self)
        self.callee = None
        return c

    def freeze(self):
        """Freezes the group.

        No direct changes to the group's identity as seen from the point of view of PRAM's engine operation can be made
        to a frozen group.  A group's identity is defined by the composition of its attributes and relations.  Groups
        are automatically frozen when added to the GroupPopulation object to prevent the user from interfering with
        PyPRAM's operation.
        """

        self.is_frozen = True

    def ga(self, name=None):
        """See :meth:`~pram.entity.Group.get_attr` method."""

        return self.get_attr(name)

    @staticmethod
    def gen_hash(attr={}, rel={}):
        """Generates the group's hash.

        Generates a hash for the attributes and relations dictionaries.  This sort of hash is desired because groups
        are judged functionally equivalent based on the content of those two dictionaries alone (i.e., the name and the
        size of a group does not affect its identity).

        The following non-cryptographic hashing algorithms have been tested:

        - hash()
        - hashlib.sha1()
        - xxhash.xxh32()
        - xxhash.xxh64()

        As is evident from the source code, each of the algorithms required a slightly different treatment of the
        attribute and relation dictionaries.  All these options are legitimate and they don't differ much in terms of
        speed.  They do howeve differ in terms of reproducability.  Namely, the results of the built-in hash() function
        cannot be compared between-runs while the other ones can.  This behavior of the hash() function is to prevent
        attackers from reusing hashes and it can be disabled by setting 'PYTHONHASHSEED=0'.

        The user can uncomment the desired method to use it, but it is not recommended.

        Args:
            attr (Mapping[str, Any]): Group's attributes.
            rel (Mapping[str, Site], optional): Group's relations.

        Returns:
            int: A hash of the attributes and relations specified.
        """

        # return hash(tuple([frozenset(attr.items()), frozenset(rel.items())]))
        # return hashlib.sha1(json.dumps((attr, rel), sort_keys=True, cls=EntityJSONEncoder).encode('utf-8')).hexdigest()
        # return xxhash.xxh32(json.dumps((attr, rel), sort_keys=True, cls=EntityJSONEncoder)).hexdigest()
        return xxhash.xxh64(json.dumps((attr, rel), sort_keys=True, cls=EntityJSONEncoder)).intdigest()  # .hexdigest()

    @classmethod
    def gen_from_db(cls, db_fpath, tbl, attr_db=[], rel_db=[], attr_fix={}, rel_fix={}, attr_rm=[], rel_rm=[], rel_at=None, limit=0, fn_live_info=None):
        """Generate groups from a relational database.

        In this method, lists are sometimes converted to allow for set operations (e.g., union or difference) and the
        results of those operations are converted back to lists for nice output printout (e.g., '[]' is more succinct
        than 'set()', which is what an empty set is printed out as).

        This is a central method for generating complete group populations from relational databases in that it
        automatically calls the :meth:`Site.gen_from_db() ~pram.entity.Site.gen_from_db` when necessary.

        For usage example see :meth:`SimulationDBI.gen_groups() pram.sim.SimulationDBI.gen_groups` and
        :meth:`Simulation.gen_groups_from_db() pram.sim.Simulation.gen_groups_from_db` methods; both invoke the current
        method internally.

        Args:
            db_fpath (str): Path to the database file (SQLite3).
            tbl (str): Table name.
            attr_db (Iterable[str]): Group attributes to be retrieved from the database (if extant).
            rel_db (Iterable[GroupDBRelSpec]): Group relation to be retrieved from the database (if extant).
            attr_fix (Mappint[str, Any]): Group attributes to be fixed for every group.
            rel_fix (Mapping[str, Site]): Group relations to be fixed for every group.
            attr_rm (Iterable[str]): Group attributes to NOT be retrieved from the database (overwrites all).
            rel_rm (Iterable[str]): Group relation to NOT be retrieved from the database (overwrites all).
            rel_at (Site, optional): A site to be set as every group's current location.
            limit (int): The maximum number of groups to be generated.  Ordinarily, this is not changed from its
                default value of zero.  It is however useful for testing, especially with very large databases.
            fn_live_info (Callable): A callable expecting a single string argument for real-time printing.

        Returns:
            list(Group): A list of groups generated.
        """

        inf = fn_live_info  # shorthand

        FS.req_file(db_fpath, f'The database does not exist: {db_fpath}')

        if inf:
            inf('    Expected in table')
            inf(f'        Attributes : {attr_db}')
            inf(f'        Relations  : {[r.col for r in rel_db]}')

        # (1) Sort out attributes and relations that do and do not exist in the table and reconcile them with the fixed ones:
        # (1.1) Use table columns to identify the attributes and relations that do and do not exist:
        with DB.open_conn(db_fpath) as c:
            columns = [i[1] for i in c.execute(f'PRAGMA table_info({tbl})')]
            attr_db_keep = [a for a in attr_db if a in columns]     # attributes to be kept based on the table schema
            rel_db_keep  = [r for r in rel_db if r.col in columns]  # relations

        if inf:
            inf( '    Found in table')
            inf(f'        Attributes : {attr_db_keep}')
            inf(f'        Relations  : {[r.col for r in rel_db]}')
            inf( '    Not found in table')
            inf(f'        Attributes : {list(set(attr_db) - set(attr_db_keep))}')
            inf(f'        Relations  : {list(set([r.col for r in rel_db]) - set([r.col for r in rel_db_keep]))}')
            inf( '    Fixed manually')
            inf(f'        Attributes : {attr_fix}')
            inf(f'        Relations  : {rel_fix}')
            inf( '    Removed manually')
            inf(f'        Attributes : {attr_rm}')
            inf(f'        Relations  : {rel_rm}')
            if len(set(attr_db_keep) & set(attr_fix)) > 0:
                inf( '    WARNING: The following exist in the table but will be masked because are manually fixed')
                inf(f'        Attributes : {list(set(attr_db_keep)             & set(attr_fix.keys()))}')
                inf(f'        Relations  : {list(set([r.col for r in rel_db])  & set(rel_fix.keys()))}')

        # (1.2) Remove the manually fixed and removed attributes and relations:
        attr_db_keep = list(set(attr_db_keep) - set(attr_fix.keys()))
        rel_db_keep  = [r for r in rel_db_keep if not r.col in rel_fix.keys()]

        attr_db_keep = list(set(attr_db_keep) - set(attr_rm))
        rel_db_keep  = [r for r in rel_db_keep if not r.col in rel_rm]

        if inf:
            inf( '    Final combination used for group forming')
            inf(f'        Attributes fixed      : {attr_fix}')
            inf(f'        Attributes from table : {attr_db_keep}')
            inf(f'        Relations  fixed      : {rel_fix}')
            inf(f'        Relations  from table : {[r.col for r in rel_db]}')

        # (2) Contruct the query:
        qry = 'SELECT COUNT(*) AS m{comma}{cols} FROM {tbl} WHERE {cols_where} GROUP BY {cols}{limit}'.format(
            tbl=tbl,
            cols=', '.join(attr_db_keep + [r.col for r in rel_db]),
            cols_where=' AND '.join([c + ' IS NOT NULL' for c in attr_db_keep + [r.col for r in rel_db]]),
            comma='' if len(attr_db_keep + [r.col for r in rel_db]) == 0 else ', ',
            limit='' if limit <= 0 else f' LIMIT {limit}'
        )

        # (3) Generate sites and groups:
        sites = {}
        groups = []
        grp_pop = 0
        site_n = 0

        gc.disable()
        with DB.open_conn(db_fpath) as c:
            # (3.1) Sites:
            for r in rel_db_keep:
                # (3.1.1) Sites have been provided:
                if r.sites is not None:
                    sites[r.name] = r.sites
                    if inf:
                        inf(f"    Using the provided {'{:,}'.format(len(sites[r.name]))} '{r.name}' sites")
                # (3.1.2) Sites have not been provided; generate them from the DB:
                else:
                    fk = DB.get_fk(c, tbl, r.col)
                    sites[r.name] = Site.gen_from_db(db_fpath, tbl=fk.tbl_to, name_col=fk.col_to)
                    site_n += len(sites[r.name])
                    if inf:
                        inf(f"    Generated {'{:,}'.format(len(sites[r.name]))} '{r.name}' sites from the '{fk.tbl_to}' table")

            # (3.2) Groups:
            row_cnt = DB.get_cnt(c, tbl)
            for row in c.execute(qry).fetchall():
                g_attr = {}  # group attributes
                g_rel  = {}  # group relations

                g_attr.update(attr_fix)
                g_rel.update(rel_fix)

                g_attr.update({ a: row[a] for a in attr_db_keep })
                g_rel.update({ r.name: sites[r.name][row[r.col]] for r in rel_db_keep })

                if rel_at is not None:
                    g_rel.update({ Site.AT: g_rel.get(rel_at) })

                groups.append(cls(m=row['m'], attr=g_attr, rel=g_rel))
                grp_pop += int(row['m'])
        gc.enable()

        if inf:
            inf( '    Summary')
            inf(f'        Records in table: {"{:,}".format(row_cnt)}')
            inf(f'        Groups formed: {"{:,}".format(len(groups))}')
            inf(f'        Sites formed: {"{:,}".format(site_n)}')
            inf(f'        Agent population accounted for by the groups: {"{:,}".format(grp_pop)} ({grp_pop / row_cnt * 100:.0f}% of the table)')

        return groups

    @classmethod
    def gen_from_db_tmp1(cls, sim, db_fpath, tbl, attr={}, rel={}, attr_db=[], rel_db=[], rel_at=None, limit=0):
        """A legacy method pending non-immediate removal.

        An unsuccessful attempt to internalize generation of sites from the DB.  As of 2019.04.24 this is no longer the
        main development route as another has proved more fruitful.  Keeping this method here in case it comes useful
        at some point.

        .. @attrs()
        .. class GroupDBRelSpec(object):
        ..     tbl      : str  = attrib()
        ..     col_from : str  = attrib()
        ..     col_to   : str  = attrib()
        ..     name     : str  = attrib()
        ..     sites    : dict = attrib(default=None)
        """

        if not os.path.isfile(db_fpath):
            raise ValueError(f'The database does not exist: {db_fpath}')

        # (1) Remove attribute that don't exist in the DB:
        with DB.open_conn(db_fpath) as c:
            columns = [i[1] for i in c.execute(f'PRAGMA table_info({tbl})')]
            attr_db_keep = [a for a in attr_db if a in columns]

        # (2) Contruct the query:
        qry = 'SELECT COUNT(*) AS m{comma}{cols} FROM {tbl} WHERE {cols_where} GROUP BY {cols}{limit}'.format(
            tbl=tbl,
            cols=', '.join(attr_db_keep + [r.col_from for r in rel_db]),
            cols_where=' AND '.join([c + ' IS NOT NULL' for c in attr_db_keep + [r.col_from for r in rel_db]]),
            comma='' if len(attr_db_keep + [r.col_from for r in rel_db]) == 0 else ', ',
            limit='' if limit <= 0 else f' LIMIT {limit}'
        )

        for i in rel_db:
            # i.sites = Site.gen_from_db(db_fpath, i.tbl, name_col=i.col_to, rel_name=i.name)
            Site.gen_from_db(sim, db_fpath, i.tbl, name_col=i.col_to, rel_name=i.name)

        # (3) Generate groups:
        groups = []
        with DB.open_conn(db_fpath) as c:
            for row in c.execute(qry).fetchall():
                g_attr = {}
                g_rel  = {}

                g_attr.update(attr)
                g_rel.update(rel)

                g_attr.update({ a: row[a] for a in attr_db_keep })
                # g_rel.update({ i.name: i.entities[row[i.col]] for i in rel_db })  # original
                # g_rel.update({ i.name: i.sites[row[i.col]] for i in rel_db })     # same but with sites internalized
                for i in rel_db:                                                    # new
                    # site = i.sites[row[i.col_from]]
                    site = sim.pop.sites[row[i.col_from]]
                    # g_rel.update({ Site.AT: site })
                    g_rel.update({ i.name: site })
                    # sim.add_site(site)

                if rel_at is not None:
                    g_rel.update({ Site.AT: g_rel.get(rel_at) })

                print(f'A:{g_attr}  R:{g_rel}')
                groups.append(cls(m=row['m'], attr=g_attr, rel=g_rel))

        return groups

    @classmethod
    def gen_from_db_tmp2(cls, db_fpath, tbl, attr_db=[], rel_db=[], attr_fix={}, rel_fix={}, rel_at=None, limit=0, fn_live_info=None):
        """A legacy method pending non-immediate removal.

        Before generating sites based on the DB schema.  Works.

        In this method, lists are sometimes converted to allow for set operations (e.g., union or difference) and the
        results of those operations are converted back to lists for nice output printout (e.g., '[]' is more succinct
        than 'set()', which is what an empty set is printed out as).
        """

        inf = fn_live_info  # shorthand

        if not os.path.isfile(db_fpath):
            raise ValueError(f'The database does not exist: {db_fpath}')

        if inf:
            inf('    Expected in table')
            inf(f'        Attributes : {attr_db}')
            inf(f'        Relations  : {[r.col for r in rel_db]}')

        # (1) Sort out attributes and relations that do and do not exist in the table and reconcile them with the fixed ones:
        # (1.1) Use table columns to identify the attributes and relations that do and do not exist:
        with DB.open_conn(db_fpath) as c:
            columns = [i[1] for i in c.execute(f'PRAGMA table_info({tbl})')]
            attr_db_keep = [a for a in attr_db if a in columns]     # attributes to be kept based on the table schema
            rel_db_keep  = [r for r in rel_db if r.col in columns]  # relations

        if fn_live_info:
            inf( '    Found in table')
            inf(f'        Attributes : {attr_db_keep}')
            inf(f'        Relations  : {[r.col for r in rel_db]}')
            inf( '    Not found in table')
            inf(f'        Attributes : {list(set(attr_db) - set(attr_db_keep))}')
            inf(f'        Relations  : {list(set([r.col for r in rel_db]) - set([r.col for r in rel_db_keep]))}')
            inf( '    Fixed manually')
            inf(f'        Attributes : {attr_fix}')
            inf(f'        Relations  : {rel_fix}')
            if len(set(attr_db_keep) & set(attr_fix)) > 0:
                inf( '    WARNING: The following exist in the table but will be masked because are manually fixed')
                inf(f'        Attributes : {list(set(attr_db_keep)             & set(attr_fix.keys()))}')
                inf(f'        Relations  : {list(set([r.col for r in rel_db])  & set(rel_fix.keys()))}')

        # (1.2) Remove the fixed attributes and relations:
        attr_db_keep = list(set(attr_db_keep) - set(attr_fix.keys()))
        rel_db_keep  = [r for r in rel_db_keep if not r.col in rel_fix.keys()]

        if fn_live_info:
            inf( '    Final combination used for group forming')
            inf(f'        Attributes fixed      : {attr_fix}')
            inf(f'        Attributes from table : {attr_db_keep}')
            inf(f'        Relations  fixed      : {rel_fix}')
            inf(f'        Relations  from table : {[r.col for r in rel_db]}')

        # (2) Contruct the query:
        qry = 'SELECT COUNT(*) AS n{comma}{cols} FROM {tbl} WHERE {cols_where} GROUP BY {cols}{limit}'.format(
            tbl=tbl,
            cols=', '.join(attr_db_keep + [r.col for r in rel_db]),
            cols_where=' AND '.join([c + ' IS NOT NULL' for c in attr_db_keep + [r.col for r in rel_db]]),
            comma='' if len(attr_db_keep + [r.col for r in rel_db]) == 0 else ', ',
            limit='' if limit <= 0 else f' LIMIT {limit}'
        )

        # (3) Generate groups:
        groups = []
        grp_n_tot = 0

        with DB.open_conn(db_fpath) as c:
            row_cnt = DB.get_cnt(c, tbl)
            for row in c.execute(qry).fetchall():
                g_attr = {}  # group attributes
                g_rel  = {}  # group relations

                g_attr.update(attr_fix)
                g_rel.update(rel_fix)

                g_attr.update({ a: row[a] for a in attr_db_keep })
                g_rel.update({ spec.name: spec.entities[row[spec.col]] for spec in rel_db })

                if rel_at is not None:
                    g_rel.update({ Site.AT: g_rel.get(rel_at) })

                groups.append(cls(m=row['m'], attr=g_attr, rel=g_rel))
                grp_n_tot += int(row['m'])

        if inf:
            inf( '    Summary')
            inf(f'        Records in table: {row_cnt}')
            inf(f'        Groups formed: {len(groups)}')
            inf(f'        Agent population accounted for by the groups: {grp_n_tot}')

        return groups

    @staticmethod
    def gen_dict(d_in, d_upd=None, k_del=None):
        """Generates a dictionary.

        This method is used to create new dictionaries based on existing ones and given changes to those existing ones.
        Specifically, a new dictionary is based on the 'd_in' dictionary with values updated based on the 'd_upd'
        dictionary and keys deleted based on the 'k_del' iterable.

        Args:
            d_in (Mapping[str, Any]): Original mapping.
            d_upd (Mapping[str, Any], optional): Key-values to be set on the original mapping.
            k_del (Iterable[str], optional): An iterable of keys to be removed from the original mapping.

        Returns:
            Mapping: A shallow copy of the updated original mapping.

        Notes:
            A shallow copy of the dictionary is returned at this point.  That is to avoid creating unnecessary copies
            of entities that might be stored as relations.  A more adaptive mechanism can be implemented later if
            needed.

        Todo:
            Consider: https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
        """

        ret = d_in.copy()

        if d_upd is not None:
            ret.update(d_upd)

        if k_del is not None and len(k_del) > 0:
            for k in k_del:
                if k in ret:
                    del ret[k]

        return ret

    def get_attr(self, name=None):
        """Retrieves attribute's value.

        Args:
            name (str): Attribute's name.

        Returns:
            Any: Attribute's value.
        """

        if name and self.attr_used is not None:
            self.attr_used.add(name)

        # return self.attr[name] if name is not None else self.attr
        return self.attr.get(name) if name is not None else self.attr

    def get_hash(self):
        """Get the group's hash.

        Groups are hashed on their attributes and relations, both being dictionaries.

        Calling this method should be the only way to get the group's hash.  Wanna do it otherwise?  Get ready for a
        world of hurt.
        """

        return self.__hash__()

    def get_site_at(self):
        """Get the site the group is currently located at.

        Returns:
            Site
        """

        # return self.get_rel(Site.AT)  # old sites handling

        if Site.AT not in self.rel.keys():
            return None
        if self.pop:
            return self.pop.sites[self.rel[Site.AT]]
        else:
            return self.rel[Site.AT]

    def get_rel(self, name=None):
        """Retrieves relation's value or all relations if no name is provided.

        Args:
            name (str, optional): Relation's name.

        Returns:
            Any: Relation's value or all relations.
        """

        if name and self.rel_used is not None:
            self.rel_used.add(name)

        # return self.rel[name] if name is not None else self.rel  # old
        # return self.rel.get(name) if name else self.rel  # old sites handling

        if name not in self.rel.keys():
            return None
        if self.pop:
            return self.pop.sites[self.rel[name]]
        else:
            return self.rel[name]

    def get_mass(self):
        return self.m

    def gr(self, name=None):
        """See :meth:`~pram.entity.Group.get_rel` method."""

        return self.get_rel(name)

    def ha(self, qry):
        """See :meth:`~pram.entity.Group.has_attr` method."""

        return self.has_attr(qry)

    def has_attr(self, qry):
        """Checks if the group has the specified attributes.

        See :meth:`~pram.entity.Group._has` method for details on what ``qry`` can be and the specifics of the check.

        Returns:
            bool
        """

        return self._has(self.attr, qry, self.attr_used)

    def has_rel(self, qry):
        """Checks if the group has the specified relations.

        See :meth:`~pram.entity.Group._has` method for details on what ``qry`` can be and the specifics of the check.

        Returns:
            bool
        """

        return self._has(self.rel, qry, self.rel_used)

    def has_sites(self, sites):
        """Checks if the groups has the sites specified.

        Args:
            sites (Iterable[Site]): Sites the existence of which should be checked for.

        Returns:
            bool
        """

        return self._has(self.rel, sites, self.rel_used)

    def hr(self, qry):
        """See :meth:`~pram.entity.Group.has_rel` method."""

        return self.has_rel(qry)

    def is_at_site(self, site):
        """ Is the groups currently at the site specified? """

        # return self.has_rel({ Site.AT: site })  # old sites handling
        return self.get_site_at() == site

    def is_at_site_name(self, name):
        """ Is the groups currently at the site with the name specified (that the group has as a relation)? """

        # return self.has_rel({ Site.AT: self.get_rel(name) })  # old sites handling

        if self.rel.get(name) is None:
            return False
        return self.rel.get(Site.AT) and self.rel[Site.AT] == self.rel.get(name)

    def is_void(self):
        """Checks if the group is a VOID group (i.e., it should be removed from the simulation).

        Returns:
            bool: True for VOID group; False otherwise.
        """

        return self.ha(Group.VOID)

    def link_to_site_at(self):
        """Links the group to the site it currently resides at.

        Returns:
            self: For method call chaining.
        """

        at = self.rel.get(Site.AT)
        if at:
            # at.add_group_link(self)  # old sites handling
            if self.pop:
                self.pop.sites[at].add_group_link(self)
        return self

    def matches_qry(self, qry):
        """Checks if the group matches the group query specified.

        Args:
            qry (GroupQry): The query.  A group automatially matches a None qry.

        Returns:
            bool: True if the group matches the query; False otherwise.
        """

        if not qry:
            return True

        # if qry.full:
        #     return qry.attr == self.attr and qry.rel == self.rel and all([fn(self) for fn in qry.cond])
        # else:
        #     return qry.attr.items() <= self.attr.items() and qry.rel.items() <= self.rel.items() and all([fn(self) for fn in qry.cond])

        # Check where most time is spent when evaluating a group-query match:
        if qry.full:
            if len(qry.cond) == 0:
                return self.matches_qry_full_cond0(qry)
            else:
                return self.matches_qry_full_cond1(qry)
        else:
            if len(qry.cond) == 0:
                return self.matches_qry_part_cond0(qry)
            else:
                return self.matches_qry_part_cond1(qry)

    def matches_qry_full_cond0(self, qry):
        return qry.attr == self.attr and qry.rel == self.rel

    def matches_qry_full_cond1(self, qry):
        return self.matches_qry_full_cond0(qry) and all([fn(self) for fn in qry.cond])

    def matches_qry_part_cond0(self, qry):
        return qry.attr.items() <= self.attr.items() and qry.rel.items() <= self.rel.items()

    def matches_qry_part_cond1(self, qry):
        return self.matches_qry_part_cond0(qry) and all([fn(self) for fn in qry.cond])

    def set_attr(self, name, value, do_force=True):
        """Sets a group's attribute.

        Args:
            name (str): Attribute's name.
            value (Any): Attribute's value.
            do_force (bool): Flag: Force despite the group being frozen? Currently unused and pending possible removal.

        Raises:
            GroupFrozenError

        Returns:
            self: For method call chaining.

        Todo:
            Remove the ``do_force`` argument?
        """

        if self.is_frozen:
            raise GroupFrozenError('Attempting to set an attribute of a frozen group.')

        # if self.attr.get(name) is not None and not do_force:
        #     raise ValueError("Group '{}' already has the attribute '{}'.".format(self.name, name))

        self.attr[name] = value
        self.hash = None

        return self

    def set_attrs(self, attr, do_force=True):
        """Sets multiple group's attributes.

        This method is not implemented yet.

        Args:
            attr (Mapping[str, Any]): Attributes.
            do_force (bool): Flag: Force despite the group being frozen? Currently unused and pending possible removal.

        Raises:
            GroupFrozenError

        Returns:
            self: For method call chaining.

        Todo:
            - Implement method.
            - Remove the ``do_force`` argument?
        """

        if self.is_frozen:
            raise GroupFrozenError('Attempting to set attributes of a frozen group.')

        raise Error('Not implemented yet')

    def set_rel(self, name, value, do_force=True):
        """Sets a group's relation.

        Args:
            name (str): Relation's name.
            value (Any): Relation's value.
            do_force (bool): Flag: Force despite the group being frozen? Currently unused and pending possible removal.

        Raises:
            GroupFrozenError

        Returns:
            self: For method call chaining.

        Todo:
            Remove the ``do_force`` argument?
        """

        if self.is_frozen:
            raise GroupFrozenError('Attempting to set a relation of a frozen group.')

        # if name == Site.AT:
        #     raise ValueError("Relation name '{}' is restricted for internal use.".format(Site.AT))

        if self.rel.get(name) and not do_force:
            raise ValueError("Group '{}' already has the relation '{}'.".format(self.name or '.', name))

        self.rel[name] = value
        self.hash = None

        if self.pop and name == Site.AT:
            self.link_to_site_at()

        return self

    def set_rels(self, rel, do_force=True):
        """Sets multiple group's relation.

        This method is not implemented yet.

        Args:
            rel (Mapping[str, Site]): Relations.
            do_force (bool): Flag: Force despite the group being frozen? Currently unused and pending possible removal.

        Raises:
            GroupFrozenError

        Returns:
            self: For method call chaining.

        Todo:
            - Implement method.
            - Remove the ``do_force`` argument?
        """

        if self.is_frozen:
            raise GroupFrozenError('Attempting to set relations of a frozen group.')

        raise Error('Not implemented yet')

    def split(self, specs):
        """Splits the group into new groups according to the split specs.

        The probabilities defining the population mass distribution among the new groups need to add up to 1.
        Complementing of the last one of those probabilities is done automatically (i.e., it does not need to be
        provided and is in fact outright ignored).

        A note on performance.  The biggest performance hit is likley going to be generating a hash which happens as
        part of instantiating a new Group object.  While this may seem like a good reason to avoid crearing new groups,
        that line of reasoning is deceptive in that a group's hash is needed regardless.  Other than that, a group
        object is light so its impact on performance should be negligible.  Furthermore, this also grants access to
        full functionality of the Group class to any function that uses the result of the present method.

        Args:
            specs (Iterable[GroupSplitSpec]): Group split specs.
        """

        # (1) Compute masses of new groups:
        p_sum = 0.0  # sum of split proportions (being probabilities, they must sum up to 1)
        m_sum = 0.0  # sum of total mass redistributed via group splitting
        m_lst = [0.0] * len(specs)

        for (i,s) in enumerate(specs):
            if i == len(specs) - 1:  # last group spec
                p = 1 - p_sum        # complement the probability
                m = self.m - m_sum   # make sure we're not missing anybody due to floating-point arithmetic
            else:
                p = s.p
                m = self.m * p
                # m = math.floor(self.m * p)  # conservative floor() use to make sure we don't go over due to rounding
            m_lst[i] = m

            p_sum += p
            m_sum += m

            if p_sum == 1.0:  # the remaining split specs must have p=0 so might as well skip them
                break

        # (2) Round masses of new groups to integers:
        if not self.pop.sim.get_pragma_fractional_mass():
            m_lst = saferound(m_lst,0)

        # (3) Instantiate new groups:
        groups = []  # split result (i.e., new groups; note that those groups may already exist in the simulation)

        for (i,s) in enumerate(specs):
            m = m_lst[i]
            if m == 0:  # don't instantiate empty groups
                continue

            attr = Group.gen_dict(self.attr, s.attr_set, s.attr_del)
            rel  = Group.gen_dict(self.rel,  s.rel_set,  s.rel_del)

            # g = Group('{}.{}'.format(self.name, i), m, attr, rel)
            # if g == self:
            #     g.name = self.name
            # groups.append(g)

            # groups.append(Group(None, m, attr, rel))  # None means we do not use group names any more
            groups.append(Group(self.name, m, attr, rel))  # use the same group name

        return groups

    def unfreeze(self):
        """Makes the group ammenable to changes via its API.

        Warnings:
            Unless you're doing something low level, you do not want to call this method.
        """

        self.is_frozen = False


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    rand_seed = 1928

    np.random.seed(rand_seed)

    # (1) Agents:
    print('(1) Agents')

    print(Agent('smith', AttrSex.M, 99.99, school=None, work='matrix', location='matrix'))

    # (1.1) Generation - Individual:
    print(Agent.gen('duncan'))
    print(Agent.gen('gurney'))
    print(Agent.gen('irulan'))
    print(Agent.gen('paul'))

    # (1.2) Generation - List:
    for a in Agent.gen_lst(5):
        print(a)

    # (2) Groups:
    print('\n(2) Groups')

    # (2.1) Hashing:
    print('(2.1) Hashing')

    # One dictionary:
    h1a = lambda d: hash(tuple(sorted(d.items())))
    assert(h1a({ 'a':1, 'b':2 }) == h1a({ 'b':2, 'a':1  }))
    assert(h1a({ 'a':1, 'b':2 }) != h1a({ 'b':2, 'a':10 }))

    h1b = lambda d: hash(frozenset(d.items()))
    assert(h1b({ 'a':1, 'b':2 }) == h1b({ 'b':2, 'a':1  }))
    assert(h1b({ 'a':1, 'b':2 }) != h1b({ 'b':2, 'a':10 }))

    # Two dictionaries:
    h2a = lambda a,b: hash(tuple([tuple(sorted(a.items())), tuple(sorted(b.items()))]))
    assert(h2a({ 'a':1, 'b':2 }, { 'c':3, 'd':4 }) == h2a({ 'b':2, 'a':1 }, { 'd':4, 'c':3  }))
    assert(h2a({ 'a':1, 'b':2 }, { 'c':3, 'd':4 }) != h2a({ 'b':2, 'a':1 }, { 'd':4, 'c':30 }))

    h2b = lambda a,b: hash(tuple([frozenset(a.items()), frozenset(b.items())]))
    assert(h2b({ 'a':1, 'b':2 }, { 'c':3, 'd':4 }) == h2b({ 'b':2, 'a':1 }, { 'd':4, 'c':3  }))
    assert(h2b({ 'a':1, 'b':2 }, { 'c':3, 'd':4 }) != h2b({ 'b':2, 'a':1 }, { 'd':4, 'c':30 }))

    # (2.2) Splitting:
    print('\n(2.2) Splitting')

    # Argument value check:
    for p in [-0.1, 0.1, 0.9, 1.1, 1, '0.9']:
        try:
            GroupSplitSpec(p=p)
            print('p={:4}  ok'.format(p))
        except ValueError:
            print('p={:4}  value error'.format(p))
        except TypeError:
            print('p={:4}  type error ({})'.format(p, type(p)))

    # Splitting:
    g1 = Group('g.1', 200, { 'sex': 'f', 'income': 'l' }, { 'location': 'home' })

    print()
    print(g1)
    print(GroupSplitSpec(p=0.1416, attr_del={ 'income' }))

    g1_split = g1.split([
        GroupSplitSpec(p=0.1416, attr_del={ 'income' }),
        GroupSplitSpec(          rel_set={ 'location': 'work' })
    ])

    print()
    for g in g1_split:
        print(g)
