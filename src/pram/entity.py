import copy
import itertools
import math
import numpy as np
import os
import sqlite3

from abc import ABC
from attr import attrs, attrib, converters, validators
from collections.abc import Iterable
from enum import IntEnum
from scipy.stats import rv_continuous

from .util import Err, DB


# ----------------------------------------------------------------------------------------------------------------------
class GroupFrozenError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
class AttrSex(IntEnum):
    F = 1
    M = 2


class AttrFluStage(IntEnum):
    NO     = 1
    ASYMPT = 2
    SYMPT  = 3


# @attrs()
# class Attr(object):
#     pass
#
#
# @attrs(slots=True)
# class AttrSex(Attr):
#     name : str = 'sex'
#     val  : AttrSexEnum = attrib(default=AttrSexEnum.m, validator=validators.in_(AttrSexEnum))
#
#
# @attrs(slots=True)
# class AttrFluStage(Attr):
#     name : str = 'flu-stage'
#     val  : AttrFluStageEnum = attrib(default=AttrFluStageEnum.no, validator=validators.in_(AttrFluStageEnum))


# ----------------------------------------------------------------------------------------------------------------------
class DistributionAgeSchool(rv_continuous):
    # TODO: Finish.
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)


class DistributionAgeWork(rv_continuous):
    # TODO: Finish.
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------------------------------------------------------------
class EntityType(IntEnum):
    AGENT    = 1
    GROUP    = 2
    SITE     = 3  # e.g., home, school, etc.
    RESOURCE = 4  # e.g., a public bus


class Entity(ABC):
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
    '''
    A resource shared by the agents (e.g., a public bus).

    This is a basic implementation of a shared resource and can safely be used only within a single simulation.  A more
    elaborate implementation based on synchronization mechanisms will be provided later and will accommodate multiple
    concurrent simulations of different and interacting systems with the agent population moving seamlessly between
    them.
    '''

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
        if do_all:
            return self.accommodate_all(n)
        else:
            return self.accommodate_any(n)

    def allocate_any(self, n):
        ''' Return the number of not accommodated agents (i.e., those over the max capacity). '''

        n_accommodated = self.capacity_max - n
        self.capacity += n_accommodated
        return n - n_accommodated

    def allocate_all(self, n):
        ''' Returns True if all agents can be accommodated, and False otherwise. '''

        if self.capacity + n <= self.capacity_max:
            self.capacity += n
            return True
        else:
            return False

    def can_accommodate_all(self, n):
        return self.capacity + n <= self.capacity_max

    def can_accommodate_any(self, n):
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
        if self.capacity == 0:
            return

        self.capacity = max(0, self.capacity - n)


# ----------------------------------------------------------------------------------------------------------------------
class Site(Resource):
    '''
    A physical site (e.g., a school or a store) agents can reside at.

    A site has a sensible interface which makes it useful.  For example, it makes sense to ask about the size and
    composition of population (e.g., groups) that are at that location.  However, because this information (and other,
    similar pieces of information) may be desired at arbitrary times, it makes most sense to compute it lazily and
    memoize it.  For that reason, a site stores a link to the population it is associated with; it queries that
    population to compute quantities of interested when they are needed.  An added benefit of this design is fostering
    proper composition; that is, updating the state of a site should be done by a site, not the population.

    A Site is a Resource which means a Site may choose to utilize the capacity property with all its related methods.
    A good example of when this would be useful is a hospital with a limited patient capacity that may be reached
    during an epidemic outbreak.
    '''

    AT = '__at__'  # relation name for the group's current location

    __slots__ = ('attr', 'rel_name', 'pop', 'groups')

    def __init__(self, name, attr=None, rel_name=AT, pop=None, capacity_max=1):
        # super().__init__(EntityType.SITE, '')
        super().__init__(name, capacity_max)

        self.rel_name = rel_name  # name of the relation the site is the object of
        self.attr = attr or {}
        self.pop = pop  # pointer to the population (can be set elsewhere too)
        self.groups = None  # None indicates the groups at the site might have changed and need to be retrieved again from the population

    def __eq__(self, other):
        '''
        We will make two sites identical if their keys are equal (i.e., object identity is not necessary).  This will
        let us recognize sites even if they are instantiated multiple times.
        '''

        return isinstance(self, type(other)) and (self.__key() == other.__key())

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return '{}({} {} {})'.format(self.__class__.__name__, self.name, self.__hash__(), self.attr)

    def __str__(self):
        return '{}  name: {:16}  hash: {}  attr: {}'.format(self.__class__.__name__, self.name, self.__hash__(), self.attr)

    def __key(self):
        return (self.name)

    @classmethod
    def gen_from_db(cls, db_fpath, tbl, name_col, rel_name, attr=[], limit=0):
        if not os.path.isfile(db_fpath):
            raise ValueError(f'The database does not exist: {db_fpath}')

        sites = {}
        with DB.open_conn(db_fpath) as c:
            for row in c.execute('SELECT {} FROM {}{}'.format(','.join(attr + [name_col]), tbl, '' if limit <= 0 else f' LIMIT {limit}')).fetchall():
                s = cls(row[name_col], { a: row[a] for a in attr }, rel_name=rel_name)
                sites[s.get_hash()] = s

        return sites

    def get_hash(self):
        return self.__hash__()

    def get_groups_here(self, qry=None, non_empty_only=True):
        '''
        Returns groups which currently are at this site.

        qry: GroupQry
        '''

        # TODO: Implement memoization (probably of only all the groups, i.e., not account for the 'qry').

        # if self.groups is None:

        qry = qry or GroupQry()
        # qry.rel.update({ self.rel_name: self.get_hash() })
        qry.rel.update({ self.rel_name: self })
        groups = self.pop.get_groups(qry)

        if non_empty_only:
            return [g for g in groups if g.n > 0]
        else:
            return groups

    def get_pop_size(self, qry=None):
        return sum(g.n for g in self.get_groups_here(qry))

    def invalidate_pop(self):
        self.groups = None

    def set_pop(self, pop):
        self.pop = pop


# ----------------------------------------------------------------------------------------------------------------------
class Agent(Entity):
    __slots__ = ('name', 'sex', 'age', 'flu', 'school', 'work', 'location')

    AGE_MIN =   0
    AGE_MAX = 120
    AGE_M   =  40
    AGE_SD  =  20

    P_STUDENT = 0.25  # unconditional prob. of being a student
    P_WORKER  = 0.60  # unconditional prob. of being a worker

    def __init__(self, name=None, sex=AttrSex.F, age=AGE_M, flu=AttrFluStage.NO, school=None, work=None, location='home'):
        super().__init__(EntityType.AGENT, '')

        self.name     = name or '.'
        self.sex      = sex
        self.age      = age
        self.flu      = flu
        self.school   = school
        self.work     = work
        self.location = location

    def __repr__(self):
        return '{}(name={}, sex={}, age={}, flu={}, school={}, work={}, location={})'.format(self.__class__.__name__, self.name, self.sex.name, round(self.age, 2), self.flu.name, self.school, self.work, self.location)

    def __str__(self):
        return '{}  name: {:12}  sex:{}  age: {:3}  flu: {:6}  school: {:16}  work: {:16}  location: {:12}'.format(self.__class__.__name__, self.name, self.sex.name, round(self.age), self.flu.name, self.school or '.', self.work or '.', self.location or '.')

    @classmethod
    def gen(cls, name=None):
        ''' Generates a singular agent. '''

        name     = name or '.'
        sex      = Agent.random_sex()
        age      = Agent.random_age()
        school   = None
        work     = None
        flu      = Agent.random_flu()
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

        return cls(name, sex, age, flu, school, work, location)

    @classmethod
    def gen_lst(cls, n):
        ''' Generates a list of agents (with auto-incrementing names). '''

        if n <= 0:
            return []
        return [cls.gen('a.{}'.format(i)) for i in range(n)]

    @staticmethod
    def random_age():
        return min(Agent.AGE_MAX, max(Agent.AGE_MIN, np.random.normal(Agent.AGE_M, Agent.AGE_SD)))

    @staticmethod
    def random_flu():
        return AttrFluStage(np.random.choice(AttrFluStage))

    @staticmethod
    def random_sex():
        return AttrSex(np.random.choice(AttrSex))


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class GroupQry(object):
    '''
    Group query.

    Objects of this simple class are used to select groups from a group population using attribute- and relation-based
    search criteria.

    It would make sense to declare this class frozen (i.e., 'frozen=True'), but as is revealsed by the following two
    measurements, performance suffers slightly when slotted classes get frozen.

    python -m timeit -s "import attr; C = attr.make_class('C', ['x', 'y', 'z'], slots=True)"             "C(1,2,3)"
    python -m timeit -s "import attr; C = attr.make_class('C', ['x', 'y', 'z'], slots=True,frozen=True)" "C(1,2,3)"
    '''

    attr : dict = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
    rel  : dict = attrib(factory=dict, converter=converters.default_if_none(factory=dict))


@attrs(kw_only=True, slots=True)
class GroupSplitSpec(object):
    '''
    A single group-split specification.

    These specifications are oridinarily provided in a list to indicate new groups that one other group is being split
    into.

    TODO: At this point, attributes and relations to be removed are assumed to be identified by their names only and
          not their values (i.e., we use a set to hold the keys that should be removed from the dictionaries for
          attributes and relations).  Perhaps this is not the way to go and we should instead be using both names and
          values.
    '''

    p        : float = attrib(default=0.0, converter=float)  # validator=attr.validators.instance_of(float))
    attr_set : dict  = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
    attr_del : set   = attrib(factory=set, converter=converters.default_if_none(factory=set))
    rel_set  : dict  = attrib(factory=dict, converter=converters.default_if_none(factory=dict))
    rel_del  : set   = attrib(factory=set, converter=converters.default_if_none(factory=set))

    @p.validator
    def is_prob(self, attribute, value):
        if not isinstance(value, float):
            raise TypeError(Err.type('p', 'float'))
        if not (0 <= value <= 1):
            raise ValueError("The probability 'p' must be in [0,1] range.")


@attrs()
class GroupDBRelSpec(object):
    name     : str  = attrib()
    col      : str  = attrib()
    entities : dict = attrib()


# ----------------------------------------------------------------------------------------------------------------------
class Group(Entity):
    __slots__ = ('name', 'n', 'attr', 'rel', 'is_frozen', 'hash', 'callee')

    attr_used = None  # a set of attribute that has been conditioned on by at least one rule
    rel_used  = None  # ^ for relations

    def __init__(self, name=None, n=0.0, attr={}, rel={}, callee=None):
        super().__init__(EntityType.GROUP, '')

        self.name = name or '.'
        self.n    = float(n)
        self.attr = attr or {}
        self.rel  = rel  or {}

        self.is_frozen = False
        self.hash = None  # computed lazily
        self.callee = callee  # used only throughout the process of creating group; unset by commit()

    def __eq__(self, other):
        '''
        When comparing groups, only attributes and relations matter; name and size are irrelevant.  Note that we need
        to implement this method regardless, because the one inherited from the 'object' class works by object identity
        only which is largely useless for us.
        '''

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
        return '{}(name={}, n={}, attr={}, rel={})'.format(__class__.__name__, self.name, self.n, self.attr, self.rel)

    def __str__(self):
        return '{}  name: {:16}  n: {:8}  attr: {}  rel: {}'.format(self.__class__.__name__, self.name, round(self.n, 2), self.attr, self.rel)

    @staticmethod
    def _has(d, qry, used_set):
        '''
        Compares the dictionary 'd' against 'qry' which can be a dictionary, an iterable (excluding a string), and a
        string.  Depending on the type of 'qry', the method performs the following checks:

            string: 'qry' must be a key in 'd'
            iterable: all items in 'qry' must be keys in 'd'
            dictionary: all items in 'qry' must exist in 'd'

        The 'used_set' is a set of attributes or relations that stores the ones that have been conditioned upon by the
        simulation rules.
        '''

        if isinstance(qry, dict):
            if used_set is not None:
                used_set.update(qry.keys())
            return qry.items() <= d.items()

        if isinstance(qry, str):  # needs to be above the Iterable check because a string is an Iterable
            if used_set is not None:
                used_set.add(qry)
            return qry in d.keys()

        if isinstance(qry, Iterable):
            if used_set is not None:
                used_set.update(qry)
            return all(i in list(d.keys()) for i in qry)

        raise TypeError(Err.type('qry', 'dictionary, Iterable, or string'))

    def apply_rules(self, pop, rules, iter, t, is_setup=False):
        '''
        Applies the list of rules, each of which may split the group into (possibly already extant) subgroups.  A
        sequential rule application scheme is (by the definition of sequentiality) bound to produce order effects which
        are undesirable.  To mitigate that problem, a Cartesian product of all the rule outcomes (i.e., split
        specifications; GroupSplitSpec class) is computed and the resulting cross-product spit specs are used to do the
        actual splitting.

        When creating the product of split specs created by the individual rules, the probabilities associated with
        those individual split specs are multiplied because the rules are assumed to be independent.  Any dependencies
        are assumed to have been handles inside the rules themselves.

        TODO: Think if the dependencies between rules could (or perhaps even should) be read from some sort of a
              graph.  Perhaps then multiplying the probabilities would not be appropriate.
        '''

        # Apply all the rules and get their respective split specs (ss):
        if is_setup:
            ss_rules = [r.setup(pop, self) for r in rules]
        else:
            ss_rules = [r.apply(pop, self, iter, t) for r in rules if r.is_applicable(self, iter, t)]

        ss_rules = [i for i in ss_rules if i is not None]
        if len(ss_rules) == 0:
            return None

        # Create a Cartesian product of the split specs (ss):
        ss_prod = []
        for ss_lst in itertools.product(*ss_rules):
            ss_comb = GroupSplitSpec(p=1.0, attr_set={}, attr_del=set(), rel_set={}, rel_del=set())  # the combined split spec
            for i in ss_lst:
                ss_comb.p *= i.p  # this assumes rule independence
                ss_comb.attr_set.update(i.attr_set)
                ss_comb.attr_del.update(i.attr_del)
                ss_comb.rel_set.update(i.rel_set)
                ss_comb.rel_del.update(i.rel_del)
            ss_prod.append(ss_comb)

        return self.split(ss_prod)

    def commit(self):
        ''' Ends creating the group by notifing the callee that has begun the group creation. '''

        if self.callee is None:
            return None

        c = self.callee
        self.callee.commit_group(self)
        self.callee = None
        return c

    def freeze(self):
        self.is_frozen = True

    @staticmethod
    def gen_hash(attr, rel):
        '''
        Generates a hash for the attributes and relations dictionaries.  This sort of hash is desired because groups
        are judged functionally equivalent or not based on the content of those two dictionaries alone and nothing else
        (e.g., the name and the size of a group does not affect its identity assessment).
        '''

        # TODO: The current implementation assumes dictionaries are not nested.  Properly hash nested dictionaries (via
        #       recursion) when that becomes necessary.
        #
        #       https://stackoverflow.com/questions/5884066/hashing-a-dictionary

        # TODO: The current implementation guarantees equality of hashes only within the lifespan of a Python
        #       interpreter.  Use another, deterministic, hashing algorithm when moving to a concurrent/distributed
        #       computation paradigm.
        #
        #       https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions/27522708#27522708

        attr = attr or {}
        rel  = rel  or {}

        return hash(tuple([frozenset(attr.items()), frozenset(rel.items())]))

    @classmethod
    def gen_from_db(cls, db_fpath, tbl, attr={}, rel={}, attr_db=[], rel_db=[], rel_at=None, limit=0):
        if not os.path.isfile(db_fpath):
            raise ValueError(f'The database does not exist: {db_fpath}')

        qry = 'SELECT COUNT(*) AS n{comma}{cols} FROM {tbl} WHERE {cols_where} GROUP BY {cols}{limit}'.format(
            tbl=tbl,
            cols=', '.join(attr_db + [r.col for r in rel_db]),
            cols_where=' AND '.join([c + ' IS NOT NULL' for c in attr_db + [r.col for r in rel_db]]),
            comma='' if len(attr_db + [r.col for r in rel_db]) == 0 else ', ',
            limit='' if limit <= 0 else f' LIMIT {limit}'
        )

        groups = []
        with DB.open_conn(db_fpath) as c:
            for row in c.execute(qry).fetchall():
                g_attr = {}
                g_rel  = {}

                g_attr.update(attr)
                g_rel.update(rel)

                g_attr.update({ a: row[a] for a in attr_db })
                g_rel.update({ spec.name: spec.entities[row[spec.col]] for spec in rel_db })

                if rel_at is not None:
                    g_rel.update({ Site.AT: g_rel.get(rel_at) })

                groups.append(cls(n=row['n'], attr=g_attr, rel=g_rel))

        return groups

    @staticmethod
    def gen_dict(d_in, d_upd=None, k_del=None):
        '''
        Returns a new dictionary based on the 'd_in' dictionary with values updated based on the 'd_upd' dictionary and
        keys deleted based on the 'k_del' iterable.

        A shallow copy of the dictionary is returned at this point.  That is to avoid creating unnecessary copies of
        entities that might be stored as relations.  A more adaptive mechanism can be implemented later if needed.
        '''

        # TODO: Consider: https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression

        ret = d_in.copy()

        if d_upd is not None:
            ret.update(d_upd)

        if k_del is not None:
            for k in k_del:
                if k in ret:
                    del ret[k]

        return ret

    def get_attr(self, name=None):
        if name is not None:
            self.attr_used.add(name)

        return self.attr[name] if name is not None else self.attr

    def get_hash(self):
        return self.__hash__()

    def get_rel(self, name=None):
        if name is not None:
            self.rel_used.add(name)

        return self.rel[name] if name is not None else self.rel

    def get_size(self):
        return self.n

    def has_attr(self, qry):
        return Group._has(self.attr, qry, self.attr_used)

    def has_rel(self, qry):
        return Group._has(self.rel, qry, self.rel_used)

    def set_attr(self, name, value, do_force=True):
        # if self.is_frozen:
        #     raise GroupFrozenError('Attempting to set an attribute of a frozen group.')
        #
        # if self.attr.get(name) is not None and not do_force:
        #     raise ValueError("Group '{}' already has the attribute '{}'.".format(self.name, name))

        self.attr[name] = value
        self.hash = None

        return self

    def set_attrs(self, attr, do_force=True):
        if self.is_frozen:
            raise GroupFrozenError('Attempting to set attributes of a frozen group.')

        raise Error('Not implemented yet')

    def set_rel(self, name, value, do_force=True):
        if self.is_frozen:
            raise GroupFrozenError('Attempting to set a relation of a frozen group.')

        # if name == Site.AT:
        #     raise ValueError("Relation name '{}' is restricted for internal use.".format(Site.AT))

        if self.rel.get(name) is not None and not do_force:
            raise ValueError("Group '{}' already has the relation '{}'.".format(self.name, name))

        self.rel[name] = value
        self.hash = None

        return self

    def set_rels(self, rel, do_force=True):
        if self.is_frozen:
            raise GroupFrozenError('Attempting to set relations of a frozen group.')

        raise Error('Not implemented yet')

    def split(self, specs):
        '''
        Splits the group into new groups according to the specs (i.e., a list of GroupSplitSpec objects).

        The probabilities defining the population mass distribution among the new groups need to add up to 1.
        Complementing of the last one of those probabilities is done automatically (i.e., it does not need to be
        provided and is in fact outright ignored).

        A note on performance.  The biggest performance hit is likley going to be generating a hash which happens as
        part of instantiating a new Group object.  While this may seem like a good reason to avoid crearing new groups,
        that line of reasoning is deceptive in that a group's hash is needed regardless.  Other than that, a group
        object is light so its impact on perfornace should be negligeable.  Furthermore, this also grants access to
        full functionality of the Group class to any function that uses the result of the present method.
        '''

        groups = []  # split result (i.e., new groups)
        p_sum = 0.0
        n_sum = 0.0

        for (i,s) in enumerate(specs):
            if i == len(specs) - 1:  # last group spec
                p = 1 - p_sum        # complement the probability
                n = self.n - n_sum   # make sure we're not missing anybody due to floating-point arithmetic
            else:
                p = s.p
                n = self.n * p
                # n = math.floor(self.n * p)  # conservative floor() use to make sure we don't go over due to rounding

            p_sum += p
            n_sum += n

            if n == 0:  # preventing instantiating empty groups
                continue

            attr = Group.gen_dict(self.attr, s.attr_set, s.attr_del)
            rel  = Group.gen_dict(self.rel,  s.rel_set,  s.rel_del)

            g = Group('{}.{}'.format(self.name, i), n, attr, rel)
            if g == self:
                g.name = self.name
            groups.append(g)

        return groups

    def unfreeze(self):
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
