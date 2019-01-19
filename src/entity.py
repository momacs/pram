import copy
import itertools
import math
import numpy as np

from attr import attrs, attrib
from enum import Enum
from scipy.stats import rv_continuous


class AttrSex(Enum):
    f = 0
    m = 1


class AttrFluStatus(Enum):
    no     = 0
    asympt = 1
    sympt  = 2


class DistributionAgeSchool(rv_continuous):
    # TODO: Finish.
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)


class DistributionAgeWork(rv_continuous):
    # TODO: Finish.
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)


# ======================================================================================================================
class EntityType(Enum):
    agent    = 1
    group    = 2
    site     = 3  # e.g., home, school, etc.
    resource = 4  # e.g., a public bus


class Entity(object):
    DEBUG_LVL = 1  # 0=none, 1=normal, 2=full

    __slots__ = ('type', 'id')

    def __init__(self, type, id):
        self.type = type
        self.id   = id

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return '{}  type: {}   id: {}'.format(self.__class__, self.type.name, self.id)

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1: print(msg)


class Site(Entity):
    '''
    A physical site (e.g., a school or a store) agents can reside at.

    A site has a sensible interface which makes it useful.  For example, it makes sense to ask about the size and
    composition of population (e.g., groups) that are at that location.  However, because this information (and other,
    similar pieces of information) may be desired at arbitrary times, it makes most sense to compute it lazily and
    memoize it.  For that reason, a site stores a link to the population it is associated with; it queries that
    population to compute quantities of interested when they are needed.  An added benefit of this design is fostering
    composition.  That is, it is the site's responsibility to updating its own state, not the population's.
    '''

    DEF_REL_NAME = 'location'  # default relation name

    __slots__ = ('name', 'rel_name', 'pop')

    def __init__(self, name, rel_name=DEF_REL_NAME, pop=None):
        super().__init__(EntityType.site, '')
        self.name = name
        self.rel_name = rel_name  # name of the relation the site is the object of
        self.pop = pop  # pointer to the population (can be set elsewhere too)

    def __eq__(self, other):
        '''
        We will make two sites identical if their keys are equal (i.e., object identity is not necessary).  This will
        let us recognize sites even if they are instantiated multiple times.
        '''

        return isinstance(self, type(other)) and (self.__key() == other.__key())

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.type, self.name)

    def __str__(self):
        return '{}  name: {}'.format(self.__class__.__name__, self.name)

    def __key(self):
        return (self.name)

    def get_hash(self):
        return self.__hash__()

    def get_groups(self, non_empty_only=True):
        '''
        Non empty groups (i.e., those with size greater than 0) are returned by default, but that behavior can be
        changed (non_empty_only=False)
        '''

        ret = self.pop.get_groups(None, { self.rel_name: self.get_hash() })
        if non_empty_only:
            ret = [i for i in ret if i.n > 0]
        return ret

    def get_pop_size(self):
        return sum([g.n for g in self.pop.get_groups(None, { self.rel_name: self.get_hash() })])

    def set_pop(self, pop):
        self.pop = pop


class Home(Site):
    __slots__ = ()

    def __init__(self):
        super().__init__('home')


class Resource(object):
    '''
    A resource shared by the agents (e.g., a public bus).
    '''

    pass


# ======================================================================================================================
class Agent(Entity):
    __slots__ = ('name', 'sex', 'age', 'flu', 'school', 'work', 'location')

    AGE_MIN =   0
    AGE_MAX = 120
    AGE_M   =  40
    AGE_SD  =  20

    P_STUDENT = 0.25  # unconditional prob. of being a student
    P_WORKER  = 0.60  # unconditional prob. of being a worker

    def __init__(self, name=None, sex=AttrSex.f, age=AGE_M, flu=AttrFluStatus.no, school=None, work=None, location='home'):
        super().__init__(EntityType.agent, '')

        self.name     = name if name is not None else '.'
        self.sex      = sex
        self.age      = age
        self.flu      = flu
        self.school   = school
        self.work     = work
        self.location = location

    def __repr__(self):
        return '{}(name={}, sex={}, age={}, flu={}, school={}, work={}, location={})'.format(self.__class__.__name__, self.name, self.sex.name, round(self.age, 2), self.flu.name, self.school, self.work, self.location)

    def __str__(self):
        return '{}  name: {:12}   sex:{}   age: {:3}   flu: {:6}   school: {:16}   work: {:16}   location: {:12}'.format(self.__class__.__name__, self.name, self.sex.name, round(self.age), self.flu.name, self.school or '.', self.work or '.', self.location or '.')

    @classmethod
    def gen(cls, name=None):
        ''' Generates a singular agent. '''

        if (name is None):
            name = '.'

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
        ''' Generates a list of agents (with monotnously increasing names). '''

        if n <= 0:
            return []
        return [cls.gen('a.{}'.format(i)) for i in range(n)]

    @staticmethod
    def random_age():
        return min(Agent.AGE_MAX, max(Agent.AGE_MIN, np.random.normal(Agent.AGE_M, Agent.AGE_SD)))

    @staticmethod
    def random_flu():
        return np.random.choice(list(AttrFluStatus))

    @staticmethod
    def random_sex():
        return np.random.choice(list(AttrSex))


# ======================================================================================================================
@attrs(frozen=True)
class GroupQry:
    '''
    Group query.

    Objects of this simple class are used to select groups from a group population.
    '''

    attr : dict = attrib(default=dict())
    rel  : dict = attrib(default=dict())


@attrs(kw_only=True)
class GroupSplitSpec:
    '''
    A single group-split specification.

    These specifications are oridinarily provided in a list to indicate new groups that one other group is being split
    into.

    TODO: At this point, attributes and relations to be removed are assumed to be identified by their names only and
          not their values (i.e., we use a set to hold the keys that should be removed from the dictionaries for
          attributes and relations).  Perhaps this is not the way to go and we should instead be using both names and
          values.
    '''

    p        : float = attrib(default=0.0)  # validator=attr.validators.instance_of(float))
    attr_upd : dict  = attrib(default=dict())
    attr_del : set   = attrib(default=set())
    rel_upd  : dict  = attrib(default=dict())
    rel_del  : set   = attrib(default=set())  # repr=False

    @p.validator
    def is_prob(self, attribute, value):
        if not isinstance(value, float):
            raise TypeError("The probability 'p' must be of type float.")
        if not (0 <= value <= 1):
            raise ValueError("The probability 'p' must be in [0,1] range.")


# ======================================================================================================================
class Group(Entity):
    __slots__ = ('name', 'n', 'attr', 'rel', '_hash')

    def __init__(self, name=None, n=0.0, attr={}, rel={}):
        super().__init__(EntityType.group, '')

        self.name = name if name is not None else '.'
        self.n    = float(n)
        self.attr = attr if attr is not None else {}
        self.rel  = rel  if rel  is not None else {}

        self._hash = Group.gen_hash(self.attr, self.rel)  # we could compute this lazily, but we will need it anyway

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
        return self._hash

    def __repr__(self):
        return '{}(name={}, n={}, attr={}, rel={})'.format(__class__.__name__, self.name, self.n, self.attr, self.rel)

    def __str__(self):
        return '{}  name: {:16}  n: {:8}  attr: {}  rel: {}'.format(self.__class__.__name__, self.name, self.n, self.attr, self.rel)

    @staticmethod
    def _has(d, qry):
        '''
        Compares the dictionary 'd' against an iterable or a dictionary 'qry'.  If 'qry' is an iterable, the method
        checks if all items in it are keys in 'd'.  If 'qry' is a dictionary, the method checks if all the items in
        that dictionary exist in 'd'.
        '''

        if isinstance(qry, dict):
            return qry.items() <= d.items()

        return all(i in list(d.keys()) for i in qry)

    def apply_rules(self, rules, t):
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
        ss_rules = [r.apply(self, t) for r in rules]
        ss_rules = [i for i in ss_rules if i is not None]
        if len(ss_rules) == 0:
            return None

        # Create a Cartesian product of the split specs (ss):
        ss_prod = []
        for ss_lst in itertools.product(*ss_rules):
            ss_comb = GroupSplitSpec(p=1.0, attr_upd={}, attr_del=set(), rel_upd={}, rel_del=set())  # the combined split spec
            for i in ss_lst:
                ss_comb.p *= i.p  # this assumes rule independence
                ss_comb.attr_upd.update(i.attr_upd)
                ss_comb.attr_del.update(i.attr_del)
                ss_comb.rel_upd.update(i.rel_upd)
                ss_comb.rel_del.update(i.rel_del)
            ss_prod.append(ss_comb)

        return self.split(ss_prod)

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

    @staticmethod
    def gen_dict(d_in, d_upd=None, k_del=None):
        '''
        Returns a new dictionary based on the 'd_in' dictionary with values updated based on the 'd_upd' dictionary and
        keys deleted based on the 'k_del' iterable.

        A shallow copy of the dictionary is returned at this point.  That is to avoid creating unnecessary copies of
        Sites and other entities that might be stored as relations.  A more adaptive mechanism can be implemented
        later if needed.
        '''

        # TODO: Consider: https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression

        ret = dict(d_in)  # shallow copy

        if d_upd is not None:
            ret.update(d_upd)

        if k_del is not None:
            for k in k_del:
                if k in ret: del ret[k]

        return ret

    def get_attr(self, name):
        return self.attr[name]

    def get_hash(self):
        return self._hash

    def get_rel(self, name):
        return self.rel[name]

    def has_attr(self, qry):
        return Group._has(self.attr, qry)

    def has_rel(self, qry):
        return Group._has(self.rel, qry)

    def split(self, specs):
        '''
        Splits the group into new groups according to the specs (i.e., a list of GroupSplitSpec objects).

        The probabilities defining the population mass distribution among the new groups need to add up to 1.
        Complementing of the last one of those probabilities is done automatically (i.e., it does not need to be
        provided and is in fact outright ignored).

        A note on performance.  The biggest performance hit is likley going to be generating a hash which happens as
        part of instantiating a new Group object.  While this may seem like a good reason to avoid crearing new groups,
        that line of reasoning is deceptive in that a group's hash is needed anyway and therefore would need to be
        computed regardless.  Other than that, a group object is light so it seems to me that going the extra mile
        right here makes the most sense, especially that all the work done here happens in context and that fosters
        compartmentalization.  Furthermore, this also grants access to full functionality of the Group class to any
        function that uses the result of the present method.
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

            attr = Group.gen_dict(self.attr, s.attr_upd, s.attr_del)
            rel  = Group.gen_dict(self.rel,  s.rel_upd,  s.rel_del)

            g = Group('{}.{}'.format(self.name, i), n, attr, rel)
            if g == self:
                g.name = self.name
            groups.append(g)

        return groups


# ======================================================================================================================
class EntityMan(object):
    '''
    Entity manager.

    Allows for entities to be kept in one place.  Entities can be added, removed, and retrieved (if they exist).
    Entity type (i.e., EntityType class) check enforces homogeneity of the collection, which I suspect may help to
    prevent bugs.

    For every entity stored inside of the manager, a key used for later retrieval can be provided.  There is no
    restriction on that key and therefore it could be a name of the entity or something equally human-friendly.  If no
    key is provided, the class falls back on using the entity's hash as the key.
    '''

    __slots__ = ('type', 'dict')

    def __init__(self, type):
        if not isinstance(type, EntityType):
            raise TypeError("Argument 'type' must take the EntityType enumeration type values.")

        self.type = type
        self.dict = {}

    def add(self, key, entity):
        '''
        The 'key' is associated with the entity and can be used to retrieve that entity later on.  If no key is
        provided (i.e., key=None), the entity's hash is used as the key.
        '''

        if entity.type != self.type:
            raise TypeError("Entity of type '{}' cannot be added to an entity manager of type '{}'.".format(entity.type, self.type))

        key = key or entity.get_hash()

        self.dict[key] = entity

    def rem(self, entity):
        try:
            del self.dict[entity.get_hash()]
        except KeyError:
            pass

    def get(self, key):
        return self.dict.get(key)


# ======================================================================================================================
if __name__ == '__main__':
    rand_seed = 1928

    np.random.seed(rand_seed)

    # (1) Agents:
    print('(1) Agents')

    print(Agent('smith', AttrSex.m, 99.99, school=None, work='matrix', location='matrix'))

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
    print(GroupSplitSpec(p=0.1416, attr_del=['income']))

    g1_split = g1.split([
        GroupSplitSpec(p=0.1416, attr_del=['income']),
        GroupSplitSpec(          rel_upd={ 'location': 'work' })
    ])

    print()
    for g in g1_split:
        print(g)
