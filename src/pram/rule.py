import math
import random

from abc         import abstractmethod, ABC
from attr        import attrs, attrib
from dotmap      import DotMap
from enum        import IntEnum
from scipy.stats import lognorm, poisson, rv_discrete

from .entity import GroupQry, GroupSplitSpec, Site
from .util   import Err, Time as TimeU


# from enum        import IntEnum
# A = IntEnum('State', 'S E I R')
# print(list(A))
# b = [member.value for name, member in A.__members__.items()]
# print(b)


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class Time(object):
    pass


@attrs(slots=True)
class TimeAlways(Time):
    pass


@attrs(slots=True)
class TimePoint(Time):
    t: float = attrib(default=0.00, converter=float)


@attrs(slots=True)
class TimeInt(Time):
    t0: float = attrib(default= 0.00, converter=float)
    t1: float = attrib(default=24.00, converter=float)


# ----------------------------------------------------------------------------------------------------------------------
class Rule(ABC):
    '''
    A rule that can be applied to a group and may augment that group or split it into multiple subgroups.

    A rule will be applied if the simulation timer's time (external to this class) falls within the range defined by
    the time specification 't'.  Every time a rule is applied, it is applied to all groups it is compatible with.  For
    instance, a rule that renders a portion of a group infection-free (i.e., marks it as recovered) can be applied to a
    group of humans currently infected with some infectious disease.  The same rule, however, would not be applied to
    a group of city buses.  Each rule knows how to recognize a compatible group.
    '''

    T_UNIT_MS = TimeU.MS.h
    NAME = 'Rule'
    ATTRS = {}  # a dict of attribute names as keys and the list of their values as values

    pop = None
    compile_spec = None

    def __init__(self, name, t, name_human=None, memo=None):
        '''
        t: Time
        '''

        Err.type(t, 't', Time)

        self.name = name
        self.t = t
        self.memo = memo
        self.name_human = name_human or name

        self.t_unit_ms = None  # set via set_t_unit() by the simulation every time it runs
        self.t_mul = 0.00      # ^
        self.T = {}

    def __repr__(self):
        if isinstance(self.t, TimeAlways):
            return '{}(name={}, t=.)'.format(self.__class__.__name__, self.name)

        if isinstance(self.t, TimePoint):
            return '{}(name={}, t={:>4})'.format(self.__class__.__name__, self.name, round(self.t.t, 1))

        if isinstance(self.t, TimeInt):
            return '{}(name={}, t=({:>4},{:>4}))'.format(self.__class__.__name__, self.name, round(self.t.t0, 1), round(self.t.t1, 1))

    def __str__(self):
        if isinstance(self.t, TimeAlways):
            return 'Rule  name: {:16}  t: .'.format(self.name)

        if isinstance(self.t, TimePoint):
            return 'Rule  name: {:16}  t: {:>4}'.format(self.name, round(self.t.t, 1))

        if isinstance(self.t, TimeInt):
            return 'Rule  name: {:16}  t: ({:>4},{:>4})'.format(self.name, round(self.t.t0, 1), round(self.t.t1, 1))

    @abstractmethod
    def apply(self, pop, group, iter, t):
        pass

    def cleanup(self, pop, group):
        '''
        Run once at the end of a simulation run.  Symmetrical to the setup() method.  Also uses the group-Splitting
        mechanism.
        '''

        pass

    def is_applicable(self, group, iter, t):
        ''' Verifies if the rule is applicable given the context. '''

        if isinstance(self.t, TimeAlways):
            return True

        if isinstance(self.t, TimePoint):
            return self.t.t == t

        if isinstance(self.t, TimeInt):
            return self.t.t0 <= t <= self.t.t1

        raise TypeError("Type '{}' used for specifying rule timing not yet implemented (Rule.is_applicable).".format(type(self.t).__name__))

    def set_t_unit(self, ms):
        self.t_unit_ms = ms
        self.t_mul = float(self.__class__.T_UNIT_MS) / float(self.t_unit_ms)

    def setup(self, pop, group):
        '''
        A rule's setup place.  If the rule relies on groups having a certain set of attributes and relations, this is
        where they should be set.  For example, a rule might set an attribute of all the groups like so:

            return [GroupSplitSpec(p=1.0, attr_set={ 'did-attend-school-today': False })]

        This reuses the group splitting mechanism; here, each group will be split into a (possibly non-extant) new
        group and the entirety of the group's mass will be moved into that new group.

        This is also where a rule should do any other population initialization required.  For example, a rule that may
        introduce a new Site object to the simulation, should make the population object aware of that site like so:

            pop.add_site(Site('new-site'))

        Every rule's setup() method is called only once by Simulation.run() method before a simulation run commences.
        '''

        pass

    @staticmethod
    def tp2rv(tp, a=0, b=23):
        ''' Converts a time probability distribution function (PDF) to a discrete random variable. '''

        return rv_discrete(a,b, values=(tuple(tp.keys()), tuple(tp.values())))


# ----------------------------------------------------------------------------------------------------------------------
class ProbabilisticAutomaton(Rule, ABC):
    '''
    Probabilistic automaton.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class MarkovChain(ProbabilisticAutomaton):
    '''
    Time-homogenous Markov chain with finite state space.

    The following example transition model for the variables named X:

                   x_1^t   x_2^t
        x_1^{t+1}    0.1     0.3
        x_2^{t+1}    0.9     0.7

    Should be specified as:

        { 'x1': [0.1, 0.9], 'x2': [0.3, 0.7] }

    ----

    code:
        MarkovChain('flu', { 's': [0.95, 0.05, 0.00], 'i': [0.00, 0.50, 0.50], 'r': [0.10, 0.00, 0.90] })

    init:
        tm = {
            s: [0.95, 0.05, 0.00],
            i: [0.00, 0.50, 0.50],
            r: [0.10, 0.00, 0.90]
        }  # right stochastic matrix

    is-applicable:
        has-attr: flu

    apply:
        tm_i = tm[group.attr.flu]
        move-mass:
            tm_i[0] -> A: flu = s
            tm_i[1] -> A: flu = i
            tm_i[2] -> A: flu = r
    '''

    def __init__(self, var, tm, name='markov-chain', t=TimeAlways(), memo=None):
        super().__init__(name, t, memo)

        if sum([i for x in list(tm.values()) for i in x]) != float(len(tm)):
            raise ValueError(f"'{self.__class__.__name__}' class: Probabilities in the transition model must add up to 1")

        self.var = var
        self.tm = tm
        self.states = list(self.tm.keys())  # simplify and speed-up lookup in apply()

    def apply(self, pop, group, iter, t):
        tm = self.tm.get(group.get_attr(self.var))
        if tm is None:
            raise ValueError(f"'{self.__class__.__name__}' class: Unknown state '{group.get_attr(self.var)}' for attribute '{self.var}'")
        return [GroupSplitSpec(p=tm[i], attr_set={ self.var: self.states[i] }) for i in range(len(self.states)) if tm[i] > 0]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ self.var ])


# ----------------------------------------------------------------------------------------------------------------------
class TimeVaryingMarkovChain(MarkovChain):
    '''
    Time-homogenous Markov chain with finite state space with a time-varying transition matrix.

    Possible alternative names:
        Time-dependant MC
        Non-homogenous MC
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class StochasticProcess(Rule, ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class MarkovProcess(StochasticProcess):
    pass


# ----------------------------------------------------------------------------------------------------------------------
class BernoulliScheme(MarkovProcess):
    '''
    Bernoulli scheme or Bernoulli shift is a generalization of the Bernoulli process to more than two possible
    outcomes.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class BernoulliProcess(BernoulliScheme):
    '''
    A Bernoulli scheme is a special case of a Markov chain where the transition probability matrix has identical rows,
    which means that the next state is even independent of the current state (in addition to being independent of the
    past states). A Bernoulli scheme with only two possible states is known as a Bernoulli process.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class RandomWalk(MarkovProcess):
    '''
    Markov processes in discreet time.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class WienerProcess(MarkovProcess):
    '''
    Markov processes in continuous time.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class PoissonProcess(MarkovProcess):
    '''
    Markov processes in continuous time.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class JumpProcess(PoissonProcess):
    '''
    A jump process is a type of stochastic process that has discrete movements, called jumps, with random arrival
    times, rather than continuous movement, typically modelled as a simple or compound Poisson process.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class CellularAutomaton(StochasticProcess):
    '''
    Cellular automata are a discrete-time dynamical system of interacting entities, whose state is discrete.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class StochasticCellularAutomaton(CellularAutomaton):
    '''
    Probabilistic cellular automata (PCA), random cellular automata, or locally interacting Markov chains.


    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class InteractingParticleSystem(StochasticProcess):
    '''
    Continuous-time Markov jump processes describing the collective behavior of stochastically interacting components.
    IPS are the continuous-time analogue of stochastic cellular automata.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class BirthDeathProcess(PoissonProcess):
    '''
    A jump process is a type of stochastic process that has discrete movements, called jumps, with random arrival
    times, rather than continuous movement, typically modelled as a simple or compound Poisson process.
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class BirthDeathProcess(MarkovProcess):
    '''
    A special case of continuous-time Markov process where the state transitions are of only two types: "births", which
    increase the state variable by one and "deaths", which decrease the state by one.

    A homogeneous Poisson process is a pure birth process.

    The following infinitesimal generator of the process (\lambda are birth rates and \mu are death rates):

        | -l_0           l_0             0             0    0  ... |
        |  m_1  -(l_1 + m_1)           l_1             0    0  ... |
        |    0           m_2  -(l_2 + m_2)           l_2    0  ... |
        |    0             0           m_3  -(l_3 + m_3)  l_3  ... |
        |    .             .             .             .    .  ... |
        |    .             .             .             .    .  ... |

    should be provided as:

        [l_0, [l_1, m_1], [l_2, m_2], [l_3, m_3], ..., m_n]

    Sojourn times have exponential p.d.f. $\lambda e^{-\lambda t}$.

    --------------------------------------------------------------------------------------------------------------------

    Birth process examples [1]:
    - Radioactive transformations
    - Conversion of fibroge molecules into fibrin molecules follows a birth process (blood clotting)

    Application areas [1]:
    - Epidemics
    - Queues, inventories, and reliability
    - Production management
    - Computer communication systems
    - Neutron propagation
    - Optics
    - Chemical reactions
    - Construction and mining
    - Compartmental models

    References:
    [1] Birth and Death Processes (Chapter 4)
        http://neutrino.aquaphoenix.com/ReactionDiffusion/SERC5chap4.pdf
    [2] https://cs.nyu.edu/mishra/COURSES/09.HPGP/scribe3
    '''

    def __init__(self, ig, name='birth-death-process', t=TimeAlways()):
        super().__init__(name, t)

        self.ig = ig  # inifinitesimal generator of the process

    def apply(self, pop, group, iter, t):
        return None


# ----------------------------------------------------------------------------------------------------------------------
class SISModel(MarkovChain):
    '''
    The SIS epidemiological model without vital dynamics.

    Model parameters:
        beta  - transmission rate
        gamma - recovery rate

    ----

    code:
        SISModel('flu', 0.05, 0.10)
    '''

    def __init__(self, var, beta, gamma, name='sis-model', t=TimeAlways(), memo=None):
        super().__init__(var, { 's': [1 - beta, beta], 'i': [gamma, 1 - gamma] }, name, t, memo)


# ----------------------------------------------------------------------------------------------------------------------
class SIRSModel(MarkovChain):
    '''
    The SIR(S) epidemiological model without vital dynamics.

    Model parameters:
        beta  - transmission rate
        gamma - recovery rate
        alpha - immunity loss rate (alpha = 0 implies life-long immunity)

    ----

    code:
        SIRSModel('flu', 0.05, 0.20, 0.10)  # SIRS
        SIRSModel('flu', 0.05, 0.20, 0.00)  # SIR
    '''

    def __init__(self, var, beta, gamma, alpha, name='sir-model', t=TimeAlways(), memo=None):
        super().__init__(var, { 's': [1 - beta, beta, 0.00], 'i': [0.00, 1 - gamma, gamma], 'r': [alpha, 0.00, 1 - alpha] }, name, t, memo)


# ----------------------------------------------------------------------------------------------------------------------
class PoissonIncidenceProcess(PoissonProcess):
    '''
    Homogenous and inhomogenous Poisson point process for disease incidence.

    The non-homogeneity of the Poisson process is handled by allowing the user to specify how the rate (i.e., the
    $\lambda$ parameter) of the process changes over time.  The change to that rate is given as the combination of a
    delta on it and the number of iterations for that delta to take place.  The rate parameter is allowed to change
    gradually (i.e., at every step of the simulation) or in strict increments (e.g., only every 10 iterations) and is
    controlled by the 'inc_smooth' parameter.

    The rate can change in the following ways (as determined by the 'rate_delta_mode' parameter which takes values of
    the 'RateDeltaMode' enumeration):

    - NC   No change (i.e., the Poisson process is stationary)
    - ADD  Additive
    - MUL  Multiplicative
    - EXP  Exponential
    - FN   User-provided lambda function called at every iteration

    A Cox process, also known as a doubly stochastic Poisson process is a point process which is a generalization of a
    Poisson process where the time-dependent intensity is itself a stochastic process.  The Cox process can be
    implemented by passing as an argument a function which will be called at every iteration to determine the current
    Poisson rate.  'rate_delta_mode=RateDeltaMode.FN' must be used in that case.

    An example model given as "AD incidence doubles every five years after 65 yo" can be instantiated by using the
    delta of two and the number of iterations of five.

    --------------------------------------------------------------------------------------------------------------------

    Epidemiology of Alzheimer's disease and other forms of dementia in China, 1990-2010: a systematic review and analysis.
    https://www.thelancet.com/action/showPdf?pii=S0140-6736%2813%2960221-4

    --------------------------------------------------------------------------------------------------------------------

    SRC: https://en.wikipedia.org/wiki/Incidence_(epidemiology); retrieved on 2019.06.05

    # Incidence (epidemiology)
    --------------------------

    Incidence in epidemiology is a measure of the probability of occurrence of a given medical condition in a
    population within a specified period of time. Although sometimes loosely expressed simply as the number of new
    cases during some time period, it is better expressed as a proportion or a rate[1] with a denominator.

    Incidence proportion (also known as cumulative incidence) is the number of new cases within a specified time period
    divided by the size of the population initially at risk. For example, if a population initially contains 1,000
    non-diseased persons and 28 develop a condition over two years of observation, the incidence proportion is 28 cases
    per 1,000 persons per two years, i.e. 2.8% per two years.

    (...)


    ## Incidence vs. prevalence
    ---------------------------

    Incidence should not be confused with prevalence, which is the proportion of cases in the population at a given
    time rather than rate of occurrence of new cases. Thus, incidence conveys information about the risk of contracting
    the disease, whereas prevalence indicates how widespread the disease is. Prevalence is the proportion of the total
    number of cases to the total population and is more a measure of the burden of the disease on society with no
    regard to time at risk or when subjects may have been exposed to a possible risk factor. Prevalence can also be
    measured with respect to a specific subgroup of a population (see: denominator data). Incidence is usually more
    useful than prevalence in understanding the disease etiology: for example, if the incidence rate of a disease in a
    population increases, then there is a risk factor that promotes the incidence.

    For example, consider a disease that takes a long time to cure and was widespread in 2002 but dissipated in 2003.
    This disease will have both high incidence and high prevalence in 2002, but in 2003 it will have a low incidence
    yet will continue to have a high prevalence (because it takes a long time to cure, so the fraction of individuals
    that are affected remains high). In contrast, a disease that has a short duration may have a low prevalence and a
    high incidence. When the incidence is approximately constant for the duration of the disease, prevalence is
    approximately the product of disease incidence and average disease duration, so prevalence = incidence × duration.
    The importance of this equation is in the relation between prevalence and incidence; for example, when the
    incidence increases, then the prevalence must also increase. Note that this relation does not hold for age-specific
    prevalence and incidence, where the relation becomes more complicated.[6]

    (...)

    ----

    code:
        PoissonIncidenceProcess('ad', 65, 0.01, 2, 5, rate_delta_mode=PoissonIncidenceProcess.RateDeltaMode.EXP))

    init:
        age_0=65, l_0=0.01, c=2, t_c=5

    is-applicable:
        has-attr: age, ad

    apply:
        if (group.attr.age >= age_0):
            l = double_rate(l)  # l0 * c^{(group.attr.age - age_0) / t_c}
            p_0 = PoissonPMF(l,0)
            move-mass:
                    p_0 -> A: age = group.attr.age + 1
                1 - p_0 -> A: age = group.attr.age + 1, A: ad = True

    ----

    if (group.attr.age >= age_0):
        l = double_rate(l)  # l0 * c^{(group.attr.age - age_0) / t_c}
        p_0 = PoissonPMF(l,0)
        p_0     -> A: age = group.attr.age + 1
        1 - p_0 -> A: age = group.attr.age + 1, A: ad = True

    ----

    mu = e^(lambda * x)
    log(mu) = lambda * x

    log(mu/t) = lambda * x
    log(mu/t) = lambda * x
    log(mu) - log(t) = lambda * x
    log(mu) = lambda * x + log(t)

    log(incidence) = lambda * x + log(age - age_0)
    '''

    RateDeltaMode = IntEnum('RateDeltaMode', 'NC ADD MUL EXP FN')  # incidence (or rate) change mode

    def __init__(self, attr, age_0, rate_0, delta, delta_t, rate_delta_mode=RateDeltaMode.NC, fn_calc_lambda=None, is_smooth=True, name='change-incidence', t=TimeAlways()):
        super().__init__(name, t)

        self.attr = attr            # the attribute name for the disease
        self.age_0 = age_0          # lower bound cut-off age for getting AD
        self.rate_0 = rate_0        # base rate of the Poisson process (i.e., at the cut-off age)
        self.rate = rate_0          # current rate of the Poisson process
        self.delta = delta          # increase the rate by this factor
        self.delta_t = delta_t      # increase the rate every this many time units
        self.is_smooth = is_smooth  # flag: smooth increment? (or happen in the defined increments)

        self.rate_delta_mode = rate_delta_mode
        self.fn_calc_lambda = {
            self.RateDeltaMode.NC  : self.calc_lambda_nc,
            self.RateDeltaMode.ADD : self.calc_lambda_add,
            self.RateDeltaMode.MUL : self.calc_lambda_mul,
            self.RateDeltaMode.EXP : self.calc_lambda_exp,
            self.RateDeltaMode.FN  : rate_delta_mode
        }.get(self.rate_delta_mode)

    def apply(self, pop, group, iter, t):
        return self.get_split_specs(group)

    def calc_lambda_nc(self, age):
        ''' No rate change (i.e., a stationary Poisson point process). '''

        return self.rate_0

    def calc_lambda_add(self, age):
        ''' Additive rate change. '''

        if self.is_smooth:
            return self.rate_0 + self.delta *            ((age - self.age_0) / self.delta_t)
        else:
            return self.rate_0 + self.delta *  math.floor((age - self.age_0) / self.delta_t)

    def calc_lambda_mul(self, age):
        ''' Multiplicative rate change. '''

        if self.is_smooth:
            return self.rate_0 * self.delta *            ((age - self.age_0) / self.delta_t)
        else:
            return self.rate_0 * self.delta *  math.floor((age - self.age_0) / self.delta_t)

    def calc_lambda_exp(self, age):
        ''' Exponential rate change. '''

        if self.is_smooth:
            return self.rate_0 * self.delta **           ((age - self.age_0) / self.delta_t)
        else:
            return self.rate_0 * self.delta ** math.floor((age - self.age_0) / self.delta_t)

    def calc_lambda_hmm(self, age):
        '''
        Poisson hidden Markov models (PHMM) are special cases of hidden Markov models where a Poisson process has a
        rate which varies in association with changes between the different states of a Markov model.
        '''

        return self.rate_0

    def get_split_specs(self, group, age_inc=1):
        '''
        age_inc - how much to increment the 'age' attribute
        '''

        age = group.ga('age')

        if age < self.age_0:
            return [GroupSplitSpec(p=1.00, attr_set={ 'age': age + age_inc, self.attr: False })]

        self.rate = self.fn_calc_lambda(age)
        p0 = poisson(self.rate).pmf(0)
        # print(f'n: {round(group.n,2):>12}  age: {age:>3}  l: {l:>3}  p0: {round(p0,2):<4}  p1: {round(1-p0,2):<4}')

        return [
            GroupSplitSpec(p=    p0, attr_set={ 'age': age + age_inc, self.attr: False }),
            GroupSplitSpec(p=1 - p0, attr_set={ 'age': age + age_inc, self.attr: True  })
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha(['age', self.attr])

    def setup(self, pop, group):
        if group.ha('age'):
            return self.get_split_specs(group, 0)
        else:
            return None


# ----------------------------------------------------------------------------------------------------------------------
class OneDecayProcess(Rule):
    '''
    Radioactive one-decay process.

    N(t) = N_0 e^{-\lambda t}
    '''

    pass


# ----------------------------------------------------------------------------------------------------------------------
class SegregationModel(Rule):
    '''
    Segregation model.

    ----

    code:
        SegregationModel('team', 2)

    init:
        p_migrate = 0.05

    is-applicable:
        has-attr: team
        has-rel: @

    apply:
        p_team = n@_{attr.team = group.attr.team} / n@
        if group.attr.flu = 'r':
            move-mass:
                p_migrate -> R: @ = get_random_site()

    ----

    p_team = n@_{attr.team = group.attr.team} / n@
    if (p_team < 0.5) then p_migrate -> R: @ = get_random_site()
    '''

    def __init__(self, attr, attr_dom_card, p_migrate=0.05):
        '''
        attr_dom_card - cardinality of the attribute values set
        p_migrate     - the proportion of the population to migrate if repelled
        '''

        super().__init__('segregation-model', TimeAlways())

        self.attr = attr
        self.p_migrate = p_migrate           # proportion of the population that will migrate if repelled
        self.p_repel = 1.00 / attr_dom_card  # population will be repelled (i.e., will move) if the site that population is at has a proportion of same self.attr lower than this

    def apply(self, pop, group, iter, t):
        attr   = group.ga(self.attr)
        site   = group.gr(Site.AT)
        n      = site.get_pop_size()
        n_team = site.get_pop_size(GroupQry(attr={ self.attr: attr }))

        if n == 0:
            return None

        p_team = n_team / n  # proportion of same self.attr

        if p_team < self.p_repel:
            site_rnd = self.get_random_site(pop, site)
            return [
                GroupSplitSpec(p=    self.p_migrate, rel_set={ Site.AT: site_rnd }),
                GroupSplitSpec(p=1 - self.p_migrate)
            ]
        else:
            return None

    def get_random_site(self, pop, site):
        ''' Returns a random site different than the specified one. '''

        s = random.choice(list(pop.sites.values()))
        while s == site:
            s = random.choice(list(pop.sites.values()))
        return s

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha([self.attr]) and group.hr([Site.AT])


# ----------------------------------------------------------------------------------------------------------------------
class SEIRModel(Rule, ABC):
    '''
    The SEIR compartmental epidemiological model.

    Disease states (SEIR)
        S Susceptible
        E Exposed (i.e., incubation period)
        I Infectious (can infect other agents)
        R Recovered

    State transition timing
        S  Until E (either by another agent or on import)
        E  Random sample
        I  Random sample
        R  Indefinite
    '''

    State = IntEnum('State', 'S E I R')

    ATTR = 'seir-state'

    T_UNIT_MS = TimeU.MS.d
    NAME = 'SEIR model'
    ATTRS = { ATTR: [member.value for name, member in State.__members__.items()] }


    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, name='seir', t=TimeAlways(), susceptibility=1.0, p_start_E=0.05, do_clean=True, name_human=None, memo=None):
        super().__init__(name, t, name_human, memo)

        self.susceptibility = susceptibility
        self.p_start_E = p_start_E    # prob of starting in the E state
        self.do_clean = do_clean      # flag: remove traces of the disease when recovered?

    # ------------------------------------------------------------------------------------------------------------------
    def apply(self, pop, group, iter, t):
        if iter == 0:
            return [
                GroupSplitSpec(p=    self.p_start_E, attr_set={ self.__class__.ATTR: self.__class__.State.E }),
                GroupSplitSpec(p=1 - self.p_start_E, attr_set={ self.__class__.ATTR: self.__class__.State.S })
            ]

        state = group.get_attr(self.__class__.ATTR)
        return {
            self.__class__.State.S: self.apply_S,
            self.__class__.State.E: self.apply_E,
            self.__class__.State.I: self.apply_I,
            self.__class__.State.R: self.apply_R,
        }.get(state, self.apply_err)(pop, group, iter, t, state)

    # ------------------------------------------------------------------------------------------------------------------
    def apply_S(self, pop, group, iter, t, state):
        return None

    def apply_E(self, pop, group, iter, t, state):
        # s  = self.T['E_rvs']()
        # tE = group.get_attr('tE') or 0
        # p  = min(1.00, tE / s)
        # print(f'E: {round(s,2)}  {tE}  {round(p,2)}')

        tE = group.get_attr('tE') or 0
        if tE < self.T['E_min'] * self.t_mul:
            p = 0.00
        elif tE > self.T['E_max'] * self.t_mul:
            p = 1.00
        else:
            p = self.T['E_rv'].cdf(tE / self.t_mul)

        # print(f'E: {round(p,4)}  {tE}')

        if self.do_clean:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.I }, attr_del=['tE']),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.E, 'tE': tE + 1 })
            ]
        else:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.I, 'tE': tE + 1 }),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.E, 'tE': tE + 1 })
            ]

    def apply_I(self, pop, group, iter, t, state):
        # s  = self.T['I_rvs']()
        # tI = group.get_attr('tI') or 0
        # p  = min(1.00, tI / s)
        # print(f'I: {round(s,2)}  {tI}  {round(p,2)}')

        tI = group.get_attr('tI') or 0
        if tI < self.T['I_min'] * self.t_mul:
            p = 0.00
        elif tI > self.T['I_max'] * self.t_mul:
            # print(iter)
            p = 1.00
        else:
            p = self.T['I_rv'].cdf(tI / self.t_mul)

        # if tI > self.T['I_max'] * self.t_mul:
        #     print(group.n)

        # print(f'I: {round(p,4)}  {tI}')

        if self.do_clean:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.R }, attr_del=['tI']),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.I, 'tI': tI + 1 })
            ]
        else:
            return [
                GroupSplitSpec(p=    p, attr_set={ self.__class__.ATTR: self.__class__.State.R, 'tI': t + 1 }),
                GroupSplitSpec(p=1 - p, attr_set={ self.__class__.ATTR: self.__class__.State.I, 'tI': t + 1 })
            ]

    def apply_R(self, pop, group, iter, t, state):
        return None

    def apply_err(self, pop, group, iter, t, state):
        raise ValueError(f'Invalid value for attribute {self.__class__.ATTR}: {state}')

    # ------------------------------------------------------------------------------------------------------------------
    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.get_attr(self.__class__.ATTR) in self.__class__.State

    def setup(self, pop, group):
        return [GroupSplitSpec(p=1.0, attr_set={ self.__class__.ATTR: __class__.State.S })]

    def cleanup(self, pop, group):
        return [GroupSplitSpec(p=1.0, attr_del=['tE', 'tI'])]


# ----------------------------------------------------------------------------------------------------------------------
class SEIRFluModel(SEIRModel):
    '''
    The SEIR model for the influenza.

    --------------------------------------------------------------------------------------------------------------------
    This part is taken from the "FRED daily rules" by David Sinclair and John Grefenstette (5 Feb, 2019)
    --------------------------------------------------------------------------------------------------------------------

    Disease states (SEIR)
        S  Susceptible
        E  Exposed (i.e., incubation period)
        IS Infectious & Symptomatic (can infect other agents)
        IA Infectious & Asymptomatic (can infect other agents)
        R  Recovered

    State transition probability
        S  --> E   1.00
        E  --> IS  0.67
        E  --> IA  0.33
        IS --> R   1.00
        IA --> R   1.00

    State transition timing
        S      Until E (either by another agent or on import)
        E      Median = 1.9, dispersion = 1.23 days (M= 45.60, d=29.52 hours)
        IS/IA  Median = 5.0, dispersion = 1.50 days (M=120.00, d=36.00 hours)
        R      Indefinite

    Probability of staying home
        IS/IA  0.50

    Time periods
        Drawn from a log-normal distribution
            median = exp(mean)
        Dispersion is equivalent to the standard deviation, but with range of [median/dispersion, median*dispersion]
            E      [1.54, 2.34] days ([36.96,  56.16] hours)
            IS/IA  [3.33, 7.50] days ([79.92, 180.00] hours)

    --------------------------------------------------------------------------------------------------------------------
    This part is taken from FRED docs on parameters (github.com/PublicHealthDynamicsLab/FRED/wiki/Parameters)
    --------------------------------------------------------------------------------------------------------------------

    The median and dispersion of the lognormal distribution are typically how incubation periods and symptoms are
    reported in the literature (e.g., Lessler, 2009).  They are translated into to the parameters of the lognormal
    distribution using the formulas:

        location = log(median)
        scale = 0.5 * log(dispersion)

    We can expect that about 95% of the draws from this lognormal distribution will fall between (median / dispersion)
    and (median * dispersion).

    Example (Lessler, 2009):

        influenza_incubation_period_median     = 1.9
        influenza_incubation_period_dispersion = 1.81
        influenza_symptoms_duration_median     = 5.0
        influenza_symptoms_duration_dispersion = 1.5

    These values give an expected incubation period of about 1 to 3.5 days, and symptoms lasting about 3 to 7.5 days.

    --------------------------------------------------------------------------------------------------------------------
    This part is based on my own reading of the following article:
    Lessler (2009) Incubation periods of acute respiratory viral infections -- A systematic review.
    --------------------------------------------------------------------------------------------------------------------

    Influenza A
        incubation period median     = 1.9
        incubation period dispersion = 1.22

    Timer
        1. Sample once per time step and store the current time in state as group attribute.
        2. Sample once and register a timer event with the global simulation timer which will perform the appropriate
           mass distribution.
    '''

    ATTR = 'flu-state'
    NAME = 'SEIR flu model'

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, t=TimeAlways(), susceptibility=1.0, p_start_E=0.05, do_clean=True, name_human=None, memo=None):
        super().__init__('flu', t, susceptibility, p_start_E, do_clean, name_human, memo)

        # self.T['E_median']     = lambda: 1.90 * self.t_mul
        # self.T['E_dispersion'] = 1.23
        # self.T['E_min']        = lambda: self.T['E_median']() / self.T['E_dispersion']
        # self.T['E_max']        = lambda: self.T['E_median']() * self.T['E_dispersion']
        # self.T['I_median']     = lambda: 5.00 * self.t_mul
        # self.T['I_dispersion'] = 1.50
        # self.T['I_min']        = lambda: self.T['I_median']() / self.T['I_dispersion']
        # self.T['I_max']        = lambda: self.T['I_median']() * self.T['I_dispersion']
        #
        # self.T['E_rvs'] = lambda t: min(self.T['E_max'](), max(self.T['E_min'](), lognorm(0.607 * math.log(self.T['E_dispersion']), 0, self.T['E_median']()).pdf(t)))
        # self.T['I_rvs'] = lambda t: min(self.T['I_max'](), max(self.T['I_min'](), lognorm(0.607 * math.log(self.T['I_dispersion']), 0, self.T['I_median']()).pdf(t)))

        self.T['E_median']     = 1.90
        self.T['E_dispersion'] = 1.23
        self.T['E_min']        = self.T['E_median'] / self.T['E_dispersion']
        self.T['E_max']        = self.T['E_median'] * self.T['E_dispersion']

        self.T['I_median']     = 5.00
        self.T['I_dispersion'] = 1.50
        self.T['I_min']        = self.T['I_median'] / self.T['I_dispersion']
        self.T['I_max']        = self.T['I_median'] * self.T['I_dispersion']

        self.T['E_rv'] = lognorm(0.607 * math.log(self.T['E_dispersion']), 0, self.T['E_median'])  # incubation period sampling distribution
        self.T['I_rv'] = lognorm(0.607 * math.log(self.T['I_dispersion']), 0, self.T['I_median'])  # infectious period sampling distribution


# ----------------------------------------------------------------------------------------------------------------------
class GoToRule(Rule):
    '''
    Changes the location of a group from the designated site to the designated site.  Both of the sites are
    specificed by relation name (e.g., 'store').  The rule will only apply to a group that (a) is currently located at
    the "from" relation and has the "to" relation.  If the "from" argument is None, all groups will qualify as long as
    they have the "to" relation.  The values of the relations need to be of type Site.

    Only one 'from' and 'to' location is handled by this rule (i.e., lists of locations are not supported).

    The group's current location is defined by the 'Site.AT' relation name and that's the relation that this rule
    updates.

    Example uses:
        - Compel a portion of agents that are at 'home' go to 'work' or vice versa
    '''

    NAME = 'Goto'

    def __init__(self, t, p, rel_from, rel_to, name_human=None, memo=None):
        super().__init__('goto', t, name_human, memo)

        Err.type(rel_from, 'rel_from', str, True)
        Err.type(rel_to, 'rel_to', str)

        self.p = p
        self.rel_from = rel_from  # if None, the rule will not be conditional on current location
        self.rel_to = rel_to

    def __repr__(self):
        return '{}(name={}, t={}, p={}, rel_from={}, rel_to={})'.format(self.__class__.__name__, self.name, self.t.t, self.p, self.rel_from, self.rel_to)

    def __str__(self):
        return 'Rule  name: {:16}  t: {}  p: {}  rel: {} --> {}'.format(self.name, self.t, self.p, self.rel_from, self.rel_to)

    def apply(self, pop, group, iter, t):
        return [
            GroupSplitSpec(p=self.p, rel_set={ Site.AT: group.get_rel(self.rel_to) }),
            GroupSplitSpec(p=1 - self.p)
        ]

    def is_applicable(self, group, iter, t):
        if not super().is_applicable(group, iter, t):
            return False

        # Moving from the designated location:
        if self.rel_from is not None:
            return (
                group.has_sites([self.rel_to, self.rel_from]) and
                group.get_rel(Site.AT) == group.get_rel(self.rel_from)
            )
        # Moving from any location:
        else:
            return group.has_sites(self.rel_to)


# ----------------------------------------------------------------------------------------------------------------------
class GoToAndBackTimeAtRule(Rule):
    '''
    Compels agents to go to the designated site and back; agents stay at the destination for a specificed amount of
    time.

    Both 'to' and 'back' sites need to be specified.  Moreover, the proportion of agents moving to the destination can
    be speficied as a probability distribution function (PDF) as can the time agents should spend at the destination.
    As a consequence of the latter, this rule tracks the time agents spend at the destination via a group attribute
    't_at_attr'.  Furthermore, agents can be forced to the original location at the end of the rule's time interval.
    For example, students can be made go back home at the time schools close, irrespective of how long they have been
    at school.

    This rule is only applicable to groups with both to and from sites and only those that are currently located at the
    latter.
    '''

    # TODO: Switch from PDF to CDF because it's more natural.

    NAME = 'Goto and back'

    TIME_PDF_TO_DEF   = { 8: 0.5, 12:0.5 }
    TIME_PDF_BACK_DEF = { 1: 0.05, 3: 0.2, 4: 0.25, 5: 0.2, 6: 0.1, 7: 0.1, 8: 0.1 }

    def __init__(self, t=TimeInt(8,16), to='school', back='home', time_pdf_to=None, time_pdf_back=None, t_at_attr='t@', do_force_back=True, name_human=None, memo=None):
        super().__init__('to-and-back', t, name_human, memo)

        self.to   = to
        self.back = back

        self.time_pdf_to   = time_pdf_to   or GoToAndBackTimeAtRule.TIME_PDF_TO_DEF
        self.time_pdf_back = time_pdf_back or GoToAndBackTimeAtRule.TIME_PDF_BACK_DEF

        self.rv_to   = Rule.tp2rv(self.time_pdf_to)
        self.rv_back = Rule.tp2rv(self.time_pdf_back)

        self.t_at_attr = t_at_attr          # name of the attribute that stores time-at the 'to' location
        self.do_force_back = do_force_back  # force all agents to go back at the end of rule time (i.e., 't.t1')?

    def apply(self, pop, group, iter, t):
        if group.has_rel({ Site.AT: group.get_rel(self.back) }) and not group.has_attr(self.t_at_attr):
            return self.apply_to(group, iter, t)

        if group.has_rel({ Site.AT: group.get_rel(self.to) }):
            return self.apply_back(group, iter, t)

    def apply_to(self, group, iter, t):
        p = self.rv_to.cdf(t)

        return [
            GroupSplitSpec(p=p, attr_set={ self.t_at_attr: 0 }, rel_set={ Site.AT: group.get_rel(self.to) }),
            GroupSplitSpec(p=1 - p)
        ]

    def apply_back(self, group, iter, t):
        # TODO: Make this method generalize to arbitrary time steps; currently, 1h increments are asssumed.

        t_at = group.get_attr(self.t_at_attr)
        p = self.rv_back.cdf(t_at)

        if self.do_force_back and t >= self.t.t1:
            p = 1.0

        return [
            GroupSplitSpec(p=p,     attr_set={ self.t_at_attr: (t_at + 1) }, rel_set={ Site.AT: group.get_rel(self.back) }),
            GroupSplitSpec(p=1 - p, attr_set={ self.t_at_attr: (t_at + 1) })
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_sites([self.to, self.back])


# ----------------------------------------------------------------------------------------------------------------------
class ResetRule(Rule):
    NAME = 'Reset'

    def __init__(self, t=TimePoint(5), attr_del=None, attr_set=None, rel_del=None, rel_set=None, name_human=None, memo=None):
        super().__init__('reset', t, name_human, memo)

        self.attr_del = attr_del
        self.attr_set = attr_set
        self.rel_del  = rel_del
        self.rel_set  = rel_set

    def apply(self, pop, group, iter, t):
        return [GroupSplitSpec(p=1.0, attr_del=self.attr_del, attr_set=self.attr_set, rel_del=self.rel_del, rel_set=self.rel_del)]


# ----------------------------------------------------------------------------------------------------------------------
class ResetSchoolDayRule(ResetRule):
    NAME = 'Reset school day'

    def __init__(self, t=TimePoint(5), attr_del=['t-at-school'], attr_set=None, rel_del=None, rel_set=None, name_human=None, memo=None):
        super().__init__(t, attr_del, attr_set, rel_del, rel_set, name_human, memo)
        self.name = 'reset-school-day'

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_sites(['home', 'school'])


# ----------------------------------------------------------------------------------------------------------------------
class ResetWorkDayRule(ResetRule):
    NAME = 'Reset work day'

    def __init__(self, t=TimePoint(5), attr_del=None, attr_set=None, rel_del=None, rel_set=None, name_human=None, memo=None):
        super().__init__(t, attr_del, attr_set, rel_del, rel_set, name_human, memo)
        self.name = 'reset-work-day'

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_sites(['home', 'work'])


# ----------------------------------------------------------------------------------------------------------------------
class RuleAnalyzerTestRule(Rule):
    NAME = 'Rule analyzer test'

    def __init__(self, t=TimeInt(8,20), name_human=None, memo=None):
        super().__init__('progress-flu', t, name_human, memo)

    def an(self, s): return f'b{s}'  # attribute name
    def rn(self, s): return f's{s}'  # relation  name

    def apply(self, group, iter, t):
        if group.has_attr({ 'flu-stage': 's' }):
            pass
        elif group.has_attr({ 'flu-stage': 'i' }):
            pass
        elif group.has_attr({ 'flu-stage': 'r' }):
            pass

    def is_applicable(self, group, iter, t):
        g = group
        c01, c02, c03, c04, c05 = 'cc01', 'cc02', 'cc03', 'cc04', 'cc05'  # attribute names stored in local variables
        s01, s02, s03, s04, s05 = 'ss01', 'ss02', 'ss03', 'ss04', 'ss05'  # ^ (relation)

        return (
            super().is_applicable(group, iter, t) and

            g.has_attr('a01') and g.has_attr([ 'a02', 'a03' ]) and g.has_attr({ 'a04':1, 'a05':2 }) and
            g.has_attr(c01) and g.has_attr([ c02, c03 ]) and g.has_attr({ c04:1, c05:2 }) and
            g.has_attr(self.an('01')) and g.has_attr([ self.an('02'), self.an('03') ]) and g.has_attr({ self.an('04'):1, self.an('05'):2 }) and

            g.has_rel('r01') and g.has_rel([ 'r02', 'r03' ]) and g.has_rel({ 'r04':1, 'r05':2 }) and
            g.has_rel(s01) and g.has_rel([ s02, s03 ]) and g.has_rel({ s04:1, s05:2 }) and
            g.has_rel(self.rn('01')) and g.has_rel([ self.rn('02'), self.rn('03') ]) and g.has_rel({ self.rn('04'):1, self.rn('05'):2 })
        )


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluProgressRule(Rule):
    '''
    Describes how a population transitions between the flu states of susceptible, infected, and recovered.

    ----

    code:
        SimpleFluProgressRule()

    init:
        .

    is-applicable:
        has-attr: flu

    apply:
        if group.attr.flu = 's':
            p_inf = n@_{attr.flu = 'i'} / n@
            move-mass:
                p_inf -> A: flu = 'i'
        if group.attr.flu = 'i':
            move-mass:
                0.2 -> A: flu = 'r'
        if group.attr.flu = 'r':
            move-mass:
                0.1 -> A: flu = 's'

    ----

    if (flu = s)
        p_inf = n@_{attr.flu = i} / n@
        p_inf -> A: flu = 'i'
    if (flu = i) then 0.2 > A: flu = r
    if (flu = r) then 0.1 > A: flu = s
    '''

    ATTRS = { 'flu': [ 's', 'i', 'r' ] }
    NAME = 'Simple flu progression model'

    def __init__(self, t=TimeAlways(), name_human=None, memo=None):
        super().__init__('flu-progress-simple', t, name_human, memo)

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total    population at current location
            n_i = at.get_pop_size(GroupQry(attr={ 'flu': 'i' }))  # infected population at current location

            p_infection = float(n_i) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'i' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Infected:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.8, attr_set={ 'flu': 'i' }),
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' })
            ]

        raise ValueError('Unknown flu state')

    def setup(self, pop, group):
        return [
            GroupSplitSpec(p=0.9, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=0.1, attr_set={ 'flu': 'i' })
        ]


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluProgressMoodRule(Rule):
    '''
    Describes how a population transitions between the states of susceptible, infected, and recovered.  Includes the
    inconsequential 'mood' attribute which may improve exposition of how the PRAM framework works.
    '''

    NAME = 'Simple flu progression model with mood'

    def __init__(self, t=TimeAlways(), name_human=None, memo=None):
        super().__init__('flu-progress-simple', t, name_human, memo)

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total    population at the group's current location
            n_i = at.get_pop_size(GroupQry(attr={ 'flu': 'i' }))  # infected population at the group's current location

            p_infection = float(n_i) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'i', 'mood': 'annoyed' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Infected:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r', 'mood': 'happy'   }),
                GroupSplitSpec(p=0.5, attr_set={ 'flu': 'i', 'mood': 'bored'   }),
                GroupSplitSpec(p=0.3, attr_set={ 'flu': 'i', 'mood': 'annoyed' })
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' })
            ]

        raise ValueError('Unknown flu state')


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluLocationRule(Rule):
    '''
    Describes how student population changes location conditional upon being exposed to the flu.

    ----

    code:
        SimpleFluLocationRule()

    init:
        .

    is-applicable:
        has-attr: flu
        has-rel: home, school

    apply:
        if group.attr.flu = 'i':
            if group.attr.income = 'l':
                move-mass:
                    0.1 -> R: @ = group.rel.home
            if group.attr.income = 'm':
                move-mass:
                    0.6 -> R: @ = group.rel.home
        if group.attr.flu = 'r':
            move-mass:
                0.8 -> R: @ = group.rel.school

    ----

    if (flu = i)
        if (income = l) then 0.1 > R: @ = home
        if (income = m) then 0.6 > R: @ = home
    if (flu = r) then 0.8 > R: @ = school
    '''

    ATTRS = { 'flu': [ 's', 'i', 'r' ], 'income': ['l', 'm'] }
    NAME = 'Simple flu location model'

    def __init__(self, t=TimeAlways(), name_human=None, memo=None):
        super().__init__('flu-location', t, name_human, memo)

    def apply(self, pop, group, iter, t):
        # Infected and low income:
        if group.has_attr({ 'flu': 'i', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Infected and medium income:
        if group.has_attr({ 'flu': 'i', 'income': 'm' }):
            return [
                GroupSplitSpec(p=0.6, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.4)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.8, rel_set={ Site.AT: group.get_rel('school') }),
                GroupSplitSpec(p=0.2)
            ]

        return None
