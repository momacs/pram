import math

from abc         import abstractmethod, ABC
from attr        import attrs, attrib
from dotmap      import DotMap
from enum        import IntEnum
from scipy.stats import lognorm, rv_discrete

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
class MCRule(Rule):
    '''
    Time-homogenous Markov chain with finite state space.

    The following example transition model for the variabled named X:

                   x_1^t   x_2^t
        x_1^{t+1}    0.1     0.3
        x_2^{t+1}    0.9     0.7

    Should be specified as:

        { 'x1': [0.1, 0.9], 'x2': [0.3, 0.7] }
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
class SEIRRule(Rule, ABC):
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
class SEIRFluRule(SEIRRule):
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
    Describes how a population transitions between the flu states of susceptible, exposed, and recovered.
    '''

    ATTRS = { 'flu': [ 's', 'e', 'r' ] }
    NAME = 'Simple flu progression model'

    def __init__(self, t=TimeAlways(), name_human=None, memo=None):
        super().__init__('flu-progress-simple', t, name_human, memo)

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total   population at current location
            n_e = at.get_pop_size(GroupQry(attr={ 'flu': 'e' }))  # exposed population at current location

            p_infection = float(n_e) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Exposed:
        if group.has_attr({ 'flu': 'i' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.8, attr_set={ 'flu': 'e' }),
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' })
            ]

        raise ValueError('Unknown flu state')

    def setup(self, pop, group):
        return [
            GroupSplitSpec(p=0.9, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=0.1, attr_set={ 'flu': 'e' })
        ]


# ----------------------------------------------------------------------------------------------------------------------
class SimpleFluProgressMoodRule(Rule):
    '''
    Describes how a population transitions between the states of susceptible, exposed, and recovered.  Includes the
    inconsequential 'mood' attribute which improves exposition of how the PRAM framework works.

    Introduced in Cohen (2019).
    '''

    NAME = 'Simple flu progression model with mood'

    def __init__(self, t=TimeAlways(), name_human=None, memo=None):
        super().__init__('flu-progress-simple', t, name_human, memo)

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total   population at the group's current location
            n_e = at.get_pop_size(GroupQry(attr={ 'flu': 'e' }))  # exposed population at the group's current location

            p_infection = float(n_e) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'e', 'mood': 'annoyed' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Exposed:
        if group.has_attr({ 'flu': 'e' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r', 'mood': 'happy'   }),
                GroupSplitSpec(p=0.5, attr_set={ 'flu': 'e', 'mood': 'bored'   }),
                GroupSplitSpec(p=0.3, attr_set={ 'flu': 'e', 'mood': 'annoyed' })
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
    '''

    ATTRS = { 'flu': [ 's', 'e', 'r' ], 'income': ['l', 'm'] }
    NAME = 'Simple flu location model'

    def __init__(self, t=TimeAlways(), name_human=None, memo=None):
        super().__init__('flu-location', t, name_human, memo)

    def apply(self, pop, group, iter, t):
        # Exposed and low income:
        if group.has_attr({ 'flu': 'e', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Exposed and medium income:
        if group.has_attr({ 'flu': 'e', 'income': 'm' }):
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
