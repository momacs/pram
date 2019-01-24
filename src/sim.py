# ----------------------------------------------------------------------------------------------------------------------
#
# PRAM - Probabilitistic Relational Agent-based Models
#
# BSD 3-Clause License
#
# Copyright (c) 2018-2019, momacs, University of Pittsburgh
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Contributors
#     Paul R Cohen  (prcohen@pitt.edu)                                                         [2018.10.01 - ...]
#     Tomek D Loboda  (tomek.loboda@gmail.com)                                                 [2018.12.16 - ...]
#
# ----------------------------------------------------------------------------------------------------------------------
#
# TODO
#     Quick
#         - Antagonistic rules (e.g., GoHome and GoToWork), if applied at the same time will always result in order
#           effects, won't they? Think about that.
#         - It is possible to have groups with identical names.  If we wanted to disallow that, the best to do that is
#           in GroupPopulation.add_group().
#         Abastract Group.commit() up a level in the hierarchy, to Entity, which will benefit Site.
#         Abstract hashing to Entity.
#         Should all entities have attributes or only groups, like it is now?
#         - Consider making attribute and relation values objects.  This way they could hold an object and its hash which
#           could allow for more robust copying.
#     Paradigm shift
#         Implement everything on top of a relational database
#     Internal mechanics
#         Group querying
#             Implement a more flexible way to query groups (GroupPopulation.get_groups())
#         Group size
#             - Should we allow fractional group sizes?  At first glance it might make little sense to have a group of
#               500.38 agents, but it's possible that if we don't allow that agents will be "slipping" between groups
#               in a hard-to-control manner.  Perhaps floating-point arithmetic is the way to go given that we can
#               always round but not vice versa.
#               * Current solution: allow floating-point group size
#         Timer-based group states
#             - Add group states that respond to the passage of time and are part of a mechanism.  For example,
#               'flu-asympt-time' could be a count-down attribute that determines for how long the group stays
#               asymptomatic before it turns symptomatic.  At some point, portion by portion, the population of
#               infected but asymptomatic agents would become symptomatic.  In that respect, a presense of a time-based
#               state would imply a time-based rule that needed to be aplied at some point (possibly conditionally).
#         Efficient group management
#             1. GroupPopulation class [DONE]
#             2. Move to frozen hash tables whenever possible (for runtime safety and performance)
#                    frozendict
#                        https://pypi.org/project/frozendict
#                    frozenmap
#                        https://pypi.org/project/frozenmap
#         Account for time
#             Generate unique entity run-time IDs
#             Scale probabilities with time
#                 1. Add process length and use it to scale group-splitting probabilities (i.e., those inside rules)
#                 2. Add cooldown to a rule; the rule does not fire until the cooldown is over; reset cooldown.
#             Option 2 appears as more palatable at the moment, at least to me
#         Account for space and spacial proximity
#             - Are xy- or geo-coordinates necessary?  Perhaps relations can handle this already.  For example, a
#               group of agents already shares the same location so we know they are in proximy of one another.
#               Moreover, if we knew how many agents work or shop at a particular location, we would know that number
#               of agents will form a group based on location.  Perhaps it would be enough for a Site (e.g., a store)
#               and a Resource (e.g., a public bus) to have a capacity property.
#         Convert between list of agents and groups
#             This should be a natural way of enabling FRED-PRAM interoperability
#             In a more general sense, this would enable parallel runs comparisons between ABMs, PRAMs, etc.
#         Representational expressiveness
#             - Move from discrete models (i.e., those implying states) to continuous or hybrid models (i.e., those
#               implying equations)
#         Type safety
#             Use static type checking
#                 https://medium.com/@ageitgey/learn-how-to-use-static-type-checking-in-python-3-6-in-10-minutes-12c86d72677b
#             Add type checks
#         Optimization (when the time comes, but not later)
#             Move to PyPy
#             __slots__
#                 https://stackoverflow.com/questions/472000/usage-of-slots
#             Memoize GroupQry queries
#                 They should be reset (or handled inteligently) after mass redistribution
#     Simulation contsruction
#         Define simulation definition language
#             - Provide probabilities as a transition matrix.  Perhaps the entire simulation could be specified as a
#               Bayesian network which would naturally imply a conditional probability distributions (or densities).
#         Define simulation based on a relational database
#     Data collection
#         - Add monitors which would take various measurements around a simulation, keep history of those measurements,
#           and persist that history if desired
#     Visualization and UI
#         Visualize simulations
#         Make the visualization controlable (ideally MVC)
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Milestones: Environment
#     Development started (Python 3.6.1; MacOS 10.12.6)                                        [2018.12.16]
#     GitHub repository created                                                                [2019.01.18]
#     Ported to FreeBSD (Python 3._._; 12.0R)                                                  []
#     Ported to Ubuntu (Python 3._._; 18.04 LTS)                                               []
#     Moved to PyPy                                                                            []
#     Containerized (Docker)                                                                   []
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Milestones: Code
#     Simulation
#         c Simulation class                                                                   [2018.12.16 - ...]
#             f Read and output time in time format (e.g., 'hh:mm')                            []
#             f Convert between list of agents and groups                                      []
#         Entities
#             c Agent                                                                          [2018.12.16 - 2019.01.05]
#             c Entity                                                                         [2018.12.16 - ...]
#             c Group                                                                          [2019.12.16 - ...]
#                 f Splitting                                                                  [2019.01.04 - 2019.01.16]
#             c GroupSplitSpec                                                                 [2019.01.06 - 2019.01.21]
#             c GroupQry                                                                       [2019.01.12 - 2019.01.21]
#             c Site                                                                           [2018.12.16 - ...]
#                 f Canonical functionality                                                    [2010.01.19 - 2019.01.21]
#             c Resource                                                                       [2010.01.18]
#         Rules
#             c Time                                                                           [2019.01.06]
#             c TimePoint                                                                      [2019.01.22]
#             c TimeInterval                                                                   [2019.01.06]
#             c Rule                                                                           [2019.01.03]
#             c GotoRule                                                                       [2019.01.21]
#             c AttendSchoolRule                                                               [2019.01.22 - 2019.01.23]
#             c ResetDayRule                                                                   [2019.01.23]
#             c ProgressFluRule                                                                [2019.01.03 - 2019.01.16]
#             f Setup                                                                          [2019.01.23]
#         Population
#             c Population                                                                     [2019.01.07]
#             c AgentPopulation                                                                [2019.01.07]
#             c GroupPopulation                                                                [2019.01.07 - ...]
#                 f Selecting groups                                                           [2019.01.07 - 2019.01.21]
#                 f Rule application                                                           [2019.01.04 - 2019.01.17]
#                 f Mass redistribution                                                        [2019.01.04 - 2019.01.17]
#
#     Data collection
#         c Probe                                                                              [2019.01.18 - 2019.01.19]
#         c GroupSizeProbe                                                                     [2019.01.18 - 2019.01.19]
#
#     Logging
#         c Log                                                                                [2019.01.06]
#         c LogMan                                                                             [2019.01.06]
#
#     Communication and control
#         c TCPServer                                                                          []
#         c TCPClient                                                                          []
#         c HTTPServer                                                                         []
#         s Protocol                                                                           []
#
#     Tests
#         c GroupTestCase                                                                      [2019.01.11]
#         c SimulationTestCase                                                                 [2019.01.19]
#         c SiteTestCase                                                                       [2019.01.21]
#
#     Visualization and UI
#         l WebUI                                                                              []
#
#     Utilities
#         c Data                                                                               [2019.01.06]
#         c Err                                                                                [2019.01.21]
#         c FS                                                                                 [2019.01.06]
#         c Hash                                                                               [2019.01.06 - ...]
#         c MPCounter                                                                          [2019.01.06]
#         c Size                                                                               [2019.01.06]
#         c Str                                                                                [2019.01.06]
#         c Tee                                                                                [2019.01.06]
#         c Time                                                                               [2019.01.06]
#
#     Optimization
#         String interning (sys.intern; not needed in Python 3)                                [2019.01.06 - 2019.01.17]
#         __slots__                                                                            [2019.01.17]
#
#     Legend
#         c Class
#         f Functionality
#         l Layer
#         s Specification
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Philosophical issues
#     Dynamical system
#         Dynamical system use differential equations as a state-and-space model.  It would be interesting to
#         investigate if a similar approach would be viable here.  Perhaps an end-to-end differentible system is
#         possible.  I suspect, however, it wouldn't, mostly on the grounds of computational intractability for
#         larger simulations.
#     State-space models
#         In a more general sense, how does PRAM correpond to state-space models?
#     Logistic map
#         My early and simple simulations resulted with the population distribution reaching what appears to be some
#         sort of a stationary distribution.  That convergence was rather fast (around 20 time steps) and while I don't
#         know what's driving it, I wonder if it's in any way related to the logistic map:
#
#             x_{n+1} = r x_n (1-x_n)
#
#         If there is a connection, perhaps it can be described with polynomial mapping of degree higher than two.  I'm
#         not sure how to visualize those higher dimensions.
#
#         Having thought of this a tiny bit more, I started to question my initial intuition because the logistic
#         function models population size change (i.e., growth or decline) and not distribution.  Perhaps it would
#         still apply to subpopulation... not sure.
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Simulation ideas
#     Smoking effects on lifespan
#         Groups will have an attribute 'lifespan'
#         Rule will make a portion of the group to smoke
#         Smoking will negatively impact lifespan
#         Being in the proximity of other smokers will increase likelihood of smoking
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Solved or being solved
#     ...
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Resources: Development
#     Simulations
#         SimPy
#             Discrete event simulation for Python
#             https://simpy.readthedocs.io
#
#     Agent-based modeling
#         Mesa
#             https://mesa.readthedocs.io/en/latest/index.html
#
#     Data structures
#         collections
#             https://docs.python.org/3.6/library/collections.html
#         Time Complexity
#             https://wiki.python.org/moin/TimeComplexity
#         attrs
#             More memory-efficient and more feature-rich than property classes or the 'namedtuple' data type
#             https://www.attrs.org
#
#     Multithreading
#         Stackless
#             https://github.com/stackless-dev/stackless/wiki
#             http://doc.pypy.org/en/latest/stackless.html
#
#     Multiprocessing
#         multiprocessing
#             Process-based parallelism
#             https://docs.python.org/3.6/library/multiprocessing.html
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Resources: Statistics
#     Sampling
#         How to get embarrassingly fast random subset sampling with Python
#             https://medium.freecodecamp.org/how-to-get-embarrassingly-fast-random-subset-sampling-with-python-da9b27d494d9
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Resources: Other
#     ...
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Deployment: MacOS  (Python 3.6.1)
#     dir=/Volumes/D/pitt/sci/pram
#
#     if [ $(which brew | wc -l | bc) == 0 ]
#     then/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
#     else brew update
#     fi
#
#     brew install git python3
#
#     python -m venv $dir && cd $dir
#     source ./bin/activate
#
#     pip install --upgrade pip && easy_install -U pip
#
#     python -m pip install numpy attrs
#
#     mkdir -p src && cd src
#     git clone https://github/...
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Deployment: FreeBSD  (12.0R, Python 3._._)
#         ...
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Deployment: Ubuntu  (18.04 LTS, Python 3._._)
#         ...
#
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

from data   import GroupSizeProbe
from entity import Agent, AttrFluStatus, AttrSex, Group, GroupQry, Home, Site
from pop    import GroupPopulation


class Simulation(object):
    '''
    A PRAM simulation.

    The discrete time is unitless; it is the simulation context that defines the appropriate granularity.  For
    instance, million years (myr) might be appropriate for geological processes while Plank time might be appropriate
    for modeling quantum phenomena.
    '''

    __slots__ = ('t', 't_step_size', 't_step_cnt', 'rand_seed', 'pop', 'rules', 'probes')

    DEBUG_LVL = 0  # 0=none, 1=normal, 2=full

    def __init__(self, t=4, t_step_size=1, t_step_cnt=24, rand_seed=None):
        '''
        One interpretation of 't=4' is that the simulation starts at 4am.  Similarily, 't_step_size=1' could mean that
        the simulation time increments in one-hour intervals.
        '''

        self.t = t
        self.t_step_size = t_step_size
        self.t_step_cnt = t_step_cnt

        if rand_seed is not None:
            self.rand_seed = rand_seed
            np.random.seed(self.rand_seed)

        self.pop = GroupPopulation()
        self.rules = []
        self.probes = []

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.t, self.t_step_size, self.t_step_cnt, self.rand_seed)

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1: print(msg)

    def add_probe(self, probe):
        self.probes.append(probe)
        probe.set_pop(self.pop)
        return self

    def add_rule(self, rule):
        self.rules.append(rule)
        return self

    def add_site(self, site):
        self.pop.add_site(site)
        return self

    def add_sites(self, sites):
        self.pop.add_sites(sites)
        return self

    def clear_probes(self):
        self.probes.clear()
        return self

    def clear_rules(self):
        self.rules.clear()
        return self

    def commit_group(self, group):
        self.pop.add_group(group)
        return self

    def create_group(self, n, attr=None, rel=None):
        self.pop.create_group(n, attr, rel)
        return self

    def new_group(self, name=None, n=0.0):
        return Group(name or self.pop.get_next_group_name(), n, callee=self)

    def rem_probe(self, probe):
        self.probes.discard(probe)
        return self

    def rem_rule(self, rule):
        self.rules.discard(rule)
        return self

    def run(self, t_step_cnt=None):
        self.pop.apply_rules(self.rules, self.t, True)

        if t_step_cnt is not None:
            self.t_step_cnt = t_step_cnt

        for i in range(self.t_step_cnt):
            self._debug('t: {}'.format(self.t))

            self.pop.apply_rules(self.rules, self.t)

            for p in self.probes:
                p.run(self.t)

            # Population size by location:
            # print('{:2}  '.format(self.t), end='')
            # for s in self.pop.sites.values():
            #     print('{}: {:>7}  '.format(s.name, round(s.get_pop_size(), 1)), end='')
            # print('')

            # Groups by location:
            # print('{:2}  '.format(self.t), end='')
            # for s in self.pop.sites.values():
            #     print('{}: ( '.format(s.name), end='')
            #     for g in s.get_groups_here():
            #         print('{} '.format(g.name), end='')
            #     print(')  ', end='')
            # print('')

            self.t = (self.t + self.t_step_size) % 24

        return self

    def summary(self, part=(True, False, False, False, False), end_line_cnt=(0,0)):
        ''' Prints the simulation summary. '''

        print('\n' * end_line_cnt[0], end='')

        if part[0]:
            print('Simulation')
            print('    Random seed: {}'.format(self.rand_seed))
            print('    Timer')
            print('        Start      : {}'.format(self.t))
            print('        Step size  : {}'.format(self.t_step_size))
            print('        Iterations : {}'.format(self.t_step_cnt))
            print('        Sequence   : {}'.format([self.t + self.t_step_size * i for i in range(5)]))
            print('    Population')
            print('        Size        : {}'.format(round(self.pop.get_size(), 1)))
            print('        Groups      : {}'.format(self.pop.get_group_cnt()))
            print('        Groups (ne) : {}'.format(self.pop.get_group_cnt(True)))
            print('        Sites       : {}'.format(self.pop.get_site_cnt()))
            print('        Rules       : {}'.format(len(self.rules)))
            print('        Probes      : {}'.format(len(self.probes)))

        if part[1]:
            if len(self.pop.groups) > 0: print('    Groups ({})\n'.format(len(self.pop.groups)) + '\n'.join(['        {}'.format(g) for g in self.pop.groups.values()]))
        if part[2]:
            if len(self.pop.sites)  > 0: print('    Sites ({})\n' .format(len(self.pop.sites))  + '\n'.join(['        {}'.format(s) for s in self.pop.sites.values()]))
        if part[3]:
            if len(self.rules)      > 0: print('    Rules ({})\n' .format(len(self.rules))      + '\n'.join(['        {}'.format(r) for r in self.rules]))
        if part[4]:
            if len(self.probes)     > 0: print('    Probes ({})\n'.format(len(self.probes))     + '\n'.join(['        {}'.format(p) for p in self.probes]))

        print('\n' * end_line_cnt[1], end='')

        return self


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    rand_seed = 1928

    # ------------------------------------------------------------------------------------------------------------------
    # (1) Simulations testing the basic operations on groups and rules:

    # from rule import TimeInt, GotoRule, ProgressFluRule
    #
    # sites = { 'home': Site('home'), 'work': Site('work-a') }
    #
    # probe_grp_size_flu = GroupSizeProbe('flu', [GroupQry(attr={ 'flu-status': fs }) for fs in AttrFluStatus])

    # (1.1) A single-group, single-rule (1g.1r) simulation:
    # s = Simulation(6,1,16, rand_seed=rand_seed)
    # s.create_group(1000, { 'flu-status': AttrFluStatus.no }, {})
    # s.add_rule(ProgressFluRule())
    # s.add_probe(probe_grp_size_flu)
    # s.run()

    # (1.2) A single-group, two-rule (1g.2r) simulation:
    # s = Simulation(6,1,16, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.add_rule(RuleProgressFlu())
    # s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
    # s.add_probe(probe_grp_size_flu)
    # s.run()

    # (1.3) As above (1g.2r), but with reversed rule order (which should not, and does not, change the results):
    # s = Simulation(6,1,16, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
    # s.add_rule(ProgressFluRule())
    # s.add_probe(probe_grp_size_flu)
    # s.run()

    # (1.4) A two-group, two-rule (2g.2r) simulation (the groups don't interact via rules):
    # s = Simulation(6,1,16, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, attr={ 'flu-status': AttrFluStatus.no })
    # s.create_group(2000, rel={ Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.add_rule(ProgressFluRule())
    # s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
    # s.add_probe(probe_grp_size_flu)
    # s.run()

    # (1.5) A two-group, two-rule (2g.2r) simulation (the groups do interact via one rule):
    # s = Simulation(6,1,16, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.create_group(2000, { 'flu-status': AttrFluStatus.no })
    # s.add_rule(ProgressFluRule())
    # s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
    # s.add_probe(probe_grp_size_flu)
    # s.run()

    # (1.6) A two-group, three-rule (2g.3r) simulation (same results, as expected):
    # s = Simulation(6,1,16, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.create_group(2000, { 'flu-status': AttrFluStatus.no }, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash() })
    # s.add_rule(ProgressFluRule())
    # s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'home', 'work'))
    # s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
    # s.add_probe(probe_grp_size_flu)
    # s.run()


    # ------------------------------------------------------------------------------------------------------------------
    # (2) Simulations testing rule interactions:

    # from rule import TimeInt, GotoRule, ProgressFluRule
    #
    # sites = { 'home': Site('home'), 'work': Site('work-a') }
    #
    # probe_grp_size_flu = GroupSizeProbe('flu', [GroupQry(attr={ 'flu-status': fs }) for fs in AttrFluStatus])
    # probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])

    # (2.1) Antagonistic rules overlapping entirely in time (i.e., goto-home and goto-work; converges to a stable distribution):
    # s = Simulation(9,1,14, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, {}, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'home', 'work'))
    # s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
    # s.add_probe(probe_grp_size_loc)
    # s.run()

    # (2.2) Antagonistic rules overlapping mostly in time (i.e., goto-home and hoto-work; second rule "wins"):
    # s = Simulation(9,1,14, rand_seed=rand_seed)
    # s.add_sites(sites.values())
    # s.create_group(1000, {}, { Site.AT: sites['home'].get_hash(), 'home': sites['home'].get_hash(), 'work': sites['work'].get_hash() })
    # s.add_rule(GotoRule(TimeInt(10,16), 0.4, 'home', 'work'))
    # s.add_rule(GotoRule(TimeInt(10,22), 0.4, 'work', 'home'))
    # s.add_probe(probe_grp_size_loc)
    # s.run()

    # ------------------------------------------------------------------------------------------------------------------
    # (3) A multi-group, multi-rule, multi-site simulations:

    # from rule import TimeInt, GotoRule, ProgressFluRule
    #
    # sites = {
    #     'home'    : Site('home'),
    #     'work-a'  : Site('work-a'),
    #     'work-b'  : Site('work-b'),
    #     'work-c'  : Site('work-c'),
    #     'store-a' : Site('store-a'),
    #     'store-b' : Site('store-b')
    # }
    #
    # probe_grp_size_flu = GroupSizeProbe('flu', [GroupQry(attr={ 'flu-status': fs }) for fs in AttrFluStatus])
    # probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])
    #
    # (Simulation(6,1,24, rand_seed=rand_seed).
    #     add_sites(sites.values()).
    #     new_group('g0', 1000).
    #         set_attr('flu-status', AttrFluStatus.no).
    #         set_rel(Site.AT, sites['home'].get_hash()).
    #         set_rel('home',  sites['home'].get_hash()).
    #         set_rel('work',  sites['work-a'].get_hash()).
    #         set_rel('store', sites['store-a'].get_hash()).
    #         commit().
    #     new_group('g1', 1000).
    #         set_attr('flu-status', AttrFluStatus.no).
    #         set_rel(Site.AT, sites['home'].get_hash()).
    #         set_rel('home',  sites['home'].get_hash()).
    #         set_rel('work',  sites['work-b'].get_hash()).
    #         set_rel('store', sites['store-b'].get_hash()).
    #         commit().
    #     new_group('g2', 100).
    #         set_attr('flu-status', AttrFluStatus.no).
    #         set_rel(Site.AT, sites['home'].get_hash()).
    #         set_rel('home',  sites['home'].get_hash()).
    #         set_rel('work',  sites['work-c'].get_hash()).
    #         commit().
    #     # add_rule(RuleProgressFlu()).
    #     add_rule(GotoRule(TimeInt( 8,12), 0.4, 'home',  'work',  'some agents leave home to go to work')).
    #     add_rule(GotoRule(TimeInt(16,20), 0.4, 'work',  'home',  'some agents return home from work')).
    #     add_rule(GotoRule(TimeInt(16,21), 0.2, 'home',  'store', 'some agents go to a store after getting back home')).
    #     add_rule(GotoRule(TimeInt(17,23), 0.3, 'store', 'home',  'some shopping agents return home from a store')).
    #     add_rule(GotoRule(TimeInt(24,24), 1.0, 'store', 'home',  'all shopping agents return home after stores close')).
    #     add_rule(GotoRule(TimeInt( 2, 2), 1.0, None,    'home',  'all still-working agents return home')).
    #     # add_probe(probe_grp_size_flu).
    #     add_probe(probe_grp_size_loc).
    #     summary(True, (0,1)).
    #     run().
    #     run(4))

    # print(probe_grp_size_flu.get_msg())
    # print(probe_grp_size_loc.get_msg())

    # ------------------------------------------------------------------------------------------------------------------
    # (4) Simulation testing a group-state-aware school attending rule:

    from rule import TimeInt, TimePoint, AttendSchoolRule, ResetDayRule

    sites = {
        'home'     : Site('home'),
        'school-a' : Site('school-a'),
        'school-b' : Site('school-b')
    }

    probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])

    (Simulation(7,1,10, rand_seed=rand_seed).
        add_sites(sites.values()).
        new_group('0', 500).
            set_attr('is-student', True).
            set_rel(Site.AT,  sites['home'].get_hash()).
            set_rel('home',   sites['home'].get_hash()).
            set_rel('school', sites['school-a'].get_hash()).
            commit().
        new_group('1', 500).
            set_attr('is-student', True).
            set_rel(Site.AT,  sites['home'].get_hash()).
            set_rel('home',   sites['home'].get_hash()).
            set_rel('school', sites['school-b'].get_hash()).
            commit().
        add_rule(ResetDayRule(TimePoint(7))).
        add_rule(AttendSchoolRule()).
        add_probe(probe_grp_size_loc).
        # summary((True, True, True, True, True), (0,1)).
        # run(1).summary((False, True, False, False, False), (1,1)).
        # run(1).summary((False, True, False, False, False), (1,1)).
        # run(1).summary((False, True, False, False, False), (1,1)).
        # run(1).summary((False, True, False, False, False), (1,1))
        # run().summary((False, True, False, False, False), (1,1))
        run()
    )

    # ------------------------------------------------------------------------------------------------------------------
    # (5) Simulation testing a location-aware flu progression rule:

    # from rule import TimeInt, ProgressFlu02
    #
    # sites = {
    #     'home'     : Site('home'),
    #     'school-a' : Site('school-a'),
    #     'school-b' : Site('school-b')
    # }
    #
    # probe_grp_size_flu = GroupSizeProbe('flu', [GroupQry(attr={ 'flu-status': fs }) for fs in AttrFluStatus])
    # probe_grp_size_loc = GroupSizeProbe('loc', [GroupQry(rel={ Site.AT: s.get_hash() }) for s in sites.values()])
    #
    # (Simulation(6,1,24, rand_seed=rand_seed).
    #     add_sites(sites.values()).
    #     new_group('A', 500).
    #         set_attr('flu-status', AttrFluStatus.no).
    #         set_rel(Site.AT,  sites['home'].get_hash()).
    #         set_rel('home',   sites['home'].get_hash()).
    #         set_rel('school', sites['school-a'].get_hash()).
    #         commit().
    #     new_group('B', 500).
    #         set_attr('flu-status', AttrFluStatus.no).
    #         set_rel(Site.AT,  sites['home'].get_hash()).
    #         set_rel('home',   sites['home'].get_hash()).
    #         set_rel('school', sites['school-b'].get_hash()).
    #         commit().
    #     add_rule(GotoRule(TimeInt( 8, 12), 1.0, 'school', 'home', 'all still-studying agents return home')).
    #     add_rule(ProgressFlu02Rule(TimeInt(8, 20))).
    #     add_rule(GotoRule(TimeInt( 20, 20), 1.0, 'school', 'home', 'all still-studying agents return home')).
    #     add_probe(probe_grp_size_flu).
    #     summary(False, (0,1)).
    #     run())
