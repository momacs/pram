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
#     Paul R Cohen  (prcohen@pitt.edu; the idea of PRAMs and the original implementation)      [2018.10.01 - ...]
#     Tomek D Loboda  (tomek.loboda@gmail.com; the present design and implementation)          [2018.12.16 - ...]
#
# ----------------------------------------------------------------------------------------------------------------------
#
# TODO
#     Quick
#         - Abstract the single-group group splitting mechanism as a dedicated method.  Currently, that is used in
#           ResetDayRule.apply() and AttendSchoolRule.setup() methods.
#         Allow executing rule sets in the order provided
#             E.g., some rules need to be run after all others have been applied
#             This may not be what we want because the idea of PRAMs is to avoid order effects.
#         Prevent creating empty groups in Group.split().
#         - Antagonistic rules (e.g., GoHome and GoToWork), if applied at the same time will always result in order
#           effects, won't they? Think about that.
#         - It is possible to have groups with identical names.  If we wanted to disallow that, the best place to do
#           that in is GroupPopulation.add_group().
#         Abastract Group.commit() up a level in the hierarchy, to Entity, which will benefit Site.
#         Abstract hashing up to Entity.
#         - Consider making attribute and relation values objects.  This way they could hold an object and its hash
#           which could allow for more robust copying.
#     Paradigm shift
#         Implement everything on top of a relational database
#     Internal mechanics
#         Population mass
#             Allow the population mass to grow or shrink, most likely only via rules.
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
#             Memoize GroupQry queries
#                 They should be reset (or handled inteligently) after mass redistribution
#             Persist probe results on a batch basis (e.g., via 'executemany()')
#             Remove groups unused in certain number of simulation steps to account for high group-dynamic scenarios
#     Simulation contsruction
#         Define simulation definition language
#             - Provide probabilities as a transition matrix.  Perhaps the entire simulation could be specified as a
#               Bayesian network which would naturally imply a conditional probability distributions (or densities).
#         Define simulation based on a relational database
#         Hand-drawn input
#             Implement a UI that turns hand-drawn graphs into PRAM simulations
#             Possible approaches
#                 Image moments
#             Quick literature search
#                 https://www.researchgate.net/publication/302980429_Hand_Drawn_Optical_Circuit_Recognition
#                 http://homepages.inf.ed.ac.uk/thospeda/papers/yu2016sketchanet.pdf
#                 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5578632/
#                 http://cs231n.stanford.edu/reports/2017/pdfs/420.pdf
#                 DeepSketch2
#                     https://www.semanticscholar.org/paper/DeepSketch-2%3A-Deep-convolutional-neural-networks-Dupont-Seddati/feac7ae2ce66e6dcc8e8c64223dc5624edea8d08
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
#     Published to PyPI                                                                        []
#     Containerized (Docker)                                                                   []
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Milestones: Code
#     Simulation
#         c Simulation class                                                                   [2018.12.16 - 2019.02.11]
#             f Read and output time in time format (e.g., 'hh:mm')                            []
#             f Convert between list of agents and groups                                      []
#         Entities
#             c Agent                                                                          [2018.12.16 - 2019.01.05]
#             c Entity                                                                         [2018.12.16 - ...]
#             c Group                                                                          [2019.12.16 - ...]
#                 f Splitting                                                                  [2019.01.04 - 2019.02.11]
#             c GroupSplitSpec                                                                 [2019.01.06 - 2019.01.21]
#             c GroupQry                                                                       [2019.01.12 - 2019.01.21]
#             c Site                                                                           [2018.12.16 - 2019.02.11]
#                 f Canonical functionality                                                    [2010.01.19 - 2019.01.21]
#             c Resource                                                                       [2010.01.18 - 2019.01.08]
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
#         c DB                                                                                 [2019.02.11]
#         c Err                                                                                [2019.01.21]
#         c FS                                                                                 [2019.01.06]
#         c Hash                                                                               [2019.01.06 - ...]
#         c MPCounter                                                                          [2019.01.06]
#         c Size                                                                               [2019.01.06 - 2019.02.11]
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
#         [2019.01]
#         My early and simple simulations resulted with the population distribution reaching what appears to be some
#         sort of a stationary distribution.  That convergence was rather fast (around 20 time steps) and while I don't
#         know what's driving it, I wonder if it's in any way related to the logistic map:
#
#             x_{n+1} = r x_n (1-x_n)
#
#         If there is a connection, perhaps it can be described with polynomial mapping of degree higher than two.  I'm
#         not sure how to visualize those higher dimensions.
#
#         [2019.01]
#         Having thought of this a tiny bit more, I started to question my initial intuition because the logistic
#         function models population size change (i.e., growth or decline) and not distribution.  Perhaps it would
#         still apply to subpopulation... not sure.
#
#         [2019.02.07]
#         I am now almost certain that I was wrong all along.  I think that the observed convergence is that of a
#         time-homogenous Markov chain with a finite state space tending to its stationary distribution \pi:
#
#             \pi = \pi P,
#
#         where $P$ is the chain's transition probability matrix.  While the flu progression simulation is equivalent
#         to such a Markov chain, more complicated simulations (e.g., the flu transmission model) won't reduce to
#         stationary chains.
#     Evaluation of PRAM
#         Historical data
#             - Use the initial state given by historical data to seed a PRAM simulation.  Compare how short of the
#              rest of the data the simulation is at specific milestone points.
#             - Additionally, historical data could be used to infer the rules.
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
# Resources: Publishing
#     Bayesian Nets plate notation graphs
#         LaTeX
#             https://github.com/jluttine/tikz-bayesnet
#         Python
#             http://daft-pgm.org
#     Graphs
#         http://www.graphviz.org
#         http://latexdraw.sourceforge.net
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Resources: Other
#     ...
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Deployment: MacOS  (Python 3.6.1)
#     dir=...
#
#     if [ $(which brew | wc -l | bc) == 0 ]
#     then/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
#     else brew update
#     fi
#
#     brew install git python3
#
#     [ -d $dir ] && echo "Directory already exists" && exit 1
#     python -m venv $dir && cd $dir
#     source ./bin/activate
#
#     pip install --upgrade pip && easy_install -U pip
#
#     python -m pip install numpy attrs
#
#     git clone https://github.com/momacs/pram
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

from .data   import GroupSizeProbe
from .entity import Agent, Group, GroupQry, Home, Site
from .pop    import GroupPopulation


class Simulation(object):
    '''
    A PRAM simulation.

    The discrete time is unitless; it is the simulation context that defines the appropriate granularity.  For
    instance, million years (myr) might be appropriate for geological processes while Plank time might be appropriate
    for modeling quantum phenomena.
    '''

    __slots__ = ('t', 't_step_size', 't_step_cnt', 'rand_seed', 'pop', 'rules', 'probes', 'is_setup_done', 'do_disp_t')

    DEBUG_LVL = 0  # 0=none, 1=normal, 2=full

    def __init__(self, t=0, t_step_size=1, t_step_cnt=0, do_disp_t=False, rand_seed=None):
        '''
        One interpretation of 't=4' is that the simulation starts at 4am.  Similarily, 't_step_size=1' could mean that
        the simulation time increments in one-hour intervals.
        '''

        self.t = t
        self.t_step_size = t_step_size
        self.t_step_cnt = t_step_cnt

        self.rand_seed = rand_seed
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)

        self.pop = GroupPopulation()
        self.rules = []
        self.probes = []

        self.is_setup_done = False
            # ensures simulation setup is performed only once while enabling multiple incremental simulation runs of
            # arbitrary length thus promoting user-interactivity

        self.do_disp_t = do_disp_t

    def __repr__(self):
        return f'{self.__class__.__name__}({self.t}, {self.t_step_size}, {self.t_step_cnt}, {self.rand_seed})'

    def _debug(self, msg):
        if self.DEBUG_LVL >= 1:
            print(msg)

    def add_group(self, group):
        self.pop.add_site(group)
        return self

    def add_groups(self, groups):
        self.pop.add_groups(groups)
        return self

    def add_probe(self, probe):
        self.probes.append(probe)
        probe.set_pop(self.pop)
        return self

    def add_probes(self, probes):
        for p in probes:
            self.add_probe(p)
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

    def run(self, t_step_cnt=0):
        # Do setup:
        if not self.is_setup_done:
            self.pop.apply_rules(self.rules, self.t, True)
            self.is_setup_done = True

        # Run the simulation
        if t_step_cnt > 0:
            self.t_step_cnt = t_step_cnt

        for i in range(self.t_step_cnt):
            self._debug(f't: {self.t}')

            if self.do_disp_t: print(f't:{self.t}')

            self.pop.apply_rules(self.rules, i, self.t)

            for p in self.probes:
                p.run(i, self.t)

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
            print( 'Simulation')
            print(f'    Random seed: {self.rand_seed}')
            print( '    Timer')
            print(f'        Start      : {self.t}')
            print(f'        Step size  : {self.t_step_size}')
            print(f'        Iterations : {self.t_step_cnt}')
            print(f'        Sequence   : {[self.t + self.t_step_size * i for i in range(5)]}')
            print( '    Population')
            print(f'        Size        : {round(self.pop.get_size(), 1)}')
            print(f'        Groups      : {self.pop.get_group_cnt()}')
            print(f'        Groups (ne) : {self.pop.get_group_cnt(True)}')
            print(f'        Sites       : {self.pop.get_site_cnt()}')
            print(f'        Rules       : {len(self.rules)}')
            print(f'        Probes      : {len(self.probes)}')

        if part[1]:
            if len(self.pop.groups) > 0: print(f'    Groups ({len(self.pop.groups)})\n' + '\n'.join(['        {}'.format(g) for g in self.pop.groups.values()]))
        if part[2]:
            if len(self.pop.sites)  > 0: print(f'    Sites ({len(self.pop.sites)})\n'   + '\n'.join(['        {}'.format(s) for s in self.pop.sites.values()]))
        if part[3]:
            if len(self.rules)      > 0: print(f'    Rules ({len(self.rules)})\n'       + '\n'.join(['        {}'.format(r) for r in self.rules]))
        if part[4]:
            if len(self.probes)     > 0: print(f'    Probes ({len(self.probes)})\n'     + '\n'.join(['        {}'.format(p) for p in self.probes]))

        print('\n' * end_line_cnt[1], end='')

        return self
