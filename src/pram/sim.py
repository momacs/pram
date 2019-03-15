#
# ----------------------------------------------------------------------------------------------------------------------
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
#         - Store school info (e.g., size) in the probe DB and associate with the probes
#         - Allow simulation to output warning to a file or a database
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

import ast
import gc
import gzip
import inspect
import numpy as np
import os
import pickle

from collections import namedtuple, Counter
from dotmap import DotMap

from .data   import GroupSizeProbe
from .entity import Agent, Group, GroupQry, Site
from .pop    import GroupPopulation


# ----------------------------------------------------------------------------------------------------------------------
class SimulationConstructionError(Exception): pass
class SimulationConstructionWarning(Warning): pass


# ----------------------------------------------------------------------------------------------------------------------
class RuleAnalyzer(object):
    '''
    Analyzes the syntax (i.e., abstract syntax trees or ASTs) of rule objects to identify group attributes and
    relations these rules condition on.

    Apart from attempting to deduce the attributes and rules, this class keeps track of the numbers of recognized and
    unrecognized attributes and relations (compartmentalized by method type, e.g., 'has_attr' and 'get_attr').

    References
        https://docs.python.org/3.6/library/dis.html
        https://docs.python.org/3.6/library/inspect.html
        https://docs.python.org/3.6/library/ast.html

        https://github.com/hchasestevens/astpath
        https://astsearch.readthedocs.io/en/latest
    '''

    def __init__(self):
        self.attr = set()
        self.rel  = set()

        self.cnt_rec   = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })  # recognized
        self.cnt_unrec = Counter({ 'get_attr': 0, 'get_rel': 0, 'has_attr': 0, 'has_rel': 0 })  # unrecognized

    def _dump(self, node, annotate_fields=True, include_attributes=False, indent='  '):
        '''
        Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default
        '''

        if not isinstance(node, ast.AST):
            raise TypeError('expected AST, got %r' % node.__class__.__name__)
        return self._format(node, 0, annotate_fields, include_attributes, indent)

    def _format(self, node, level=0, annotate_fields=True, include_attributes=False, indent='  '):
        '''
        Source: https://bitbucket.org/takluyver/greentreesnakes/src/default/astpp.py?fileviewer=file-view-default
        '''

        if isinstance(node, ast.AST):
            fields = [(a, self._format(b, level, annotate_fields, include_attributes, indent)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(a, self._format(getattr(node, a), level, annotate_fields, include_attributes, indent))
                               for a in node._attributes])
            return ''.join([
                node.__class__.__name__,
                '(',
                ', '.join(('%s=%s' % field for field in fields)
                           if annotate_fields else
                           (b for a, b in fields)),
                ')'])
        elif isinstance(node, list):
            lines = ['[']
            lines.extend((indent * (level + 2) + self._format(x, level + 2, annotate_fields, include_attributes, indent) + ','
                         for x in node))
            if len(lines) > 1:
                lines.append(indent * (level + 1) + ']')
            else:
                lines[-1] += ']'
            return '\n'.join(lines)
        return repr(node)

    def _analyze_test(self, node):
        if isinstance(node, ast.AST):
            fields = [(a, self._analyze_test(b)) for a,b in ast.iter_fields(node)]
            return ''.join([node.__class__.__name__, '(', ', '.join((b for a,b in fields)), ')'])
        elif isinstance(node, list):
            lines = []
            lines.extend((self._analyze_test(x) + ',' for x in node))
            return '\n'.join(lines)
        return repr(node)

    def _analyze(self, node):
        '''
        Processe a node of the AST recursively looking for method calls that suggest group attribute and relation names
        conditioned on by the rule.  It also updates counts of all known and unknown names (compartmentalized by the
        method name).

        References
            https://docs.python.org/3.6/library/ast.html#abstract-grammar
        '''

        if isinstance(node, ast.AST):
            for _,v in ast.iter_fields(node):
                self._analyze(v)

            if node.__class__.__name__ == 'Call':
                call_args = list(ast.iter_fields(node))[1][1]

                if list(ast.iter_fields(node))[0][1].__class__.__name__ == 'Attribute':
                    attr = list(ast.iter_fields(node))[0][1]
                    attr_name = list(ast.iter_fields(attr))[1][1]

                    if attr_name in ('get_attr', 'get_rel', 'has_attr', 'has_rel'):
                        call_args = call_args[0]
                        if call_args.__class__.__name__ == 'Str':
                            if attr_name in ('get_attr', 'has_attr'):
                                self.attr.add(RuleAnalyzer.get_str(call_args))
                            else:
                                self.rel.add(RuleAnalyzer.get_str(call_args))
                            self.cnt_rec[attr_name] += 1
                        elif call_args.__class__.__name__ in ('List', 'Dict'):
                            for i in list(ast.iter_fields(call_args))[0][1]:
                                if i.__class__.__name__ == 'Str':
                                    if attr_name in ('get_attr', 'has_attr'):
                                        self.attr.add(RuleAnalyzer.get_str(i))
                                    else:
                                        self.rel.add(RuleAnalyzer.get_str(i))
                                    self.cnt_rec[attr_name] += 1
                                else:
                                    self.cnt_unrec[attr_name] += 1
                                    # print(list(ast.iter_fields(i)))
        elif isinstance(node, list):
            for i in node:
                self._analyze(i)

    def analyze(self, rule):
        tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef): continue  # skip non-classes

            for node_fn in node.body:
                if not isinstance(node_fn, ast.FunctionDef): continue  # skip non-methods
                self._analyze(node_fn.body)

                # if node_fn.name in ('is_applicable'): print(self._analyze_01(node_fn.body))

    def dump(self, rule):
        tree = ast.fix_missing_locations(ast.parse(inspect.getsource(rule.__class__)))
        print(self._dump(tree))

    @staticmethod
    def get_str(node):
        return list(ast.iter_fields(node))[0][1]


# ----------------------------------------------------------------------------------------------------------------------
class Simulation(object):
    '''
    A PRAM simulation.

    The discrete time is unitless; it is the simulation context that defines the appropriate granularity.  For
    instance, million years (myr) might be appropriate for geological processes while Plank time might be appropriate
    for modeling quantum phenomena.

    Each simulation stores certain statistics related to the most recent of its runs (in the 'self.last_run' DotMap).
    '''

    def __init__(self, t0=0, t_step_size=1, t_step_cnt=0, rand_seed=None):
        '''
        One interpretation of 't=4' is that the simulation starts at 4am.  Similarily, 't_step_size=1' could mean that
        the simulation time increments in one-hour intervals.
        '''

        self.t0 = t0
        self.t = t0
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

        self.rule_analyzer = RuleAnalyzer()

        self.last_run = DotMap()  # dict of the most recent run

        self.pragma = DotMap()
        self.pragma.analyze = True            # analyze the simulation and show improvement suggestions
        self.pragma.autoprune_groups = False  # remove attributes and relations not referenced by rules

    def __repr__(self):
        return f'{self.__class__.__name__}({self.t}, {self.t_step_size}, {self.t_step_cnt}, {self.rand_seed})'

    def add_group(self, group):
        if len(self.rules) == 0:
            raise SimulationConstructionError('A group is being added but no rules are present; rules need to be added before groups.')

        if len(self.pop.groups) == 0:  # run when the first group is being added (because that marks the end of adding rules)
            self.analyze_rules_pre_run()

        self.pop.add_group(group)
        return self

    def add_groups(self, groups):
        for g in groups:
            self.add_group(g)
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
        if len(self.pop.groups) > 0:
            raise SimulationConstructionError('A rule is being added but groups already exist; rules need be added before groups.')

        self.rules.append(rule)
        return self

    def add_site(self, site):
        self.pop.add_site(site)
        return self

    def add_sites(self, sites):
        self.pop.add_sites(sites)
        return self

    def analyze_rules_pre_run(self):
        '''
        Analyze rule conditioning prior to running the simulation.  This is done by analyzing the syntax (i.e.,
        abstract syntax trees or ASTs) of rule objects to identify group attributes and relations these rules condition
        on.  This is a difficult problem and the current implementation leaves room for future improvement.  In fact,
        the current algorithm works only if the names of attributes and relations are specified in the code of rules as
        string literals.  For example, if an attribute name is stored in a variable or is a result of a method call, it
        will not be captured by this algorithm.

        As a consequence, it is not possible to perform error-free groups auto-pruning based on this analysis.
        '''

        for r in self.rules:
            self.rule_analyzer.analyze(r)

        return self

    def analyze_rules_post_run(self):
        '''
        Analyze rule conditioning after running the simulation.  This is done by processing group attributes and
        relations that the simulation has recorded as accessed by at least one rule.  The evidence of attributes and
        relations having been actually accessed is a strong one.  However, as tempting as it may be to use this
        information to prune groups, it's possible that further simulation iterations depend on other sets of
        attributes and relations.

        As a consequence, it is not possible to perform error-free groups auto-pruning based on this analysis.
        '''

        lr = self.last_run
        lr.clear()

        lr.attr_used = getattr(Group, 'attr_used')  # attributes conditioned on by at least one rule
        lr.rel_used  = getattr(Group, 'rel_used')   # ^ (relations)

        lr.attr_groups = set()  # attributes defining groups
        lr.rel_groups  = set()  # ^ (relations)
        for g in self.pop.groups.values():
            for ga in g.attr.keys(): lr.attr_groups.add(ga)
            for gr in g.rel.keys():  lr.rel_groups. add(gr)

        lr.attr_unused = lr.attr_groups - lr.attr_used  # attributes not conditioned on by even one rule
        lr.rel_unused  = lr.rel_groups  - lr.rel_used   # ^ (relations)

        if self.pragma.analyze and (len(lr.attr_unused) > 0 or len(lr.rel_unused) > 0):
            if len(lr.attr_unused) > 0 and len(lr.rel_unused) == 0:
                print('Based on the most recent simulation run, the following group attributes are superfluous:')
                print(f'    {list(lr.attr_unused)}')

            if len(lr.attr_unused) == 0 and len(lr.rel_unused) > 0:
                print('Based on the most recent simulation run, the following group relations are superfluous:')
                print(f'    {list(lr.rel_unused)}')

            if len(lr.attr_unused) > 0 and len(lr.rel_unused) > 0:
                print('Based on the most recent simulation run, the following group attributes A and relations R are superfluous:')
                print(f'    A: {list(lr.attr_unused)}')
                print(f'    R: {list(lr.rel_unused )}')

        return self

    def clear_probes(self):
        self.probes.clear()
        return self

    def clear_rules(self):
        self.rules.clear()
        return self

    def commit_group(self, group):
        self.add_group(group)
        return self

    def gen_groups_from_db(self, fpath_db, tbl, attr={}, rel={}, attr_db=[], rel_db=[], rel_at=None, limit=0, fpath=None, is_verbose=False):
        groups = None

        # Load:
        if fpath is not None and os.path.isfile(fpath):
            if is_verbose: print('Loading groups... ', end='')
            with gzip.GzipFile(fpath, 'rb') as f:
                gc.disable()
                groups = pickle.load(f)
                gc.enable()
            if is_verbose: print('done.')

        # Generate:
        else:
            if is_verbose: print('Generating groups... ', end='')
            groups = Group.gen_from_db(fpath_db, tbl, attr, rel, attr_db, rel_db, rel_at, limit)
            if is_verbose: print('done.')

            if fpath is not None:
                if is_verbose: print('Saving groups... ', end='')
                with gzip.GzipFile(fpath, 'wb') as f:
                    pickle.dump(groups, f)
                if is_verbose: print('done.')

        self.add_groups(groups)
        return self

    def gen_sites_from_db(self, fpath_db, fn_gen=None, fpath=None, is_verbose=False):
        sites = None

        # Load:
        if fpath is not None and os.path.isfile(fpath):
            if is_verbose: print('Loading sites... ', end='')
            with gzip.GzipFile(fpath, 'rb') as f:
                gc.disable()
                sites = pickle.load(f)
                gc.enable()
            if is_verbose: print('done.')

        # Generate:
        elif fn_gen is not None:
            if is_verbose: print('Generating sites... ', end='')
            sites = fn_gen(fpath_db)
            if is_verbose: print('done.')

            if fpath is not None:
                if is_verbose: print('Saving sites... ', end='')
                with gzip.GzipFile(fpath, 'wb') as f:
                    pickle.dump(sites, f)
                if is_verbose: print('done.')

        return sites

    def new_group(self, name=None, n=0.0):
        return Group(name or self.pop.get_next_group_name(), n, callee=self)

    def rem_probe(self, probe):
        self.probes.discard(probe)
        return self

    def rem_rule(self, rule):
        self.rules.discard(rule)
        return self

    def run(self, t_step_cnt=0, do_disp_t=False):
        '''
        One by-product of running the simulation is that the simulation stores all group attributes and relations that
        are conditioned on by at least one rule.  After the run is over, a set of unused attributes and relations is
        produced, unless silenced.  That output may be useful for making future simulations more efficient by allowing
        the modeller to remove the unused bits which unnecesarily partition the group space.
        '''

        # Rule conditioning 01 -- Init:
        setattr(Group, 'attr_used', set())
        setattr(Group, 'rel_used',  set())

        # Do setup:
        if not self.is_setup_done:
            self.pop.apply_rules(self.rules, 0, self.t, True)
            self.is_setup_done = True

        # Run the simulation:
        t_step_cnt = t_step_cnt if t_step_cnt > 0 else self.t_step_cnt

        for i in range(t_step_cnt):
            if do_disp_t: print(f't:{self.t}')

            self.pop.apply_rules(self.rules, i, self.t)

            for p in self.probes:
                p.run(i, self.t)

            self.t = (self.t + self.t_step_size) % 24

        # Rule conditioning 02 -- Analyze and deinit:
        self.analyze_rules_post_run()

        getattr(Group, 'attr_used').clear()
        getattr(Group, 'rel_used' ).clear()

        return self

    def set_pragma_analyze(self, value):
        self.pragma.analyze = value
        return self

    def set_pragma_autoprune_groups(self, value):
        self.pragma.autoprune_groups = value
        return self

    def show_rule_analysis(self):
        # Rule analyzer:
        ra = self.rule_analyzer

        print('Rule analyzer')
        print('    Used')
        print(f'        Attributes : {list(ra.attr)}')
        print(f'        Relations  : {list(ra.rel)}')
        print('    Counts')
        print(f'        Recognized   : get_attr:{ra.cnt_rec["get_attr"]} get_rel:{ra.cnt_rec["get_rel"]} has_attr:{ra.cnt_rec["has_attr"]} has_rel:{ra.cnt_rec["has_rel"]}')
        print(f'        Unrecognized : get_attr:{ra.cnt_unrec["get_attr"]} get_rel:{ra.cnt_unrec["get_rel"]} has_attr:{ra.cnt_unrec["has_attr"]} has_rel:{ra.cnt_unrec["has_rel"]}')

        # Post-run:
        lr = self.last_run

        print('Most recent simulation run')
        print('    Used')
        print(f'       Attributes : {list(lr.attr_used)}')
        print(f'       Relations  : {list(lr.rel_used)}')
        print('    Groups')
        print(f'       Attributes : {list(lr.attr_groups)}')
        print(f'       Relations  : {list(lr.rel_groups)}')
        print('    Superfluous')
        print(f'       Attributes : {list(lr.attr_unused)}')
        print(f'       Relations  : {list(lr.rel_unused)}')

        return self

    def summary(self, do_header=True, n_groups=8, n_sites=8, n_rules=8, n_probes=8, end_line_cnt=(0,0)):
        '''
        Prints the simulation summary.  The summary can be printed at any stage of a simulation (i.e., including at
        the very beginning and end) and parts of the summary can be shown and hidden and have their length specified.
        '''

        print('\n' * end_line_cnt[0], end='')

        if do_header:
            print( 'Simulation')
            print(f'    Random seed: {self.rand_seed}')
            print( '    Timer')
            print(f'        Start      : {"{:,}".format(self.t0)}')
            print(f'        Step size  : {"{:,}".format(self.t_step_size)}')
            print(f'        Iterations : {"{:,}".format(self.t_step_cnt)}')
            print(f'        Sequence   : {[self.t0 + self.t_step_size * i for i in range(5)]}')
            print( '    Population')
            print(f'        Size        : {"{:,.2f}".format(round(self.pop.get_size(), 1))}')
            print(f'        Groups      : {"{:,}".format(self.pop.get_group_cnt())}')
            print(f'        Groups (ne) : {"{:,}".format(self.pop.get_group_cnt(True))}')
            print(f'        Sites       : {"{:,}".format(self.pop.get_site_cnt())}')
            print(f'        Rules       : {"{:,}".format(len(self.rules))}')
            print(f'        Probes      : {"{:,}".format(len(self.probes))}')

        if n_groups > 0:
            if len(self.pop.groups) > 0: print(f'    Groups ({"{:,}".format(len(self.pop.groups))})\n' + '\n'.join(['        {}'.format(g) for g in list(self.pop.groups.values())[:n_groups]]))
            if len(self.pop.groups) > n_groups: print('        ...')

        if n_sites > 0:
            if len(self.pop.sites)  > 0: print(f'    Sites ({"{:,}".format(len(self.pop.sites))})\n'   + '\n'.join(['        {}'.format(s) for s in list(self.pop.sites.values())[:n_sites]]))
            if len(self.pop.sites) > n_sites: print('        ...')

        if n_rules > 0:
            if len(self.rules)      > 0: print(f'    Rules ({"{:,}".format(len(self.rules))})\n'       + '\n'.join(['        {}'.format(r) for r in self.rules[:n_rules]]))
            if len(self.rules) > n_rules: print('        ...')

        if n_probes > 0:
            if len(self.probes)     > 0: print(f'    Probes ({"{:,}".format(len(self.probes))})\n'     + '\n'.join(['        {}'.format(p) for p in self.probes[:n_probes]]))
            if len(self.probes) > n_probes: print('        ...')

        print('\n' * end_line_cnt[1], end='')

        return self
