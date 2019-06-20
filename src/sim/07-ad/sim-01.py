'''
An Alzheimer's Disease simulation that comes in two flavors depending on which of the two following rules are used: A
deterministic one or a probabilistic one.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation
from pram.util   import Time as TimeU


# ----------------------------------------------------------------------------------------------------------------------
class GetADDetRule(Rule):
    '''
    Deterministic AD rule (i.e., after certain age an agent has AD and below that age they don't).
    '''

    T_UNIT_MS = TimeU.MS.y

    AGE = 65 - 1  # cutoff age for getting AD

    def __init__(self):
        super().__init__('get-ad-bin', TimeAlways())

    def apply(self, pop, group, iter, t):
        c = self.__class__  # shorthand
        age = group.get_attr('age')
        if age >= c.AGE:
            return [GroupSplitSpec(p=1.00, attr_set={ 'age': age + 1, 'ad': True  })]
        else:
            return [GroupSplitSpec(p=1.00, attr_set={ 'age': age + 1, 'ad': False })]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr(['age', 'ad'])

    def setup(self, pop, group):
        return [GroupSplitSpec(p=1.00, attr_set={ 'ad': group.get_attr('age') >= self.__class__.AGE })]


# ----------------------------------------------------------------------------------------------------------------------
class GetADProbRule(Rule):
    '''
    Probabilistic AD rule (i.e., after certain age an agent has a probabiity of having AD and below that age they are
    healthy).
    '''

    T_UNIT_MS = TimeU.MS.y

    AGE = 65 - 1  # cutoff age for getting AD
    P = 0.5       # the prob of getting AD after the cutoff age

    def __init__(self):
        super().__init__('get-ad-prob', TimeAlways())

    def apply(self, pop, group, iter, t):
        return self.get_split_specs(group)

    def get_split_specs(self, group, age_inc=1):
        '''
        age_inc - how much to increment the 'age' group attribute
        '''

        age = group.get_attr('age')
        p = max(min(self.__class__.P * (age - self.__class__.AGE), 1.00), 0.00)
        return [
            GroupSplitSpec(p=    p, attr_set={ 'age': age + age_inc, 'ad': True  }),
            GroupSplitSpec(p=1 - p, attr_set={ 'age': age + age_inc, 'ad': False })
        ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr(['age', 'ad'])

    def setup(self, pop, group):
        return self.get_split_specs(group, 0)


# ----------------------------------------------------------------------------------------------------------------------
# r = GetADDetRule()
r = GetADProbRule()

(Simulation().
    set().
        pragma_autocompact(False).
        pragma_live_info(True).
        done().
    add().
        rule(r).
        group(Group(n=100, attr={ 'age': 60 })).
        done().
    run(7).
    summary(False, 1024,0,0,0, (1,0))
)
