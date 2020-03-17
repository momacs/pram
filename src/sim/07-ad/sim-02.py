'''
An Alzheimer's Disease simulation which models the following scenario: "AD incidence doubles every 5 years after
65 yo."
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import math

from scipy.stats import poisson

from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation
from pram.util   import Time as TimeU


# ----------------------------------------------------------------------------------------------------------------------
class DoubleIncidenceADRule(Rule):
    '''
    Doubles the incidence of the AD every five years.  The incidence is modeled by the rate parameter $\lambda$ of the
    Poisson distribution.
    '''

    T_UNIT_MS = TimeU.MS.y

    AGE_0 = 65       # cutoff age for getting AD
    LAMBDA_0 = 0.01  # base Poisson rate (i.e., at cutoff age and until the first increase)
    INC_FACTOR = 2   # increase the rate by this much
    INC_EVERY = 5    # increase the rate every this many time units

    def __init__(self):
        super().__init__('double-incidence-of-ad', TimeAlways())

    def apply(self, pop, group, iter, t):
        return self.get_split_specs(group)

    def get_lambda(self, age):
        return self.LAMBDA_0 * self.INC_FACTOR ** math.floor((age - self.AGE_0) / self.INC_EVERY)

    def get_split_specs(self, group, age_inc=1):
        '''
        age_inc - how much to increment the 'age' attribute
        '''

        age = group.ga('age')

        if age < self.AGE_0:
            return [GroupSplitSpec(p=1.00, attr_set={ 'age': age + age_inc, 'ad': False })]

        l = self.get_lambda(age)
        p0 = poisson(l).pmf(0)
        print(f'n: {round(group.m,2):>12}  age: {age:>3}  l: {round(l,2):>3}  p0: {round(p0,2):<4}  p1: {round(1-p0,2):<4}')

        return [GroupSplitSpec(p=1.00, attr_set={ 'age': age + age_inc, 'ad': False })]  # testing so don't move mass

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.ha(['age', 'ad'])

    def setup(self, pop, group):
        if group.ha('age'):
            return self.get_split_specs(group, 0)
        else:
            return None


# ----------------------------------------------------------------------------------------------------------------------
(Simulation().
    set().
        pragma_autocompact(True).
        pragma_live_info(False).
        done().
    add().
        rule(DoubleIncidenceADRule()).
        group(Group(m=100, attr={ 'age': 60 })).
        done().
    run(40)
)
