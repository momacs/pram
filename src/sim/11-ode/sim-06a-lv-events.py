'''
The Lotka-Volterra two-species system with two desease events: One for predator population and for one for prey
population.  These events disturb the size of the two populations at a specific point in time (i.e., the population
dynamics process is affected directly).
'''

import matplotlib.pyplot as plt

from pram.entity import Group, GroupSplitSpec
from pram.rule   import ODESystemAttr, Event, TimeAlways
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (0) Forward Euler method:

# import numpy as np
# from decimal import Decimal
#
# alpha = Decimal(1.1)  # baboon
# beta  = Decimal(0.4)
# delta = Decimal(0.1)  # cheetah
# gamma = Decimal(0.4)
#
# s = [Decimal(10.0), Decimal(10.0)]  # x,y
# h = [[], [], []]  # s[0], s[1], t
#
# for t in np.arange(0.0, 3.0, 0.1):
#     d = [s[0] * (alpha - beta * s[1]), s[1] * (-gamma + delta * s[0])]
#
#     s[0] = s[0] + d[0] * Decimal(t)
#     s[1] = s[1] + d[1] * Decimal(t)
#
#     h[0].append(s[0])
#     h[1].append(s[1])
#     h[2].append(t)
#
# plt.plot(h[2], h[0], 'b-', h[2], h[1], 'r-')  # red - predator
# plt.show()
#
# sys.exit(0)


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init:

alpha = 1.1  # baboon
beta  = 0.4
delta = 0.1  # cheetah
gamma = 0.4


def f_lotka_volterra(t, state):
    x,y = state
    return [x * (alpha - beta * y), y * (-gamma + delta * x)]


class PredatorDiseaseEvent(Event):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='predator-disease-evt')

    def apply(self, pop, group, iter, t):
        return [GroupSplitSpec(p=1.00, attr_set={ 'y': 1 })]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and iter == 440


class PreyDiseaseEvent(Event):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='prey-disease-evt')

    def apply(self, pop, group, iter, t):
        return [GroupSplitSpec(p=1.00, attr_set={ 'x': 3 })]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and iter == 1100


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

s = (
    Simulation().
    set_pragma_autocompact(True).
    add([
        ODESystemAttr(f_lotka_volterra, ['x', 'y'], dt=0.1),
        PredatorDiseaseEvent(),
        PreyDiseaseEvent(),
        Group(m=1, attr={ 'x': 10, 'y': 10 })
    ]).
    run(1500)
)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Results:

h = s.rules[0].get_hist()

# Time series plot:
fig = plt.figure(figsize=(20,4), dpi=150)
plt.plot(h[0], h[1][0], lw=2, linestyle='-', color='blue', mfc='none', antialiased=True)
plt.plot(h[0], h[1][1], lw=2, linestyle='-', color='red',  mfc='none', antialiased=True)
plt.legend(['Prey', 'Predator'], loc='upper right')
plt.xlabel('Iteration')
plt.ylabel('Attribute value')
plt.grid(alpha=0.25, antialiased=True)
plt.show()

# plt.plot(h[0], h[1][0], 'b-', h[0], h[1][1], 'r-')  # red - predator
# plt.show()

# Phase plot:
# plt.plot(h[1][0], h[1][1], 'k-')
# plt.show()


# https://mbe.modelica.university/components/components/population/#population-components
