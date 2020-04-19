'''
The Lotka-Volterra two-species system interacting with a climate calamity process.  That process models increasingly
hostile natural environment which decreases the reproductive rate of the prey population.  The process does not
directly disturb the sizes of the two populations and affects the population dynamics process indirectly.  This is a
catch-all process, and as such it combines all relevant influences examples of which include:

- Increasing global baseline temperature leading to increased frequency of drought
- Decreasing biodiversity leading to lower adaptability of plants leading to lower prey food availability
- Strenghtening bacteria leading to increased incidence of prey disease

The process changes the prey reproductive parameter in very small decrements but it still leads to the eventual
extinction of the predator population (due to insufficient prey population) and then the prey population itself.  While
the model parameters have not been validated biologically nor ecologically, the predator species goes extinct in about
80 years and the prey population goes extinct in about 140 years.
'''

import matplotlib.pyplot as plt

from pram.entity import Group, GroupSplitSpec
from pram.rule   import ODESystemAttr, Process, TimeAlways
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (1) Init:

alpha = 1.1  # baboon
beta  = 0.4
delta = 0.1  # cheetah
gamma = 0.4


def f_lotka_volterra(t, state):
    x,y = state
    return [x * (alpha - beta * y), y * (-gamma + delta * x)]


class ClimateCalamityProcess(Process):
    def __init__(self, t=TimeAlways()):
        super().__init__(t=t, name='climate-calamity-proc')

    def apply(self, pop, group, iter, t):
        global alpha
        alpha = alpha - 0.001
        return None


# ----------------------------------------------------------------------------------------------------------------------
# (2) Simulation:

s = (
    Simulation().
    set_pragma_autocompact(True).
    add([
        ODESystemAttr(f_lotka_volterra, ['x', 'y'], dt=0.1),
        ClimateCalamityProcess(),
        Group(m=1, attr={ 'x': 10, 'y': 10 })
    ]).
    run(2000)
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
