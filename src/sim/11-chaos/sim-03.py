import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import ODESystemSelf
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (1) Lorentz system:

'''
Lorenz system.

A simplified mathematical model for atmospheric convection specified by the Lorenz equations:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = xy - beta * z

The equations relate the properties of a two-dimensional fluid layer uniformly warmed from below and cooled from above.
In particular, the equations describe the rate of change of three quantities with respect to time: x is proportional to
the rate of convection, y to the horizontal temperature variation, and z to the vertical temperature variation. The
constants sigma, rho, and beta are system parameters proportional to the Prandtl number, Rayleigh number, and certain
physical dimensions of the layer itself.

From a technical standpoint, the Lorenz system is nonlinear, non-periodic, three-dimensional and deterministic.

SRC: https://en.wikipedia.org/wiki/Lorenz_system
'''

# rho = 28.0
# sigma = 10.0
# beta = 8.0 / 3.0
#
# def f_lorentz(t, state):
#     x,y,z = state
#     return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
#
# r = ODESystemSelf(f_lorentz, [1.0, 1.0, 1.0], dt=0.01)
#
# (Simulation().add([r, Group(n=1)]).run(5000))
#
# h = r.get_hist()
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(h[1][0], h[1][1], h[1][2])
# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# (2) Lotka-Volterra:

alpha = 1.1  # baboon
beta  = 0.4
delta = 0.1  # cheetah
gamma = 0.4

def f_lotka_volterra(t, state):
    x,y = state
    return [x * (alpha - beta * y), -y * (gamma - delta * x)]

r = ODESystemSelf(f_lotka_volterra, [10, 10], dt=0.1)

(Simulation().add([r, Group(n=1)]).run(500))

h = r.get_hist()

import matplotlib.pyplot as plt
plt.plot(h[0], h[1][0], 'b-', h[0], h[1][1], 'r-')  # red - predator
plt.show()

# Phase-space plot:
# plt.plot(h[1][0], h[1][1], 'k-')
# plt.show()
