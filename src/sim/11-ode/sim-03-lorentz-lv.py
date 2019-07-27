import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.rule   import ODESystem
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (1) Lorentz system:

# rho = 28.0
# sigma = 10.0
# beta = 8.0 / 3.0
#
# def f_lorentz(t, state):
#     x,y,z = state
#     return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
#
# r = ODESystem(f_lorentz, [1.0, 1.0, 1.0], dt=0.01)
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
    return [x * (alpha - beta * y), y * (-gamma + delta * x)]

r = ODESystem(f_lotka_volterra, [10, 10], dt=0.1)

(Simulation().add([r, Group(n=1)]).run(500))

h = r.get_hist()

import matplotlib.pyplot as plt
plt.plot(h[0], h[1][0], 'b-', h[0], h[1][1], 'r-')  # red - predator
plt.show()

# Phase-space plot:
# plt.plot(h[1][0], h[1][1], 'k-')
# plt.show()
