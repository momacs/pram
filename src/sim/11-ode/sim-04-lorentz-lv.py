'''
Two ODE systems (i.e., Lorenz and Lotka-Volterra) implemented on group attributes.
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.entity import Group
from pram.rule   import ODESystemAttr
from pram.sim    import Simulation
from pram.util   import Time


# ----------------------------------------------------------------------------------------------------------------------
# (1) Lorenz system:

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f_lorenz(t, state):
    x,y,z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

r = ODESystemAttr(f_lorenz, ['x', 'y', 'z'], dt=0.01)

(Simulation().set_pragma_autocompact(True).add([r, Group(n=1, attr={'x': 0.0, 'y': 10, 'z': 0.0})]).run(5000))

h = r.get_hist()

# Phase plot (here, Lorenz attractor):
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(h[1][0], h[1][1], h[1][2])
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# (2) Lotka-Volterra:

# alpha = 1.1  # baboon
# beta  = 0.4
# delta = 0.1  # cheetah
# gamma = 0.4
#
# def f_lotka_volterra(t, state):
#     x,y = state
#     return [x * (alpha - beta * y), y * (-gamma + delta * x)]
#
# r = ODESystemAttr(f_lotka_volterra, ['x', 'y'], dt=0.1)
#
# (Simulation().set_pragma_autocompact(True).add([r, Group(n=1, attr={ 'x': 10, 'y': 10 })]).run(500))
#
# h = r.get_hist()
#
# # Time series plot:
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(20,4), dpi=150)
# plt.plot(h[0], h[1][0], lw=2, linestyle='-', color='blue', mfc='none', antialiased=True)
# plt.plot(h[0], h[1][1], lw=2, linestyle='-', color='red',  mfc='none', antialiased=True)
# plt.legend(['Prey', 'Predator'], loc='upper right')
# plt.xlabel('Iteration')
# plt.ylabel('Attribute value')
# plt.grid(alpha=0.25, antialiased=True)
# plt.show()

# plt.plot(h[0], h[1][0], 'b-', h[0], h[1][1], 'r-')  # red - predator
# plt.show()

# Phase portrait plot:
# import matplotlib.pyplot as plt
# plt.plot(h[1][0], h[1][1], 'k-')
# plt.show()
