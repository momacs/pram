'''
Lorenz system.

A simplified mathematical model for atmospheric convection. The model is a system of three ordinary differential
equations now known as the Lorenz equations:

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

------------------------------------------------------------------------------------------------------------------------

A chaotic map is a map (= evolution function) that exhibits some sort of chaotic behavior.

List of chaotic maps (https://en.wikipedia.org/wiki/List_of_chaotic_maps)
    Map                     Time dom    Space dom  Space dim  Params
    ----------------------------------------------------------------
    Logistic map            discrete    real               1       1
    Lotka-Volterra system   continuous  real               3       4
    Lorenz system           continuous  real               3       3
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


# Declarations and definitions:
def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

state0 = [1.0, 1.0, 1.0]  # the initial state

# Loopless:
# t = np.arange(0.0, 40.0, 0.01)
# states = odeint(f, state0, t)

# Loop (compatible with a PRAM simulation):
t = np.arange(0.0, 40.0, 0.01)
states = []
s = state0
t0 = 0.0
for t1 in t[1:]:
    # print(s)
    s = odeint(f, s, [t0,t1])
    t0 = t1
    states.append(s)
# states = odeint(f, state0, t)

# Visualize:
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
plt.show()
