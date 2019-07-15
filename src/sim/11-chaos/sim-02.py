# https://stackoverflow.com/questions/36829446/taking-a-single-ode-step

# from scipy.integrate import ode
# obj = ode(lambda t, y: -y)
# obj.set_initial_value(1)
# while obj.t < 1:
#     y_new = obj.integrate(1, step=True)
# print(obj.t) # prints 1.037070648009345


# ----------------------------------------------------------------------------------------------------------------------
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
# https://scicomp.stackexchange.com/questions/11616/how-to-get-ode-solution-at-specified-time-points

# from scipy.integrate import ode
#
# y0, t0 = [1.0j, 2.0], 0
#
# def f(t, y, arg1):
#     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
# def jac(t, y, arg1):
#     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
#
# r = ode(f, jac).set_integrator('zvode', method='bdf', with_jacobian=True)
# r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
# t1 = 10
# dt = 1
# while r.successful() and r.t < t1:
#     r.integrate(r.t+dt)
#     print("%g %g" % (r.t, r.y))


# ----------------------------------------------------------------------------------------------------------------------
# from scipy.integrate import ode
#
# rho = 28.0
# sigma = 10.0
# beta = 8.0 / 3.0
#
# def f(t, state):
#     x, y, z = state  # unpack the state vector
#     return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]  # derivatives
#
# state0 = [1.0, 1.0, 1.0]  # the initial state
# t0 = 0.0
# t1 = 1.0
# dt = 0.1
#
# r = ode(f).set_integrator('zvode', method='bdf')
# # r = ode(f).set_integrator('lsoda')
# r.set_initial_value(state0, t0)
# while r.successful() and r.t < t1:
#     r.integrate(r.t + dt)
#     print(f'{round(r.t,1)}: {r.y}')
#     # print(r.__dict__)




# ----------------------------------------------------------------------------------------------------------------------
from scipy.integrate import ode

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(t, state):
    x, y, z = state  # unpack the state vector
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]  # derivatives

state0 = [1.0, 1.0, 1.0]  # the initial state
t0 = 0.0
t1 = 1.0
dt = 0.1

r = ode(f).set_integrator('zvode', method='bdf')
r.set_initial_value(state0, t0)

r.integrate(r.t + dt)
print(f'{round(r.t,1)}: {r.y}')

r.integrate(r.t + dt)
print(f'{round(r.t,1)}: {r.y}')

r.integrate(r.t + dt)
print(f'{round(r.t,1)}: {r.y}')

r.integrate(r.t + dt)
print(f'{round(r.t,1)}: {r.y}')
