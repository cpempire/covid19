import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# define the ode systems
def seir(y, t, N, beta, a, gamma):
    s, e, i, r = y
    dydt = [-beta*i*s/N, beta*i*s/N - a*e, a*e - gamma*i, gamma*i]
    return dydt

# parameter values
N = 1000            # total population
beta = 2.           # transmission rate
a = 1.              # incubation rate
gamma = 0.1         # recover rate

# initial values
y0 = [480, 1, 0, 0]

# time interval and steps
t = np.linspace(0, 100, 1001)

# solve the ode systems
sol = odeint(seir, y0, t, args=(N, beta, a, gamma))

# plot solutions
plt.figure()
plt.plot(t, sol[:, 0], 'b', label='susceptible')
plt.plot(t, sol[:, 1], 'g', label='exposed')
plt.plot(t, sol[:, 2], 'r', label='infected')
plt.plot(t, sol[:, 3], 'k', label='recovered')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.savefig("seir_example.pdf")
plt.show()
plt.close()