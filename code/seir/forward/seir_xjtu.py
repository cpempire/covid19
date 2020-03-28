# implementation of the SEIR model from XJTU in following paper
# https://www.mdpi.com/2077-0383/9/2/462

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# define the ode systems
def seir(y, t, theta, c, beta, q, sigma, lamb, rho, delta_I, delta_q, gamma_I, gamma_A, gamma_H, alpha):

    S, E, I, A, Sq, Eq, H, R = y

    dydt = [-(beta*c+c*q*(1-beta))*S*(I+theta*A) + lamb*Sq,
            beta*c*(1-q)*S*(I+theta*A) - sigma*E,
            sigma*rho*E - (delta_I+alpha+gamma_I)*I,
            sigma*(1-rho)*E - gamma_A*A,
            (1-beta)*c*q*S*(I+theta*A) - lamb*Sq,
            beta*c*q*S*(I+theta*A) - delta_q*Eq,
            delta_I*I + delta_q*Eq - (alpha+gamma_H)*H,
            gamma_I*I + gamma_A*A + gamma_H*H]

    return dydt

# parameter values
theta   = 1.            # weight I + theta*A
c       = 14.781        # contact rate
beta    = 2.1011e-8     # transmission rate per contact
q       = 1.8887e-7     # quarantined rate of exposed people
sigma   = 1./7          # incubation rate -- transition from exposed to infected
lamb    = 1./14         # rate of quarantined uninfected to be released
rho     = 0.86834       # infected symptomatic probability
delta_I = 0.13266       # transition rate of symptomatic infected to quarantined infected
delta_q = 0.1259        # transition rate of quarantined exposed to quarantined infected
gamma_I = 0.33029       # recovery rate of symptomatic infected
gamma_A = 0.13978       # recovery rate of asymptomatic infected
gamma_H = 0.11624       # recovery rate of quarantined infected
alpha   = 1.7826e-5     # disease induced death rate

# initial values
y0 = [11081000, 105.1, 27.679, 53.839, 739, 1.1642, 1., 2.]

# time interval and steps
t = np.linspace(0, 50, 501)

# solve the ode systems
sol = odeint(seir, y0, t, args=(theta, c, beta, q, sigma, lamb, rho, delta_I, delta_q, gamma_I, gamma_A, gamma_H, alpha))

# plot solutions
plt.figure()
plt.semilogy(t, sol[:, 0], 'b', label='$S$ susceptible')
plt.semilogy(t, sol[:, 1], 'g', label='$E$ exposed')
plt.semilogy(t, sol[:, 2], 'k.', label='$I$ symptomatic infected')
plt.semilogy(t, sol[:, 3], 'k', label='$A$ asymptomatic infected')
plt.semilogy(t, sol[:, 4], 'y', label='$Sq$ quarantined susceptible')
plt.semilogy(t, sol[:, 5], 'm', label='$Eq$ quarantined exposed')
plt.semilogy(t, sol[:, 6], 'c', label='$H$ quarantined infected')
plt.semilogy(t, sol[:, 7], 'r', label='$R$ recovered')
plt.legend(loc='best')
plt.xlabel('time t')
plt.ylabel('# people')
plt.grid()
plt.savefig("seir_xjtu.pdf")
plt.show()
plt.close()