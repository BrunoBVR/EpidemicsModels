# Bruno Vieira Ribeiro - SEIR for COVID-19 numerical simulation
# IFB - 2020

# x[] -> S,E,I,R

import numpy as np
import matplotlib.pyplot as plt

# Defines functions for this project
from funcs import *

    
#Main body of algorithm:

            # Data parameters:
# t_inc (incubation time = 5 days (avg) )
# t_inf (infectious time = 1.61 day (avg) )
# R0 = basic reproduction number (average of individuals infected by an infected individual during infectation)
# R0 approx 2.74

t_inc = 5.0
t_inf = 1.61
R0 = 2.74

            # Equation parameters
# Parameters in F: beta, sigma, gamma

sigma = 1.0/t_inc
gamma = 1.0/t_inf
beta = R0/t_inf

print(40*"=")
print(f"sigma = {sigma}")
print(f"gamma = {gamma}")
print(f"beta = {beta}")
print(f"R0 = {R0}")
            # Simulation parameters
# time parameters given in days
# population can be rescaled depending on location

dt = 0.1
t_max = 200.0

# Choose n=4 for the SEIR model (intend to add more eqs to model)
n = 4

# Maximum number of infecteds to start quarantine
max_inf = 200
#reach = False # flag to check first hit on max_inf infected

            # Setting intitial conditions
# xa[] = [S0 , E0, I0, R0]
# I0 = 1/eps
# eps = 1/city population

# Change this line to account for real data:
city_pop = 10000


# eps = 1.0/city_pop
eps = 1

xa = np.array([city_pop-eps, 0.0, eps, 0.0])
#x_seir = np.zeros(n)


print(40*'=')
print("Initial population [S, E, I, R] = ", xa)
            # Setting time counter and time arrays

# These are for plotting and data purposes
t = 0
time = np.array(t)
S = np.array(xa[0])
E = np.array(xa[1])
I = np.array(xa[2])
R = np.array(xa[3])

[Q_start, time, S, E, I, R] = time_loop(t_max, dt, xa, n, beta, sigma, gamma, city_pop, max_inf,
             S, E, I, R, reach = False)


plt.plot(time, S, 'r-')
plt.plot(time, E, 'b-')
plt.plot(time, I, 'g-')
plt.plot(time, R, 'k--')
plt.xlim(0,200)
plt.xlabel('Time (days)')
plt.ylabel('Individuals')
plt.grid()
plt.legend(["S", "E", "I", "R"], loc='best')
plt.show()

plt.plot(time, I, 'g-')
plt.axvline(x = Q_start, linewidth=2, color='r', ls='--')
plt.xlabel('Time (days)')
plt.ylabel('Infected individuals')
plt.grid()
plt.show()

plt.plot(time, I+R, 'r-')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Infected individuals')
plt.grid()
plt.show()

plt.plot(time, I+R, 'r-')
plt.plot(time, R, 'b-')
plt.legend(["Cumulative Infected", "Recovered"], loc='best')
plt.xlabel('Time (days)')
plt.ylabel('Recovered individuals')
plt.grid()
plt.show()


# In[2]:


# Simple reduction in R0 and running again

r = 0.2
names = []
while r < 0.7:

    xa = np.array([city_pop-eps, 0.0, eps, 0.0])

    t=0
    time = np.array(t)
    S_r = np.array(xa[0])
    E_r = np.array(xa[1])
    I_r = np.array(xa[2])
    R_r = np.array(xa[3])

    # # Remember: beta = R0/t_inf
    # Reduction in R0 -> reduction in beta
    # Percentage of reduction in R0 during quarantine

    beta_red = (1-r)*beta

    [Q_start_r, time, S_r, E_r, I_r, R_r] = time_loop(t_max, dt, xa, n, beta_red, sigma, gamma, city_pop, max_inf,
                 S_r, E_r, I_r, R_r, reach = False)

    plt.plot(time, I_r)
    names.append("{:.1f}".format(r))
    r+= 0.2

names.append("No reduction")
plt.plot(time, I, 'r-')
plt.legend(names, loc='best')
plt.xlabel('Time (days)')
plt.ylabel('Infected individuals')
plt.grid()
plt.show()


# Quarantine period
t_max_Q = 200
xa = np.array([city_pop-eps, 0.0, eps, 0.0])
x_seir = np.zeros(n)


print(40*'=')
print("Initial population [S, E, I, R] = ", xa)
            # Setting time counter and time arrays

# These are for plotting and data purposes
t=0
time = np.array(t)
S_Q = np.array(xa[0])
E_Q = np.array(xa[1])
I_Q = np.array(xa[2])
R_Q = np.array(xa[3])


# Remember: beta = R0/t_inf
# Reduction in R0 -> reduction in beta
Q_dur = 30
Q_stop = Q_start + Q_dur

# Percentage of reduction in R0 during quarantine
R_red = 0.7



[time, S_Q, E_Q, I_Q, R_Q] = time_loop(t_max_Q, dt, xa, n, beta, sigma, gamma, city_pop, max_inf,
                              S_Q, E_Q, I_Q, R_Q, R_red, Q_start, Q_dur, reach=True, Quarantine=True)

plt.plot(time, S_Q, 'r-')
plt.plot(time, E_Q, 'b-')
plt.plot(time, I_Q, 'g-')
plt.plot(time, R_Q, 'k--')

plt.axvspan(Q_start, Q_stop, facecolor='g', alpha=0.25)

plt.xlabel('Time (days)')
plt.ylabel('Individuals')
plt.grid()
plt.legend(["S", "E", "I", "R"], loc='best')
plt.show()


# In[4]:


plt.plot(time, I_Q, 'g-')
plt.axvline(x = Q_start, linewidth=2, color='r', ls='--')

plt.axvspan(Q_start, Q_stop, facecolor='g', alpha=0.25)

plt.xlabel('Time (days)')
plt.ylabel('Infected individuals')
plt.grid()
plt.show()


# In[5]:


plt.plot(time, I_Q, 'g-')
plt.plot(time[:int(t_max/dt)+2], I, 'r-')
plt.legend([f"R0 reduced {R_red*100}%", "No Quarantine"], loc='best')
plt.axvline(x = Q_start, linewidth=2, color='r', ls='--')

plt.axvspan(Q_start, Q_stop, facecolor='g', alpha=0.25)

plt.xlabel('Time (days)')
plt.ylabel('Infected individuals')
plt.grid()
plt.show()


# In[ ]:




