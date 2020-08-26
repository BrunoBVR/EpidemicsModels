#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

def f(n, x, beta, sigma, gamma, N):
    
    fx = np.zeros(n)
    
    fx[0] = -(beta*x[2]*x[0])/N
    fx[1] = (beta*x[2]*x[0])/N - sigma*x[1]
    fx[2] = sigma*x[1] - gamma*x[2]
    fx[3] = gamma*x[2]
    
    return fx
    


def RK4(xa, n, dt, beta, sigma, gamma, N):
    
    k1=np.zeros(n)
    k2=np.zeros(n)
    k3=np.zeros(n)
    k4=np.zeros(n)
    F=np.zeros(n)
    x=np.zeros(n)
    
    F = f(n, xa, beta, sigma, gamma, N)
    
    for i in range(n):
        k1[i] = F[i]
        x[i] = xa[i] + 0.5*dt*k1[i]
        
    F = f(n, x, beta, sigma, gamma, N)
        
    for i in range(n):
        k2[i] = F[i]
        x[i] = xa[i] + 0.5*dt*k2[i]
        
    F = f(n, x, beta, sigma, gamma, N)
    
    for i in range(n):
        k3[i] = F[i]
        x[i] = xa[i] + dt*k3[i]
        
    F = f(n, x, beta, sigma, gamma, N)
    
    for i in range(n):
        k4[i] = F[i]
        x[i] = xa[i] + (dt/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
        xa[i] = x[i]
    
    
    return x

def time_loop(t_max, dt, xa, n, beta, sigma, gamma, city_pop, max_inf,
             S, E, I, R, R_red = 0, Q_start = 0, Q_dur = 0, reach = False, Quarantine = False):
    
    t = 0

    time = np.array(t)
    
    x_seir = np.zeros(n)
    
    ########### Start of time loop
    while t < t_max:
        t += dt

    #     print(f"time: {t}", end='\r')
    #     print(t)

        if not Quarantine:
            x_seir = RK4(xa, n, dt, beta, sigma, gamma, city_pop)
        
        elif Quarantine:
            if t >= Q_start and t < (Q_start+Q_dur):
                x_seir = RK4(xa, n, dt, (1 - R_red)*beta, sigma, gamma, city_pop)

            else:
                x_seir = RK4(xa, n, dt, beta, sigma, gamma, city_pop)
                
        if x_seir[2] >= max_inf and reach == False:
            Q_start = t
            print(f'Days to reach {max_inf}: {t}')
            reach = True


        xa = x_seir

        time = np.append(time, t)
        S = np.append(S, x_seir[0])
        E = np.append(E, x_seir[1])
        I = np.append(I, x_seir[2])
        R = np.append(R, x_seir[3])
    ########### End of time loop
    if not Quarantine:
        return Q_start, time, S, E, I, R
    
    else:
        return time, S, E, I, R