#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

# Initialize with a time vector to run the whole simulation

def f(x, beta, sigma, gamma):
    
    fx = np.zeros(4)
    
    N = np.sum(x)
    
    fx[0] = -(beta*x[2]*x[0])/N
    fx[1] = (beta*x[2]*x[0])/N - sigma*x[1]
    fx[2] = sigma*x[1] - gamma*x[2]
    fx[3] = gamma*x[2]
    
    return fx
    


def RK4(xa, dt, beta, sigma, gamma):
    
    k1=np.zeros(4)
    k2=np.zeros(4)
    k3=np.zeros(4)
    k4=np.zeros(4)
    F=np.zeros(4)
    x=np.zeros(4)
    
    F = f(xa, beta, sigma, gamma)
    
    for i in range(4):
        k1[i] = F[i]
        x[i] = xa[i] + 0.5*dt*k1[i]
        
    F = f(x, beta, sigma, gamma)
        
    for i in range(4):
        k2[i] = F[i]
        x[i] = xa[i] + 0.5*dt*k2[i]
        
    F = f(x, beta, sigma, gamma)
    
    for i in range(4):
        k3[i] = F[i]
        x[i] = xa[i] + dt*k3[i]
        
    F = f(x, beta, sigma, gamma)
    
    for i in range(4):
        k4[i] = F[i]
        x[i] = xa[i] + (dt/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])
        xa[i] = x[i]
    
    
    return x

def time_loop(time, xa, beta, sigma, gamma):
    
    x_seir = np.zeros(4)
    dt = time[1] - time[0]
    
    S = np.array(xa[0])
    E = np.array(xa[1])
    I = np.array(xa[2])
    R = np.array(xa[3])
    ########### Start of time loop
    for t in time:

        x_seir = RK4(xa, dt, beta, sigma, gamma)

        xa = x_seir

        S = np.append(S, x_seir[0])
        E = np.append(E, x_seir[1])
        I = np.append(I, x_seir[2])
        R = np.append(R, x_seir[3])
    ########### End of time loop

    return S, E, I, R

def seir_model(t_max, S, E, I, R, beta, sigma, gamma):
    
    x_seir = np.zeros(4)
    dt = 0.1
    t = 0
    
    xa = np.zeros(4)
    xa[0] = S
    xa[1] = E
    xa[2] = I
    xa[3] = R
    
    
    ########### Start of time loop
    while t <= t_max:

        x_seir = RK4(xa, dt, beta, sigma, gamma)

        xa = x_seir
        
        t += dt
        

    ########### End of time loop
    return x_seir[2] + x_seir[3]
#     return x_seir