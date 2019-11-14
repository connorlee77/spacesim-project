#!/usr/bin/env python3

import numpy as np
import scipy as sp
import math as mt
import matplotlib.pyplot as plt

"""
For open loop simulation of the 
3DOF dynamics of the spacecraft simulator 

Given : Control Input, Intial Condition and Model Parameters 

Euler Method for Time Integration
"""

def SS3dofDyn(x,u,param):
    m = param[0]
    I = param[1]
    l = param[2]
    b = param[3]

    state = x

    mi = 1/m
    lI = l/(2*I)
    bI = b/(2*I)

    B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])

    R = np.array([[mt.cos(state[2]),mt.sin(state[2]),0],[-mt.sin(state[2]),mt.cos(state[2]),0],[0,0,1]])

    F = np.mat(R)*np.mat(B)*np.mat(u)

    A = np.zeros([6,6])
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1
    
    # Dynamics 

    dxdt = np.mat(A)*np.mat(np.reshape(state,(6,1))) + np.concatenate((np.array([[0],[0],[0]]),F),axis=0)
    return np.reshape(np.array(dxdt),(6))


def EulerInt(dxdt,dt,xt):
    xt1 = xt + dxdt*dt   
    return xt1


def LinearDyn(x,u,param,dt):

    state = x
    m = param[0]
    I = param[1]
    l = param[2]
    b = param[3]

    mi = 1/m
    lI = l/(2*I)
    bI = b/(2*I)

    B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])
    dR = np.array([[-mt.sin(state[2]),mt.cos(state[2]),0],[-mt.cos(state[2]),-mt.sin(state[2]),0],[0,0,0]])

    u_dummy = np.mat(dR)*np.mat(B)*np.mat(u)

    A = np.zeros([6,6])

    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1
    A[3,2] = u_dummy[0]
    A[4,2] = u_dummy[1]

    F = np.identity(6) + np.array(A)*dt
    return F



if __name__ == "__main__":
    xinit = np.array([[1],[0],[0],[0],[0],[0]])
    x0 = np.reshape(xinit,(6))
    u = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
    param = np.array([1,1,1,1])
    t = np.linspace(0, 10, 100)

    n = len(t)
    x = np.zeros([6,n])
    x[:,0] = x0
    
    # Propagation Example Discrete Dynamics

    for i in range(0,n-1):
        dx = SS3dofDyn(x[:,i],u,param)
        x[:,i+1] = EulerInt(dx,t[1]-t[0],x[:,i])
    
    # plots 

    F = LinearDyn(x0,u,param,t[1]-t[0])

    print(F)

    plt.plot(t, x[0, :], 'b', label='x(t)')
    plt.plot(t, x[2, :], 'g', label='y(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()