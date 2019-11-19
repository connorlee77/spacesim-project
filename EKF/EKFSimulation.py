#!/usr/bin/env python3

import numpy as np
import scipy as sp
import math as mt
import sympy as sp

import matplotlib.pyplot as plt
import SensorModel as sensor
import SpacecraftSim3DOF as dyn
import SpacecraftSim3DOFEKF as ekf

"""
Simulation Using the Library 

"""
def main():

    param = np.array([1,1,1,1])
    # Initial Conditions for Dynamics
    xinit_dyn = np.array([[1],[0],[0],[0],[0],[0]])
    x0_dyn = np.reshape(xinit_dyn,(6))
    # control input
    u = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])

    Sigma = np.zeros((6,6))
    Sigma[0,0] = 0.005
    Sigma[1,1] = 0.005
    Sigma[2,2] = 0.005
    Sigma[3,3] = 0.005
    Sigma[4,4] = 0.005
    Sigma[5,5] = 0.005
    
    # Sensor Matrix

    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])



    #Intial Conditions for EKF

    # state
    xinit_ekf = np.array([[-3],[-3],[0.3],[0],[0],[0]])
    x0_ekf = np.reshape(xinit_ekf,(6))
    
    # Covariance 

    P = np.identity(6)*100000000

    # Process Covariance 

    Q = np.identity(6)*1

    sig = np.zeros((6,6))

    sig[0,0] = 0.1
    sig[1,1] = 0.1
    sig[2,2] = 0.01
    sig[3,3] = 0.1
    sig[4,4] = 0.1
    sig[5,5] = 0.1
    
    R = sig

    t = np.linspace(0, 20, 200)

    dt = t[1]-t[0]

    n = len(t)
    x_dyn = np.zeros([6,n])
    x_dyn[:,0] = x0_dyn

    x_ekf = np.zeros([6,n])
    x_ekf[:,0] = x0_ekf



    for i in range(0,n-1):
        # dynamics 
        dx = dyn.SS3dofDyn(x_dyn[:,i],u,param, predict_fricfunc=False)
        x_dyn[:,i+1] = dyn.EulerInt(dx,dt,x_dyn[:,i])
        # Sensor Data Simulator 
        yg = sensor.GPS(x_dyn[:,i+1])
        ya = sensor.IMU_3DOF(x_dyn[:,i],x_dyn[:,i+1],dt)

        y = np.concatenate((yg,ya),axis=0)
        # EKF loop to estimate above dynamics
        #   # Propagation Step
        xu_ekf = ekf.state_prop(x_ekf[:,i],u,dt,param)
        Pu = ekf.covar_prop(P,Q,x_ekf[:,i],u,dt,param)
        #   # Update Step
        K = ekf.Kalmangain(Pu,H,R)
        x_ekf[:,i+1] = ekf.state_update(xu_ekf,y,H,K)
        P = ekf.covar_update(Pu,H,K)

    plt.plot(t, x_dyn[4, :], 'b', label='x_dyn(t)')
    plt.plot(t, x_ekf[4, :], 'g', label='x_ekf(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()     


if __name__ == "__main__":
    main()
