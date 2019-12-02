#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:26:57 2019

@author: jspratt
"""

#!/usr/bin/env python3

import numpy as np
import scipy as sp
import math as mt
import sympy as sp

import matplotlib.pyplot as plt
import SensorModel_LIMITED as sensor
import SpacecraftSim3DOF as dyn
import SpacecraftSim3DOFEKF as ekf

np.random.seed(0)
def EKF(predictOrNone, t):
    param = np.array([1,1,1,1])
    # Initial Conditions for Dynamics
    xinit_dyn = np.array([[1],[0],[0],[0],[0],[0]])
    x0_dyn = np.reshape(xinit_dyn,(6))
    # control input
    u = np.array([[1],[0],[1],[0],[0],[0],[0],[0]])
    
    # Sensor Matrix

    #H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1]])

    #Intial Conditions for EKF

    # state
    xinit_ekf = np.array([[-3],[-3],[-1.3],[0],[0],[0]])
    x0_ekf = np.reshape(xinit_ekf,(6))
    
    # Covariance 
    #P = np.identity(6)*100000000
    P = np.identity(6)

    # Process Covariance 
    Q = np.identity(6)*0.1

    sig = np.zeros((3,3))

    sig[0,0] = 0.1
    sig[1,1] = 0.1
    sig[2,2] = 0.01
    #sig[3,3] = 0.1
    #sig[4,4] = 0.1
    #sig[5,5] = 0.1
    
    R = sig

    dt = t[1]-t[0]
    n = len(t)
    x_dyn = np.zeros([6,n])
    x_dyn[:,0] = x0_dyn

    x_ekf = np.zeros([6,n])
    x_ekf[:,0] = x0_ekf

    y_plot = np.zeros((3,100))

    for i in range(0,n-1):

        # dynamics 
        dx = dyn.SS3dofDyn(x_dyn[:,i],u,param, fric_func='true', dt=dt)
        x_dyn[:,i+1] = dyn.EulerInt(dx,dt,x_dyn[:,i])
        # Sensor Data Simulator 
        yg = sensor.GPS(x_dyn[:,i+1])
        ya = sensor.IMU_3DOF(x_dyn[:,i],x_dyn[:,i+1],dt)

        y = np.concatenate((yg,ya),axis=0)
        # EKF loop to estimate above dynamics
        #   # Propagation Step
        xu_ekf = ekf.state_prop(x_ekf[:,i],u,dt,param, fric_func=predictOrNone)
        Pu = ekf.covar_prop(P,Q,x_ekf[:,i],u,dt,param)
        #   # Update Step
        K = ekf.Kalmangain(Pu,H,R)
        x_ekf[:,i+1] = ekf.state_update(xu_ekf,y,H,K)
        P = ekf.covar_update(Pu,H,K)
        y_plot[:,i+1] = y.reshape((3))
    
    

    xerror = np.linalg.norm(x_ekf[0:2, :] - x_dyn[0:2,:], axis=0)
    xdoterror = np.linalg.norm(x_ekf[3:5, :] - x_dyn[3:5,:], axis=0)
    print(np.mean(xerror))
    print(np.mean(xdoterror))  

    return x_dyn, x_ekf, y_plot

"""
Simulation Using the Library 

"""
def main():

    t = np.linspace(0, 10, 100)
    x_dyn, x_ekf, y_plot = EKF('none', t)
    x_dyn_prd, x_ekf_prd, y_plot_prd = EKF('predict', t)

    ylabel = [r'$x$', r'$y$', r'$sin(\theta)$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$']
    for i in range(0,6):
        
        if i==2:
            plt.plot(t, np.sin(x_dyn[i, :]), 'b', label='Simulated Data')
            plt.plot(t, np.sin(x_ekf[i, :]), 'g', label='EKF Estimate')
        elif i == 0:
            plt.plot(t, y_plot[i, :],'rx',label='measurements')
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
        elif i == 1:
            plt.plot(t, y_plot[i, :],'rx',label='measurements')
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
        elif i == 5:
            plt.plot(t, y_plot[2, :],'rx',label='measurements')
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
        else:
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
        plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.ylabel(ylabel[i])
        plt.grid()  
        #plt.savefig('figures/' + str(i) + '.png')
        #plt.close()
        plt.show()

    for i in range(0,6):
        
        if i==2:
            plt.plot(t, np.sin(x_dyn[i, :]), 'b', label='Simulated Data')
            plt.plot(t, np.sin(x_ekf[i, :]), 'g', label='EKF Estimate')
            plt.plot(t, np.sin(x_ekf_prd[i, :]), 'r', label='EKF Estimate w/ lrn. func')
        elif i == 0:
            plt.plot(t, y_plot[i, :],'rx',label='measurements')
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
            plt.plot(t, x_ekf_prd[i, :], 'r', label='EKF Estimate w/ lrn. func')
        elif i == 1:
            plt.plot(t, y_plot[i, :],'rx',label='measurements')
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
            plt.plot(t, x_ekf_prd[i, :], 'r', label='EKF Estimate w/ lrn. func')
        elif i == 5:
            plt.plot(t, y_plot[2, :],'rx',label='measurements')
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
            plt.plot(t, x_ekf_prd[i, :], 'r', label='EKF Estimate w/ lrn. func')
        else:
            plt.plot(t, x_dyn[i, :], 'b', label='Simulated Data')
            plt.plot(t, x_ekf[i, :], 'g', label='EKF Estimate')
            plt.plot(t, x_ekf_prd[i, :], 'r', label='EKF Estimate w/ lrn. func')
        plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.ylabel(ylabel[i])
        plt.grid()  
        #plt.savefig('figures/' + str(i) + 'prd.png')
        #plt.close()
        plt.show()

if __name__ == "__main__":
    main()
