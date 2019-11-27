#!/usr/bin/env python3

import numpy as np
import scipy as sp
import math as mt
import sympy as sp
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

import matplotlib.pyplot as plt
import SensorModel as sensor
import SpacecraftSim3DOF as dyn
import SpacecraftSim3DOFUKF as ukf

"""
Simulation Using the Library 

"""
def main():

    param = np.array([1,1,1,1])
    # Initial Conditions for Dynamics
    xinit_dyn = np.array([[1],[0],[0],[0],[0],[0]])
    x0_dyn = np.reshape(xinit_dyn,(6))
    # control input
<<<<<<< HEAD
    u = np.array([[1],[0],[1],[0],[0],[0],[0],[0]])
=======
    u = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
>>>>>>> 08705f9e00777eb4ef941cdf5922c4b47eeb5c98

    Sigma = np.zeros((6,6))
    Sigma[0,0] = 0.005
    Sigma[1,1] = 0.005
    Sigma[2,2] = 0.005
    Sigma[3,3] = 0.005
    Sigma[4,4] = 0.005
    Sigma[5,5] = 0.005
    
    # Sensor Matrix

    #H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1]])
    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

    #Intial Conditions for UKF

    # state
<<<<<<< HEAD
    xinit_ukf = np.array([[-3],[-3],[-1.3],[0.1],[0.1],[1]])
=======
    xinit_ukf = np.array([[-3],[-3],[0.3],[0],[0],[0]])
>>>>>>> 08705f9e00777eb4ef941cdf5922c4b47eeb5c98
    #xinit_ukf = np.array([[1],[0],[0],[0],[0],[0]])
    x0_ukf = np.reshape(xinit_ukf,(6))
    
    # Covariance scp 

    P = np.identity(6)*100000000

    # Process Covariance 

    #Q = np.identity(6)*0.05
    Q = np.identity(6)*1
    

    sig = np.zeros((6,6))

    sig[0,0] = 0.1
    sig[1,1] = 0.1
    sig[2,2] = 0.01
    sig[3,3] = 0.1
    sig[4,4] = 0.1
    sig[5,5] = 0.1

    R = sig

<<<<<<< HEAD
    t = np.linspace(0, 10, 100)
=======
    t = np.linspace(0, 20, 200)
>>>>>>> 08705f9e00777eb4ef941cdf5922c4b47eeb5c98

    dt = t[1]-t[0]

    n = len(t)
    x_dyn = np.zeros([6,n])
    x_dyn[:,0] = x0_dyn

    x_ukf = np.zeros([6,n])
    x_ukf[:,0] = x0_ukf

    # UKF
    
    y=[]
    # Simulated data
    for i in range(0,n-1):
<<<<<<< HEAD
        dx = dyn.SS3dofDyn(x_dyn[:,i],u,param, predict_fricfunc=True)
=======
        dx = dyn.SS3dofDyn(x_dyn[:,i],u,param, predict_fricfunc=False)
>>>>>>> 08705f9e00777eb4ef941cdf5922c4b47eeb5c98
        x_dyn[:,i+1] = dyn.EulerInt(dx,dt,x_dyn[:,i])
        yg = sensor.GPS(x_dyn[:,i+1])
        ya = sensor.IMU_3DOF(x_dyn[:,i],x_dyn[:,i+1],dt)
        
        y.append(np.concatenate((yg,ya),axis=0))
    
    alpha = 0.001
    beta = 2
    kappa = 0
    
    #def fx(x,dt):
    #    return ukf.state_prop(x,u,dt,param)
    
    def hx(x):
        return np.dot(H,x)
    
    sig_points = MerweScaledSigmaPoints(6, alpha = alpha, beta=beta, kappa=kappa)
    
    kf = UnscentedKalmanFilter(dim_x=6,dim_z=6,dt=dt,fx=ukf.state_prop,hx=hx,points=sig_points)
    kf.x = x0_ukf.reshape((6))
    kf.P = P
    kf.R = R
    kf.Q = Q
    
    zs = y
    print(zs[0])
    
    xs = x_ukf
    for idx, z in enumerate(zs):
        z=z.reshape((6))
        kf.predict(u=u,param=param)
        kf.update(z)
        xs[:,idx+1] = kf.x      
        
        
<<<<<<< HEAD
    xerror = np.linalg.norm(xs[0:2,:]-x_dyn[0:2,:],axis=0)
    xdoterror = np.linalg.norm(xs[3:5,:]-x_dyn[3:5,:],axis=0)
    print(np.mean(xerror))
    print(np.mean(xdoterror))
    
    plt.figure(0)
    plt.plot(t, x_dyn[0, :], 'b', label='Simulated Data')
    plt.plot(t, xs[0, :], 'g', label='UKF Estimate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.show()
    
    plt.figure(1)
    plt.plot(t, x_dyn[1, :], 'b', label='Simulated Data')
    plt.plot(t, xs[1, :], 'g', label='UKF Estimate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.show()
    
    sintheta_dyn = np.sin(x_dyn[2,:])
    sintheta_ukf = np.sin(xs[2,:])    
    
    plt.figure(2)
    plt.plot(t, sintheta_dyn, 'b', label='Simulated Data')
    plt.plot(t, sintheta_ukf, 'g', label='UKF Estimate')
    plt.legend(loc='upper left')
    plt.xlabel('t')
    plt.ylabel(r'$sin(\theta)$')
    plt.grid()
    plt.show()
    
    plt.figure(3)
    plt.plot(t, x_dyn[3, :], 'b', label='Simulated Data')
    plt.plot(t, xs[3, :], 'g', label='UKF Estimate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel(r'$\dot{x}$')
    plt.grid()
    plt.show()
    
    plt.figure(4)
    plt.plot(t, x_dyn[4, :], 'b', label='Simulated Data')
    plt.plot(t, xs[4, :], 'g', label='UKF Estimate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel(r'$\dot{y}$')
    plt.grid()
    plt.show()
    
    plt.figure(5)
    plt.plot(t, x_dyn[5, :], 'b', label='Simulated Data')
    plt.plot(t, xs[5, :], 'g', label='UKF Estimate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel(r'$\dot{\theta}$')
    plt.grid()
=======
    plt.plot(x_dyn[4, 2:])
    plt.plot(xs[4, 2:])
>>>>>>> 08705f9e00777eb4ef941cdf5922c4b47eeb5c98
    plt.show()


if __name__ == "__main__":
    main()
