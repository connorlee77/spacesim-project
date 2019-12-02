#!/usr/bin/env python3

import numpy as np
import scipy as sp
import math as mt
import sympy as sp

import matplotlib.pyplot as plt
import SensorModel as sensor
import SpacecraftSim3DOF as dyn

"""
Discrete EKF Implementation of the Spacecraft Simulator 

Given : Control Input and Model Parameters
        and sensor model

        All inputs are outputs are array data types
"""

param = np.array([1,1,1,1])

# Sensor Matrix 
def SensorMatrix(x):

    state = np.reshape(x,(6,1))

    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1]])

    Hx = np.mat(H)*np.mat(state)

    return np.array(Hx) 


# Propagation Step

def state_prop(x,dt,u,param):
    # Process Noise 
    """
    Sigma = np.zeros((6,6))
    Sigma[0,0] = 0.005
    Sigma[1,1] = 0.005
    Sigma[2,2] = 0.005
    Sigma[3,3] = 0.005
    Sigma[4,4] = 0.005
    Sigma[5,5] = 0.005
    """

    # Propagation Example Discrete Dynamics

    xp_dummy = dyn.SS3dofDyn(x,u,param, predict_fricfunc=False)
    xp_dummy = dyn.SS3dofDyn(x,u,param, predict_fricfunc=True)
    xp = dyn.EulerInt(xp_dummy,dt,x) 
    # + Sigma*np.random.randn(6,1)

    return xp


def covar_prop(P,Q,x,u,dt,param):

    # Q is process Covariance

    F = dyn.LinearDyn(x,u,param,dt)

    Pn = np.mat(F)*np.mat(P)*np.mat(np.transpose(F)) + np.mat(Q)

    return np.array(Pn)

# Update Step

def Kalmangain(P,H,R):
    
    # R Sensor Covariance 

    S = np.mat(H)*np.mat(P)*np.mat(np.transpose(H)) + np.mat(R)

    Inv =  np.linalg.inv(np.array(S)) 

    K = np.mat(P)*np.mat(np.transpose(H))*np.mat(Inv)    

    return np.array(K)


def state_update(x,y,H,K):


    state = np.reshape(x,(6,1))

    E = np.mat(y) - np.mat(H)*np.mat(state)

    X = state + np.mat(K)*E

    return np.reshape(np.array(X),(6))


def covar_update(P,H,K):

    Pn = (np.mat(np.identity(6))- np.mat(K)*np.mat(H))*np.mat(P)

    return np.array(Pn)

