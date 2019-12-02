#!/usr/bin/env python3

import numpy as np
import scipy as sp

"""
Project: "Spacecraft Simulator" ACM154
Discrete Sensor Models that immitate : 1) Vicon and 2) IMU
"""
def Vicon(state):
    """
    State = (x,y,\Theta,\dot{x},\dot{y},\dot{\Theta})
    w.r.t to Inertial frame
    """
    num_state = len(state) 

    # total states have to be '6' for the code
    
    # Sensor Matrix
    H = np.identity(num_state)
    
    # Sensor Noise
    Sigma = np.zeros((6,6))
    Sigma[0,0] = 0.005
    Sigma[1,1] = 0.005
    Sigma[2,2] = 0.003
    Sigma[3,3] = 0.01
    Sigma[4,4] = 0.01
    Sigma[5,5] = 0.01
    
    # Sensor Output
    sensor_output = H*state + Sigma*np.random.randn(6,1)

    return sensor_output 


def IMU_3DOF(x0,x1,T):
    """
    6axis IMU 
    State = (x,y,Theta,dot{x},dot{y},dot{Theta})
    t0 is for earlier time 
    t1 is for current time 
    T = t1-t0 : time step between the readings
    
    We compute Angular velocity dot{Theta} with respect to the body frame
    and Acceleration (ddot x and ddot y) w.r.t body frame 
    Accel below are w.r.t inertial frame for simplicity
    """
    # num_state = len(state_t0)
    state_t1 = np.reshape(x1,(6))
    state_t0 = np.reshape(x0,(6))
    sigma_theta = 0.1
    sigma_x = 0.1
    sigma_y = 0.1
    dtheta = state_t1[5]  + sigma_theta*np.random.randn(1,1)
    accel_x = (state_t1[3] - state_t0[3])/T + sigma_x*np.random.randn(1,1)
    accel_y =  (state_t1[4] - state_t0[4])/T  + sigma_y*np.random.randn(1,1)

    vel_x = state_t1[3] + sigma_x*np.random.randn(1,1)
    vel_y = state_t1[4] + sigma_y*np.random.randn(1,1)

    theta = np.arctan2(accel_y,accel_x) + sigma_theta*np.random.randn(1,1)

    return np.reshape(np.array([[theta],[vel_x],[vel_y],[dtheta]]),(4,1))
    #return np.reshape(np.array([dtheta]),(1,1))

    
def GPS(state):
    """
    Gives absolute position with respect to an Inertial Frame 
    """
    sigma_x = 0.1 
    sigma_y = 0.1

    x = state[0] + sigma_x*np.random.randn(1,1)
    y = state[1] + sigma_y*np.random.randn(1,1)
    return np.reshape(np.array([[x],[y]]),(2,1))


if __name__ == "__main__":
    n = 6
    state1 = np.array([[1],[1],[1],[1],[1],[1]])
    state2 = np.array([[1],[0],[1],[0],[1],[0]])
    sensor = Vicon(state1)
    print(sensor)
    data = IMU_3DOF(state1,state2,0.01)
    print(data)



