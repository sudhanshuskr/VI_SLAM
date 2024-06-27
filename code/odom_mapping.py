import numpy as np
from load_data import *
D_PER_TIC = 0.0022 #m
V_SUBSAM = 2
W_SUBSAM = 2
LID_SUBSAM = 3

from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y





def nextState(x_t,v_t,w_t,tau_t,theta_t):
    delx = np.array([[(v_t*np.sinc((w_t*tau_t)/2)*np.cos(theta_t + ((w_t*tau_t)/2)))],
                     [(v_t*np.sinc((w_t*tau_t)/2)*np.sin(theta_t + ((w_t*tau_t)/2)))],
                     [np.zeros(len(theta_t)) + w_t]]).reshape(3,len(theta_t))
    x_t = x_t + (tau_t*delx)
    return x_t


def pos2velocity(counts = encoder_counts,timestep = encoder_stamps):
    [fr,fl,rr,rl] = counts
    vr = ((fr + rr)/2)*D_PER_TIC
    vl = ((fl + rl)/2)*D_PER_TIC
    vel = (vr + vl)/2
    vel = vel[0:-1]
    tf = timestep[1:]
    ti = timestep[0:-1]
    t = tf - ti
    vel = np.divide(vel,t)
    return vel,timestep[0:-1]

def downsample_vel(vel,timestep):
    return vel[::V_SUBSAM],timestep[::V_SUBSAM]

    
    
'''
print(encoder_counts)
print(encoder_stamps)
     
print(encoder_counts.shape)
print(encoder_stamps.shape)

vel,t = pos2velocity(encoder_counts,encoder_stamps)
    
print(vel)
print(t)
     
print(vel.shape)
print(t.shape)
'''