#project estimattion goes here :D


#base layout of code

#combining the wheel turns with whats seen by the camera (the aruco marker)
#Setup EKF fusing wheel-encoder odometry
    #includes range and bearing based on visual placement of aruco marker (x,y,theta)

#wheel odometry tells you how far the robot has travelled through the encoder :P
#Begin with an estimation (guess) then be like okay well how can I recover from this which is what
#the EKF is for 

#NO GROUND TRUTH EVER

#imports taken directly from assessment 2 file!
import numpy as np
import os
import sys
import time
import zipfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import control as ct

from sim.environment import Environment #connection environment class from environment file
from sim.config import ENCODER_NOISE, LIDAR_NOISE


def Odometry_readings():
    env = Environment() #env object
    theta_readings = env.get_wheel_angles() #i want wheel angles thx

def Recursive_filter(): #used to predict&correct on x hat and p every step
    #Kf = lqe(A, B, Qd, Rn)
    A = 
    B = 
    Qd = 
    Rn = 
    KF = ct.lqe(A, B, Qd, Rn)
