#project estimattion goes here :D

import numpy as np

#base layout of code

#set up anchors through the charging station (?)
#Setup EKF fusing wheel-encoder odometry
    #includes range and bearing based on visual placement of anchor (x,y,theta)

#wheel odometry tells you how far the robot has travelled through the encoder :P
#Begin with an estimation (guess) then be like okay well how can I recover from this which is what
#the EKF is for 