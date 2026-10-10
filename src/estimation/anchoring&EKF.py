#project estimattion goes here :D


#base layout of code

#set up anchors through the charging station (?)
#Setup EKF fusing wheel-encoder odometry
    #includes range and bearing based on visual placement of anchor (x,y,theta)

#wheel odometry tells you how far the robot has travelled through the encoder :P
#Begin with an estimation (guess) then be like okay well how can I recover from this which is what
#the EKF is for 

#imports taken directly from assessment 2 file!
import numpy as np
import os
import sys
import time
import zipfile
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# Machine Vision Toolbox (RVC3)
from machinevisiontoolbox import Image, CentralCamera, PointCloud

# Spatial Maths: rigid-body transforms
from spatialmath import SE3
import spatialmath.base as smb

import cv2
import open3d as o3d


print("Hello test")