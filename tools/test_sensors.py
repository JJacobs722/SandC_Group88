import cv2
import matplotlib.pyplot as plt
from src.sim.environment import Environment #loads environment blueprint
from src.sim import config #loads settings

env = Environment(seed=0) #build environment from blueprint
env.reset(x=1.6, y=0.0, yaw=0.0)   #place robot past the orange box, facing the dock

# ---- Camera + ArUco detection ----
img = env.get_camera_image() #take photo from robot camera
bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #swap colours to suite OpenCV
cv2.imwrite("camera_test.png", bgr) #save photo as "camera-test.png"

detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)) #creates a marker detector looking for ArUco marker
corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) #searches the photo for markers: first turns photo black and white: returns corners: pixel positions of each markers 4 corners, id (which marker number it found) '_' rejected shapes that looked a bit like markers (not needed)
print("Detected marker IDs:", None if ids is None else ids.ravel()) #prints which markers it found...none found: ids is none prints none... otherwise .ravel() flattens the result into a simple list like [0].
print("Camera K:\n", env.get_camera_intrinsics()) #prints the camera's K matrix.

# ---- LiDAR ----
angles, ranges = env.get_lidar()
print("LiDAR straight ahead (m):", ranges[config.LIDAR_BEAMS // 2]) #prints the distance for the beam pointing straight ahead.

ax = plt.subplot(projection="polar") #makes a polar graph
ax.plot(angles, ranges, ".", markersize=2) #plots each beam as a small dot
ax.set_title("LiDAR scan (robot at centre)") #title
plt.savefig("lidar_test.png") #saves graph
print("Saved camera_test.png and lidar_test.png")