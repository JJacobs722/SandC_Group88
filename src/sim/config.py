# All environment settings (here in one place)

MODEL_FILE = "assets/robotis_tb3/world.xml"

CONTROL_DT = 0.02 #s - controller runs at 50hz (robot thinks 50 times per sec --> reads sensors --> then decides wheel speeds)

CAMERA_NAME = "front_cam"
CAMERA_WIDTH = 640 #picture size
CAMERA_HEIGHT = 480

LIDAR_SITE = "lidar_site"
LIDAR_BEAMS = 360 #one laser beam per degree, all the way around
LIDAR_MAX_RANGE = 3.5 #m - same as real Turtlebot lidar

#sensor noise <-- how much random error to add to the readings (standard deviations) - for noise free run set to 0
ENCODER_NOISE = 0.002 #rad
LIDAR_NOISE = 0.01 #m

MARKER_ID = 0 #which marker to look for (marker 0) 
MARKER_SIZE = 0.20 #marker size (m) - only counting marker w/out white boarder