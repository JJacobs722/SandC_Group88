import cv2

#ArUco ID 0 from the 4x4 dictionary, with a white border so it can be detected
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) #Picks marker with 4x4 grid of squares from a library of 50 possible markers (must be used amongst everyone)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 400) #Draw marker number 0, 400 pixels wide
marker = cv2.copyMakeBorder(marker, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255) #Add a white frame around it, 50 pixels on each side.
marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR) #Converts to colour image
cv2.imwrite("assets/robotis_tb3/aruco_0.png", marker) #Saves picture into same folder as world file, so world can find it
print("Saved assets/robotis_tb3/aruco_0.png") #Print that its saved.