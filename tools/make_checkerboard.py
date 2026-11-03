"""Draw a checkerboard picture for camera calibration.

10 x 7 squares (so 9 x 6 inner corners where four squares meet), plus a white border.
On the board in the simulation each square is 2.5 cm.
"""
import cv2
import numpy as np

SQUARES_X, SQUARES_Y, PX = 10, 7, 60     # squares across, squares down, pixels per square in the picture
board = np.zeros((SQUARES_Y * PX, SQUARES_X * PX), np.uint8)   # start all black
for i in range(SQUARES_Y):
    for j in range(SQUARES_X):
        if (i + j) % 2 == 0:                                   # every other square turns white
            board[i*PX:(i+1)*PX, j*PX:(j+1)*PX] = 255
board = cv2.copyMakeBorder(board, PX, PX, PX, PX, cv2.BORDER_CONSTANT, value=255)   # one square of white border
cv2.imwrite("assets/robotis_tb3/checkerboard.png", cv2.cvtColor(board, cv2.COLOR_GRAY2BGR))
print("Saved assets/robotis_tb3/checkerboard.png")