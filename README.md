# STOC-Machine-projects
# Description
This code was written to work in tandem with other St. Olaf/Carleton Engineering Team projects.
The final goal of this project is to run this code on a drone in order to aid in autonomous movement.
Calculates speed of objects using OpenCV's optical flow library and point detection functions.
Eventually will be used in tandem with neural networks to track the speed of roombas.
# Usage
Currently code operates on its own but will ideally work with neural networks and height detection in order to
output velocity of roomba movement in m/s.
# Files/Folders
test_inputs: contains all inputs and input creators that will be used to test optical flow and circle detection.

circle_detection.py: detects circles using OpenCV's HoughCircles function

GridSquares.py: I couldn't figure out how to call code from another GitHub repository so I copied this code: https://github.com/STOC-Machine/vision.
This code outputs the height of the camera above a grid.

*** RUN THIS CODE TO GET OPTICAL FLOW DATA ***
optical_flow.py: Uses Lucas-Kanade method of optical flow to calculate the speed of objects moving across the screen. Currently
returns velocities in pixels/second.

