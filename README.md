# STOC-Machine-projects
# This is optical flow code that will allow a drone to navigate through a space
# The drone will need to be able to detect rumbas so the sepcifics of the code allows for circle detection
# This was made for the St. Olaf Carleton Engineering Team 
# Optical.py is the most up to date
Most up-to-date version of code is Optical.py. It does not currently incorporate height detection and, as a result, real-world speed detection so velocity values are given in pixels/second. More testing needs to be done to decrease error in velocity results. Currently, real-world velocity detection would be dependent on two things; the accuracy of height detection software and physical specifications of the camrea. Physical specifications of the camera affect the FOV of the camera and thus the ratio of pixels to a given meter (height also affects this). Thus, for testing, this optical flow program should be limited to pixels/second to keep the testing in as a controlled environment as possible. 
