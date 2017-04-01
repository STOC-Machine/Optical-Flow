# Author: Malcolm Grossman
# This code looks for an object of certain color(s) on the screen and tracks it
# The 
import cv2
import numpy as np
from collections import deque
import argparse
# constructs and parses pare arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
# lower and upper limits for black (the color of the rumba)
color_lower = (29,86,6)
color_upper = (64, 255, 255)
#initialize tracker points and coordinate deltas
pts = deque(maxlen=args["buffer"])


# Startup of camera, has failsafe in case nothing is returned from cv2.VideoCatpture
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

while(True):
    ret, frame = cap.read()
    # blurs the frame and converts it to the hsv color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # constructs a mask for the color black and then removes any blobs
    mask = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)

    # find the rumba and initialize the current (x,y) center
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum
        # enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"] / M["m00"]))

        # radius needs to be a certain size
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255),2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    pts.appendleft(center)
    # loop over tracked points
    for i in np.arange(1, len(pts)):
        # if tracked points are None ignore them
        if pts[i-1] is None or pts[i] is None:
            continue
        #draw the lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
                
       
    cv2.imshow("Color Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    
