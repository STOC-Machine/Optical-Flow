import cv2
import numpy as np
# This program detects circles using OpenCV's HoughCircles function
cv2.namedWindow("Circle Detection")
# initializes camera 
cam = cv2.VideoCapture(0)
if cam.isOpened():
    ret, frame = cam.read()
else:
    ret = False
while(True):
    ret, frame = cam.read()
    # blurrs and turns frame into single-channel image that can be
    # used as an input for HoughCircles
    img = cv2.medianBlur(frame,5)
    # bad at detecting black circles, fix color space (invert?) if we want to detect black
    # or darker circles
    imgg = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # intakes a 8-bit single-channel image and returns a list of circles
    # of the form [x, y, radius]
    circles = cv2.HoughCircles(imgg,cv2.HOUGH_GRADIENT,1,75,
                               param1=40,param2=87, minRadius=1,maxRadius=400)
    

    # draws the circles on the frame
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    
    cv2.imshow("Circle Detection",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

