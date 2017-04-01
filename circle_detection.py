import cv2
import numpy as np

cv2.namedWindow("Circle Detection")
cam = cv2.VideoCapture(0)
if cam.isOpened():
    ret, frame = cam.read()
else:
    ret = False
while(True):
    ret, frame = cam.read()
    img = cv2.medianBlur(frame,5)
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    circles = cv2.HoughCircles(imgg,cv2.HOUGH_GRADIENT,1,75,
                               param1=15,param2=75, minRadius=1,maxRadius=100)
    print(circles)

    
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

