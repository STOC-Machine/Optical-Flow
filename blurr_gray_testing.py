import cv2
import numpy as np
cv2.namedWindow("Blurr & Grayscale testing")
cam = cv2.VideoCapture(0)
if cam.isOpened():
    ret, frame = cam.read()
else:
    ret = False
while(True):
    ret, frame = cam.read()
    img = cv2.bilateralFilter(frame, 15, 75, 75)
    imgg = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY )
    cv2.imshow("Blur Testing", img)
    cv2.imshow("No Blur", frame)
    cv2.imshow("Color w/o Blur", imgg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
