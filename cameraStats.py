import cv2
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
height, width, channels =  frame.shape
print(width)
print(height)
print(channels)