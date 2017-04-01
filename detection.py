import cv2
import numpy as np
from collections import deque
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default = 32, help="max buffer size")
args = vars(ap.parse_args())
# Initializing the optical flow portion
lk_params = dict( winSize = (15, 15), maxLevel = 2,
                  criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 50, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
track_len = 10
detect_interval = 5
tracks = []
frame_idx = 0
# lower and upper limits for black (the color of the rumba)
black_lower = (0,0,0)
black_upper = (40, 40, 40)
#initialize tracker points and coordinate deltas
pts = deque(maxlen=args["buffer"])
print(pts)
counter = 0
(dX, dY) = (0,0)
direction = ""
# Initializing circle tracking portion
circles = None
cbox = (297,297,30,30)
# Tracker for tracking the circles (uses KCF tracking)
tracker = cv2.Tracker_create("KCF")

# Startup of camera, has failsafe in case nothing is returned from cv2.VideoCatpture
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

ret = tracker.init(frame, cbox)
while(True):
    ret, frame = cap.read()
    # blurs the frame and converts it to the hsv color space
    """
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # constructs a mask for the color black and then removes any blobs
    mask = cv2.inRange(hsv, black_lower, black_upper)
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
            #check if points accumulated in buffer
            if counter >= 10 and i == 1 and pts[-10] is not None:
                # get difference between x and y and re-initialize the
                # direction text var
                dX = pts[-10][0] - pts[i][0]
                dY = pts[-10][1] - pts[i][1]
                (dirX, dirY) = ("", "")
                #make sure there's actually movement in x and y direcitons
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dY) == 1 else "West"
                if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"

                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

                else:
                    direction = dirX if dirX != "" else dirY

                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
                
            cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255),3)
            cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY), (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0,0,255), 1)
    
    """            
    # converts frame into single channel image for HoughCircle detection
    thresh = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    #ret, thresh = cv2.threshold(gray_frame,20,255,cv2.THRESH_BINARY_INV)
    if circles is None:
        # returns a list of circles detected with [x, y, radius]
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 200, param1 = 50, param2 = 30,
                                minRadius = 100, maxRadius = 10000)
        continue
    #selects a single circle
    circle = circles[0][0]
    #print(circle)
    #cbox = (circle[0]-10, circle[1]-10, circle[0]+10, circle[1]+10)
    #the first circle in the circles list will be tracked
    ret, cbox = tracker.update(frame)
    #the object has been lost, reset to find a circle

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 200, param1 = 50, param2 = 30,
                            minRadius = 100, maxRadius = 10000)
    if len(tracks) > 0:
        img0, img1 = prev_thresh, thresh
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        dist = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = dist < 1
        new_tracks = []
        
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1,2), good):
            if not good_flag:
                continue
            tr.append((x,y))
            
            if len(tr) > track_len:
                del tr[0]
                
            new_tracks.append(tr)
            cv2.circle(frame, (x, y), 2, (0,255,0), -1)
            
        tracks = new_tracks
        cv2.polylines(frame, [np.int32(tr) for tr in tracks], False, (255, 0, 0))
        for i in range(len(tracks)-1):
            cv2.line(thresh, (tracks[i][0][0], tracks[i][0][1]),
                     (tracks[i][1][0], tracks[i][1][1]), (0,255,0), 1)
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(thresh)
        mask[:] = 255        
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x,y), 5, 0, -1)
                
        p = cv2.goodFeaturesToTrack(thresh, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1,2):
                tracks.append([(x,y)])

    frame_idx += 1
    prev_thresh = thresh
                  
            
    if circles is None:
        cv2.imshow("Circle Flow", frame)
        continue
    for i in circles[0,:]:
        #p3 = (int(i[0]-10), int(i[1]-10))
        #p4 = (int(i[0]+10), int(i[1]+10))
        p3 = (int(cbox[0]), int(cbox[1]))
        p4 = (int(cbox[0]+cbox[2]), int(cbox[1]+cbox[3]))
        cv2.rectangle(frame, p3, p4, (0,0,255))
        cv2.circle(frame, (i[0],i[1]), i[2], (0,255,0),2)
        cv2.circle(frame, (i[0],i[1]),2,(0,0,255),3)
        cv2.imshow("Circle Flow", frame)
        counter += 1
        
    cv2.imshow("frame", frame)
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
            
cap.release()
cv2.destroyAllWindows()
