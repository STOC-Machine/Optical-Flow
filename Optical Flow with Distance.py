
import cv2
import numpy as np
import time
import GridSquares
import math

lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

feature_params = dict( maxCorners = 500, 
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        # speed in pixels/second
        self.velocitor = []
        # speed in centimeters/second
        self.velocity = []
        self.cam = cv2.VideoCapture(0)#'speed_detection3.mp4' 
        # camera is listed as having an FOV of 60 degrees
        # DFOV = Diagonal Field of View
        # Horizontal FOV = 2 * atan(tan(DFOV/2)*cos(atan(9/16)))
        # Vertical FOV = 2 * atan(tan(DFOV/2)*sin(atan(9/16)))
        # I am using these values halved so HFOV/2 and VFOV/2
        # sets the camera resolution to 1280 x 720 p MIGHT NOT WORK FOR ALL CAMERAS
        # using Logitech 720p webcam
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.DFOV = 60*(math.pi/180)
        self.HFOV = 2*math.atan(math.tan(self.DFOV/2)*math.cos(math.atan(9/16)))
        self.VFOV = 2*math.atan(math.tan(self.DFOV/2)*math.sin(math.atan(9/16)))
        self.frame_idx = 0
        self.calibration_values = None
        self.times = []

    def run(self):
        while True:
            ret, frame = self.cam.read()
            # gets square objects from Daniel's code. Using to get height of drone
            gridsquares = GridSquares.run(frame)
            squares = gridsquares[0]
            matrix = gridsquares[1]
            if self.calibration_values == None:
                self.calibration_values = cv2.calibrationMatrixValues(matrix, (1280,720),3,3)
                #print(self.calibration_values)
                self.HFOV = self.calibration_values[0]*(math.pi/180)
                self.VFOV = self.calibration_values[1]*(math.pi/180)
            img = cv2.medianBlur(frame, 5)
            t = time.clock()
            self.times.append(t)
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            vis = frame.copy()
            # make more precise parameters at competition to account for lighting and other variables
            circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 75,
                                       param1=45, param2=87, minRadius=1,maxRadius=300)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1,1,2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1,2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x,y), good_flag in zip(self.tracks, p1.reshape(-1,2),good):
                    if not good_flag:
                        continue
                    tr.append((x,y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    if len(self.velocitor) > self.track_len:
                        del self.velocitor[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                for i in range(len(self.tracks)-1):
                    #print(self.tracks)
                    tracklen = len(self.tracks[i])-1
                    x1=self.tracks[i][tracklen][0]-self.tracks[i][0][0]
                    y1=self.tracks[i][tracklen][1]-self.tracks[i][0][1]
                    self.velocitor.append([x1/(self.times[self.frame_idx-1]-self.times[self.frame_idx]),y1/(self.times[self.frame_idx-1]-self.times[self.frame_idx])])
                    cv2.line(vis, (self.tracks[i][0][0], self.tracks[i][0][1]), (self.tracks[i][1][0], self.tracks[i][1][1]), (0,255,0),1)
                    #print(self.velocitor[i])
                if squares != []:
                    # first value in squares is the most accurate
                    # Not fully working, check with David
                    height = -squares[0].location[2]
                    #print(height)
                    # x is the distance from one side of the screen to the other in cm
                    # y is the distance from the top to the bottom of the screen in cm
                    # x = 2*height*tan(HFOV/2)
                    # y = 2*height*tan(VFOV/2)
                    # vertical FOV = 15.304295 degrees for this camera
                    # horizaontal FOV = 26.71174 degrees for this camera
                    x = 2*height*math.tan(self.HFOV)
                    y = 2*height*math.tan(self.VFOV)
                    for j in range(len(self.velocitor)-1):
                        #velocity = (x(cm)/x(pxls))*velocity(pxls/s)
                        self.velocity.append([self.velocitor[j][0]*(x/1280), self.velocitor[j][1]*(y/720)])
                        #if (self.velocity[j][0] > 5):
                        #print(self.velocity[j])
                        
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('circle tracks', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                for i in self.velocitor:
                    print(i)
                break

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0
    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


import cv2
import numpy as np
import time
import GridSquares
import math

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        # speed in pixels/second
        self.velocitor = []
        # speed in centimeters/second
        self.velocity = []
        self.cam = cv2.VideoCapture(0)#'speed_detection3.mp4'
        # camera is listed as having an FOV of 60 degrees
        # DFOV = Diagonal Field of View
        # Horizontal FOV = 2 * atan(tan(DFOV/2)*cos(atan(9/16)))
        # Vertical FOV = 2 * atan(tan(DFOV/2)*sin(atan(9/16)))
        # I am using these values halved so HFOV/2 and VFOV/2
        # sets the camera resolution to 1280 x 720 p MIGHT NOT WORK FOR ALL CAMERAS
        # using Logitech 720p webcam
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.DFOV = 60*(math.pi/180)
        self.HFOV = 2*math.atan(math.tan(self.DFOV/2)*math.cos(math.atan(9/16)))
        self.VFOV = 2*math.atan(math.tan(self.DFOV/2)*math.sin(math.atan(9/16)))
        self.frame_idx = 0
        self.calibration_values = None
        self.times = []

    def run(self):
        while True:
            ret, frame = self.cam.read()
            # gets square objects from Daniel's code. Using to get height of drone
            gridsquares = GridSquares.run(frame)
            squares = gridsquares[0]
            matrix = gridsquares[1]
            if self.calibration_values == None:
                self.calibration_values = cv2.calibrationMatrixValues(matrix, (1280,720),3,3)
                #print(self.calibration_values)
                self.HFOV = self.calibration_values[0]*(math.pi/180)
                self.VFOV = self.calibration_values[1]*(math.pi/180)
            img = cv2.medianBlur(frame, 5)
            t = time.clock()
            self.times.append(t)
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            vis = frame.copy()
            # make more precise parameters at competition to account for lighting and other variables
            circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 75,
                                       param1=45, param2=87, minRadius=1,maxRadius=300)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1,1,2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1,2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x,y), good_flag in zip(self.tracks, p1.reshape(-1,2),good):
                    if not good_flag:
                        continue
                    tr.append((x,y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    if len(self.velocitor) > self.track_len:
                        del self.velocitor[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                for i in range(len(self.tracks)-1):
                    #print(self.tracks)
                    tracklen = len(self.tracks[i])-1
                    x1=self.tracks[i][tracklen][0]-self.tracks[i][0][0]
                    y1=self.tracks[i][tracklen][1]-self.tracks[i][0][1]
                    self.velocitor.append([x1/(self.times[self.frame_idx-1]-self.times[self.frame_idx]),y1/(self.times[self.frame_idx-1]-self.times[self.frame_idx])])
                    cv2.line(vis, (self.tracks[i][0][0], self.tracks[i][0][1]), (self.tracks[i][1][0], self.tracks[i][1][1]), (0,255,0),1)
                    #print(self.velocitor[i])
                if squares != []:
                    # first value in squares is the most accurate
                    # Not fully working, check with David
                    height = -squares[0].location[2]
                    #print(height)
                    # x is the distance from one side of the screen to the other in cm
                    # y is the distance from the top to the bottom of the screen in cm
                    # x = 2*height*tan(HFOV/2)
                    # y = 2*height*tan(VFOV/2)
                    # vertical FOV = 15.304295 degrees for this camera
                    # horizaontal FOV = 26.71174 degrees for this camera
                    x = 2*height*math.tan(self.HFOV)
                    y = 2*height*math.tan(self.VFOV)
                    for j in range(len(self.velocitor)-1):
                        #velocity = (x(cm)/x(pxls))*velocity(pxls/s)
                        self.velocity.append([self.velocitor[j][0]*(x/1280), self.velocitor[j][1]*(y/720)])
                        #if (self.velocity[j][0] > 5):
                        #print(self.velocity[j])

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)

                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('circle tracks', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                for i in self.velocitor:
                    print(i)
                break

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0
    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

>>>>>>> 9b94ce6d1670be035b8cdae3299f0f36f977db50
