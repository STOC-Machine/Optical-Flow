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
        self.cam = cv2.VideoCapture(0)
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

    def run(self):
        while True:
            ret, frame = self.cam.read()
            # gets square objects from Daniel's code. Using to get height of drone
            squares = GridSquares.run(frame)
            img = cv2.medianBlur(frame, 5)
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            vis = frame.copy()
            # make more precise parameters at competition to account for lighting and other variables
            circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 75,
                                       param1=45, param2=87, minRadius=1,maxRadius=300)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    t1 = time.clock()
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1,1,2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    t2 = time.clock()
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
                        cv2.circle(vis, (x,y), 2, (0,255,0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    for i in range(len(self.tracks)-1):
                        x1=self.tracks[i][1][0]-self.tracks[i][0][0]
                        y1=self.tracks[i][1][1]-self.tracks[i][0][1]
                        self.velocitor.append([x1/(t2-t1), y1/(t2-t1)])
                        cv2.line(vis, (self.tracks[i][0][0], self.tracks[i][0][1]), (self.tracks[i][1][0], self.tracks[i][1][1]), (0,255,0),1)
                        #print(self.velocitor[i])
                    if squares != []:
                        # first value in squares is the most accurate
                        # Not fully working, check with David
                        height = -squares[0].location[2]
                        # x is the distance from one side of the screen to the other in cm
                        # y is the distance from the top to the bottom of the screen in cm
                        # x = 2*height*tan(HFOV/2)
                        # y = 2*height*tan(VFOV/2)
                        # vertical FOV = 15.304295 degrees for this camera
                        # horizaontal FOV = 26.71174 degrees for this camera
                        x = 2*height*math.tan(self.HFOV)
                        y = 2*height*math.tan(self.VFOV)
                        for j in range(len(self.velocitor)-1):
                            # velocity = (x(cm)/x(pxls))*velocity(pxls/s)
                            self.velocity.append([self.velocitor[j][0]*(x/1280), self.velocitor[j][1]*(y/720)])
                            print(self.velocity[j])
                        
                if self.frame_idx % self.detect_interval == 0:
                    for i in circles[0,:]:
                        self.tracks.append([(i[0],i[1])])
                for i in circles[0, :]:
                    cv2.circle(vis, (i[0],i[1]),i[2], (0,255,0),2)
                    cv2.circle(vis, (i[0],i[1]),2 , (0,0,255),3)


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('circle tracks', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
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
                    
