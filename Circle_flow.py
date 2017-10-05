# Logitech 720P HD camera has resolution = (640 x 480)
import math
import time
from operator import attrgetter

import cv2
import numpy as np

import GridSquares

lk_params = {'winSize': (15, 15), 'maxLevel': 2,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

feature_params = {'maxCorners': 500, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}


class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.velocitor = [] # speed in pixels/second
        self.velocity = []  # speed in centimeters/second
        self.cam = cv2.VideoCapture(0)
        # camera is listed as having an FOV of 60 degrees
        # DFOV = Diagonal Field of View
        # Horizontal FOV = 2 * atan(tan(DFOV/2)*cos(atan(9/16)))
        # Vertical FOV = 2 * atan(tan(DFOV/2)*sin(atan(9/16)))
        # I am using these values halved so HFOV/2 and VFOV/2
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.DFOV = 60*(math.pi/180)
        self.HFOV = 2*math.atan(math.tan(self.DFOV/2)*math.cos(math.atan(9/16)))
        self.VFOV = 2*math.atan(math.tan(self.DFOV/2)*math.sin(math.atan(9/16)))
        self.frame_idx = 0
        self.calibration_values = None
        self.times = []

    def run(self):
        """
        uses GridSquares to determine height
        detects circles
        determines speed of circles across screen
        converts speed into cm/s
        displays image and visual aids on circles
        """
        while True:
            ret, frame = self.cam.read()
            frame_copy = np.copy(frame)
            squares = GridSquares.computeFrameSquares(frame_copy)
            if self.calibration_values is None:
                self.calibration_values = cv2.calibrationMatrixValues(
                    np.array([[811.75165344, 0., 317.03949866], [0., 811.51686214, 247.65442989], [0., 0., 1.]]),
                    (640, 480), 3, 3)
                self.HFOV = self.calibration_values[0]*(math.pi/180)
                self.VFOV = self.calibration_values[1]*(math.pi/180)
            img = cv2.medianBlur(frame, 5)
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            vis = frame.copy()
            circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 75,
                                       param1=45, param2=87, maxRadius=300,minRadius=1)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    t1 = time.clock()
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1,1,2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    t2 = time.clock()
                    self.times.append((t1,t2))
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
                        trackleangth = len(self.tracks[i])-1
                        x1=self.tracks[i][trackleangth][0]-self.tracks[i][0][0]
                        y1=self.tracks[i][trackleangth][1]-self.tracks[i][0][1]
                        self.velocitor.append([x1 / (self.times[self.frame_idx][1] - self.times[self.frame_idx][0]),
                                               y1 / (self.times[self.frame_idx][1] - self.times[self.frame_idx][0])])
                        cv2.line(vis, (self.tracks[i][0][0], self.tracks[i][0][1]),
                                 (self.tracks[i][1][0], self.tracks[i][1][1]), (0, 255, 0), 1)
                    if squares != []:
                        for square in squares:
                            square.getPosStats()
                        for square1 in squares:
                            for square2 in squares:
                                square1.compareSquareNormals(square2,frame_copy)
                        squares.sort(key=attrgetter("score"),reverse=True)
                        height = squares[0].getHeight()

                        # x is the distance from one side of the screen to the other in cm
                        # y is the distance from the top to the bottom of the screen in cm
                        # x = 2*height*tan(HFOV/2)
                        # y = 2*height*tan(VFOV/2)
                        # vertical FOV = 15.304295 degrees for this camera
                        # horizaontal FOV = 26.71174 degrees for this camera
                        x = 2 * height * math.tan(self.HFOV)
                        y = 2 * height * math.tan(self.VFOV)
                        for j in range(len(self.velocitor) - 1):
                            # velocity = (x(cm)/x(pxls))*velocity(pxls/s)
                            self.velocity.append([self.velocitor[j][0] * (x / 640), self.velocitor[j][1] * (y / 480)])
                """
                if self.frame_idx % self.detect_interval == 0:
                    for i in circles[0,:]:
                        self.tracks.append([(i[0],i[1])])
                """
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
    """
    the main function, runs all code listed above
    """
    print(__doc__)
    App().run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
