#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
from imutils.video import FPS
import datetime
import time
import numpy as np
import cv2
from threading import Thread

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,

                       blockSize = 7 )

class WebcamVideoStream:
    def __init__(self,src=0,):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.fps_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.vlist = []


    def run(self, cam, wantCircles):
        vs = WebcamVideoStream(src=cam).start()
        fps = 30
        t1 = time.clock()
        while True:
            frame = vs.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:

                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                for i in range(len(p0)):
                    v = ((p1[i][0][0]-p0[i][0][0])* (fps),
                         (p1[i][0][1]-p0[i][0][1])*(fps))
                    self.vlist.append(v)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []

                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            # gets new fps every fps interval
            if self.frame_idx % self.fps_interval == 0:
                t2 = time.clock()
                fps = self.fps_interval/(t2-t1)
                t1 = t2

            if self.frame_idx % self.detect_interval == 0:
                if wantCircles:
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(frame, (x, y), 5, 0, -1)
                    if wantCircles:
                        circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=75,
                                                   param1=45, param2=100, maxRadius=300, minRadius=0)

                        if circles is not None:
                            for i in circles:
                                for j in i:
                                    cv2.circle(frame, (j[0], j[1]), j[2], (0, 0, 0), 5, 8, 0)

                        if circles is not None:
                            for x, y, r in np.float32(circles).reshape(-1, 3):
                                self.tracks.append([(x, y)])
                else:
                    mask = np.zeros_like(frame_gray)
                    del self.vlist[:]
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            cv2.waitKey(1)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                print("velocity: " + str(self.vlist))
                cv2.destroyAllWindows()
                vs.stop()
                break

def main():
    import sys
    cam = input("Enter the int representation of your camera (default should be 0): ")
    cam = int(cam)
    circles = input("Do you want to track circles (y/n): ")
    if circles == "y":
        wantCircles = True
    else:
        wantCircles = False
    print(__doc__)
    App().run(cam,wantCircles)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()