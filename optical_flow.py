#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow. Uses goodFeaturesToTrack or HoughCircles
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
    """
    This class is using threading on top of OpenCV's VideoCapture function in order to increase
    fps performance.
    """
    def __init__(self,src=0,):
        # initializes the camera the gets the first frames
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # begins threading operation on video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # checks if feed has stopped
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # reads video stream and returns an array representing a frame
        return self.frame

    def stop(self):
        # stops the feed
        self.stopped = True



class App:
    def __init__(self):
        # initialized values
        # for how many frames the tracked points previous locations are stored
        # e.g. the for each point tracked, the location of that point in the previous 10
        # frames is stored
        self.track_len = 10
        # a new set of points is detected very n frames
        self.detect_interval = 5
        # a new fps is calculated every n frames
        self.fps_interval = 5
        # points that are being tarcked
        self.tracks = []
        # the current frames, increases by 1 for every frame read
        self.frame_idx = 0
        # list of velocities relating to movement of points frame to frame
        self.vlist = []


    def run(self, cam, wantCircles):
        # capture begins
        vs = WebcamVideoStream(src=cam).start()
        # inital fps will be recalculated after self.fps_interval frames
        fps = 30
        t1 = time.clock()
        while True:
            frame = vs.read()
            # transforms the image to gray for using in calcOpticalFlowPyrLK b/c
            # this function only takes gray images
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            # if there are points to track do the following
            if len(self.tracks) > 0:

                #initialize the previous frame and the current frame
                img0, img1 = self.prev_gray, frame_gray
                # reshapes the list of tracked points into a [1,2] matrix
                """
                [[[x1,y1]]
                 [[x2,y2]]
                 [[x3,y3]]
                ]
                """
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                # Calculates the location of points p0 from img0 in the current frame img1
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # Grabs points in reverse of previous function in order to check error
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)


                # If the difference between the actual original points (p0) and the calculated (p0r) is greater
                # than 1 for a point, there is too much error and it should not be added to the
                # list of tracked points.
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1


                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    # if d > 1 do not add to tracks
                    if not good_flag:
                        continue
                    tr.append((x, y))

                    # If there are too many positions being tracked for a given point,
                    # delete the oldest position
                    if len(tr) > self.track_len:
                        del tr[0]

                    new_tracks.append(tr)
                    # draw a circle on the position where the point currently is
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                # update self.tracks with the new tracks
                self.tracks = new_tracks

                # calculates velocity
                for tr in self.tracks:
                    distances = []
                    prev_x,prev_y = tr[0][0],tr[0][1]

                    # for every position for a given point get the distance between
                    # two consecutive points, average that distance and calculate the
                    # speed in pxls/s
                    for i in range(len(tr)):
                        #
                        curr_x,curr_y = tr[i][0],tr[i][1]
                        x,y=curr_x-prev_x,curr_y-prev_y
                        prev_x, prev_y = curr_x, curr_y
                        # if the distance between two points is less than 1 pixel,
                        # ignore it
                        if abs(x) < 1.0 and abs(y) < 1.0:
                            continue
                        distances.append([x,y])
                    total = [0,0]

                    # sum up all the distances for a given point
                    for distance in distances:
                        total[0] += distance[0]
                        total[1] += distance[1]

                    # get the average of the summed points and calculate velocity of that
                    if len(distances) != 0:
                        avg_distance = [total[0]/(len(distances)), total[1]/(len(distances))]
                        v = [avg_distance[0]*fps, avg_distance[1]*fps]
                        self.vlist.append(v)

                # creates 'tails' for the points. Draws a line between all of the stored positions
                # of points. Makes those cool green lines
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            # gets new fps every fps_interval
            if self.frame_idx % self.fps_interval == 0:
                t2 = time.clock()
                fps = self.fps_interval/(t2-t1)
                t1 = t2

            # captures a new set of points every detect_interval
            if self.frame_idx % self.detect_interval == 0:
                # if the user chooses the track circles this runs
                if wantCircles:
                    # draw a circle centered on a tracked point
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(frame, (x, y), 5, 0, -1)

                    # get the coordinates of all circles in the image in the form [x,y,r]
                    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=75,
                                                param1=45, param2=100, maxRadius=300, minRadius=0)

                    if circles is not None:
                        for i in circles:
                            for j in i:
                                # draw more circles
                                cv2.circle(frame, (j[0], j[1]), j[2], (0, 0, 0), 5, 8, 0)

                        # add the x and y coordinates of the circle's center to the list of
                        # tracked points
                        for x, y, r in np.float32(circles).reshape(-1, 3):
                            self.tracks.append([(x, y)])
                else:
                    # mask is an array of of size frame specifying a region of interest
                    # for goodFeaturesToTrack for us this is the entire frame so it is filled
                    # with the int 255.
                    mask = np.zeros_like(frame_gray)
                    # resets vlist for new points
                    del self.vlist[:]
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    # grabs a bunch of 'interesting' points from the frame. i.e. points of high
                    # contrast, usually on edges of objects
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

            self.frame_idx += 1
            # current frame = old frame
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            cv2.waitKey(1)

            # press escape to exit the program
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