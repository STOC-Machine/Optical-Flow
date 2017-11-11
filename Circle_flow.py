# Logitech 720P HD camera has resolution = (640 x 480)
from __future__ import print_function
from imutils.video import FPS
import datetime
import time
import numpy as np
import cv2
from threading import Thread
import math
from operator import attrgetter
import GridSquares

lk_params = {'winSize': (15, 15), 'maxLevel': 2,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class WebcamVideoStream:
    def __init__(self,src=0):
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

class OpticalFlow:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.fps_interval = 5
        self.camera_values = np.array(
            [[376.60631072, 0., 334.94985263], [0., 376.37590044, 245.47987032], [0., 0., 1.]])
        self.distortion_coefficients = np.array([-3.30211385e-01, 1.58724644e-01, -1.87573090e-04, 4.55691783e-04,
                                                 -4.98096761e-02])
        self.camera_resolution = (640, 480)
        self.matrix_size = (3, 3)
        self.fps = 14

    def setFrameWidthAndHeight(self, cam):
        """
        Resizes the camera object to have a width of 1280pxls and a height of 720pxls
        :param cam: a camera object
        :return: cam
        """
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cam

    def getFrame(self, cam):
        """
        :param cam: a camera object
        :return: frame: a single frame from the camera object
        """
        frame = cam.read()
        return frame

    def undistortFrame(self, frame):
        """
        :param frame: a 3-channel image pulled from a webcam
        :return: undistorted_frame: a frame that has been undistorted using distortion_coefficients
        """
        undistorted_frame = cv2.undistort(frame, self.camera_values, self.distortion_coefficients)
        return undistorted_frame

    def prepFrame(self, frame):
        """
        :param frame: a 3-channel image pulled from a webcam
        :return: grayed_frame: a frame that has been blurred and grayed
        This function intakes an image pulled directly from a webcam and returns
        that frame in a blurred and grayed form
        """
        blurred_frame = cv2.medianBlur(frame, 5)
        grayed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGRA2GRAY)
        return grayed_frame

    def setCalibartionValues(self):
        """
        :return: calibration_values = a matrix that represents the calibrated camera
        """
        calibration_values = cv2.calibrationMatrixValues(np.array(self.camera_values),
                                                         self.camera_resolution, self.matrix_size[0],
                                                         self.matrix_size[1])
        return calibration_values

    def setFieldOfView(self, calibration_values):
        """

        :param calibration_values: a matrix that represents the calibrated camera
        :return: field of views: a tuple of length 2 that has the vertical field of view and
        horizontal field of view calculated using calibration_values
        """
        horizontal_field_of_view = calibration_values[0] * (math.pi / 180)
        vertical_field_of_view = calibration_values[1] * (math.pi / 180)
        field_of_views = (horizontal_field_of_view, vertical_field_of_view)
        return field_of_views

    def getPoints(self, tracked_points, frame0, frame1):
        """

        :param tracked_points: a list of points that are being tracked
        :param frame0: a frame from a webcam
        :param frame1: another frame from a webcam
        :return: point0: a point representing the center of a circle we want to track from frame0
                 point1: a point representing the center of the same circle as point0 but in frame1
        """
        point0 = np.float32([tr[-1] for tr in tracked_points]).reshape(-1, 1, 2)
        point1, _st, _err = cv2.calcOpticalFlowPyrLK(frame0, frame1, point0, None, **lk_params)
        return (point0, point1)

    def addSpeed(self, point0, point1, speed_list, fps):
        """
        :param point0: a point representing a circle
        :param point1: a point representing the same circle in a different location
        :param speed_list: a list holding the speed at which the object moved between the two frames in pxls/s
        :param fps: the frames per second of the camera
        :return: speed_list
        """
        speed = [(point0[0][0][0] - point1[0][0][1]) / fps,
                 (point0[0][0][1] - point1[0][0][1]) / fps]
        speed_list.append(speed)
        return speed_list

    def checkDistance(self, point0, point1):
        """
        :param point0: a size 2 tuple with x and y coordinates representing the location of an object
        :param point1: a size 2 tuple with x and y coordinates representing the location of the same object
        in a different frame
        :return: good: An array of True/False statements
        """
        distance = abs(point0 - point1).reshape(-1, 2).max(-1)
        good = distance < 1
        return good

    def addGoodTracks(self, tracked_points, point1, good):
        """
        :param tracked_points: the list of points that are being tracked throughout the program
        :param point1: a size 2 tuple with x and y coordinates representing the location of an object
        :param good: An array of True/False statements
        :return: new_tracks: a list of tracked points that are confirmed to be valid tracks
        """
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracked_points, point1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            new_tracks.append(tr)
        return new_tracks

    def keepTracksSmall(self, tracked_points):
        """
        If the number of points being tracked exceeds self.track_len, delete a point
        :param tracked_points: A list of tracked points
        """
        for tr in tracked_points:
            if len(tr) > self.track_len:
                del tr[0]

    def sortSquares(self, squares, frame):
        """
        Takes an array of square objects, finds a value associated to them and sorts them by that value
        :param squares: An array of square objects
        :param frame: A 3-channel image pulled from a webcam
        :return: squares
        """
        #if squares != []:
        for square in squares:
            square.getPosStats()
        for square1 in squares:
            for square2 in squares:
                square1.compareSquareNormals(square2, frame)
        squares.sort(key=attrgetter("score"), reverse=True)
        return squares

    def getHeight(self, frame, cameraMatrix):
        """
        Takes an array of square objects and returns the height of the first square in the array
        :param frame: A 4-channel image pulled from a webcam
        :param cameraMatrix: A matrix containing values of a calibrated camera
        :return: height: The height of the camera
        """
        height = 0
        squares,temp = GridSquares.getSquareStats(frame,self.camera_values,self.distortion_coefficients)
        if len(squares) > 0:
            squares[0]
        return height

    def getDimensions(self, height, field_of_views):
        """
        :param height: the height of the camera
        :param field_of_views: A length 2 tuple that contains the vertical and horizontal field of view
        :return: dimensions: A length 2 tuple that contains the dimensions in cm that the camera is viewing
        """
        x = 2 * height * math.tan(field_of_views[0])
        y = 2 * height * math.tan(field_of_views[1])
        dimensions = (x, y)
        return dimensions

    def addVelocity(self, speed_list, dimensions, velocity_list):
        """
        :param speed_list: a list holding the speeds at which the objects moved between two frames in pxls/s
        :param dimensions: A length 2 tuple that contains the dimensions in cm that the camera is viewing
        :param velocity_list: a list holding the velocities at which the objects moved between two frames in cm/s
        :return: velocity_list
        """
        for distance in speed_list:
            velocity = [distance[0] * (dimensions[0] / self.camera_resolution[0]),
                        distance[1] * (dimensions[1] / self.camera_resolution[1])]
            velocity_list.append(velocity)
        return velocity_list

    def addCircles(self, tracked_points, circles):
        """
        Adds the x,y values of the circle objects as points to be tracked
        :param tracked_points: A list of tracked points
        :param circles: A list of circle objects
        """
        if circles is not None:
            for x, y, r in np.float32(circles).reshape(-1, 3):
                tracked_points.append([(x, y)])

    def addPoints(self, tracked_points, grayed_frame):
        """
        Tracks interesting points instead of circles.
        Interesting points would be points of high contrast or sharp lines
        :param tracked_points: A list of tracked points
        :param grayed_frame: A single channel grayscaled image
        :return:
        """
        mask = np.zeros_like(grayed_frame)
        mask[:] = 255
        p = cv2.goodFeaturesToTrack(grayed_frame, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracked_points.append([(x, y)])
        return tracked_points

    def getTime(self):
        t = time.clock()
        return t

    def getFPS(self,t1,t2):
        fps = self.fps_interval/(t2-t1)
        return fps


    def run(self):
        cam = WebcamVideoStream(src=0).start()
        tracked_points = []
        speed_list = []
        velocity_list = []
        frame_idx = 0
        fps = 20
        t1 = self.getTime()
        calibration_values = self.setCalibartionValues()
        field_of_views = self.setFieldOfView(calibration_values)
        prev_grayed_frame = None
        while True:
            frame = self.getFrame(cam)
            grayed_frame = self.prepFrame(frame)

            if len(tracked_points) > 0:
                frame0, frame1 = prev_grayed_frame, grayed_frame
                point0, point1 = self.getPoints(tracked_points, frame0, frame1)
                speed_list = self.addSpeed(point0, point1, speed_list, fps)
                good = self.checkDistance(point0, point1)
                new_tracked_points = self.addGoodTracks(tracked_points, point1.reshape(-1, 2), good)
                self.keepTracksSmall(tracked_points)
                tracked_points = new_tracked_points
                height = self.getHeight(frame,calibration_values)
                dimensions = self.getDimensions(height, field_of_views)
                velocity_list = self.addVelocity(speed_list, dimensions, velocity_list)

            if frame_idx % self.detect_interval == 0:
                tracked_points = self.addPoints(tracked_points,grayed_frame)
                circles = cv2.HoughCircles(grayed_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=75,
                                           param1=45, param2=75, maxRadius=300, minRadius=1)

                if circles is not None:
                    for i in circles:
                        for j in i:
                            cv2.circle(frame,(j[0],j[1]),j[2],(0,0,0),5,8,0)
                self.addCircles(tracked_points, circles)

            if frame_idx % self.fps_interval == 0:
                t2 = self.getTime()
                fps = self.getFPS(t1,t2)
                print(fps)
                t1 = t2

            prev_grayed_frame = grayed_frame
            frame_idx += 1
            cv2.imshow('circle tracks', frame)
            cv2.waitKey(1)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                print(velocity_list)
                cv2.destroyAllWindows()
                cam.stop()
                break


def main():
    """
    the main function, runs all code listed above
    """
    opticalflow = OpticalFlow()
    opticalflow.run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
