# Logitech 720P HD camera has resolution = (640 x 480)
import math
import time
from operator import attrgetter

import cv2
import numpy as np

import GridSquares

lk_params = {'winSize': (15, 15), 'maxLevel': 2,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}


class OpticalFlow:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.camera_values = np.array(
            [[376.60631072, 0., 334.94985263], [0., 376.37590044, 245.47987032], [0., 0., 1.]])
        self.distortion_coefficients = np.array([-3.30211385e-01, 1.58724644e-01, -1.87573090e-04, 4.55691783e-04,
                                                 -4.98096761e-02])
        self.camera_resolution = (640, 480)
        self.matrix_size = (3, 3)

    def addTime(self, time_list):
        """
        takes the time and adds it to the time list
        :param time_list: a list where the times that different frames were taken at will be stored
        :return: time_list
        """
        t = time.clock()
        time_list.append(t)
        return time_list

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
        ret, frame = cam.read()
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

    def getSquares(self, frame):
        """
        :param frame: a 3-channel image pulled from a webcam
        :return: squares: a list of square objects
        """
        squares = GridSquares.computeFrameSquares(frame)
        return squares

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
        point1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, point0, None, **lk_params)
        return point0, point1

    def addSpeed(self, point0, point1, time1, time2, speed_list):
        """
        :param point0: a point representing a circle
        :param point1: a point representing the same circle in a different location
        :param time1: the time that point0 was taken at
        :param time2: the time that point1 was taken at
        :param speed_list: a list holding the speed at which the object moved between the two frames in pxls/s
        :return: speed_list
        """
        speed = [(point0[0][0][0] - point1[0][0][1]) / (time1 - time2),
                 (point0[0][0][1] - point1[0][0][1]) / (time1 - time2)]
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
        if squares != []:
            for square in squares:
                square.getPosStats()
            for square1 in squares:
                for square2 in squares:
                    square1.compareSquareNormals(square2, frame)
            squares.sort(key=attrgetter("score"), reverse=True)
        return squares

    def getGoodHeight(self, squares):
        """
        Takes an array of square objects and returns the height of the first square in the array
        :param squares: An array of square objects
        :return: height: The height of the camera
        """
        height = 0
        if squares != []:
            height = squares[0].getHeight()
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

    def run(self, cam):
        tracked_points = []
        speed_list = []
        velocity_list = []
        time_list = []
        frame_idx = 0
        cam = self.setFrameWidthAndHeight(cam)
        calibration_values = self.setCalibartionValues()
        field_of_views = self.setFieldOfView(calibration_values)
        prev_grayed_frame = None
        while True:
            time_list = self.addTime(time_list)
            frame = self.getFrame(cam)
            squares = self.getSquares(frame)
            undistorted_frame = self.undistortFrame(frame)
            grayed_frame = self.prepFrame(undistorted_frame)
            if frame_idx % self.detect_interval == 0:
                circles = cv2.HoughCircles(grayed_frame, cv2.HOUGH_GRADIENT, 1, 75,
                                           param1=45, param2=75, maxRadius=300, minRadius=1)
                self.addCircles(tracked_points, circles)

            if len(tracked_points) > 0:
                frame0, frame1 = prev_grayed_frame, grayed_frame
                point0, point1 = self.getPoints(tracked_points, frame0, frame1)
                speed_list = self.addSpeed(point0, point1, time_list[frame_idx], time_list[frame_idx - 1], speed_list)
                good = self.checkDistance(point0, point1)
                new_tracked_points = self.addGoodTracks(tracked_points, point1.reshape(-1, 2), good)
                self.keepTracksSmall(tracked_points)
                tracked_points = new_tracked_points
                squares = self.sortSquares(squares, frame)
                height = self.getGoodHeight(squares)
                dimensions = self.getDimensions(height, field_of_views)
                velocity_list = self.addVelocity(speed_list, dimensions, velocity_list)
            prev_grayed_frame = grayed_frame
            frame_idx += 1
            cv2.imshow('circle tracks', frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    """
    the main function, runs all code listed above
    """
    print(__doc__)
    cam = cv2.VideoCapture(0)
    opticalflow = OpticalFlow()
    opticalflow.run(cam)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
