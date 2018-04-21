import numpy as np
import cv2
from operator import attrgetter
from vector import dot, distance, cross

# Length of the grid squares in cm
SQUARE_LENGTH = 28.5

# Width of gaps between grid squares
SQUARE_GAP = 2.5


def compute_frame_squares(img):
    """
    Takes an image, and returns a list of all "squares" in the image
    :param img:
    :return:
    """

    # Create a new image with the minimum of all three channels
    red, green, blue = cv2.split(img)
    gray = np.minimum(np.minimum(red, green), blue)

    # Run a threshold to find only white lines.
    ret, out = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) # Logitech camera
    cv2.imshow('Threshold', out)

    # Get contours in image in a list.
    dump, contours, hierarchy = cv2.findContours(out, cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area, since there are lots of contours with 0 area
    contours.sort(key=cv2.contourArea, reverse=True)
    squares = []
    for i in range(len(contours)):
        # Stop once area is too small
        if cv2.contourArea(contours[i]) < 1000:
            break

        # Simplify contour, check if it's a square
        candidate = GridSquare(contours[i])
        epsilon = 0.01 * cv2.arcLength(candidate.contour, True)
        candidate.corners = cv2.approxPolyDP(candidate.contour, epsilon, True)
        if len(candidate.corners) == 4:
            # It's a square!
            squares.append(candidate)

    return squares


class GridSquare:
    cam_rot = None # Rotation vector of camera aligned to grid square axes.
    normal = [] # Normal vector pointing out of the square
    rvec = None # rvec returned by solvePNP
    cam_pos = None # Equivalent to location. Used for internal testing reasons
    tvec = None # tvec returned by solvePNP ransac
    score = 0 # How good is this square. See compare square normals for more
    corners = [] # Image coordinates of square corners
    contour = None # Unsimplified image coordinates of square corners
    location = [] # Square location in camera coordinates

    def __init__(self, contour):
        self.contour = contour

    def update_pos_stats(self, cam_matrix, distortion_coefficients, object_points):
        """
        Takes the square's corners found in the image, corresponding 3d
        coordinates, and intrinsic camera information. Sets fields related to
        extrinsic camera information: cam_rot, normal, cam_pos, location.
        Note that the square might be bogus until you check its score, and
        camera extrinsics are unaligned until align squares is called.
        :param cam_matrix:
        :param distortion_coefficients:
        :param object_points:
        :return: nothing
        """

        temp_corners = self.corners.reshape(4, 2, 1).astype(float)

        # Gets vector from camera to center of square
        inliers, self.rvec, self.tvec = cv2.solvePnP(object_points, temp_corners,
                                                     cam_matrix, distortion_coefficients)
        rot_matrix = cv2.Rodrigues(self.rvec)
        cam_pos = np.multiply(cv2.transpose(rot_matrix[0]), -1).dot(self.tvec)
        cam_to_grid_transform = np.concatenate((cv2.transpose(rot_matrix[0]), cam_pos), axis=1)
        grid_to_cam_transform = np.linalg.inv(np.concatenate((cam_to_grid_transform, np.array([[0, 0, 0, 1]])), axis=0))

        self.cam_rot = list(cam_to_grid_transform.dot(np.array([0, 0, 1, 0])))
        self.normal = grid_to_cam_transform.dot(np.array([0, 0, 1, 0]))
        self.cam_pos = cam_pos
        self.location = [cam_pos[0][0],cam_pos[1][0],cam_pos[2][0]]

    def align_squares(self, guess, cam_matrix, distortion_coefficients, object_points):
        """
        Needed because ABCD has different camera location from BCDA. Takes a
        square's corners, camera intrisics, object information, and a camera
        rotation to align to. Finds the camera position that gives a rotation
        vector (unit vector in cam direction using grid axes) closest to guess.
        :param guess:
        :param cam_matrix:
        :param distortion_coefficients:
        :param object_points:
        :return: nothing
        """

        alignment_scores = [dot(self.cam_rot, guess), 0, 0, 0]

        # Loop through possible orders: ABCD, BCDA, CDAB, DABC
        for rot in range(1, 4):
            # Shift to the rot permutation
            temp_corners = np.roll(self.corners, rot, axis=0)

            # Get vector from camera to center of square
            temp_corners = temp_corners.reshape(4,2,1).astype(float)
            inliers, self.rvec, self.tvec = cv2.solvePnP(object_points, temp_corners,
                                                         cam_matrix, distortion_coefficients)

            # Get rotation information for change to square center at origin
            rot_matrix = cv2.Rodrigues(self.rvec)

            cam_pos = np.multiply(cv2.transpose(rot_matrix[0]), -1).dot(self.tvec)
            cam_to_grid_transform = np.concatenate((cv2.transpose(rot_matrix[0]), cam_pos), axis=1)

            # Compare camera unit vector to guess
            alignment_scores[rot] = dot(list(cam_to_grid_transform.dot(np.array([0, 0, 1, 0]))), guess)

        # Pick the orientation that was best and recompute camera location
        self.corners = np.roll(self.corners, alignment_scores.index(max(alignment_scores)), axis=0)
        self.update_pos_stats(cam_matrix, distortion_coefficients, object_points)

    def compare_square_normals(self, square, dim):
        """
        Increments the score of two squares based on how parallel their normal
        vectors are.
        :param square:
        :param dim: the dimensions of the image
        :return: nothing
        """

        # Using 1-cross product due to larger change of sin when parallel
        temp_cross = cross(self.normal, square.normal)

        edge = False
        for point in square.corners:
            if point[0][0] < 1 or point[0][0] > dim[1] - 2 or point[0][1] < 1 \
                    or point[0][1] > dim[0] - 2:
                # Contours on the edge don't improve scores
                edge = True

        if not edge:
            # Increment the score
            self.score += 1 - abs(distance(temp_cross) / (distance(square.normal)*distance(self.normal)))


# Get Square Stats
#
def get_square_stats(img, cam_matrix, distortion_coefficients, cam_rot_guess):
    """
    Returns list of squares with locations, sorted by score, aligned to
    cam_rot_guess. Needs camera intrinsics. If cam_rot_guess is 0, will pick an
    arbitrary square to align to.
    :param img:
    :param cam_matrix:
    :param distortion_coefficients:
    :param cam_rot_guess:
    :return:
    """

    # 3d grid square coordinates
    object_points = np.array([[[-SQUARE_LENGTH/2, -SQUARE_LENGTH/2, 0]],
                              [[-SQUARE_LENGTH/2, SQUARE_LENGTH/2, 0]],
                              [[SQUARE_LENGTH/2, SQUARE_LENGTH/2, 0]],
                              [[SQUARE_LENGTH/2, -SQUARE_LENGTH/2, 0]]],
                             np.float32).reshape(4, 3, 1)

    # Find squares in the image
    squares = compute_frame_squares(img)
    for square in squares:
        # Get camera location for those squares
        square.update_pos_stats(cam_matrix, distortion_coefficients, object_points)

    # Compare each pair of squares
    for i in range(len(squares)):
        for j in range(len(squares)):
            squares[i].compare_square_normals(squares[j], img.shape)

    # Sort by score
    squares.sort(key=attrgetter('score'),reverse=True)

    # Filter out low scores if there are squares
    if len(squares)>0:
        threshold = max(.95*squares[0].score, 1.0)
        squares = list(filter(lambda x: x.score > threshold, squares))

    # Align squares
    for square in squares:
        if cam_rot_guess == 0:
            # If we don't have a guess, use the highest score square
            cam_rot_guess = squares[0].cam_rot

        square.align_squares(cam_rot_guess, cam_matrix, distortion_coefficients, object_points)

    return squares, cam_rot_guess

