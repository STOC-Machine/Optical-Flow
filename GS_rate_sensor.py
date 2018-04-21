import cv2
import sys
import numpy as np
import glob
import grid_squares as grid
import vector as vec

FONT = cv2.FONT_HERSHEY_SIMPLEX

#camera_matrix = np.array([[376.60631072, 0., 334.94985263], [0., 376.37590044, 245.47987032], [0., 0., 1.]])
#distortion_coefficients = np.array([-3.30211385e-01, 1.58724644e-01, -1.87573090e-04, 4.55691783e-04, -4.98096761e-02])

# Logitech values
camera_matrix = np.array([[811.75165344, 0, 317.03949866],
                          [0, 811.51686214, 247.65442989],
                          [0, 0, 1]])
distortion_coefficients = np.array([-3.00959078e-02, -2.22274786e-01,
                                    -5.31335928e-04, -3.74777371e-04,
                                    1.80515550e+00])
distortion_coefficients = distortion_coefficients.reshape(5, 1)


def find_grid(img, cam_rot_guess):
    """
    Takes an undistorted image and a camera rotation guess and finds the position
    of the camera relative to the grid, as well as identifying all visible
    squares in the grid.
    :param img: The image to process
    :param cam_rot_guess:
    :return: (squares, cam_rot, base_object_points)
    """

    birds_view = np.zeros([1000, 1000, 3], dtype=np.uint8)
    cv2.circle(birds_view,
               (int(birds_view.shape[0] / 2), int(birds_view.shape[1] / 2)), 5,
               (255, 255, 0), -1)
    squares, cam_rot_guess = grid.get_square_stats(img, camera_matrix,
                                                   np.array([[]]),
                                                   cam_rot_guess)

    cam_rot = cam_rot_guess

    square_length = 28.5
    square_gap = 2
    base_object_points = [[-square_length / 2, -square_length / 2, 0],
                          [-square_length / 2, square_length / 2, 0],
                          [square_length / 2, square_length / 2, 0],
                          [square_length / 2, -square_length / 2, 0]]

    for square in squares:
        temp_vec = vec.sub(square.location, squares[0].location)

        temp_vec[0] = (square_length + square_gap) * round(
            temp_vec[0] / (square_length + square_gap), 0)
        temp_vec[1] = (square_length + square_gap) * round(
            temp_vec[1] / (square_length + square_gap), 0)
        temp_vec[2] = 0
        # Where the magic happens. Gets vector from camera to center of square

    return squares, cam_rot, base_object_points


def display_grid(img, squares, cam_rot, base_object_points):
    """
    Display the output of find_grid in a pretty way.
    :param img: the image in which the grid was found
    :param squares: the squares in the image
    :param cam_rot: the deduced camera orientation
    :param base_object_points: ??
    :return: nothing
    """
    
    birds_view = np.zeros([1000, 1000, 3], dtype=np.uint8)
    cv2.circle(birds_view,
               (int(birds_view.shape[0] / 2), int(birds_view.shape[1] / 2)), 5,
               (255, 255, 0), -1)

    if cam_rot != 0:
        cam_line = vec.add(vec.scalar_mult(cam_rot, 50),
                                [birds_view.shape[0] / 2,
                                 birds_view.shape[0] / 2, 0])

        cv2.line(birds_view, (int(cam_line[0]), int(cam_line[1])),
                 (int(birds_view.shape[0] / 2), int(birds_view.shape[1] / 2)),
                 (0, 0, 255), 1, cv2.LINE_AA)

    for square in squares:
        for edge_index in range(4):
            temp_draw_vec = vec.add(square.location,
                                         base_object_points[edge_index])
            temp_draw_vec2 = vec.add(square.location,
                                          base_object_points[edge_index - 1])

            cv2.line(birds_view, (int(temp_draw_vec[0] + birds_view.shape[0] / 2),
                                  int(temp_draw_vec[1] + birds_view.shape[1] / 2)),
                     (int(temp_draw_vec2[0] + birds_view.shape[0] / 2),
                      int(temp_draw_vec2[1] + birds_view.shape[1] / 2)),
                     (255, 255, 255), 3, cv2.LINE_AA)

        x = sum(point[0][0] for point in square.corners) // 4
        y = sum(point[0][1] for point in square.corners) // 4

        cv2.putText(img, '{} {}'.format(int(abs(square.location[2])),
                                        int(square.score * 100)), (x, y), FONT,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.polylines(img, [square.corners], True, (255, 0, 0))
        cv2.drawContours(img, square.contour, True, (0, 255, 0))

    if len(squares) > 0:
        cam_line2 = vec.add(vec.scalar_mult(cam_rot, 50),
                            [birds_view.shape[0] / 2,
                             birds_view.shape[0] / 2, 0])

        cv2.line(birds_view, (int(cam_line2[0]), int(cam_line2[1])),
                 (int(birds_view.shape[0] / 2), int(birds_view.shape[1] / 2)),
                 (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Squares", img)
    cv2.imshow("Birds eye view", birds_view)


def run_from_camera(camera):
    """
    Run bird's view from camera input.
    :param camera: the cv2 camera file handle
    :return: nothing
    """

    cam_rot_guess = 0

    # Run until q is pressed
    while cv2.waitKey(1) & 0xFF != ord('q'):
        ret, img = camera.read()

        if type(img) is not np.ndarray:
            print('Error: image did not read properly, skipping')
            continue

        img = cv2.undistort(img, camera_matrix, distortion_coefficients)
        squares, cam_rot, base_object_points = find_grid(img, cam_rot_guess)
        display_grid(img, squares, cam_rot, base_object_points)

        if len(squares) > 0:
            cam_rot_guess = squares[0].cam_rot


def run_from_files(files):
    """
    Run bird's view from file input.
    :param files: a directory/filename with wildcards
    :return: nothing
    """
    files = glob.glob(files)

    while len(files) > 0:
        file = files.pop(0)
        img = cv2.imread()

        if img is None:
            print('Error: could not read image file {}, skipping.'.format(file))
            continue

        img = cv2.undistort(img, camera_matrix, distortion_coefficients)
        squares, cam_rot, base_object_points = find_grid(img, 0)
        display_grid(img, squares, cam_rot, base_object_points)

        # Wait for keypress to continue, close old windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Use /dev/video0 by default. Change from 0 if using different /video
        run_from_camera(cv2.VideoCapture(0))
    else:
        run_from_files(sys.argv[1])
