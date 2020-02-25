import cv2 as cv
import numpy as np


def find_contours(img_frame):
    """
    find and draw contours on the main frame
    :param img_frame: a frame from the video
    :return: image with contours and the contours matrix
    """
    _, frame_thresh = cv.threshold(cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY), 220, 255, 0)
    contours, _ = cv.findContours(frame_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_tag_orientation(img_frame, dimension):
    """ get orientation from the image frame
    :param img_frame: image frame from the video
    :param dimension: dimension to modify the new world frame
    :return: orientation of the tag along with new world points based on the orientation
    """
    # Check get_H_matrix function in superimpose for orientation notation
    orientations = {0: 0, 1: 0, 2: 0, 3: 0}
    # Orientation: Bottom Right
    for i in range(250, 301):
        for j in range(250, 301):
            orientations[0] += img_frame[i, j]
    # Orientation: Bottom Left
    for i in range(250, 301):
        for j in range(100, 151):
            orientations[1] += img_frame[i, j]
    # Orientation: Top Right
    for i in range(100, 151):
        for j in range(250, 301):
            orientations[2] += img_frame[i, j]
    # Orientation: Top Left
    for i in range(100, 151):
        for j in range(100, 151):
            orientations[3] += img_frame[i, j]

    orientation = max(orientations, key=orientations.get)
    # World frame for top-left orientation
    new__world_frame = np.array([[dimension - 1, dimension - 1], [0, dimension - 1], [0, 0], [dimension - 1, 0]],
                                dtype="float32")
    # Modify the world frame w.r.t the orientation of the tag
    # Orientation: Bottom Right
    if orientation == 0:
        new__world_frame = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]],
                                    dtype="float32")
        return orientation, new__world_frame
    # Orientation: Bottom Left
    elif orientation == 1:
        new__world_frame = np.array([[0, dimension - 1], [0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1]],
                                    dtype="float32")
        return orientation, new__world_frame
    # Orientation: Top Right
    elif orientation == 2:
        new__world_frame = np.array([[dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1], [0, 0]],
                                    dtype="float32")
        return orientation, new__world_frame

    return orientation, new__world_frame


def get_tag_id(img_frame, orientation):
    """
    :param img_frame:
    :param orientation:
    :return:
    """
    tag_id = ''
    keys = []
    # Check get_H_matrix function in superimpose.py for orientation notation
    if orientation == 0:
        keys = [1, 0, 2, 3]
    elif orientation == 1:
        keys = [3, 1, 0, 2]
    elif orientation == 2:
        keys = [2, 3, 1, 0]
    elif orientation == 3:
        keys = [0, 2, 3, 1]
    structure = {0: [200, 250, 200, 250], 1: [150, 200, 200, 250], 2: [200, 250, 150, 200], 3: [150, 200, 150, 200]}

    total = 0
    for key in keys:
        for i in range(structure[key][0], structure[key][1]):
            for j in range(structure[key][2], structure[key][3]):
                total += img_frame[i][j]

        if (total / 2500) > 220:
            tag_id += '1'
        else:
            tag_id += '0'
    return tag_id
