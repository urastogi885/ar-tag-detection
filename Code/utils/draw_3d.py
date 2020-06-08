import numpy as np
from cv2 import line
from cv2 import projectPoints
from cv2 import drawContours


def get_krt_matrix(inv_h):
    """
    get matrices to transform 3-D points
    :param inv_h: inverse homography matrix
    :return: a tuple of 3 transformation matrices
    """
    k_mat = np.array(
        [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T
    inv_k_mat = np.linalg.inv(k_mat)
    b_mat = np.matmul(inv_k_mat, inv_h)
    b1 = b_mat[:, 0].reshape(3, 1)
    b2 = b_mat[:, 1].reshape(3, 1)
    r3 = np.cross(b_mat[:, 0], b_mat[:, 1])
    b3 = b_mat[:, 2].reshape(3, 1)
    scalar = 2 / (np.linalg.norm(inv_k_mat.dot(b1)) + np.linalg.norm(inv_k_mat.dot(b2)))
    t = scalar * b3
    r1 = scalar * b1
    r2 = scalar * b2
    r3 = (r3 * scalar * scalar).reshape(3, 1)
    r_mat = np.concatenate((r1, r2, r3), axis=1)
    return r_mat, t, k_mat


def draw_cube(video_frame, three_d_points, krt_matrices):
    """
    draw cube on the current video frame
    :param video_frame: current video frame
    :param three_d_points: pre-defined 3-D points
    :param krt_matrices: a tuple of 3 transformation matrices
    :return: frame with cube drawn on it
    """
    # Get new 3-D points using pre-defined 3-D points and transformation matrices
    three_d_points, _ = projectPoints(three_d_points, krt_matrices[0], krt_matrices[1], krt_matrices[2], np.zeros(4))
    three_d_points = np.int32(three_d_points).reshape(-1, 2)
    video_frame = drawContours(video_frame, [three_d_points[:4]], -1, (0, 255, 0), 2)
    for i, j in zip(range(4), range(4, 8)):
        video_frame = line(video_frame, tuple(three_d_points[i]), tuple(three_d_points[j]), (0, 0, 255), 2)
    video_frame = drawContours(video_frame, [three_d_points[4:]], -1, (255, 0, 0), 2)
    return video_frame
