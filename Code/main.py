import numpy as np
import cv2 as cv
import random
from copy import copy
import detector
import superimpose as si
import draw_3d

if __name__ == '__main__':
    # Create cv object of video-capture
    tag = cv.VideoCapture('videos/Tag2.mp4')
    lena = cv.imread('images/Lena.png')
    lena_x, lena_y, lena_channels = lena.shape
    lena_gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)
    x_ref, y_ref = 400, 400
    three_d_axis = np.float32(
        [[0, 0, 0], [0, 500, 0], [500, 500, 0], [500, 0, 0], [0, 0, -300], [0, 500, -300], [500, 500, -300],
         [500, 0, -300]])
    while True:
        # Read the video frame by frame
        video_frame_exists, video_frame = tag.read()
        if not video_frame_exists:
            break
        # Store original frame for comparison
        vf_original = copy(video_frame)
        vf_grayscale = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
        # Store all size parameters of the frame
        rows, cols, channels = video_frame.shape
        # Get contours from the video frame
        contours = detector.find_contours(video_frame)
        # Define array to store max contour area
        max_contour_area = np.zeros((1, 1, 2), dtype=int)
        for contour in contours:
            contour_area = cv.contourArea(contour)
            contour_poly_curve = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, closed=True), closed=True)
            if 2000 < contour_area <= 22600:
                # Filtering Contours with 4 corners
                if len(contour_poly_curve) == 4:
                    # Draw the selected Contour matching the criteria fixed
                    cv.drawContours(video_frame, [contour], 0, (0, 0, 225), 2)
                    # Warp the video frame
                    h_mat, _ = si.get_h_matrices(contour_poly_curve, x_ref, y_ref)
                    warp_image, wi_coords = si.warp_image(h_mat, x_ref, y_ref)
                    vf_superimpose = si.superimpose_image(warp_image, vf_original, vf_grayscale, wi_coords)
                    _, vf_si_thresh = cv.threshold(vf_superimpose, 220, 255, cv.THRESH_BINARY)
                    # Get orientation and tag ID
                    orientation = detector.get_orientation(vf_superimpose)
                    tag_id = detector.get_tag_id(vf_superimpose, orientation)
                    print(orientation, tag_id)
                    # Warp Lena onto the video
                    _, inv_h = si.get_h_matrices(contour_poly_curve, lena_x, lena_y, orientation)
                    # warp_image, wi_coords = si.warp_image(inv_h, rows, cols)
                    # lena_superimpose = si.superimpose_image(warp_image, vf_original, lena_gray, wi_coords)
                    # _, lena_si_thresh = cv.threshold(lena_superimpose, 220, 255, cv.THRESH_BINARY)
                    # mask = np.zeros((rows, cols, channels), dtype=np.uint8)
                    # roi_corners2 = np.int32(contour_poly_curve)
                    # ignore_mask_color2 = (255,) * channels
                    # mask = cv.bitwise_not(mask)
                    # masked_image2 = cv.bitwise_and(vf_original, mask)
                    # lena_result = cv.bitwise_or(lena_si_thresh, masked_image2)
                    # Draw cube
                    r_mat, t, k_mat = draw_3d.get_krt_matrix(inv_h)
                    three_d_points, _ = cv.projectPoints(three_d_axis, r_mat, t, k_mat, np.zeros(4))
                    three_d_points = np.int32(three_d_points).reshape(-1, 2)
                    cv.drawContours(vf_original, [three_d_points[:4]], -1, (0, 255, 0), 3)
                    for i, j in zip(range(4), range(4, 8)):
                        vf_original = cv.line(vf_original, tuple(three_d_points[i]), tuple(three_d_points[j]),
                                              (0, 0, 255), 3)
                    vf_original = cv.drawContours(vf_original, [three_d_points[4:]], -1, (255, 0, 0), 3)

        cv.imshow('Original Frame', video_frame)
        cv.imshow('Superimpose', vf_original)
        key = cv.waitKey(1)
        if key == 27:
            break

