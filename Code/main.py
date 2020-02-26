# Import standard libraries
import numpy as np
import cv2 as cv
from copy import copy
# Import custom function scripts
import detector
import superimpose as si
import draw_3d

if __name__ == '__main__':
    # Define constants for entire project
    ref_dimension = 400
    ref_world_frame = np.array([[0, 0], [ref_dimension - 1, 0], [ref_dimension - 1, ref_dimension - 1],
                                [0, ref_dimension - 1]], dtype="float32")
    three_d_axis = np.float32([[0, 0, 0], [0, 500, 0], [500, 500, 0], [500, 0, 0], [0, 0, -300], [0, 500, -300],
                               [500, 500, -300], [500, 0, -300]])
    # Create cv objects for video and Lena image
    tag = cv.VideoCapture('videos/Tag2.mp4')
    lena = cv.imread('images/Lena.png')
    lena = cv.resize(lena, (ref_dimension - 1, ref_dimension - 1))
    lena_x, lena_y, lena_channels = lena.shape
    lena_gray = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)
    # Define output video object
    video_format = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_output = cv.VideoWriter('videos/cube_tag2.avi', video_format, 10, (640, 480))
    total_frames = 0
    # Begin loop for iterate through each frame of the video
    while True:
        # Read the video frame by frame
        video_frame_exists, video_frame = tag.read()
        if not video_frame_exists:
            break
        total_frames += 1
        # if total_frames >= 370:
        # Store original frame for comparison
        vf_original = copy(video_frame)
        vf_grayscale = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
        # Store all size parameters of the frame
        rows, cols, channels = video_frame.shape
        # Get contours from the video frame
        contours = detector.find_contours(video_frame)
        # Lena list and arrays
        warped_lena_list = []
        lena_superimpose = np.zeros((rows, cols, channels), dtype='uint8')
        # Define array to store max contour area
        max_contour_area = np.zeros((1, 1, 2), dtype=int)
        for contour in contours:
            contour_area = cv.contourArea(contour)
            contour_poly_curve = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, closed=True), closed=True)
            if 2000 < contour_area <= 22000:
                # Filtering Contours with 4 corners
                if len(contour_poly_curve) == 4:
                    # Draw the selected Contour matching the criteria fixed
                    # cv.drawContours(video_frame, [contour], 0, (0, 0, 225), 2)
                    # Warp the video frame
                    h_mat, _ = si.get_h_matrices(contour_poly_curve, ref_dimension, ref_dimension)
                    # warp_image, wi_coords = si.warp_image(h_mat, ref_dimension, ref_dimension)
                    # vf_superimpose = si.superimpose_image(warp_image, vf_original, vf_grayscale, wi_coords)
                    # _, vf_si_thresh = cv.threshold(vf_superimpose, 220, 255, cv.THRESH_BINARY)
                    vf_warp = si.get_warp_perspective(cv.transpose(video_frame), h_mat, (ref_dimension,
                                                                                         ref_dimension))
                    vf_warp_gray = cv.cvtColor(cv.transpose(vf_warp), cv.COLOR_BGR2GRAY)
                    # Get orientation and tag ID
                    orientation, new_world_frame = detector.get_tag_orientation(vf_warp_gray, ref_dimension)
                    tag_id = detector.get_tag_id(vf_warp_gray, orientation)
                    print(total_frames, orientation, tag_id)
                    # Warp Lena onto the video frame
                    h_mat, inv_h = si.get_h_matrices(contour_poly_curve, lena_x, lena_y, orientation)
                    warp_image, wi_coords = si.warp_image(inv_h, lena_x, lena_y)
                    lena_superimpose = si.superimpose_image(warp_image, video_frame, lena, wi_coords)
                    # lena_warp = si.get_warp_perspective(cv.transpose(lena), h_mat, (lena_x, lena_y))
                    # lena_warp = cv.transpose(lena_warp)
                    # warped_lena_list.append(cv.transpose(lena_warp))
                    # lw_grayscale = cv.cvtColor(lena_warp, cv.COLOR_BGR2GRAY)
                    # _, lena_gray = cv.threshold(lw_grayscale, 0, 250, cv.THRESH_BINARY_INV)
                    # mask = np.zeros((rows, cols, channels), dtype=np.uint8)
                    # roi_corners2 = np.int32(contour_poly_curve)
                    # ignore_mask_color2 = (255,) * channels
                    # mask = cv.bitwise_not(mask)
                    # masked_image2 = cv.bitwise_and(vf_original, mask)
                    # lena_result = cv.bitwise_or(lena_si_thresh, masked_image2)
                    # Get projection matrix to draw cuboid onto the video frame
                    r_mat, t, k_mat = draw_3d.get_krt_matrix(inv_h)
                    three_d_points, _ = cv.projectPoints(three_d_axis, r_mat, t, k_mat, np.zeros(4))
                    three_d_points = np.int32(three_d_points).reshape(-1, 2)
                    cv.drawContours(vf_original, [three_d_points[:4]], -1, (0, 255, 0), 3)
                    for i, j in zip(range(4), range(4, 8)):
                        vf_original = cv.line(vf_original, tuple(three_d_points[i]), tuple(three_d_points[j]),
                                              (0, 0, 255), 3)
                    vf_original = cv.drawContours(vf_original, [three_d_points[4:]], -1, (255, 0, 0), 3)
        # lena_superimposed = si.superimpose_image(warped_lena_list, vf_original, lena_superimposed, rows, cols)
        # Write output into a video
        cv.imshow("Lena", lena_superimpose)
        cv.imshow("Project", vf_original)
        key = cv.waitKey(1)
        if key == 27:
            break
        if total_frames > 1:
            video_output.write(vf_original)
    print(total_frames)
    tag.release()
    video_output.release()
    cv.destroyAllWindows()

