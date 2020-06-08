# Import standard libraries
import cv2
import numpy as np
from sys import argv
from copy import deepcopy
# Import custom function scripts
from utils import draw_3d, detector, superimpose as si

script, video_location, output_destination = argv

if __name__ == '__main__':
    # Define constants for entire project
    ref_dimension = 400
    three_d_points = np.float32([[0, 0, 0], [0, 400, 0], [400, 400, 0], [400, 0, 0], [0, 0, -400],
                                [0, 400, -400], [400, 400, -400], [400, 0, -400]])
    # Create cv objects for video and Lena image
    tag = cv2.VideoCapture(str(video_location))
    lena = cv2.imread('images/Lena.png')
    lena = cv2.resize(lena, (ref_dimension - 1, ref_dimension - 1))
    lena_x, lena_y, lena_channels = lena.shape
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    # Define output video object
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_output = cv2.VideoWriter(str(output_destination), video_format, 20.0, (1920, 1080))
    total_frames = 0
    # Begin loop for iterate through each frame of the video
    while True:
        # Read the video frame by frame
        video_frame_exists, video_frame = tag.read()
        # Exit when the video file ends
        if not video_frame_exists:
            break
        total_frames += 1
        # Store original frame for comparison
        vf_original = deepcopy(video_frame)
        vf_grayscale = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        # Store all size parameters of the frame
        rows, cols, channels = video_frame.shape
        # Get contours from the video frame
        contours = detector.find_contours(video_frame)
        # Define array to store max contour area
        max_contour_area = np.zeros((1, 1, 2), dtype=int)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            contour_poly_curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, closed=True), closed=True)
            if 2000 < contour_area < 22600 and len(contour_poly_curve) == 4:
                # Draw the selected Contour matching the criteria fixed
                cv2.drawContours(vf_original, [contour], 0, (0, 0, 225), 1)
                # Warp the video frame
                h_mat, _ = si.get_h_matrices(contour_poly_curve, ref_dimension, ref_dimension)
                vf_warp = cv2.warpPerspective(video_frame, h_mat, (ref_dimension, ref_dimension))
                vf_warp_gray = cv2.cvtColor(vf_warp, cv2.COLOR_BGR2GRAY)
                # Get orientation and tag ID
                orientation = detector.get_tag_orientation(vf_warp_gray)
                tag_id = detector.get_tag_id(vf_warp_gray, orientation)
                # Display tag ID on each frame
                print(total_frames, orientation, tag_id)
                cv2.putText(vf_original, "Tag ID: " + tag_id, (contour_poly_curve[0][0][0] - 50,
                                                               contour_poly_curve[0][0][1] - 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)
                # Warp Lena onto the video frame
                h_mat, inv_h = si.get_h_matrices(contour_poly_curve, lena_x, lena_y, orientation)
                lena_warped = cv2.warpPerspective(lena, inv_h, (cols, rows))
                # lena_warped = cv2.transpose(lena_warped)
                # Invert the video frame within the region of the tag to superimpose Lena on the video frame
                lw_grayscale = cv2.cvtColor(lena_warped, cv2.COLOR_BGR2GRAY)
                _, lw_thresh = cv2.threshold(lw_grayscale, 0, 250, cv2.THRESH_BINARY_INV)
                vf_slotted = cv2.bitwise_and(vf_original, vf_original, mask=lw_thresh)
                # Superimpose lena onto the tag detected in the video frame
                vf_original = cv2.add(vf_slotted, lena_warped)
                # Get projection matrix to draw cuboid onto the video frame
                vf_original = draw_3d.draw_cube(vf_original, three_d_points, draw_3d.get_krt_matrix(inv_h))
                # Write output into a video frame by frame
                video_output.write(vf_original)
    # Destroy all cv objects
    tag.release()
    video_output.release()
    cv2.destroyAllWindows()
