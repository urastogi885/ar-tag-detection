import numpy as np
import cv2 as cv
import random
from copy import copy
import detector
import superimpose
import draw_3d

if __name__ == '__main__':
    # Create cv object of video-capture
    tag = cv.VideoCapture('videos/multipleTags.mp4')
    # tag = cv.imread('images/video_frame.png')
    # ar_tag = None
    # ar_tag = cv.cornerHarris(cv.cvtColor(tag, cv.))
    print_once = True
    while True:
        # Read the video frame by frame
        video_frame_exists, video_frame = tag.read()
        if not video_frame_exists:
            break
        # Store original frame for comparison
        vf_original = copy(video_frame)
        # Store all size parameters of the frame
        rows, cols, channels = video_frame.shape
        # Get contours from video frame
        vf_grayscale = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
        _, vf_threshold = cv.threshold(cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY), 200, 255, 0)
        contours, _ = cv.findContours(vf_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Define array to store max contour area
        max_contour_area = np.zeros((1, 1, 2), dtype=int)
        for contour in contours:
            contour_area = cv.contourArea(contour)
            contour_poly_curve = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, closed=True), closed=True)
            if 2000 < contour_area <= 17250:
                # Filtering Contours with 4 corners
                if len(contour_poly_curve) == 4:
                    # Draw the selected Contour matching the criteria fixed
                    cv.drawContours(video_frame, [contour], 0, (0, 0, 225), 2)

        cv.imshow('Original Frame', video_frame)
        key = cv.waitKey(1)
        if key == 27:
            break

