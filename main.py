import cv2 as cv
import detector
import superimpose
import draw_3d

if __name__ == '__main__':
    # Just stub implementation
    # Checking stub functions in detector
    template_img = cv.imread('images_and_videos/ref_marker.png')
    contour_img = detector.draw_contours(template_img)
    orientation = detector.get_orientation(contour_img)
    h_mat = detector.get_h_matrix(orientation, 0, 0)
    tag_id = detector.get_tag_td(contour_img, orientation)
    # Checking stub functions in tag superimpose
    superimpose.warp_image()
    superimpose.superimpose_image()
    # Checking stub functions in tag_draw_3d
    draw_3d.get_krt_matrix()
    draw_3d.draw_cuboid()
