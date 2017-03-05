import os.path
import matplotlib.image as mpimg
import numpy as np
import cv2

from camera import undistort

def get_perspective_transform_matrices():
    if not hasattr(get_perspective_transform_matrices, 'M'):
        if not os.path.exists('test_images/straight_lines1_undistorted.jpg'):
            image = undistort(mpimg.imread('test_images/straight_lines1.jpg'))
            mpimg.imsave('test_images/straight_lines1_undistorted.jpg', image)
        else:
            image = mpimg.imread('test_images/straight_lines1_undistorted.jpg')

        nx, ny = image.shape[:2][::-1]

        src = np.float32([(0,ny-93-1), (572-1-9,ny-275-1),
                          (nx-562-1, ny-275-1), (nx-1, ny-93-1)])
        dst = np.float32([src[0], (src[0][0], 0),
                          (src[3][0], 0), src[3]])

        get_perspective_transform_matrices.M = cv2.getPerspectiveTransform(src, dst)
        get_perspective_transform_matrices.Minv = cv2.getPerspectiveTransform(dst, src)

    return get_perspective_transform_matrices.M, get_perspective_transform_matrices.Minv


def transform(image):
    M, _ = get_perspective_transform_matrices()
    return cv2.warpPerspective(image, M, image.shape[:2][::-1])

def inverse_transform(image):
    _, Minv = get_perspective_transform_matrices()
    return cv2.warpPerspective(image, Minv, image.shape[:2][::-1])
