import glob
import os.path
import pickle

import matplotlib.image as mpimg
import numpy as np
import cv2

chessboard = (9, 6) # (corners in a row, corners in a column)

def get_camera_params():
    if not hasattr(get_camera_params, 'mtx') or not hasattr(get_camera_params, 'dist'):
        images = glob.glob('camera_cal/calibration*.jpg')

        if os.path.exists('cameramatrix.p'):
            data = pickle.load(open('cameramatrix.p', 'rb'))
            get_camera_params.mtx = data['mtx']
            get_camera_params.dist = data['dist']
        else:
            objp = np.zeros((chessboard[0]*chessboard[1], 3), np.float32)
            # fixed corner coordinates
            objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            for i, fpath in enumerate(images):
                img = cv2.imread(fpath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(gray, chessboard, None)

                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)

            image = mpimg.imread('camera_cal/calibration1.jpg')
            img_shape = image.shape[:2][::-1]
            # get the camera calibration matrix based the calibration pattern in objpoints->imgpoints
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

            pickle.dump({'mtx': mtx, 'dist': dist}, open('cameramatrix.p', 'wb'))

            get_camera_params.mtx = mtx
            get_camera_params.dist = dist

    return get_camera_params.mtx, get_camera_params.dist


def undistort(image):
    mtx, dist = get_camera_params()
    return cv2.undistort(image, mtx, dist, None, mtx)
