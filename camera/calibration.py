#! /usr/bin/python

# Calibration based on opencv example

__author__ = "neduchal"
__date__ = "$19.3.2015 9:15:52$"

import numpy as np
import cv2
import json

def calibrate(camera, filename, pattern_size, square_size, min_num_of_images = 10):
    """
    
    :param patern_size: velikost sachovnice
    :type patern_size: array(1,2)
    
    """
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        return -1
    
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    
    obj_points = []
    img_points = []
    h, w = 0, 0
    i = 0
    
    while i < min_num_of_images:
        print 'processing %s. image' % str(i),
        img = capture.grab()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
        h, w = img.shape[:2]
        
        cv2.imshow('Kalibrace', img)
        cv2.waitKey(0)
        
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            cv2.imwrite('pic'+ str(i)+ '_chess.bmp', vis)
            i = i + 1
        if not found:
            print 'chessboard not found'
            continue
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)    
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h))
        print "RMS:", rms
        print "camera matrix:\n", camera_matrix
        print "distortion coefficients: ", dist_coefs.ravel()
        
        data = {}
        
        data['rms'] = rms
        data['camera_matrix'] = camera_matrix
        data['dist_coefs'] = dist_coefs
        data['rvecs'] = rvecs
        data['tvecs'] = tvecs

        json.dump(data, filename)
        
        cv2.destroyAllWindows()
    


if __name__ == "__main__":
    calibrate(0, "params.json", (9,6), 24)

