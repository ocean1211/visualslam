#! /usr/bin/python
# -*- coding: utf-8 -*-

# Calibration based on opencv example

__author__ = "neduchal"
__date__ = "$19.3.2015 9:15:52$"

import numpy as np
import cv2
import json


class Camera:

    def __init__(self, camera):
        self.camPointer = cv2.VideoCapture(camera)
        if(not self.camPointer.isOpened()):
            print "NEPODARILO SE NACIST KAMERU"
        pass
    
    def frame(self, gray = False):
        frame = self.camPointer.read()
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def calibrate(camera, filename, pattern_size, square_size, min_num_of_images = 10, sleep_time = 500):
        """
            Kalibrace kamery pomocí šachovnice

        :param camera: index kamery (jako v openCV)
        :type camera: int
        :param filename: cesta k souboru pro ulozeni kalibracnich dat
        :type filename: string
        :param patern_size: velikost sachovnice
        :type patern_size: array(1,2)
        :param square_size: velikost ctverce na sachovnici
        :type square_size: int
        :param min_num_of_images: minimalni pocet snimku pro kalibraci
        :type min_num_of_images: int (default 10)

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

        capture.read()

        while i < min_num_of_images:
            print 'processing %s. image' % str(i),
            retval, img = capture.read()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
            h, w = img.shape[:2]

            cv2.imshow('Kalibrace', img)
            cv2.waitKey(500)

            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                i = i + 1
            if not found:
                print 'chessboard not found'
                continue
            print ""    

            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)    
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h))
        print "RMS:", rms
        print "camera matrix:\n", camera_matrix
        print "distortion coefficients: ", dist_coefs.ravel()

        data = {}

        data['rms'] = rms
        data['camera_matrix0'] = [camera_matrix[0,0], camera_matrix[0,1], camera_matrix[0,2]]
        data['camera_matrix1'] = [camera_matrix[1,0], camera_matrix[1,1], camera_matrix[1,2]]
        data['camera_matrix2'] = [camera_matrix[2,0], camera_matrix[2,1], camera_matrix[2,2]]
        data['dist_coefs'] = str(dist_coefs)
        #data['rvecs'] = rvecs
        #data['tvecs'] = tvecs

        fp = open(filename, 'w')
        json.dump(data, fp, indent = 4)
        fp.close()        
        cv2.destroyWindow("Kalibrace")


