#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Detectors
---------------------

"""

# imports
import cv2
    
    
def detect(detector, image):
    """
        list of detectors : 
            ["FAST","STAR","SIFT","SURF","ORB","MSER","GFTT","HARRIS"]
    
    """
    det = cv2.FeatureDetector_create(detector)
    kpts = det.detect(image)
    return kpts    
