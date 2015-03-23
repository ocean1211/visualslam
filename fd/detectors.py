#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Detectors
---------------------

"""

# import funkcí
import sys
import os.path
import numpy as np
import scipy
import cv2


def detect(dType, frame, parameters = {}):
    if dType == "sift":
        return sift(frame)
    elif dType == "surf":
        return surf(frame, parameters)
    elif dType == "fast":
        return fast(frame, parameters)
    elif dType == "harris":
        return harris(frame, parameters)
    elif dType == "shiTomasi":
        return shiTomasi(frame, parameters)        
    
    return None


def fast(img, parameters): # TODO > Zjistit jak nastavit neighborhood
	fastDetector = cv2.FastFeatureDetector()
	fastDetector.setInt('threshold', parameters['threshold'])
	fastDetector.setBool('nonmaxSuppression', parameters['nonmaxSuppression'])
	kp = fastDetector.detect(img, None)
	return kp

def sift(img):
	sift = cv2.SIFT()
	kp = sift.detect(img, None)
        return kp 

def surf(img, parameters):
	surf = cv2.SURF(parameters['hessianTreshold'])
	kp = surf.detect(img, None)
	return kp

