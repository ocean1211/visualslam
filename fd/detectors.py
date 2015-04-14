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
        kp, features =  sift(frame)
        return kp, features
    elif dType == "surf":
        kp, features =  surf(frame)
        return kp, features
    
    return None


def fast(img, parameters): # TODO > Zjistit jak nastavit neighborhood
	fastDetector = cv2.FastFeatureDetector()
	fastDetector.setInt('threshold', parameters['threshold'])
	fastDetector.setBool('nonmaxSuppression', parameters['nonmaxSuppression'])
	kp = fastDetector.detect(img, None)
	return kp

def sift(img):
	sift = cv2.SIFT()
        kp, features = sift.detectAndCompute(img, None)
        return kp, features

def surf(img):
	surf = cv2.SURF()
        kp, features = surf.detectAndCompute(img, None)
	return kp, features

