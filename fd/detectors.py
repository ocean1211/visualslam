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



def fast(img, parameters): # TODO > Zjistit jak nastavit neighborhood
	fastDetector = cv2.FastFeatureDetector()
	fastDetector.setInt('threshold', parameters['threshold'])
	fastDetector.setBool('nonmaxSuppression', parameters['nonmaxSuppression'])
	kp = fastDetector.detect(img, None)
	features = keypointsToNp(kp)
	return features

def harris(img, parameters): # TODO > Correct!
	dst = cv2.cornerHarris(img,parameters['blockSize'], parameters['ksize'], parameters['k'])
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
	return corners

def shiTomasi(img, parameters):
	corners = cv2.goodFeaturesToTrack(img,parameters['maxCorners'],parameters['minQuality'], parameters['minDistBetweenFeatures'])
	corners = np.int0(corners)
	features = cornersToNp(corners)
	return features

def sift(img):
	sift = cv2.SIFT()
	kp = sift.detect(img, None)
	features = keypointsToNp(kp)
	return features

def surf(img, parameters):
	surf = cv2.SURF(parameters['hessianTreshold'])
	kp = surf.detect(img, None)
	features = keypointsToNp(kp)
	return features

def keypointsToNp(kp):
	features = np.zeros([len(kp), 3], dtype = np.double)
	j = 0
	for i in kp:
		features[j,1] = i.pt[0]
		features[j,0] = i.pt[1]	
		features[j,2] = i.response
		j += 1
	return features

def cornersToNp(kp):
	features = np.zeros([len(kp), 3], dtype = np.double)
	j = 0
	for i in kp:
		features[j,1] = i.ravel()[0]
		features[j,0] = i.ravel()[1]	
		features[j,2] = 0
		j += 1
	return features
