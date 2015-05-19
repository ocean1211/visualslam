#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí
import sys
import os.path
import numpy as np
import scipy

# nastaveni slozek, ve kterych se budou hledat funkce
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../utils/"))

import math_quaternion as mq

"""
Feature Class
-------------------------------

"""
class featureList:
	features = []
	
	def __init__(self): 
	
	def initializeFeature():
		half_patch_size_wi = 20
		half_patch_size_wm = 6
		excluded_band = half_patch_size_wi + 1
		max_init_attempts = 1
		init_box_size = np.zeros([2,1])
		init_box_size[:] = [60,40]
		init_box_semisize = init_box_size/2
		init_rho = 1
		std_rho = 1
		std_pxl = model.IMAGE_NOISE
		rand_attempt = 1
		not_empty_box = 1
		detected_new = 0

		# TODO : PREDICT CAMERA MEASUREMENTS
		uv_pred = np.zeros([2, self.count()])
		for i in range(self.count()):
			uv_pred[0:2,i] = self.features[i].h
		
		for i in range(max_init_attempts):
			if detected_new == 1
				break
		search_region_center = np.zeros([2,1], dtype = np.double)	
		search_region_center[0:2] = np.random.rand(2,1)
		search_region_center[0] = np.round(search_region_center[0]*((camera.colsNum-2)*(excluded_band-2)*init_box_semisize[0]))+excluded_band+initializing_box_semisize[0]
		search_region_center[1] = np.round(search_region_center[1]*((camera.colsNum-2)*(excluded_band-2)*init_box_semisize[1]))+excluded_band+initializing_box_semisize[1]
		
		# TODO : DETECT NEW FEATURE IN THE FRAME

		# TODO : Zbytek funkce...




	def deleteFeature(self, index, offset):
		for j in range(self.count()-i):
			k = i+j
			self.features[k].begin = features[k].begin - offset
		self.features.pop(i)

	def updateInfo(self):	
		for i in range(self.count()):
			if np.sum(self.features[i].h.shape) == 0:
				self.features[i].times_predicted = self.features[i].times_predicted + 1
			if ((self.features[i].li_inlier == 1)||(self.features[i].hi_inlier == 1)):
				self.features[i].times_measured = self.features[i].times_measured + 1
				self.features[i].individually_compatible = 0;
				self.features[i].li_inlier = 0;
				self.features[i].hi_inlier = 0;
				self.features[i].h = np.zeros([0,0], dtype = double)
				self.features[i].z = np.zeros([0,0], dtype = double)
				self.features[i].H = np.zeros([0,0], dtype = double)
				self.features[i].S = np.zeros([0,0], dtype = double)

	def inverseDepth2Cartesian():


	def count(self):
		return len(self.features)

class feature:

	patch_wi = np.zeros([1,1], dtype = np.double)
	patch_wm = np.zeros([1,1], dtype = np.double)
	position = np.zeros([3,1], dtype = np.double) # r_wc
	rotation = np.zeros([3,3], dtype = np.double) # R_wc
	half_patch_size_wi = 20
	half_patch_size_wm = 6
	times_predicted = 0
	times_measured = 0
	init_time = 0.0
	feature_type = 0 # 0 - inverse depth, 1 - cartesian
	init_measurement = np.zeros([2,1], dtype = np.double)
	uv_wi = np.zeros([2,1], dtype = np.double) # Mozna nepotrebne
	y = np.zeros([6,1], dtype = np.double)
	individually_compatible = 0
	li_inlier = 0 # low innovation inlier
	hi_inlier = 0 # high innovation inlier
	z = np.zeros([0,1], dtype = np.double)
	h = np.zeros([0,1], dtype = np.double)
	H = np.zeros([1,1], dtype = np.double)
	S = np.zeros([1,1], dtype = np.double) # Mozna nepotrebne
	R = np.identity(2, dtype = np.double)	
	size = 6
	measurement_size = 2
	begin = 0

	# konstruktor
	def __init__(self, pos, t, uv, newFeature, image, pBegin): #
		quat = mq.quaternion()
		self.position = pos[0:3]
		self.rotation = quat.rotationMatrix(pos[3:7])
		init_time = t
		init_measurement = uv
		uv_vi = uv 
		y = newFeature
		patch_wi = np.zeros([image.shape[0], image.shape[1]], dtype = np.double)
		patch_wm = np.zeros([image.shape[0], image.shape[1]], dtype = np.double)
		patch_wi = image		
		begin = pBegin		

