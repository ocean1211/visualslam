#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
MAP MANAGMENT
---------------------

"""

# import funkcí
import sys
import os.path
import numpy as np
import scipy

# ziskani cest ke skriptum
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../"))

# nacteni dalsich soucasti
from models import *
from utils import *
from features import *

def manage(model, featureList, camera, frame, min_num_of_features, dt): 
	deleteFeatures(model, featureList)
	measured = 0
	for i in range(features.count()):
		if ((featureList.features[i].li_inlier == 1)||(featureList.features[i].hi_inlier == 1)):
			measured = measured + 1
	featureList.updateInfo()
	inverseDepth2Cartesian(model, featureList)
	if measured == 0
		initializeFeatures(model, featureList, camera, frame, min_num_of_features, dt)
	else
		initializeFeatures(model, featureList, camera, frame, min_num_of_features - measured, dt)
	

def deleteFeatures(model, featureList):
	deletion_list = []
	# Nalezeni bodu pro vymazani z mapy
	for i in range(featureList.count()):
		if(featureList.features[i].times_measured < 0.5 * featureList.features[i].times_predicted) && (featureList.features[i].times_predicted > 5):
			deletion_list.append(i)
	if (len(deletion_list) > 0) :
		for i in deletion_list[:]:
			offset = model.deleteFeature(i,featureList.features[i])
			featureList.deleteFeature(i, offset)


def inverseDepth2Cartesian(model, featureList):
	lin_index_threshold = 0.1
	for i in range(featureList.count()):
		if featureList.features[i].type = 0 :
			begin = featureList.features[i].begin
			std_rho = np.sqrt(model.P[begin + 5, begin + 5])
			rho = model.x[begin + 5]
			std_d = std_rho/np.pow(rho,2)
			theta = model.x[begin + 3]
			phi = model.x[begin + 4]
			mi = m(phi, theta)
			x_c1 = np.zeros([3,1], dtype = np.double)
			x_c1 = model.x[begin:begin + 3]
			x_c2 = np.zeros([3,1], dtype = np.double)
			x_c2 = model.x[0:3]
			p = id2Cart(model.x[begin:begin + 6])
			d_c2p = np.linalg.norm(p-x_c2)
			cos_alpha = ((p-x_c1).T*(p-x_c2))/(np.linalg.norm(p-x_c1)*np.linalg.norm(p-x_c2))
			linearity_index = 4*std_d*cos_alpha/d_c2p;
			if linearity_index<linearity_index_threshold:
				size_x_old = model.x.shape[0]
				# Stavovy vektor
				x1 = np.zeros([model.x.shape[0], 1], dtype = np.double)
				x1[0:begin] = model.x[0:begin]
				x1[begin:begin+3] = p
				x1[begin + 3:] = model.x[begin + 3:]
				model.x = x1
        # Kovariance
				dm_dtheta = np.zeros([3,1], dtype = np.double)
				dm_dtheta[0:3] = [ np.cos(phi)*np.cos(theta), 0, -np.sin(phi)*np.cos(theta)]
				dm_dphi = np.zeros([3,1], dtype = np.double)
				dm_dphi[0:3] = [-np.sin(phi)*np.sin(theta), -np.cos(phi), -np.sin(phi)*np.cos(theta) ]
				J = np.zeros([3,6], dtype = np.double)
				J[0:3,0:3] = np.identity(3)
				J[0:3,3] = (1/rho)*dm_dtheta
				J[0:3,4] = (1/rho)*dm_dphi
				J[0:3,5] = -mi/np.pow(rho,2)				
				J_all = np.zeros([model.x.shape[0], size_x_old], dtype = np.double)
				J_all[0:begin,0:begin] = np.identity(begin-1, dtype = np.double)
				J_all[begin:begin+3, begin:begin+6] = J
				J_all[begin + 6:, begin + 6:] = np.identity(size_x_old-(begin+6))
        model.P = J_all*P*J_all.T;        
				featureList.features[i].type = 1
				return # Zjistit proc se jich neda delat vic najednou [clanek o teto problematice ve ctecce]


def id2Cart(idFeature):
	rw = idFeature[0:3]
	theta = idFeature[3]
	phi = idFeature[4]
	rho = idFeature[5]

	m = np.zeros([3,1], dtype = np.double)
	m = [np.cos(phi)*np.sin(theta), -np.sin(phi), np.cos(phi)*np.cos(theta)]
	cartesian = np.zeros([3,1], dtype = np.double)
	cartesian  = [rw[0] + (1/rho)*m[0], rw[1] + (1/rho)*m[1], rw[2] + (1/rho)*m[2]]
	return cartesian

def initializeFeatures(model, featureList, camera, frame, min_num_of_features, dt):
	max_attempts = 50
	attempts = 0
	initialized = 0

	while ((initialized < min_num_of_features) && (attempts < max_attempts)):
		attempts = attempts + 1
		# TODO inicializace jednoho landmarku
		uv = featureList.initializeFeature(model, featureList, camera, frame, dt)
		if (uv.shape[1] ~= 0)
			initialized = initialized + 1


