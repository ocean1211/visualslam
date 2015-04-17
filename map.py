#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
MAP MANAGMENT
---------------------

"""

# imports
import numpy as np
import os.path
import sys
import math



import mathematics

class map:
    # konstruktor
    def __init__(self):     
        self.features
        pass
    
    def manage(self, x, P, inparams, frame, step, min_num_of_features=10):
        self.deleteFeatures(x, P)
        measured = 0
        for feature in self.features:
            if feature.low_innovation_inlier or feature.high_innovation_inlier:
                measured = measured + 1
        self.updateFeaturesInfo()
        x, P = self.inverseDepth2XYZ(x, P)
        if (measured < min_number_of_features):
            x, P = initializeFeatures(x, P, inparams, frame, inparams, min_number_of_features - measured);        
	return x, P
    
    def deleteFeatures(self, x, P):
        if len(self.list) == 0: return x, P;
        deleteMove = 0
        for i in range(len(self.features)):
            self.features[i]['begin'] = self.features[i]['begin'] - deleteMove
            if (self.features[i]['times_measured'] < 0.5 * self.features[i]['times_predicted']):
                deleteMove = deleteMove + self.features[i]['type'] * 2
                x, P = self.deleteFeature(x, P, i)
                self.features.pop(i)
	return x, P
    
    def deleteFeature(self, x, P, f_id):
        rows2delete = self.features[f_id]['type'] * 2
        begin = self.features[f_id]['begin']
        # IN P
        P2 = np.zeros([P.shape[0]- rows2delete, P.shape[1]- rows2delete])
        P2[0:begin,0:begin] = P[0:begin,0:begin]
        if (P2.shape[0] > begin + rows2delete) :
            P2[0:begin, begin:] = P[0:begin, begin + rows2delete:]
            P2[begin:, 0:begin] = P[begin + rows2delete:, 0:begin]
            P2[begin:, begin:] = P[begin + rows2delete:, begin + rows2delete:]
        P = P2
        # IN X
        x2 = np.zeros(x.shape[0] - rows2delete)
        x2[0:begin] = x[0:begin]
        x2[begin:] =  x[begin + rows2delete:]        
        x = x2
        return x, P
    
    def updateFeaturesInfo(self):
        for i in range(len(self.features)):            
            if (self.features[i]['h'] != None):
                self.features[i]['times_predicted'] += 1
            if (self.features[i]['low_innovation_inlier'] or self.features[i]['high_innovation_inlier']):
                self.features[i]['times_measured'] += 1
        self.features[i]['individually_compatible'] = 0
        self.features[i]['low_innovation_inlier'] = 0
        self.features[i]['high_innovation_inlier'] = 0
        self.features[i]['h'] = None
        self.features[i]['z'] = None
        self.features[i]['H'] = None
        self.features[i]['S'] = None       
        pass
    
    def inverseDepth2XYZ(self, x, P):
        pass
    
    def initializeFeatures(self, x, P, inparams, frame, step, num_of_features):
        max_attempts = 50
        attempts = 0
        initialized = 0
        
        while ((initialiyed < num_of_features) and (attempts < max_attempts)):
            size = x.shape[0]
            attempts += 1
            x, P = initializeFeature(x, P, inparams, frame, step)
            if size < x.shape[0]:
                initialized += 1
        return x, P
    
    def initializeFeature(self, x, P, inparams, frame, step):
        pass     
    
    def predictCameraMeasurements(self, inparams, x):
        r = x[0:3]
        R = mathematics.q2r(x[3:7])
        begin = self.features[i]['begin']
        for i in range(len(self.features)):       
            if self.features[i]['type'] == 1:
                yi = x[begin:begin+3]
                hi = self.hi(yi, r, R, inparams)
            else:
                yi = x[begin:begin+6]
                hi = self.hi(yi, r, R, inparams) 
            self.features[i]['h'] = hi
        pass
    
    # ID = inverse depth
    def hi(self, yi, r, R, inparams):
        if yi.shape[0] > 3:
            theta, phi, rho= yi[[3,4,5]]
            mv = mathematics.m
            hrl = np.dot(R.T, (yi[0:3] - r))*rho + mv
        else:
            hrl = np.dot(R.T, (yi - r))  
        if ((math.atan2(hrl(0),hrl(2))*180/math.pi < -60) or \
                (math.atan2(hrl(0),hrl(2))*180/math.pi > 60) or \
                (math.atan2(hrl(1),hrl(2))*180/math.pi < -60) or \
                (math.atan2(hrl(1),hrl(2))*180/math.pi > 60)):                            
            return None        
        yi2 = yi(2)
        if (yi2 == 0):
            yi2 += 1
        uv_u1 = inparams.u0 + (yi(0)/yi2)* inparams.fku
        uv_u2 = inparams.v0 + (yi(1)/yi2)* inparams.fkv  
        uv_u = np.zeros(2, dtype= np.float32)
        uv_u[:] = [uv_u1, uv_u2]
        uv_d = mathematics.distort_fm(uv_u, inparams) 
        
        # is feature visible ?
        if ((uv_d[0] > 0) and (uv_d[0] < inparams.width) and \
                (uv_d[1] > 0) and (uv_d[1] < inparams.height)):
            return uv_d
        else:
            return None        
    
    def predict_features_appearance(self, x, inparams):
        r = x[0:3]
        R = mathematics.q2r(x[3:7])
        for i in range(len(self.features)):                
            if self.features[i]['type'] != None:
                begin = self.features[i]['begin']                
                xyz_w = np.zeros(3, dtype = np.float32)
                if self.features[i]['type'] == 1:
                    xyz_w = x[begin:begin+3]
                else :
                    xyz_w = mathematics.id2cartesian(x[begin:begin+6])                    
                self.feature[i]['patch'] = self.pred_patch_fc(self.features[i], r, R, xyz_w, inparams)
                    
    def pred_patch_fc(self, feature, r, R, xyz, inparams):
        pass
    
    def addFeaturesID(self, uv, x, p, newFeature, inparams, init_rho, std_rho):
        pass
    
    def addFeatureCovarianceID(self, uv, x, P, inparams, init_rho, std_rho):
        pass
    
    def addFeatureInfo(self, uv, im, x_res, step, newFeature):
        pass
    
    def generateStatePattern(self, pattern, yID, zXYZ, x):
        pass
    
    def set_as_most_supported_hypothesis(self, pos_li_inliers_ID, pos_li_inliers_XYZ):
        pass
    
    def calculateDerivatives(self, x, inparams):
        pass
    
    def calculate_Hi_XYZ(self, xv, y, inparams, f):
        pass
    
    def calculate_Hi_ID(self, xv, y, inparams, f):
        pass    