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
        pass
    
    def updateFeaturesInfo(self):
        pass
    
    def inverseDepth2XYZ(self, x, P):
        pass
    
    def initializeFeatures(self, x, P, inparams, frame, step, num_of_features):
        pass 
    
    def initializeFeature(self, x, P, inparams, frame, step):
        pass     
    
    def predictCameraMeasurements(self, inparams, x):
        pass
    
    def hi_xyz(self, hi, yi, r, R, inparams):
        pass
    
    def hi_inverseDepth(self, hi, yi, r, R, inparams):
        pass
    
    def predict_features_appearance(self, x, inparams):
        pass
    
    def pred_patch_fc(self, patch, inparams):
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