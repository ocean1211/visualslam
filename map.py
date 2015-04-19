#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
MAP MANAGMENT
---------------------

"""

# imports
import math
import mathematics
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
        rows2delete = self.features[f_id]['type'] * 3
        begin = self.features[f_id]['begin']
        # IN P
        P2 = np.zeros([P.shape[0]- rows2delete, P.shape[1]- rows2delete])
        P2[0:begin, 0:begin] = P[0:begin, 0:begin]
        if (P2.shape[0] > begin + rows2delete):
            P2[0:begin, begin:] = P[0:begin, begin + rows2delete:]
            P2[begin:, 0:begin] = P[begin + rows2delete:, 0:begin]
            P2[begin:, begin:] = P[begin + rows2delete:, begin + rows2delete:]
        P = P2
        # IN X
        x2 = np.zeros(x.shape[0] - rows2delete)
        x2[0:begin] = x[0:begin]
        x2[begin:] = x[begin + rows2delete:]        
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
        
        while ((initialized < num_of_features) and (attempts < max_attempts)):
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
                yi = x[begin:begin + 3]
                hi = self.hi(yi, r, R, inparams)
            else:
                yi = x[begin:begin + 6]
                hi = self.hi(yi, r, R, inparams) 
            self.features[i]['h'] = hi
        pass
    
    # ID = inverse depth
    def hi(self, yi, r, R, inparams):
        if yi.shape[0] > 3:
            theta, phi, rho = yi[[3, 4, 5]]
            mv = mathematics.m
            hrl = np.dot(R.T, (yi[0:3] - r)) * rho + mv
        else:
            hrl = np.dot(R.T, (yi - r))  
        if ((math.atan2(hrl(0), hrl(2)) * 180 / math.pi < -60) or \
            (math.atan2(hrl(0), hrl(2)) * 180 / math.pi > 60) or \
            (math.atan2(hrl(1), hrl(2)) * 180 / math.pi < -60) or \
            (math.atan2(hrl(1), hrl(2)) * 180 / math.pi > 60)):                            
            return None        
        yi2 = yi(2)
        if (yi2 == 0):
            yi2 += 1
        uv_u1 = inparams['u0'] + (yi(0) / yi2) * inparams['fku']
        uv_u2 = inparams['v0'] + (yi(1) / yi2) * inparams['fkv']
        uv_u = np.zeros(2, dtype=np.float32)
        uv_u[:] = [uv_u1, uv_u2]
        uv_d = mathematics.distort_fm(uv_u, inparams) 
        
        # is feature visible ?
        if ((uv_d[0] > 0) and (uv_d[0] < inparams['w']) and \
            (uv_d[1] > 0) and (uv_d[1] < inparams['h'])):
            return uv_d
        else:
            return None        
    
    def predict_features_appearance(self, x, inparams):
        r = x[0:3]
        R = mathematics.q2r(x[3:7])
        for i in range(len(self.features)):                
            if self.features[i]['h'] != None:
                begin = self.features[i]['begin']                
                xyz_w = np.zeros(3, dtype=np.float32)
                if self.features[i]['type'] == 1:
                    xyz_w = x[begin:begin + 3]
                else:
                    xyz_w = mathematics.id2cartesian(x[begin:begin + 6])                    
                    self.feature[i]['patch'] = self.pred_patch_fc(self.features[i], r, R, xyz_w, inparams)
                    
    def pred_patch_fc(self, feature, r, R, xyz, inparams):
        pass
    
    def addFeaturesID(self, uvd, x, P, newFeature, inparams, init_rho, std_rho):
        xv = x[0:13]
        xf = mathmeatics.hinv(uvd, xv, inparams, init_rho)
        x = np.concatenate(x, xf)
        P = self.addFeatureCovarianceID()
        return x, P
    
    def addFeatureCovarianceID(self, uvd, x, P, inparams, init_rho, std_rho):
        R = mathematics.q2r(x[3:7])   
        uv_u = mathematics.undistort_fm(uvd, inparams)
        XYZ_c = np.zeros(3, np.float32)
        x_c = (-(inparams['u0']-uv_u[0]) / inaparams['fku'])
        y_c = (-(inparams['v0']-uv_u[1]) / inaparams['fkv'])        
        XYZ_c[:] = [x_c, y_c, 1]
        XYZ_w = np.dot(R, XYZ_c)
        dtheta_dgw = np.zeros(3, np.float32)
        dtheta_dgw[:] = [(XYZ_w[0] / (XYZ_w[0] ** 2 + XYZ_w[2] ** 2)), 0, (-XYZ_w[0] / (XYZ_w[0] ** 2 + XYZ_w[2] ** 2))]
        dphi_dgw = np.zeros(3, np.float32)
        dphi_dgw[:] = [(XYZ_w[0] * XYZ_w[1]) / ((np.sum(XYZ_w ** 2)) * np.sqrt(XYZ_w[0] ** 2 + XYZ_w[2] ** 2)), 
            -np.sqrt(XYZ_w[0] ** 2 + XYZ_w[2] ** 2) / (np.sum(XYZ_w ** 2)),
            (XYZ_w[2] * XYZ_w[1]) / ((np.sum(XYZ_w ** 2)) * np.sqrt(XYZ_w[0] ** 2 + XYZ_w[2] ** 2))]
        dgw_dqwr = mathematics.dRq_times_a_by_dq(q, XYZ_c)
        dtheta_dqwr = np.dot(dtheta_dgw.T, dgw_dqwr)
        dphi_dqwr = np.dot(dphi_dgw.T, dgw_dqwr)
        dy_dqwr = np.zeros([6, 4], dtype=np.float32)
        dy_dqwr[3, 0:4] = dtheta_dqwr.T
        dy_dqwr[4, 0:4] = dphi_dqwr.T
        dy_drw = np.zeros([6, 3], dtype=np.float32)
        dy_drw[0:3, 0:3] = np.identity(3, dtype=np.float32)
        dy_dxv = np.zeros([6, 13], dtype=np.float32)
        dy_dxv[0:6, 0:3] = dy_drw
        dy_dxv[0:6, 4:8] = dy_drw  
        dyprima_dgw = np.zeros([5, 3], dtype=np.float32)
        dyprima_dgw[3, 0:3] = dtheta_dgw.T
        dyprima_dgw[4, 0:3] = dphi_dgw.T
        dgw_dgc = R
        dgc_dhu = np.zeros([3, 2], dtype=np.float32)
        dgc_dhu[:] = [[1 / inparams['fku'], 0, 0][1 / inparams['fkv'], 0, 0]]
        dhu_dhd = mathematics.jacob_undistord_fm(uv, inparams)
        dyprima_dhd = np.dot(np.dot(np.dot(dyprima_dgw, dgw_dgc), dgc_dhu), dhu_dhd)
        dy_dhd = np.zeros([6, 3], dtype=np.float32)
        dy_dhd[0:5, 0:2] = dyprima_dhd
        dy_dhd[5, 2] = 1         
        Padd = np.zeros([3, 3], dtype=np.float32)
        Padd[0:2, 0:2] = np.identity(2, dtype=np.float32) * inparams['sd'] ** 2
        Padd[2, 2] = std_rho
        P2 = np.zeros([P.shape[0] + 6, P.shape[1] + 6], dtype=np.float32)
        P2[0:P.shape[0], 0:P.shape[1]] = P
        P2[P.shape[0]:P.shape[0] + 6, 0:13] = np.dot(dy_dxv, P[0:13, 0:13])
        P2[0:13, P.shape[1]:P.shape[1] + 6] = np.dot(P[0:13, 0:13], dy_dxv.T)
        if (P.shape[0] > 13):
            P2[P.shape[0]:P.shape[0] + 6, 13:P.shape[1]-13] = np.dot(dy_dxv, P[0:13, 13:])
            P2[13:P.shape[0]-13, P.shape[1]:P.shape[1] + 6] = np.dot(P[13:, 0:13], dy_dxv.T)
        P2[P.shape[0]:P.shape[0] + 6, P.shape[1]:P.shape[1] + 6] = np.dot(np.dot(dy_dxv, P[0:13, 0:13]), dy_dxv.T) + \
            np.dot(np.dot(dy_dhd, Padd), dy_dhd.T)
        return P2
    
    def addFeatureInfo(self, uv, im, x_res, step, nF):
        half_size_wi = 20
        x1 = min(max(uv[0]-half_size_wi, 0), im.shape[1])
        x2 = min(max(uv[0]-half_size_wi + 41, 0), im.shape[1])
        y1 = min(max(uv[1]-half_size_wi, 0), im.shape[0])
        y2 = min(max(uv[1]-half_size_wi + 41, 0), im.shape[0])        
        im_patch = im[y1:y2, x1:x2]
        begin = 0
        if (len(self.features) != 0):
            begin = self.features[-1]['begin'] + self.features[-1]['type'] * 3
        self.features.append(self.newFeature(im_patch, uv, x_res, step, nF, begin))        
    
    def generateStatePattern(self, pattern, zID, zXYZ, x):
        
        pass
    
    def set_as_most_supported_hypothesis(self, pos_li_inliers_ID, pos_li_inliers_XYZ):
        jID = 0
        jXYZ = 0

        for i in range(len(self.features)):            
            self.features[i]['low_innovation_inlier'] = 0;            
            if self.features[i]['z'] != None:
                if self.features[i]['type'] == 1:
                    if pos_li_inliers_XYZ[jXYZ] == 1:
                        self.features[i]['low_innovation_inlier'] = 1;
                        jXYZ += 1
                else:
                    if pos_li_inliers_XYZ[jID] == 1:
                        self.features[i]['low_innovation_inlier']                    
                        jID += 1
    
    def calculateDerivatives(self, x, inparams):
        for i in range(len(self.features)):       
            if self.features[i]['h'] != None:
                begin = self.features[i]['begin']
                f = self.features[i]
                if f['type'] == 1:
                    y = x[begin:begin + 3]     
                    self.features[i]['H'] = self.H_XYZ(x[0:13], y, inparams, x.shape[0], f['begin'])                    
                else:
                    y = x[begin:begin + 6]   
                    self.features[i]['H'] = self.H_ID(x[0:13], y, inparams, x.shape[0], f['begin'])                    
    
    def H_XYZ(self, xv, y, inparams, sizeX, begin):
        zi = f['h']        
        num_of_features = len(self.features)
        Hi = np.zeros([2, sizeX], dtype=np.float32)
        Hi[:, 0:13] = xyz_dh_dxv(inparams, xv, y, zi)
        Hi[:, begin:begin + 3] = xyz_dh_dx(inparams, xv, y, zi)        
        return Hi

    def H_ID(self, xv, y, inparams, f):
        zi = f['h']        
        num_of_features = len(self.features)
        Hi = np.zeros([2, sizeX], dtype=np.float32)
        Hi[:, 0:13] = id_dh_dxv(inparams, xv, y, zi)
        Hi[:, begin:begin + 6] = id_dh_dx(inparams, xv, y, zi)        
        return Hi   