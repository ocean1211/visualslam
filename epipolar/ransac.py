#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import random
import sys

def sampsons_error(px1, px2, F):
    """
        Computation of Sampson's error    
    """
    x1 = np.ones(3)
    x1[0] = px1[0]
    x1[1] = px1[1]
    
    x2 = np.ones([3,1])
    x2[0] = px2[0]
    x2[1] = px2[1]    
    
    Fx1 = F.dot(x1)
    Fx2 = np.dot(F.T,x2) 
    Se = ((np.dot(np.dot(x2.T,F), x1))**2)/((Fx1[1]**2) + (Fx1[2]**2)+(Fx2[1]**2)+(Fx2[2]**2))
    return Se
    
def ransac(mat, kp1, kp2):
    p = 0
    
    Fc = np.zeros([3,3])
    olderror = sys.maxint
    
    while p < 300:
        
        samples = []
        samples_mat = np.zeros([8,6])
        A = np.zeros([8,9])
        
        for i in range(8):
            r = random.randint(0, len(mat))
            
            if r in samples:
                i = i - 1
                continue
            img1_idx = mat[i].queryIdx
            img2_idx = mat[i].trainIdx

            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
                            
            samples.append(r)
            
            samples_mat[i,:] = ([x1, y1, 1, x2, y2, 1])
            A[i,:] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

        u, s, v = np.linalg.svd((A))      
        
        E_est = v[:,-1]
        E_est = np.reshape(E_est, [3,3])
        eu, es, ev = np.linalg.svd((E_est))
        s_1 = np.diag([1,1,0])        
        E = np.dot(np.dot(eu,s_1), ev)
        #F = np.linalg.inv(calib_mat.T) * E * np.linalg.inv(calib_mat.T)
        F = E 
        error = 0 
        #print E
        for i in range(len(mat)):            
            error = error + sampsons_error(kp1[mat[i].queryIdx].pt, kp2[mat[i].trainIdx].pt, F)            
            pass    
        if error < olderror:
            olderror = error
            Fc = F
        p = p+1
    return Fc
        