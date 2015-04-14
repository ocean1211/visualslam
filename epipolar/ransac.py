#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import random

def sampsons_error(x1, x2, F):
    """
        Computation of Sampson's error    
    """
    Fx1 = F*x
    Fx2 = F.T * x2    

    Se = ((x2.T * F * x1)^2)/((Fx1[1]**2) + (Fx1[2]**2)+(Fx2[1]**2)+(Fx2[2]**2))
    
def ransac(mat, kp1, kp2):
    p = 0
    
    Fc = np.zeros([3,3])
    olderror = 0
    
    while p < 2000:
        
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
        print u, s, v        
        s_1 = np.diag([1,1,0])        
        E = u * s_1 * v
        #F = np.linalg.inv(calib_mat.T) * E * np.linalg.inv(calib_mat.T)
        F = E 
        error = 0 
        for i in range(len(mat)):            
            error = error + sampsons_error(kp1[mat[i].queryIdx].pt, kp2[mat[i].trainIdx].pt, F)
            pass
        if error < olderror:
            olderror = error
            Fc = F
        p = p+1
    return Fc
        