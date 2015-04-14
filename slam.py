#! /usr/bin/python
# -*- coding: utf-8 -*-

# Imports
# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------
import numpy as np
import sys
import os
import cv2
import argparse
# -----------------------------------------------------------------------------
# Parts of system
# -----------------------------------------------------------------------------
from inout import *
from fd import *
from vis import *
from epipolar import *

configFile = "config/config1.json"


def main():
    # Nacteni konfigurace
    config = cjson.load(configFile)

    # Vybrani zdroje dat
    if config["source"][0] == "camera":
        source = camera.Camera(config["source"][1])
    else: 
        source = images.ImageDirectory(config["source"][1])
    
    # Nacteni prvniho snimku
    frame = source.frame(gray = True)
   
    surf = cv2.SURF() 
    # Zpracovani prvniho snimku :
    oldframe = frame 
    oldkp, oldfeatures = detectors.detect("surf", frame)


    # MAIN LOOP
    while not frame == None:
        # Zpracovani dat
        
        # Nacteni dalsiho snimku    
        frame = source.frame(gray = True)        
        
        # Feature detection
        
        #kp, features = surf.detectAndCompute(frame, None)
        kp, features = detectors.detect("surf", frame)
            
        #print np.min(features[:,0]), np.max(features[:,1])
        
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matchesP = bf.match(features,oldfeatures)        
          
        # Sort them in the order of their distance.
        matchesP = sorted(matchesP, key = lambda x:x.distance)

        points1 = np.zeros([len(matchesP), 2])
        points2 = np.zeros([len(matchesP), 2])
        i = 0
        for mat in matchesP:
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            (x1,y1) = kp[img1_idx].pt
            (x2,y2) = oldkp[img2_idx].pt
            points1[i,:] = np.array([x1, y1])
            points2[i,:] = np.array([x2, y2])
            i = i + 1
        
        #retval, mask = cv2.findFundamentalMat(points1, points2)
        retval, mask = cv2.findHomography(points1, points2, cv2.cv.CV_RANSAC)
        
        
        
        # Draw first 10 matches.
        #matches.drawMatches(frame, kp, oldframe, oldkp, matchesP[:10])
        
        
        # Zobrazeni aktualniho snimku        
        cv2.imshow("Data", frame)
        
        # Vyckavani + preruseni pri stisku q
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        oldframe = frame             
        oldfeatures = features 
        oldkp = kp
        pass
  
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
