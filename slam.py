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
    oldkp, oldfeatures = surf.detectAndCompute(frame, None)



    # MAIN LOOP
    while not frame == None:
        # Zpracovani dat
        
        # Nacteni dalsiho snimku    
        frame = source.frame(gray = True)        
        
        # Feature detection
        
        kp, features = surf.detectAndCompute(frame, None)
        
            
        #print np.min(features[:,0]), np.max(features[:,1])
        
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matchesP = bf.match(features,oldfeatures)        
          
        # Sort them in the order of their distance.
        matchesP = sorted(matchesP, key = lambda x:x.distance)

        # Draw first 10 matches.
        matches.drawMatches(frame, kp, oldframe, oldkp, matchesP[:20])
        
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
