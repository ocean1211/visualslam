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
    
    # Zpracovani prvniho snimku :
    
    

    # MAIN LOOP
    while not frame == None:
        # Zpracovani dat
        
        # Nacteni dalsiho snimku    
        frame = source.frame(gray = True)        
        
        # Feature detection
        
        features = detectors.detect(config["detector"]["type"], frame, config["detector"]["parameters"])
        
            
        #print np.min(features[:,0]), np.max(features[:,1])
          
        
        # Zobrazeni aktualniho snimku        
        cv2.imshow("Data", frame)
        
        # Vyckavani + preruseni pri stisku q
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        pass
  
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
