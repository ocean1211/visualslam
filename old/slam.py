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
import monocularSlam as ms

configFile = "../config/config1.json"


def main():
    # Nacteni konfigurace
    config = cjson.load(configFile)

    # Vybrani zdroje dat
    if config["source"][0] == "camera":
        source = camera.Camera(config["source"][1])
    else: 
        source = images.ImageDirectory(config["source"][1])
    
    mono = ms.Slam(source, config['camera_params'])
    mono.init()

    while True:
        frame = mono.step()
        if mono.end is True:
            break
        cv2.imshow('vizualization', frame)

        key = cv2.waitKey(0)
        if key == 'e':
            mono.stepOn = False
        elif key == 's':
            mono.stepOn = True
        elif key == 'k':
            mono.end = True
 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
