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
import config
from loaders import *
from inout import *


def main():

  # Loading calibration, filenames, poses and timestamp data :  
  # sequence filenames --------------------------------------------------------    
  seq_dir = config.sequence_dir + config.sequence_num + '/'
  left_sequence_files = directories.directory(seq_dir + config.left_image)
  right_sequence_files = directories.directory(seq_dir + config.right_image) 
  # poses (ground truth) ------------------------------------------------------    
  if int(config.sequence_num) < 11 : 
    poses_data = poses.Loader(config.poses_dir + config.sequence_num + '.txt')
  # calibration and timestamps ------------------------------------------------      
  calib_files = directories.directory(config.calib_dir + config.sequence_num)    
  calib_data = calibration.Loader(calib_files.getFirst())
  times_data = timestamp.Loader(calib_files.getLast())
  
  # INIT PART WILL BE HERE  
  
  # INIT CAMERA, MAP, FILTERS, FEATURES, ...
    
  # MAIN LOOP -----------------------------------------------------------------  
  count =  left_sequence_files.getCount()
  for i in range(count):
    # INIT OF EACH LOOP
  
    # PREDICTION STEP
  
    # DATA ACQUISITION
  
    # GETTING FEATURES 
  
    # UPDATE STEP

    # VISUALIZATION
    pass

  # DATA SAVING AND EVALUATING PART

  pass
  


if __name__ == "__main__":
    main()
