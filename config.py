#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys

'''
  VISUAL SLAM PROJECT CONFIGURATION FILE

  AUTHORS : Petr Neduchal, Ivan Gruber
'''
# Abbreviations --------------------------------------------------------------
# NIY - NOT IMPLEMENTED YET



# ----------------------------------------------------------------------------
# GENERAL SETTINGS
# ----------------------------------------------------------------------------

# Data source 
# 0 - directory (images)
# 1 - camera (live stream) - NIY
data_source = 0;





# ----------------------------------------------------------------------------
# Directories (Image data, calibration data, ...)
# ----------------------------------------------------------------------------

# Platform dependent (if you have win64 than add elif or replace if condition)
# WIN 32 ---------------------------------------------------------------------
if sys.platform == 'win32':
    user = 'Petr'
    prefix = 'C:/Users/'+ user +'/Documents/Projekty/data/kitti/'
# UNIX -----------------------------------------------------------------------
else:
    user = 'neduchal'
    prefix = '/home/'+user+'/Dokumenty/Data/kitti/'
# ----------------------------------------------------------------------------
    
# Subdirectories -------------------------------------------------------------
# Image Data -----------------------------------------------------------------
left_image = 'image_0/'
right_image = 'image_1/'
# Calibration, Sequences and Poses prefixes ----------------------------------
calib_dir = prefix + 'calib/'
sequence_dir = prefix + 'sequences/'
poses_dir = prefix + 'poses/'
# Sequence Number 00 - 15
sequence_num = '00'

