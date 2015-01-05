#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys


# Directories 
# Platform dependent (if you have win64 than add elif or replace if condition)
if sys.platform == 'win32':
    user = 'Petr'
    prefix = 'C:/Users/'+ user +'/Documents/Projekty/data/kitti/'
    
else:
    user = 'neduchal'
    prefix = '/home/'+user+'/Dokumenty/Data/kitti/'    
    
    
 # General 
left_image = 'image_0/'
right_image = 'image_1/'   
    
calib_dir = prefix + 'calib/'
sequence_dir = prefix + 'sequences/'
poses_dir = prefix + 'poses/'

sequence_num = '00'

#pokus o zmenu

