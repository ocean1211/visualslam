#! /usr/bin/python
# -*- coding: utf-8 -*-

# Imports
import sys
import os
import numpy

'''
 Class poses.Loader
 
 Loads kitti dataset gray camera calibration data from file  
 
'''
class Loader:
  
  rows = 3
  cols = 4
  
  poses = {}
  
  def __init__(self, filename):
    self.read(filename)
    pass
  
  def read(self, filename):
    f = open(filename, 'r')  
    j = 0 
    for line in f:     
      lineArray = line.split(' ')   
      data = numpy.zeros([3,4], numpy.double)
      for i in (range(self.rows * self.cols)):
          data[ i/self.cols, i % self.cols ] = float(lineArray[i])  
      self.poses[j] = data
      j = j + 1
    f.close()
    
    pass
  
  
  
  
  
