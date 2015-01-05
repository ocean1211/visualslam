#! /usr/bin/python
# -*- coding: utf-8 -*-

# Imports
# -----------------------------------------------------------------------------
import sys
import os
import numpy

'''
 Class poses.Loader

 Loads kitti dataset gray camera calibration data from file

'''
class Loader:
  # Poses matrix size ---------------------------------------------------------   
  rows = 3
  cols = 4

  # Poses list (dictionary) ---------------------------------------------------
  poses = {}
  
  # Init (Constructor) --------------------------------------------------------
  def __init__(self, filename):
    self.read(filename)
    pass
    
  # Read function -------------------------------------------------------------
  def read(self, filename):
    f = open(filename, 'r')
    j = 0
    for line in f:
      lineArray = line.split(' ')
      data = numpy.zeros([self.rows,self.cols], numpy.double)
      for i in (range(self.rows * self.cols)):
          data[ i/self.cols, i % self.cols ] = float(lineArray[i])
      self.poses[j] = data
      j = j + 1
    f.close()
    pass
# -----------------------------------------------------------------------------
    
