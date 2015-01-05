#! /usr/bin/python
# -*- coding: utf-8 -*-

# Imports
# -----------------------------------------------------------------------------
import sys
import os
import numpy
# -----------------------------------------------------------------------------


'''
 Class calibration.Loader

 Loads kitti dataset gray camera calibration data from file

'''

# -----------------------------------------------------------------------------
class Loader:
  # Calibration matrix size ---------------------------------------------------
  rows = 3
  cols = 4
  # Empty calibration matrices ------------------------------------------------
  calibMatrixLeft = numpy.zeros([ rows, cols ], numpy.double)
  calibMatrixRight = numpy.zeros([ rows, cols ], numpy.double)
  
  # Init (Constructor) --------------------------------------------------------
  def __init__(self, filename):
    self.read(filename)
    pass
    
  # Read function -------------------------------------------------------------
  def read(self, filename):
    f = open(filename, 'r')
    leftStr = f.readline()
    leftArray = leftStr.split(' ')
    rightStr = f.readline()
    rightArray = leftStr.split(' ')
    for i in (range(self.rows * self.cols)):
        self.calibMatrixLeft[ i/self.cols, i % self.cols ] = leftArray[i+1]
        self.calibMatrixRight[ i/self.cols, i % self.cols ] = rightArray[i+1]
    f.close()
    pass
# -----------------------------------------------------------------------------
