#! /usr/bin/python
# -*- coding: utf-8 -*-

# Imports
# -----------------------------------------------------------------------------
import sys
import os
import numpy
import cv2
# -----------------------------------------------------------------------------

'''
 Class image.Loader

 Loads kitti dataset (one) gray image data from file

'''
# -----------------------------------------------------------------------------
class Loader:
  image = numpy.zeros([1,1], numpy.double)
  winStr = 'window'

  def __init__(self, filename, winName):
    self.read(filename)
    self.winStr = winName
    pass

  def read(self, filename):
    self.image = cv2.imread(filename)
    pass

  def show(self):
    cv2.imshow(self.winStr, self.image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

  def getData(self):
    return self.image

  def size(self):
    return self.image.shape
