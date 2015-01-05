#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
IO
---------------------

"""

# import funkcí
import sys
import os.path
import numpy as np
import scipy
import cv2

## Image Class
class image:
	
  colorData = 0 # BGR!!!
  greyData = 0

	# construktor
  def __init__(self, path): 
    self.colorData = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    self.colorData
    if len(self.colorData.shape) == 3:
      self.greyData = cv2.cvtColor(self.colorData, cv2.COLOR_BGR2GRAY)
    else:
      self.greyData = self.colorData
    pass

  def getColor(self):
    return self.colorData

  def getGrey(self):
    return self.greyData

  def show(self):
    cv2.imshow('Color Data', self.colorData)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

  def showGrey(self):
    cv2.imshow('Grey Data', self.greyData)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  pass

