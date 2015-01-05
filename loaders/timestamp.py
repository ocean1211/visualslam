#! /usr/bin/python
# -*- coding: utf-8 -*-

# Imports
import sys
import os
import numpy

'''
 Class timestamp.Loader
 
 Loads kitti dataset gray timestamp data from file  
 
'''
class Loader:
  
  timestamps = []
  
  def __init__(self, filename):
    self.read(filename)
    pass
  
  def read(self, filename):
    f = open(filename, 'r')
    for line in f:
      self.timestamps.append(float(line))
    f.close()
    pass
  
  
  
  
  
