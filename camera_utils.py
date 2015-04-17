#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
motionModel
---------------------

"""

# import funkcí
import sys
import os.path
from numpy import *
import numpy as np
import scipy

def m(phi, theta):
	res = np.zeros(3, dtype = np.double)
	res[0:3] = [np.cos(phi)*np.sin(theta),   -np.sin(phi),  np.cos(phi)*np.cos(theta)]
	return res





