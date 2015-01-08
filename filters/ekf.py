#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
EKF 
---------------------

"""

# import funkcí
import sys
import os.path
import numpy as np
import scipy

# ziskani cest ke skriptum
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../"))

# nacteni dalsich soucasti
import BaseFilter as bf
from models import *
#from utils import *

# trida rozsireneho kalmanova filtru
class ekf(bf.BaseFilter):

	# konstruktor
  def __init__(self): #
    pass
		
  # krok predikce
  def predict(self, model, delta_t):
    model.predict(delta_t) 
    pass		 

  #krok filtrace
  def update(self, model, z) :
    if (z.shape[0] > 0):
      #S = model.dH() * model.P_p * model.dH().T + model.R()
      S = np.dot(np.dot(model.dH(),model.P_p), model.dH().T) +model.R()
      #K = model.P_p * model.dH().T * np.linalg.inv(S)
      K = np.dot(np.dot(model.P_p,  model.dH().T), np.linalg.inv(S))
      model.x = model.x_p + np.dot(K,(z - model.x[0:2]))
      model.P  = model.P - (np.dot(np.dot(K,S), K.T))
      #model.P = 0.5 * model.P + 0.5 * model.P.T
    else :
      model.x = model.x_p
      model.P = model.P_p
    pass
