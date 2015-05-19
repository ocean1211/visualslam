#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Base Filter
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
from models import *

# trida rozsireneho kalmanova filtru
class BaseFilter:

	# konstruktor
  def __init__(self): #
    raise NotImplementedError
    pass
		
  # krok predikce
  def predict(self, model, delta_t):
    raise NotImplementedError  
    pass		 

  #krok filtrace
  def update(self, model, z) :
    raise NotImplementedError  
    pass
