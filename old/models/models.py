#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
General 6D Constant Velocity motion model (only for EKF ? )
---------------------
Constant velocity motion model for movement in 3D space.
x - Contains position, velocity and features data 
P - Covariance matrix
x_p, P_p - prediction versions of x and P
"""

# import funkcí
import sys
import os.path
import numpy as np
import scipy

# nastaveni slozek, ve kterych se budou hledat funkce
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../"))

#from utils import *
#from features import *

# Trida constantVelocity - pohzbovy model s konstantni rychlosti
class constantVelocity :
  x = np.zeros([4,1], dtype = np.double)
  P	= np.zeros([4,4], dtype = np.double)
  x_p = np.zeros([4,1], dtype = np.double)
  P_p	= np.zeros([4,4], dtype = np.double)
  size = 4
  LINEAR_ACCELERATION_DEV = 0.001
  IMAGE_NOISE = 1.0

  # konstruktor tridy
  def __init__(self, lin_dev): # Konstruktor
    self.LINEAR_ACCELERATION_DEV = lin_dev

	# vypocet pohybu za zmenu casu
  def motion(self,delta_t) : 
    r = self.x[0:2]
    v = self.x[2:4]
    r = r + v * delta_t
    xnew = self.x;
    xnew[0:2] = r
    self.x_p = xnew;

  # vypocet jakobianu k pohybu za zmenu casu
  def motionJacobian(self, delta_t):
    F = np.identity(self.size, dtype = np.double)
    # dxnew_by_dv = I * delta_t
    temp3x3A = np.identity(2, dtype = np.double) * delta_t
    F[0:2,2:4] = temp3x3A
    return F
    
  def predict(self, delta_t):
    self.motion(delta_t)
    F = self.motionJacobian(delta_t)
    linCov = np.power(self.LINEAR_ACCELERATION_DEV*delta_t,2)
    Pn = np.zeros([4,4], dtype = np.double)
    ident = np.identity(2, dtype = np.double)
    Pn[0:2,0:2] = (ident * linCov)  
    G = np.zeros([4,4], dtype = np.double)
    G[2:4,0:2] = ident 
    G[0:2,0:2] = (ident * delta_t)  
    Q = np.dot(np.dot(G,Pn),G.T)	
    self.P_p = self.P
    self.P_p[0:4,0:4] = np.dot(np.dot(F , self.P), F.T)	+ Q

  def dH(self):
    H = np.zeros([2,4], dtype = np.double)
    H[0:2,0:2] = np.identity(2, dtype = np.double)
    #H[0:2,2:4] = np.identity(2, dtype = np.double)    
    return H

  def R(self):
    return np.identity(2, dtype = np.double)


