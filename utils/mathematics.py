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

def QuaternionFromAngularVelocity(av):
	angle = np.sqrt(av[0]*av[0] + av[1]*av[1] + av[2]*av[2])

	if (angle > 0.0):
		s = np.sin(angle/2.0) / angle
		c = np.cos(angle/2.0)
		q = np.array([c, s * av[0],s * av[1],s * av[2]], dtype = np.double)
	else :
		q = np.array([1, 0, 0, 0], dtype = np.double)
	return  q;

def dq3_by_dq1(q):
	m = np.identity(4, dtype = np.double) 
	m[0,0] = m[1,1] = m[2,2] = m[3,3] = q[0]
	m[1,0] = m[3,2] = q[1]
	m[0,1] = m[2,3] = -q[1]
	m[2,0] = m[1,3] = q[2]
	m[0,2] = m[3,1] = -q[2]
	m[3,0] = m[2,1] = q[3]
	m[0,3] = m[1,2] = -q[3]
	return m

def dq3_by_dq2(q):
	m = np.identity(4, dtype = np.double) 
	m[0,0] = m[1,1] = m[2,2] = m[3,3] = q[0]
	m[1,0] = m[2,3] = q[1]
	m[0,1] = m[3,2] = -q[1]
	m[2,0] = m[3,1] = q[2]
	m[0,2] = m[1,3] = -q[2]
	m[3,0] = m[1,2] = q[3]
	m[0,3] = m[2,1] = -q[3]
	return m

def dq0_by_domegaA(omegaA, omega, delta_t):
	return ((-delta_t / 2.0) * (omegaA/omega) * np.sin(omega * delta_t / 2.0))

def dqA_by_domegaA(omegaA, omega, delta_t):
	return ((delta_t / 2.0) * omegaA * omegaA / (omega * omega) * \
					np.cos(omega * delta_t / 2.0) + (1.0 / omega) * (1.0 - omegaA * omegaA / (omega * omega)) * \
					np.sin(omega * delta_t / 2.0))

def dqA_by_domegaB(omegaA, omegaB, omega, delta_t):
	return ((omegaA * omegaB / (omega * omega)) * ((delta_t / 2.0) * \
				 np.cos(omega * delta_t / 2.0) - \
			   (1.0 / omega) * np.sin(omega * delta_t / 2.0) ))

def dq_omega_dt(omega, delta_t):
	omegamod = np.sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2])
	dqomegadt_by_domega = np.zeros([4,3], dtype = np.double)

  # Use generic ancillary functions to calculate components of Jacobian
	dqomegadt_by_domega[0, 0] = dq0_by_domegaA(omega[0], omegamod, delta_t);
	dqomegadt_by_domega[0, 1] = dq0_by_domegaA(omega[1], omegamod, delta_t);
	dqomegadt_by_domega[0, 2] = dq0_by_domegaA(omega[2], omegamod, delta_t);
	dqomegadt_by_domega[1, 0] = dqA_by_domegaA(omega[0], omegamod, delta_t);
	dqomegadt_by_domega[1, 1] = dqA_by_domegaB(omega[0], omega[1], omegamod, delta_t);
	dqomegadt_by_domega[1, 2] = dqA_by_domegaB(omega[0], omega[2], omegamod, delta_t);
	dqomegadt_by_domega[2, 0] = dqA_by_domegaB(omega[1], omega[0], omegamod, delta_t);
	dqomegadt_by_domega[2, 1] = dqA_by_domegaA(omega[1], omegamod, delta_t);
	dqomegadt_by_domega[2, 2] = dqA_by_domegaB(omega[1], omega[2], omegamod, delta_t);
	dqomegadt_by_domega[3, 0] = dqA_by_domegaB(omega[2], omega[0], omegamod, delta_t);
	dqomegadt_by_domega[3, 1] = dqA_by_domegaB(omega[2], omega[1], omegamod, delta_t);
	dqomegadt_by_domega[3, 2] = dqA_by_domegaA(omega[2], omegamod, delta_t);
	return dqomegadt_by_domega

def dqi_by_dqi(qi, qq):
	return (1 - (qi*qi / qq*qq))/ qq

def dqi_by_dqj(qi, qj, qq):
	return -qi * qj / (qq*qq*qq)

def dqnorm_by_dq(q):
	M = np.zeros([4,4], dtype = np.double)
	quat = mq.quaternion()
	qq = quat.norm(q)

	M[0,0] = dqi_by_dqi(q[0], qq) 
	M[0,1] = dqi_by_dqj(q[0], q[1], qq)
	M[0,2] = dqi_by_dqj(q[0], q[2], qq) 
	M[0,3] = dqi_by_dqj(q[0], q[3], qq) 
	M[1,0] = dqi_by_dqj(q[1], q[0], qq)  
	M[1,1] = dqi_by_dqi(q[1], qq) 
	M[1,2] = dqi_by_dqj(q[1], q[2], qq) 
	M[1,3] = dqi_by_dqj(q[1], q[3], qq) 
	M[2,0] = dqi_by_dqj(q[2], q[0], qq) 
	M[2,1] = dqi_by_dqj(q[2], q[1], qq) 
	M[2,2] = dqi_by_dqi(q[2], qq) 
	M[2,3] = dqi_by_dqj(q[2], q[3], qq)  
	M[3,0] = dqi_by_dqj(q[3], q[0], qq) 
	M[3,1] = dqi_by_dqj(q[3], q[1], qq)  
	M[3,2] = dqi_by_dqj(q[3], q[2], qq)  
	M[3,3] = dqi_by_dqi(q[3], qq)
	return  M



class quaternion:

	# konstruktor
	def __init__(self): #
		pass

	def multiply(self, q,r):
		t = np.zeros([4,1], dtype = np.double)
		t[0] = (r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3])
		t[1] = (r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2])
		t[2] = (r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1])
		t[3] = (r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0])
		return t

	def divide(self, q,r):
		t = np.zeros([4,1], dtype = np.double)
		t[0] = (r[0]*q[0] + r[1]*q[1] + r[2]*q[2] + r[3]*q[3])
		t[1] = (r[0]*q[1] - r[1]*q[0] - r[2]*q[3] + r[3]*q[2])
		t[2] = (r[0]*q[2] + r[1]*q[3] - r[2]*q[0] - r[3]*q[1])
		t[3] = (r[0]*q[3] - r[1]*q[2] + r[2]*q[1] - r[3]*q[0])
		normVal = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3]
		t = t/normVal
		return t

	def conjugate(self, q):
		q[1:] = -q[1:]
		return q

	def modulus(self, q):
		modul = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
		return modul

	def inv(self, q):
		normVal = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
		q = q/normVal
		return q
	
	def norm(self, q):
		normVal = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
		return normVal

	def normalize(self, q):
		modul = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
		q = q/modul
		return q

	def rotate(self, q,v): # rotate to new position
		vv = np.zeros([3,1], dtype = np.double)
		pom = np.zeros([3,1], dtype = np.double)
		pom[0] = (1 - 2*q[2]*q[2] - 2*q[3]*q[3])
		pom[1] = (2*(q[1]*q[2] + q[0]*q[3]))
		pom[2] = (2*(q[1]*q[3] - q[0]*q[2]))
		print pom.T
		print v
		vv[0] = np.dot(pom.T,v)
		pom[1] = (1 - 2*q[1]*q[1] - 2*q[3]*q[3])
		pom[0] = (2*(q[1]*q[2] - q[0]*q[3]))
		pom[2] = (2*(q[2]*q[3] + q[0]*q[1]))
		vv[1] = np.dot(pom.T,v)
		pom[2] = (1 - 2*q[1]*q[1] - 2*q[2]*q[2])
		pom[1] = (2*(q[2]*q[3] - q[0]*q[1]))
		pom[0] = (2*(q[1]*q[3] + q[0]*q[2]))
		vv[2] = np.dot(pom.T,v)
		return vv

	def rotationMatrix(q): # matrix representation of quaternion
		R = np.zeros([3,3], dtype = np.double)
		R[0,0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
		R[1,1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3]
		R[2,2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
		R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
		R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
		R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
		R[0,2] = 2*(q[1]*q[3] + q[0]*q[2])
		R[2,1] = 2*(q[2]*q[3] + q[0]*q[1])
		R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
		return R

	pass





