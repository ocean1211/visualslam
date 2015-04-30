#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
motionModel
---------------------

"""

# import funkcí
import numpy as np
from numpy import *
import os.path
import scipy
import sys

def QuaternionFromAngularVelocity(av):
    """

    :param av:
    :return:
    """
    angle = np.sqrt(np.sum(av ** 2))

    if (angle > 0.0):
        s = np.sin(angle / 2.0) / angle
        c = np.cos(angle / 2.0)
        q = np.array([c, s * av[0], s * av[1], s * av[2]], dtype=np.double)
    else:
        q = np.array([1, 0, 0, 0], dtype=np.double)
    return  q;

def dq3_by_dq1(q):
    """

    :param q:
    :return:
    """
    m = np.identity(4, dtype=np.double) 
    # m[:,:] = [[q[0], -q[1], -q[2], -q[3]],
    #           [q[1],  q[0], -q[3],  q[2]], 
    #           [q[2],  q[3],  q[0], -q[1]],
    #           [q[3], -q[2],  q[1],  q[0]]]
    m[0, 0] = m[1, 1] = m[2, 2] = m[3, 3] = q[0]
    m[1, 0] = m[3, 2] = q[1]
    m[0, 1] = m[2, 3] = -q[1]
    m[2, 0] = m[1, 3] = q[2]
    m[0, 2] = m[3, 1] = -q[2]
    m[3, 0] = m[2, 1] = q[3]
    m[0, 3] = m[1, 2] = -q[3]
    return m

def dq3_by_dq2(q):
    """

    :param q:
    :return:
    """
    m = np.identity(4, dtype=np.double) * q[0]
    m[1, 0] = m[2, 3] = q[1]
    m[0, 1] = m[3, 2] = -q[1]
    m[2, 0] = m[3, 1] = q[2]
    m[0, 2] = m[1, 3] = -q[2]
    m[3, 0] = m[1, 2] = q[3]
    m[0, 3] = m[2, 1] = -q[3]
    return m

def dq0_by_domegaA(omegaA, omega, delta_t):
    """

    :param omegaA:
    :param omega:
    :param delta_t:
    :return:
    """
    return ((-delta_t / 2.0) * (omegaA / omega) * np.sin(omega * delta_t / 2.0))

def dqA_by_domegaA(omegaA, omega, delta_t):
    """

    :param omegaA:
    :param omega:
    :param delta_t:
    :return:
    """
    return ((delta_t / 2.0) * (omegaA ** 2) / (omega ** 2) * \
            np.cos(omega * delta_t / 2.0) + (1.0 / omega) * (1.0 - (omegaA ** 2) / (omega ** 2)) * \
            np.sin(omega * delta_t / 2.0))

def dqA_by_domegaB(omegaA, omegaB, omega, delta_t):
    """

    :param omegaA:
    :param omegaB:
    :param omega:
    :param delta_t:
    :return:
    """
    return ((omegaA * omegaB / (omega ** 2)) * ((delta_t / 2.0) * \
            np.cos(omega * delta_t / 2.0) - \
            (1.0 / omega) * np.sin(omega * delta_t / 2.0)))

def dq_omega_dt(omega, delta_t):
    """

    :param omega:
    :param delta_t:
    :return:
    """
    omegamod = np.sqrt(np.sum(omega ** 2))
    dqomegadt_by_domega = np.zeros([4, 3], dtype=np.double)

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
    """

    :param qi:
    :param qq:
    :return:
    """
    return (1 - (qi ** 2 / float(qq ** 2))) / qq

def dqi_by_dqj(qi, qj, qq):
    """

    :param qi:
    :param qj:
    :param qq:
    :return:
    """
    return -qi * qj / float(qq ** 3)

def dqnorm_by_dq(q):
    """

    :param q:
    :return:
    """
    M = np.zeros([4, 4], dtype=np.double)	
    qq = np.sum(q ** 2)

    M[0, 0] = dqi_by_dqi(q[0], qq) 
    M[0, 1] = dqi_by_dqj(q[0], q[1], qq)
    M[0, 2] = dqi_by_dqj(q[0], q[2], qq) 
    M[0, 3] = dqi_by_dqj(q[0], q[3], qq) 
    M[1, 0] = dqi_by_dqj(q[1], q[0], qq)  
    M[1, 1] = dqi_by_dqi(q[1], qq) 
    M[1, 2] = dqi_by_dqj(q[1], q[2], qq) 
    M[1, 3] = dqi_by_dqj(q[1], q[3], qq) 
    M[2, 0] = dqi_by_dqj(q[2], q[0], qq) 
    M[2, 1] = dqi_by_dqj(q[2], q[1], qq) 
    M[2, 2] = dqi_by_dqi(q[2], qq) 
    M[2, 3] = dqi_by_dqj(q[2], q[3], qq)  
    M[3, 0] = dqi_by_dqj(q[3], q[0], qq) 
    M[3, 1] = dqi_by_dqj(q[3], q[1], qq)  
    M[3, 2] = dqi_by_dqj(q[3], q[2], qq)  
    M[3, 3] = dqi_by_dqi(q[3], qq)
    return  M
    
def qprod(q, r):
    """

    :param q:
    :param r:
    :return:
    """
    t = np.zeros([4, 1], dtype=np.double)
    t[0] = (r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3])
    t[1] = (r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2])
    t[2] = (r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1])
    t[3] = (r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0])

    return t    
            
def q2r(q): # matrix representation of quaternion
    """

    :param q:
    :return:
    """
    R = np.zeros([3, 3], dtype=np.double)
    R[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    R[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    R[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
    R[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    R[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    R[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    return R   

def xyz_dh_dy(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    xyz_dhrl_dy = np.linalg.inv(q2r(xv))
    return np.dot(xyz_dh_dhrl(inparams,xv, y, zi),xyz_dhrl_dy)

def xyz_dh_dhrl(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    Rrw = np.linalg.inv(q2r(xv))
    xyz_dhd_dhu = np.linalg.inv(jacob_undistort_fm(zi, inparams))
    
    hrl = np.dot(Rrw, y - x[0:3] )
    xyz_dhu_dhrl = np.zeros([2,3], dtype = np.double)
    xyz_dhu_dhrl[:,:] = [[inparams['fku']/hrl[2], 0 , -hrl[0]*inparams.fku/(hrl[2]**2)],
              [ 0 , inparams['fkv']/hrl[2] , -hrl[0]*inparams.fkv/(hrl[2]**2)]]
    a =  np.dot(xyz_dhd_dhu, xyz_dhu_dhrl)               
    return a

def xyz_dh_dxv(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    
    xyz_dhrl_drw = -1 * np.linalg.inv(q2r(xv[3:6]))
    xyz_dh_drw = np.dot(xyz_dh_dhrl(inparams, xv, y, zi), xyz_dhrl_drw)
    
    
    dqbar_dq = np.double(np.diag([1,-1,-1,-1]))    
    q = np.array([xv[3], -xv[4], -xv[5], -xv[6]], dtype = np.double)
    xyz_dhrl_dqwr = np.dot(dRq_times_a_by_dq(q,(y - xv[0:3])), dqbar_dq)    
    xyz_dh_dqwr = np.dot(xyz_dh_dhrl(inparams, xv, y, zi), xyz_dhrl_dqwr)
    
    Hi1 = np.concatenate([xyz_dh_drw, xyz_dh_dqwr, np.zeros([2,6], dtype = np.double)], axis = 1)
    
    return Hi1




def id_dh_dy(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    id_dhrl_dy = np.linalg.inv(q2r(xv))
    return np.dot(id_dh_dhrl(inparams,xv, y, zi),id_dhrl_dy)

def id_dh_dhrl(inparams, xv, y, zi):
    pass

def id_dh_dxv(inparams, xv, y, zi):
    pass





def distort_fm(uv_u, inparams):
    
    pass

def undistort_fm(uv_d, inparams):
    
    pass

def jacob_undistort_fm(uv_d, inparams):
    
    return j_un

def hinv(uv_d, xv, inparams, initRho):
    pass

def dRq_times_a_by_dq(q, a):
    pass

def dR_by_dq0(q):
    pass

def dR_by_dqx(q):
    pass

def dR_by_dqy(q):
    pass

def dR_by_dqz(q):
    pass

def rotate_with_dist_fc_c2c1(uv_c1, R_c2c1, t_c2c1, n, d, inparams):
    uv_c2 = np.zeros(3)
    return uv_c2

def rotate_with_dist_fc_c1c2(uv_c2, R_c2c1, t_c2c1, n, d, inparams):
    uv_c1 = np.zeros(3)
    return uv_c1


def m(phi, theta):
    res = np.zeros(3, dtype=np.double)
    res[0:3] = [np.cos(phi) * np.sin(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)]
    return res    
    
def id2cartesian(yi):
    pass

def std_dev(im, mean):

    std = 0
    return std

class quaternion:

    # konstruktor
    def __init__(self): #
        pass

    def multiply(self, q, r):
        t = np.zeros([4, 1], dtype=np.double)
        t[0] = (r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3])
        t[1] = (r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2])
        t[2] = (r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1])
        t[3] = (r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0])
        return t

    def divide(self, q, r):
        t = np.zeros([4, 1], dtype=np.double)
        t[0] = (r[0] * q[0] + r[1] * q[1] + r[2] * q[2] + r[3] * q[3])
        t[1] = (r[0] * q[1] - r[1] * q[0] - r[2] * q[3] + r[3] * q[2])
        t[2] = (r[0] * q[2] + r[1] * q[3] - r[2] * q[0] - r[3] * q[1])
        t[3] = (r[0] * q[3] - r[1] * q[2] + r[2] * q[1] - r[3] * q[0])
        normVal = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]
        t = t / normVal
        return t

    def conjugate(self, q):
        q[1:] = -q[1:]
        return q

    def modulus(self, q):
        modul = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        return modul

    def inv(self, q):
        normVal = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
        q = q / normVal
        return q
	
    def norm(self, q):
        normVal = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
        return normVal

    def normalize(self, q):
        modul = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        q = q / modul
        return q

    def rotate(self, q, v): # rotate to new position
        vv = np.zeros([3, 1], dtype=np.double)
        pom = np.zeros([3, 1], dtype=np.double)
        pom[0] = (1 - 2 * q[2] * q[2] - 2 * q[3] * q[3])
        pom[1] = (2 * (q[1] * q[2] + q[0] * q[3]))
        pom[2] = (2 * (q[1] * q[3] - q[0] * q[2]))
        print pom.T
        print v
        vv[0] = np.dot(pom.T, v)
        pom[1] = (1 - 2 * q[1] * q[1] - 2 * q[3] * q[3])
        pom[0] = (2 * (q[1] * q[2] - q[0] * q[3]))
        pom[2] = (2 * (q[2] * q[3] + q[0] * q[1]))
        vv[1] = np.dot(pom.T, v)
        pom[2] = (1 - 2 * q[1] * q[1] - 2 * q[2] * q[2])
        pom[1] = (2 * (q[2] * q[3] - q[0] * q[1]))
        pom[0] = (2 * (q[1] * q[3] + q[0] * q[2]))
        vv[2] = np.dot(pom.T, v)
        return vv

    def rotationMatrix(q): # matrix representation of quaternion
        R = np.zeros([3, 3], dtype=np.double)
        R[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
        R[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
        R[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
        R[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        R[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        R[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        R[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
        R[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
        R[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
        return R

	pass





