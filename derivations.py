#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import mathematics


def dtheta_dgw(h_w):
    # Theta podle [x,y,z]
    return np.array([h_w[2]/np.sum(h_w[[0, 2]]**2), 0, -h_w[0]/np.sum(h_w[[0, 2]]**2)])


def dphi_dgw(h_w):
    # Phi podle [x,y,z]
    return np.array([(h_w[0] * h_w[1])/((np.sum(h_w ** 2)) * np.sqrt(h_w[[0, 2]]**2)),
                     - np.sqrt(h_w[[0, 2]]**2)/(np.sum(h_w ** 2)),
                     (h_w[2] * h_w[1])/((np.sum(h_w ** 2)) * np.sqrt(h_w[[0, 2]]**2))
                     ])


def dgw_dqwr(x, h_c):
    # [x, y, z] podle q
    return mathematics.d_r_q_times_a_by_dq(x[3:7], h_c)


def dtheta_dqwr(p_dtheta_dgw, p_dgw_dqwr):
    return np.dot(p_dtheta_dgw.T, p_dgw_dqwr)


def dphi_dqwr(p_dphi_dgw, p_dgw_dqwr):
    return np.dot(p_dphi_dgw.T, p_dgw_dqwr)


def dxf_dqwr(p_dtheta_dqwr, p_dphi_dqwr):
    p_dxf_dqwr = np.zeros([6, 4], dtype=np.float64)
    p_dxf_dqwr[3, 0:4] = p_dtheta_dqwr.T
    p_dxf_dqwr[4, 0:4] = p_dphi_dqwr.T
    return p_dxf_dqwr


def dxf_drw():
    p_dxf_drw = np.zeros([6, 3], dtype=np.float64)
    p_dxf_drw[0:3, 0:3] = np.identity(3, dtype=np.float64)
    return p_dxf_drw


def dxf_dxv(p_dxf_drw, p_dxf_dqwr):
    p_dxf_dxv = np.zeros([6, 13], dtype=np.float64)
    p_dxf_dxv[0:6, 0:3] = p_dxf_drw
    p_dxf_dxv[0:6, 4:8] = p_dxf_dqwr
    return p_dxf_dxv


def dxfprima_dqw(p_dtheta_dgw, p_dphi_dgw):
    p_dxfprima_dgw = np.zeros([5, 3], dtype=np.float64)
    p_dxfprima_dgw[3, 0:3] = p_dtheta_dgw.T
    p_dxfprima_dgw[4, 0:3] = p_dphi_dgw.T
    return p_dxfprima_dgw


def dgc_dhu(camera):
    p_dgc_dhu = np.zeros([3, 2], dtype=np.float64)
    p_dgc_dhu[0:2, 0:2] = np.diag([1 / camera['fku'], 1 / camera['fkv']])
    return p_dgc_dhu


def dhu_dhd(xy, camera):
    p_dhu_dhd = mathematics.jacob_undistort_fm(xy, camera)
    return p_dhu_dhd


def dxfprima_dhd(p_dxfprima_dgw, p_dgw_dgc, p_dgc_dhu, p_dhu_dhd):
    return np.dot(np.dot(np.dot(p_dxfprima_dgw, p_dgw_dgc), p_dgc_dhu), p_dhu_dhd)


def dxf_dhd(p_dxfprima_dhd):
    p_dxf_dhd = np.zeros([6, 3], dtype=np.float64)
    p_dxf_dhd[0:5, 0:2] = p_dxfprima_dhd
    p_dxf_dhd[5, 2] = 1
    return p_dxf_dhd
