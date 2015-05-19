#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
    Funkce pro práci s homogenními souřadnicemi
    -------------------------------------------

"""

import numpy as np

def rem_dimension(v):
    """
        Funkce prevede

    :param v:
    :return:
    """
    uv = v[0:v.shape[0]-1]/float(v[v.shape[0]-1])
    return uv

def add_dimension(uv):
    """

    :param uv:
    :return:
    """

    v = np.ones(uv.shape[0]+1)
    v[0:uv.shape[0]] = uv
    return v

def rot3d_x(angle):
    """

    :param angle:
    :return:
    """
    s_a = np.sin(angle)
    c_a = np.cos(angle)

    mat_r = np.array([[1, 0, 0, 0],
                      [0, c_a, s_a, 0],
                      [0, -s_a, c_a, 0],
                      [0, 0, 0, 1]
                      ])
    return mat_r

def rot3d_y(angle):
    """

    :param angle:
    :return:
    """
    s_a = np.sin(angle)
    c_a = np.cos(angle)

    mat_r = np.array([[c_a, 0, -s_a, 0],
                      [0, 1, 0, 0],
                      [s_a, 0, c_a, 0],
                      [0, 0, 0, 1]
                      ])
    return mat_r

def rot3d_z(angle):
    """

    :param angle:
    :return:
    """
    s_a = np.sin(angle)
    c_a = np.cos(angle)

    mat_r = np.array([[c_a, s_a, 0, 0],
                      [-s_a, c_a, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]
                      ])
    return mat_r

def rot3d(angles):
    """

    :param angles:
    :return:
    """
    roll = angles[2]
    pitch = angles[1]
    yaw = angles[0]
    return np.dot(np.dot(rot3d_z(roll), rot3d_x(pitch)),rot3d_y(yaw))
