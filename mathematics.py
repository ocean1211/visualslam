# -*- coding: utf-8 -*-

"""
Mathematics
---------------------

"""

# import funkcí
import numpy as np
import math


def q2r(q):
    """

    :param q:
    :return:
    """
    q0 = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2
    mat_r = np.diag([q0 + 2*(q[1]**2), q0 + 2*(q[2]**2), q0 + 2*(q[2]**2)])
    mat_r[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    mat_r[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    mat_r[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    mat_r[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    mat_r[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    mat_r[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    return mat_r


def m(phi, theta):
    """

    :param phi:
    :param theta:
    :return:
    """
    return np.array([np.cos(phi) * np.sin(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)])


def distort_fm(uv, camera):
    """

    :param uv:
    :param camera:
    :return:
    """
    xu = (uv[0]-camera['u0'])*camera['d']
    yu = (uv[1]-camera['v0'])*camera['d']
    ru = np.sqrt(xu**2 + yu**2)
    rd = ru/(1+camera['kd1']*(ru**2) + camera['kd2']*(ru**4))
    for k in range(10):
        f = rd + camera['kd1'] * (rd**3) + camera['kd2']*(rd**5) - ru
        f_p = 1 + 3*camera['kd1'] * (rd**2) + camera['kd2']*(rd**4)
        rd = rd - (f/f_p)
    pass
    d = float(1+camera['kd1']*(rd**2) + camera['kd2']*(rd**4))
    uv_d = np.zeros(2, dtype=np.float32)
    uv_d[0] = (xu/d)/camera['d'] + camera['u0']
    uv_d[1] = (yu/d)/camera['d'] + camera['v0']
    return uv_d


def undistort_fm(uv_d, inparams):
    """

    :param uv_d:
    :param inparams:
    :return:
    """
    xd = (uv_d[0]-inparams['u0'])*inparams['d']
    yd = (uv_d[1]-inparams['v0'])*inparams['d']
    rd_2 = xd**2 + yd**2
    d = (1+inparams['kd1']*rd_2 + inparams['kd2']*(rd_2**2))
    uv_u = np.zeros(2, dtype=np.float32)

    uv_u[0] = (xd*d)/(inparams['d']) + inparams['u0']
    uv_u[1] = (yd*d)/(inparams['d']) + inparams['v0']
    return uv_u


def jacob_undistort_fm(uv_d, camera):
    """

    :param uv_d:
    :param camera:
    :return:
    """
    xd = (uv_d[0]-camera['u0'])*camera['d']
    yd = (uv_d[1]-camera['v0'])*camera['d']
    rd_2 = xd**2 + yd**2
    rd = np.sqrt(rd_2)
    d1 = float(1+camera['kd1']*rd_2 + camera['kd2']*(rd_2**2))
    d2 = float(camera['kd1'] + 2*camera['kd2']*rd)
    j_un = np.zeros([2, 2], dtype=np.float32)
    j_un[0, 0] = d1 + (uv_d[0] - camera['u0'])*d2*2*xd*camera['d']
    j_un[0, 1] = (uv_d[0] - camera['u0'])*d2*2*yd*camera['d']
    j_un[1, 0] = (uv_d[1] - camera['v0'])*d2*2*xd*camera['d']
    j_un[1, 1] = d1 + (uv_d[1] - camera['v0'])*d2*2*yd*camera['d']
    return j_un


def hinv(uv, x, camera, rho):
    """

    :param uv:
    :param x:
    :param camera:
    :param rho:
    :return:
    """
    uv_u = undistort_fm(uv, camera)
    h = np.array([-(camera['u0']-uv_u[0])/camera['fku'],
                  -(camera['v0']-uv_u[1])/camera['fkv'],
                  1])
    mat_r = q2r(x[3:7])
    n = np.dot(mat_r, h)
    xf = x[0:6]
    xf[3:] = [math.atan2(n[0], n[2]),
              math.atan2(-n[1], np.sqrt(n[0]**2 + n[2]**2)),
              rho]
    return xf


def d_r_q_times_a_by_dq(q, a):
    """

    :param q:
    :param a:
    :return:
    """
    drq = np.zeros([3, 4], dtype=np.float32)
    drq[0:3, 0] = np.dot(d_r_by_dq0(q), a)
    drq[0:3, 1] = np.dot(d_r_by_dqx(q), a)
    drq[0:3, 2] = np.dot(d_r_by_dqy(q), a)
    drq[0:3, 3] = np.dot(d_r_by_dqz(q), a)
    return drq


def d_r_by_dq0(q):
    """

    :param q:
    :return:
    """
    res_mat = np.zeros([3, 3], dtype=np.float32)
    res_mat[:, :] = [[2*q[0], -2*q[3], 2*q[2]],
                     [2*q[3], 2*q[0], -2*q[1]],
                     [-2*q[2], 2*q[1], 2*q[0]]]
    return res_mat


def d_r_by_dqx(q):
    """

    :param q:
    :return:
    """
    res_mat = np.zeros([3, 3], dtype=np.float32)
    res_mat[:, :] = [[2*q[1], 2*q[2], 2*q[3]],
                     [2*q[2], -2*q[1], -2*q[0]],
                     [2*q[3], 2*q[0], -2*q[1]]]
    return res_mat


def d_r_by_dqy(q):
    """

    :param q:
    :return:
    """
    res_mat = np.zeros([3, 3], dtype=np.float32)
    res_mat[:, :] = [[-2*q[2], 2*q[1], 2*q[0]],
                     [2*q[1], 2*q[2], 2*q[3]],
                     [-2*q[0], 2*q[3], -2*q[2]]]
    return res_mat


def d_r_by_dqz(q):
    """

    :param q:
    :return:
    """
    res_mat = np.zeros([3, 3], dtype=np.float32)
    res_mat[:, :] = [[-2*q[3], -2*q[0], 2*q[1]],
                     [2*q[0], -2*q[3], 2*q[2]],
                     [2*q[1], 2*q[2], 2*q[3]]]
    return res_mat