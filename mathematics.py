#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
motionModel
---------------------

"""

# import funkcí
import numpy as np
import math


def quaternion_from_angular_velocity(av):
    """

    :param av:
    :return:
    """
    # a == angle
    a = np.sqrt(np.sum(av ** 2))

    if a > 0.0:
        s = np.sin(a / 2.0) / a
        c = np.cos(a / 2.0)
        q = np.array([c, s * av[0], s * av[1], s * av[2]], dtype=np.float32)
    else:
        q = np.array([1, 0, 0, 0], dtype=np.float32)
    return q


def dq3_by_dq1(q):
    """

    :param q:
    :return:
    """
    res_mat = np.identity(4, dtype=np.float32)
    # m[:,:] = [[q[0], -q[1], -q[2], -q[3]],
    # [q[1],  q[0], -q[3],  q[2]],
    #           [q[2],  q[3],  q[0], -q[1]],
    #           [q[3], -q[2],  q[1],  q[0]]]
    res_mat[0, 0] = res_mat[1, 1] = res_mat[2, 2] = res_mat[3, 3] = q[0]
    res_mat[1, 0] = res_mat[3, 2] = q[1]
    res_mat[0, 1] = res_mat[2, 3] = -q[1]
    res_mat[2, 0] = res_mat[1, 3] = q[2]
    res_mat[0, 2] = res_mat[3, 1] = -q[2]
    res_mat[3, 0] = res_mat[2, 1] = q[3]
    res_mat[0, 3] = res_mat[1, 2] = -q[3]
    return res_mat


def dq3_by_dq2(q):
    """

    :param q:
    :return:
    """
    res_mat = np.identity(4, dtype=np.float32) * q[0]
    res_mat[1, 0] = res_mat[2, 3] = q[1]
    res_mat[0, 1] = res_mat[3, 2] = -q[1]
    res_mat[2, 0] = res_mat[3, 1] = q[2]
    res_mat[0, 2] = res_mat[1, 3] = -q[2]
    res_mat[3, 0] = res_mat[1, 2] = q[3]
    res_mat[0, 3] = res_mat[2, 1] = -q[3]
    return res_mat


def dq0_by_domega_a(omega_a, omega, d_t):
    """

    :param omega_a:
    :param omega:
    :param d_t:
    :return:
    """
    return (-d_t / 2.0) * (omega_a / omega) * np.sin(omega * d_t / 2.0)


def dq_a_by_domega_a(omega_a, omega, d_t):
    """

    :param omega_a:
    :param omega:
    :param d_t:
    :return:
    """
    return ((d_t / 2.0) * (omega_a ** 2) / (omega ** 2) *
            np.cos(omega * d_t / 2.0) + (1.0 / omega) * (1.0 - (omega_a ** 2) / (omega ** 2)) *
            np.sin(omega * d_t / 2.0))


def dq_a_by_domega_b(omega_a, omega_b, omega, d_t):
    """

    :param omega_a:
    :param omega_b:
    :param omega:
    :param d_t:
    :return:
    """
    return ((omega_a * omega_b / (omega ** 2)) * ((d_t / 2.0) *
                                                  np.cos(omega * d_t / 2.0) -
                                                  (1.0 / omega) * np.sin(omega * d_t / 2.0)))


def dq_omega_dt(omega, d_t):
    """

    :param omega:
    :param d_t:
    :return:
    """
    omegamod = np.sqrt(np.sum(omega ** 2))
    temp = np.zeros([4, 3], dtype=np.float32)

    # Use generic ancillary functions to calculate components of Jacobian
    temp[0, 0] = dq0_by_domega_a(omega[0], omegamod, d_t)
    temp[0, 1] = dq0_by_domega_a(omega[1], omegamod, d_t)
    temp[0, 2] = dq0_by_domega_a(omega[2], omegamod, d_t)
    temp[1, 0] = dq_a_by_domega_a(omega[0], omegamod, d_t)
    temp[1, 1] = dq_a_by_domega_b(omega[0], omega[1], omegamod, d_t)
    temp[1, 2] = dq_a_by_domega_b(omega[0], omega[2], omegamod, d_t)
    temp[2, 0] = dq_a_by_domega_b(omega[1], omega[0], omegamod, d_t)
    temp[2, 1] = dq_a_by_domega_a(omega[1], omegamod, d_t)
    temp[2, 2] = dq_a_by_domega_b(omega[1], omega[2], omegamod, d_t)
    temp[3, 0] = dq_a_by_domega_b(omega[2], omega[0], omegamod, d_t)
    temp[3, 1] = dq_a_by_domega_b(omega[2], omega[1], omegamod, d_t)
    temp[3, 2] = dq_a_by_domega_a(omega[2], omegamod, d_t)
    dqomegadt_by_domega = temp
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
    mat_m = np.zeros([4, 4], dtype=np.float32)
    qq = np.sum(q ** 2)

    mat_m[0, 0] = dqi_by_dqi(q[0], qq)
    mat_m[0, 1] = dqi_by_dqj(q[0], q[1], qq)
    mat_m[0, 2] = dqi_by_dqj(q[0], q[2], qq)
    mat_m[0, 3] = dqi_by_dqj(q[0], q[3], qq)
    mat_m[1, 0] = dqi_by_dqj(q[1], q[0], qq)
    mat_m[1, 1] = dqi_by_dqi(q[1], qq)
    mat_m[1, 2] = dqi_by_dqj(q[1], q[2], qq)
    mat_m[1, 3] = dqi_by_dqj(q[1], q[3], qq)
    mat_m[2, 0] = dqi_by_dqj(q[2], q[0], qq)
    mat_m[2, 1] = dqi_by_dqj(q[2], q[1], qq)
    mat_m[2, 2] = dqi_by_dqi(q[2], qq)
    mat_m[2, 3] = dqi_by_dqj(q[2], q[3], qq)
    mat_m[3, 0] = dqi_by_dqj(q[3], q[0], qq)
    mat_m[3, 1] = dqi_by_dqj(q[3], q[1], qq)
    mat_m[3, 2] = dqi_by_dqj(q[3], q[2], qq)
    mat_m[3, 3] = dqi_by_dqi(q[3], qq)
    return mat_m


def qprod(q, r):
    """

    :param q:
    :param r:
    :return:
    """
    t = np.zeros([4, 1], dtype=np.float32)
    t[0] = (r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3])
    t[1] = (r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2])
    t[2] = (r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1])
    t[3] = (r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0])

    return t


# matrix representation of quaternion
def q2r(q):
    """

    :param q:
    :return:
    """
    mat_r = np.zeros([3, 3], dtype=np.float32)
    mat_r[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    mat_r[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    mat_r[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
    mat_r[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    mat_r[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    mat_r[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    mat_r[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    mat_r[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    mat_r[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    return mat_r


def xyz_dh_dy(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    xyz_dhrl_dy = np.linalg.inv(q2r(xv))
    return np.dot(xyz_dh_dhrl(inparams, xv, y, zi), xyz_dhrl_dy)


def xyz_dh_dhrl(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    mat_r_rw = np.linalg.inv(q2r(xv))
    xyz_dhd_dhu = np.linalg.inv(jacob_undistort_fm(zi, inparams))

    hrl = np.dot(mat_r_rw, y - xv[0:3])
    xyz_dhu_dhrl = np.zeros([2, 3], dtype=np.float32)
    xyz_dhu_dhrl[:, :] = [[inparams['fku'] / hrl[2], 0, -hrl[0] * inparams.fku / (hrl[2] ** 2)],
                          [0, inparams['fkv'] / hrl[2], -hrl[0] * inparams.fkv / (hrl[2] ** 2)]]
    a = np.dot(xyz_dhd_dhu, xyz_dhu_dhrl)
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

    dqbar_dq = np.float32(np.diag([1, -1, -1, -1]))
    q = np.array([xv[3], -xv[4], -xv[5], -xv[6]], dtype=np.float32)
    xyz_dhrl_dqwr = np.dot(d_r_q_times_a_by_dq(q, (y - xv[0:3])), dqbar_dq)
    xyz_dh_dqwr = np.dot(xyz_dh_dhrl(inparams, xv, y, zi), xyz_dhrl_dqwr)

    mat_hi1 = np.concatenate([xyz_dh_drw, xyz_dh_dqwr, np.zeros([2, 6], dtype=np.float32)], axis=1)

    return mat_hi1


def id_dh_dy(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    id_dhrl_dy = np.linalg.inv(q2r(xv))
    return np.dot(id_dh_dhrl(inparams, xv, y, zi), id_dhrl_dy)


def id_dh_dhrl(inparams, xv, y, zi):
    pass


def id_dh_dxv(inparams, xv, y, zi):
    pass


def distort_fm(uv_u, inparams):
    """

    :param uv_u:
    :param inparams:
    :return:
    """
    xu = (uv_u[0]-inparams['u0'])*inparams['d']
    yu = (uv_u[1]-inparams['v0'])*inparams['d']
    ru_2 = xu**2 + yu**2
    ru = np.sqrt(ru_2)
    rd = ru/float(1+inparams['kd1']*ru_2 + inparams['kd2']*(ru_2**2))
    for k in range(10):
        f = rd + inparams['kd1']*(rd**3) + inparams['kd2']*(rd**5) - ru
        f_p = 1 + 3*inparams['kd1']*(rd**2) + inparams['kd2']*(rd**4)
        rd = rd - (f/f_p)
    pass
    rd_2 = rd**2
    d = float(1+inparams['kd1']*rd_2 + inparams['kd2']*(rd_2**2))
    uv_d = np.zeros(2, dtype=np.float32)
    uv_d[0] = (xu/d)/float(inparams['d']) + inparams['u0']
    uv_d[1] = (yu/d)/float(inparams['d']) + inparams['v0']
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
    d = float(1+inparams['kd1']*rd_2 + inparams['kd2']*(rd_2**2))
    uv_u = np.zeros(2, dtype=np.float32)
    uv_u[0] = (xd/d)/float(inparams['d']) + inparams['u0']
    uv_u[1] = (yd/d)/float(inparams['d']) + inparams['v0']
    return uv_u


def jacob_undistort_fm(uv_d, inparams):
    """

    :param uv_d:
    :param inparams:
    :return:
    """
    xd = (uv_d[0]-inparams['u0'])*inparams['d']
    yd = (uv_d[1]-inparams['v0'])*inparams['d']
    rd_2 = xd**2 + yd**2
    rd = np.sqrt(rd_2)
    d1 = float(1+inparams['kd1']*rd_2 + inparams['kd2']*(rd_2**2))
    d2 = float(inparams['kd1'] + 2*inparams['kd2']*rd)
    j_un = np.zeros([2,2], dtype=np.float32)
    j_un[0, 0] = d1 + (uv_d[0] - inparams['u0'])*d2*2*xd*inparams['d']
    j_un[0, 1] = (uv_d[0] - inparams['u0'])*d2*2*yd*inparams['d']
    j_un[1, 0] = (uv_d[1] - inparams['v0'])*d2*2*xd*inparams['d']
    j_un[1, 1] = d1 + (uv_d[1] - inparams['v0'])*d2*2*yd*inparams['d']
    return j_un


def hinv(uv_d, xv, inparams, init_rho):
    """

    :param uv_d:
    :param xv:
    :param inparams:
    :param init_rho:
    :return:
    """
    uv_u = undistort_fm(uv_d, inparams)
    h = np.ones(3, dtype=np.float32)
    h[0] = -(inparams['u0']-uv_u[0])/float(inparams['fku'])
    h[1] = -(inparams['v0']-uv_u[1])/float(inparams['fkv'])
    mat_r = q2r(xv[3:7])
    n = np.dot(mat_r, h)
    new_feature = np.zeros(6, dtype=np.float32)
    new_feature[0:3] = xv[0:3]
    new_feature[3] = math.atan2(n[0], n[2])
    new_feature[4] = math.atan2(-n[1], np.sqrt(n[0]**2 + n[2]**2))
    new_feature[5] = init_rho
    return new_feature


def d_r_q_times_a_by_dq(q, a):
    """

    :param q:
    :param a:
    :return:
    """
    drq = np.zeros([3,4], dtype=np.float32)
    drq[0:3, 0] = np.dot(d_r_by_dq0(q), a)
    drq[1:3, 0] = np.dot(d_r_by_dqx(q), a)
    drq[2:3, 0]= np.dot(d_r_by_dqy(q), a)
    drq[3:3, 0] = np.dot(d_r_by_dqz(q), a)
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


def rotate_with_dist_fc_c2c1(uv_c1, R_c2c1, t_c2c1, n, d, inparams):
    uv_c2 = np.zeros(3)
    return uv_c2


def rotate_with_dist_fc_c1c2(uv_c2, R_c2c1, t_c2c1, n, d, inparams):
    uv_c1 = np.zeros(3)
    return uv_c1


def m(phi, theta):
    """

    :param phi:
    :param theta:
    :return:
    """
    res = np.zeros(3, dtype=np.float32)
    res[0:3] = [np.cos(phi) * np.sin(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)]
    return res


def id2cartesian(yi):
    pass


def std_dev(im, mean):
    std = 0
    return std


class quaternion:
    # konstruktor
    def __init__(self):  #
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

    def rotate(self, q, v):  # rotate to new position
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

    def rotationMatrix(q):  # matrix representation of quaternion
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





