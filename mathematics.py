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
    t = np.zeros(4, dtype=np.float32)
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
    xyz_dhu_dhrl[:, :] = [[inparams['fku'] / hrl[2], 0, -hrl[0] * inparams['fku'] / (hrl[2] ** 2)],
                          [0, inparams['fkv'] / hrl[2], -hrl[0] * inparams['fkv'] / (hrl[2] ** 2)]]
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
    inv_mat_r = np.linalg.inv(q2r(xv))
    id_dhrl_dy = np.zeros([3, 6], dtype=np.float32)
    id_dhrl_dy[0:3, 0:3] = y[5] * inv_mat_r
    temp = np.zeros(3, dtype=np.float32)
    temp[:] = [math.cos(y[4])*math.cos(y[3]), 0,
               -math.cos(y[4]) * math.sin(y[3])]
    id_dhrl_dy[0:3, 3] = np.dot(inv_mat_r, temp)
    temp[:] = [-math.sin(y[4])*math.sin(y[3]), -math.cos(y[4]),
               -math.sin(y[4]) * math.cos(y[3])]
    id_dhrl_dy[0:3, 4] = np.dot(inv_mat_r, temp)
    id_dhrl_dy[0:3, 5] = np.dot(inv_mat_r, y[0:3] - xv[0:3])

    return np.dot(id_dh_dhrl(inparams, xv, y, zi), id_dhrl_dy)


def id_dh_dhrl(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    id_dhd_dhu = np.linalg.inv(jacob_undistort_fm(zi, inparams))
    inv_mat_r = np.linalg.inv(q2r(xv[3:7]))
    temp = np.zeros(3, dtype=np.float32)
    temp[:] = [math.cos(y[4])*math.sin(y[3]), -math.sin(y[4]),
               math.cos(y[4]) * math.cos(y[3])]
    hrl = np.dot(inv_mat_r, y[0:3] - xv[0:3])*y[5] + temp
    id_dhu_dhrl = np.zeros([2, 3], dtype=np.float32)
    id_dhu_dhrl[0:2, 0:2] = np.diag(
        [inparams['fku']/hrl[2],
         inparams['fkv']/hrl[2]]).astype(np.float32)
    id_dhu_dhrl[0, 2] = -hrl[0]*inparams['fku']/(hrl[2]**2)
    id_dhu_dhrl[1, 2] = -hrl[1]*inparams['fkv']/(hrl[2]**2)
    return np.dot(id_dhd_dhu, id_dhu_dhrl)


def id_dh_dxv(inparams, xv, y, zi):
    """

    :param inparams:
    :param xv:
    :param y:
    :param zi:
    :return:
    """
    mat_h = np.zeros([2, 13], dtype=np.float32)

    id_dhrl_drw = np.linalg.inv(q2r(xv[3:7]))*y[5]
    id_dh_drw = np.dot(id_dh_dhrl(inparams, xv, y, zi), id_dhrl_drw)

    mi = m(y[4], y[3])
    dqbar_dq = np.identity(4, dtype=np.float32)
    dqbar_dq[1:, 1:] *= -1
    q = xv[3:7]
    q[1:] *= -1
    if_dhrl_dqrw = np.dot(d_r_q_times_a_by_dq(q, (y[0:3] - xv[0:3]) * y[5] + mi),
                           dqbar_dq)

    id_dh_dqrw = np.dot(id_dh_dhrl(inparams, xv, y, zi), if_dhrl_dqrw)

    mat_h[0:2, 0:3] = id_dh_drw
    mat_h[0:2, 3:7] = id_dh_dqrw

    return mat_h


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
        f = rd + inparams['kd1'] * (rd**3) + inparams['kd2']*(rd**5) - ru
        f_p = 1 + 3*inparams['kd1'] * (rd**2) + inparams['kd2']*(rd**4)
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
    j_un = np.zeros([2, 2], dtype=np.float32)
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


def rotate_with_dist_fc_c2c1(uv_c1, mat_r, t, n, d, inparams):
    """

    :param uv_c1:
    :param mat_r:
    :param t:
    :param n:
    :param d:
    :param inparams:
    :return:
    """
    uv_c = undistort_fm(uv_c1, inparams)
    mat_k = np.diag([inparams['fku'], inparams['fkv'], 1]).astype(np.float32)
    mat_k[0, 2] = inparams['u0']
    mat_k[1, 2] = inparams['v0']
    mat_k2 = np.dot(mat_k, (mat_r - (np.dot(t, n.T/float(d)))))
    mat_k2 = np.dot(mat_k2, np.linalg.inv(mat_k))
    mat_k = np.linalg.inv(mat_k2)
    temp = np.ones(3, dtype=np.float32)
    temp[0:2] = uv_c
    temp = np.dot(mat_k, temp)
    uv_c2 = temp[0:2]/temp[2]
    return distort_fm(uv_c2, inparams)


def rotate_with_dist_fc_c1c2(uv_c2, mat_r, t, n, d, inparams):
    """

    :param uv_c2:
    :param mat_r:
    :param t:
    :param n:
    :param d:
    :param inparams:
    :return:
    """
    uv_c = undistort_fm(uv_c2, inparams)
    mat_k = np.diag([inparams['fku'], inparams['fkv'], 1]).astype(np.float32)
    mat_k[0, 2] = inparams['u0']
    mat_k[1, 2] = inparams['v0']
    mat_k2 = np.dot(mat_k, (mat_r - (np.dot(t, n.T/float(d)))))
    mat_k = np.dot(mat_k2, np.linalg.inv(mat_k))
    temp = np.ones(3, dtype=np.float32)
    temp[0:2] = uv_c
    temp = np.dot(mat_k, temp)
    uv_c1 = temp[0:2]/temp[2]
    return distort_fm(uv_c1, inparams)


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
    """

    :param yi:
    :return:
    """
    temp = m(yi[4], yi[3])
    return yi[0:3] + (1/float(yi[5])) * temp


def std_dev(im, mean):
    """

    :param im:
    :param mean:
    :return:
    """
    temp = np.ones(im.shape, dtype=np.float32) * mean
    std = np.sqrt(np.mean((im - temp)**2))
    return std
