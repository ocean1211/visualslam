#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import mathematics

import detectors


def add_feature(features, x, xf, uv, im, frame_num):
    """

    :param x:
    :param xf:
    :param uv:
    :param im:
    :param begin:
    :param frame_num:
    :return:
    """

    if len(features) != 0:
        begin = features[-1]['begin'] + 3 * features[-1]['type']
    else:
        begin = 13

    f = dict(predicted=0,
             measured=0,
             match=0,
             inlier=0,
             position=x[0:3],
             rotation=x[3:7],
             frame_num=frame_num,
             init_measurement=uv,
             y=xf,
             im_wm=np.zeros(im.shape, dtype=np.float64),
             im_wi=im,
             R=np.identity(2, dtype=np.float64),
             type=2,
             h=None,
             z=None,
             H=None,
             S=None,
             begin=begin)


    features.append(f)


def init_new_feature(x, cov, image, camera, detector, features, frame_num):
    """

    :param x:
    :param cov:
    :param image:
    :param camera:
    :param detector:
    :param features:
    :return:
    """
    features = predict_feature_measurement(x, camera, features)
    xy, frame_part = find_new_feature(image, camera, detector, features)
    x, cov, xf = compute_new_feature(x, cov, xy, camera)
    add_feature(features, x, xf, xy, frame_part, frame_num)

def predict_feature_measurement(x, camera, features):
    """

    :param x:
    :param camera:
    :param features:
    :return:
    """
    for i, f in enumerate(features):

        mat_r = mathematics.q2r(x[3:7])
        if f['type'] == 2:
            mi = mathematics.m(phi=x[f['begin']+4], theta=x[f['begin']+3])
            hrl = np.dot(mat_r.T, (x[f['begin']:f['begin']+3] - x[0:3])) * x[f['begin']+5] + mi
        else:
            hrl = np.dot(mat_r.T, (x[f['begin']:f['begin']+3] - x[0:3]))
        tempx = math.atan2(hrl[0], hrl[2]) * 180 / math.pi
        tempy = math.atan2(hrl[1], hrl[2]) * 180 / math.pi
        if (tempx > 60) or (tempx < -60) or (tempy > 60) or (tempy < -60):
            continue

        uvx = camera['u0'] + (hrl[0] / hrl[2]) * camera['fku']
        uvy = camera['v0'] + (hrl[1] / hrl[2]) * camera['fkv']
        uv = np.array([uvx, uvy])
        uvd = mathematics.distort_fm(uv, camera)

        if np.all(uvd > 0) and np.all(uvd < camera['size']):
            f['h'] = uvd
    return features


def find_new_feature(image, camera, detector, features, size=np.array([30, 20])):
    """

    :param image:
    :param camera:
    :param detector:
    :param features:
    :param size:
    :return:
    """
    half_size = size/2
    box = np.floor(np.random.rand(2) * (np.array(camera['size']) - 4 * half_size - 1)) + half_size

    for f in features:
        if f['h'] is not None:
            if (np.all(f['h'] > box)) and (np.all(f['h'] < box + size)):
                return None, None

    im_part = image[box[1]:box[1]+size[1], box[0]:box[0]+size[0]]
    kp = detectors.detect(detector['type'], im_part)
    if len(kp) == 0:
        return None, None
    (xp, yp) = kp[0].pt
    xp += box[0]
    yp += box[1]
    im_part = image[yp-half_size[1]:yp+half_size[1], xp-half_size[0]:xp+half_size[0]]

    return np.array([xp, yp]), im_part


def compute_new_feature(x, cov, xy, camera):
    """

    :param x:
    :param cov:
    :param xy:
    :param camera:
    :return:
    """
    rho = 1
    std_rho = 1

    # State vector
    xf = mathematics.hinv(xy, x, camera, rho)
    x = np.concatenate([x, xf])

    # Covariance
    mat_r = mathematics.q2r(x[3,7])
    uv_u = mathematics.undistort_fm(xy, camera)
    h_c = np.array([-(camera['u0']-uv_u[0])/camera['fku'],
                    -(camera['v0']-uv_u[1])/camera['fkv'],
                    1.0])

    h_w = np.dot(mat_r, h_c)

    # Theta podle [x,y,z]
    dtheta_dgw = np.array([h_w[2]/np.sum(h_w[[0, 2]]**2), 0, -h_w[0]/np.sum(h_w[[0, 2]]**2)])

    # Phi podle [x,y,z]
    dphi_dgw = np.array([(h_w[0] * h_w[1])/((np.sum(h_w ** 2)) * np.sqrt(h_w[[0, 2]]**2)),
                         - np.sqrt(h_w[[0, 2]]**2)/(np.sum(h_w ** 2)),
                         (h_w[2] * h_w[1])/((np.sum(h_w ** 2)) * np.sqrt(h_w[[0, 2]]**2))
                         ])
    # [x, y, z] podle q
    dgw_dqwr = mathematics.d_r_q_times_a_by_dq(x[3:7], h_c)

    dtheta_dqwr = np.dot(dtheta_dgw.T, dgw_dqwr)
    dphi_dqwr = np.dot(dphi_dgw.T, dgw_dqwr)

    dxf_dqwr = np.zeros([6, 4], dtype=np.float64)
    dxf_dqwr[3, 0:4] = dtheta_dqwr.T
    dxf_dqwr[4, 0:4] = dphi_dqwr.T
    dxf_drw = np.zeros([6, 3], dtype=np.float64)
    dxf_drw[0:3, 0:3] = np.identity(3, dtype=np.float64)
    dxf_dxv = np.zeros([6, 13], dtype=np.float64)
    dxf_dxv[0:6, 0:3] = dxf_drw
    dxf_dxv[0:6, 4:8] = dxf_dqwr

    dxfprima_dgw = np.zeros([5, 3], dtype=np.float64)
    dxfprima_dgw[3, 0:3] = dtheta_dgw.T
    dxfprima_dgw[4, 0:3] = dphi_dgw.T

    dgw_dgc = mat_r
    dgc_dhu = np.zeros([3, 2], dtype=np.float64)
    dgc_dhu[0:2, 0:2] = np.diag([1 / camera['fku'], 1 / camera['fkv']])
    dhu_dhd = mathematics.jacob_undistort_fm(xy, camera)

    dyprima_dhd = np.dot(np.dot(np.dot(dxfprima_dgw, dgw_dgc), dgc_dhu), dhu_dhd)

    dxf_dhd = np.zeros([6, 3], dtype=np.float64)
    dxf_dhd[0:5, 0:2] = dyprima_dhd
    dxf_dhd[5, 2] = 1

    padd = np.zeros([3, 3], dtype=np.float64)
    padd[0:2, 0:2] = np.identity(2, dtype=np.float64) * camera['sd'] ** 2
    padd[2, 2] = std_rho

    mat_p2 = np.zeros([cov.shape[0] + 6, cov.shape[1] + 6], dtype=np.float64)
    mat_p2[0:cov.shape[0], 0:cov.shape[1]] = cov
    mat_p2[cov.shape[0]:cov.shape[0] + 6, 0:13] = np.dot(dxf_dxv, cov[0:13, 0:13])
    mat_p2[0:13, cov.shape[1]:cov.shape[1] + 6] = np.dot(cov[0:13, 0:13], dxf_dxv.T)

    shp0 = cov.shape[0]
    shp1 = cov.shape[1]
    if shp0 > 13:
        mat_p2[shp0:shp0 + 6, 13:shp1] = np.dot(dxf_dxv, cov[0:13, 13:])
        mat_p2[13:shp0, shp1:shp1 + 6] = np.dot(cov[13:, 0:13], dxf_dxv.T)
        mat_p2[shp0:shp0 + 6, shp1:shp1 + 6] = (
            np.dot(np.dot(dxf_dxv, cov[0:13, 0:13]), dxf_dxv.T) +
            np.dot(np.dot(dxf_dhd, padd), dxf_dhd.T))

    cov = mat_p2

    return x, cov, xf
