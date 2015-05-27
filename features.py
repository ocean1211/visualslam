#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import mathematics

import detectors
import derivations

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


def delete_features(x, cov, features):
    """

    :param x:
    :param cov:
    :return:
    """
    if len(features) == 0:
        return x, cov
    delete_move = 0
    del_list = []
    for i in range(len(features)):
        if features[i]['times_predicted'] < 5:
            continue
        features[i]['begin'] -= delete_move
        if features[i]['times_measured'] < 0.5 * features[i]['times_predicted']:
            delete_move += features[i]['type'] * 2
            x, cov = delete_feature(x, cov, i, features)
            del_list.append(i)

    for i in del_list:
        features.pop(i)
    return x, cov


def delete_feature(x, cov, f_id, features):
    """

    :param x:
    :param cov:
    :param f_id:
    :return:
    """
    rows2delete = features[f_id]['type'] * 3
    begin = features[f_id]['begin']
    # IN P
    cov2 = np.zeros([cov.shape[0] - rows2delete, cov.shape[1] - rows2delete])
    cov2[0:begin, 0:begin] = cov[0:begin, 0:begin]
    if cov2.shape[0] > begin + rows2delete:
        cov2[0:begin, begin:] = cov[0:begin, begin + rows2delete:]
        cov2[begin:, 0:begin] = cov[begin + rows2delete:, 0:begin]
        cov2[begin:, begin:] = cov[begin + rows2delete:, begin + rows2delete:]
    cov = cov2
    # IN X
    x2 = np.zeros(x.shape[0] - rows2delete)
    x2[0:begin] = x[0:begin]
    x2[begin:] = x[begin + rows2delete:]
    x = x2
    return x, cov


 def update_features_info(features):
    """

    :return:
    """
    for i in range(len(features)):
        if features[i]['h'] is not None:
            features[i]['predicted'] += 1
        if features[i]['inlier'] :
            features[i]['measured'] += 1
        features[i]['match'] = 0
        features[i]['inlier'] = 0
        features[i]['h'] = None
        features[i]['z'] = None
        features[i]['H'] = None
        features[i]['S'] = None
    pass


def init_new_features(x, cov, image, camera, detector, features, frame_num, num_of_features):
    """

    :param x:
    :param cov:
    :param image:
    :param camera:
    :param detector:
    :param features:
    :param frame_num:
    :param num_of_features:
    :return:
    """
    max_attempts = 50
    attempts = 0
    initialized = 0

    while (initialized < num_of_features) and (attempts < max_attempts):
        size = x.shape[0]
        attempts += 1
        x, cov = init_new_feature(x, cov, image, camera, detector, features, frame_num)
        if size < x.shape[0]:
            initialized += 1
    return x, cov


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
    return x, cov

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

def compute_dxf_dxv(h_w, h_c, x):
    dtheta_dgw = derivations.dtheta_dgw(h_w)
    dphi_dgw = derivations.dphi_dgw(h_w)
    dgw_dqwr = derivations.dgw_dqwr(x, h_c)
    dtheta_dqwr = derivations.dtheta_dqwr(dtheta_dgw, dgw_dqwr)
    dphi_dqwr = derivations.dphi_dqwr(dphi_dgw, dgw_dqwr)
    dxf_dqwr = derivations.dxf_dqwr(dtheta_dqwr, dphi_dqwr)
    dxf_drw = derivations.dxf_drw()
    return derivations.dxf_dxv(dxf_drw, dxf_dqwr)

def compute_dxf_dhd(h_w, mat_r, xy, camera):
    dtheta_dgw = derivations.dtheta_dgw(h_w)
    dphi_dgw = derivations.dphi_dgw(h_w)
    dxfprima_dgw = derivations.dxfprima_dqw(dtheta_dgw, dphi_dgw)
    dgw_dgc = mat_r
    dgc_dhu = derivations.dgc_dhu(camera)
    dhu_dhd = derivations.dhu_dhd(xy, camera)
    dxfprima_dhd = derivations.dxfprima_dhd(dxfprima_dgw, dgw_dgc, dgc_dhu, dhu_dhd)
    return derivations.dxf_dhd(dxfprima_dhd)

def compute_new_feature(x, cov, xy, camera):
    """

    :param x:
    :param cov:
    :param xy:
    :param camera:
    :return:
    """
    rho = 0.1
    std_rho = 0.5
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

    dxf_dxv = compute_dxf_dxv(h_w, h_c, x)
    dxf_dhd = compute_dxf_dhd(h_w, mat_r, xy, camera)

    padd = np.diag([camera['sd']**2, camera['sd']**2, std_rho]).astype(np.float64)

    shp0 = cov.shape[0]
    shp1 = cov.shape[1]

    mat_p2 = np.zeros([shp0 + 6, shp1 + 6], dtype=np.float64)
    mat_p2[0:shp0, 0:shp1] = cov

    mat_p2[shp0:shp0 + 6, 0:shp1] = np.dot(dxf_dxv, cov[0:13, :])
    mat_p2[0:shp0, shp1:shp1 + 6] = np.dot(cov[:, 0:13], dxf_dxv.T)

    mat_p2[shp0:shp0 + 6, shp1:shp1 + 6] = (
        np.dot(np.dot(dxf_dxv, cov[0:13, 0:13]), dxf_dxv.T) +
        np.dot(np.dot(dxf_dhd, padd), dxf_dhd.T))

    cov = mat_p2
    return x, cov, xf
