#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
MAP MANAGMENT
---------------------

"""

# imports
import detectors
import math
import mathematics
import numpy as np


class Map:
    """
        Map class
    """

    # constructor
    def __init__(self):
        self.features = []
        pass

    @staticmethod
    def new_feature(im, uv, x, step, xf, begin):
        """

        :param im:
        :param uv:
        :param x:
        :param step:
        :param xf:
        :param begin:
        :return:
        """
        feature = dict(measurement_size=2,
                       size=6,
                       times_predicted=0,
                       times_measured=0,
                       individually_compatible=0,
                       low_innovation_inlier=0,
                       high_innovation_inlier=0,
                       position=x[0:3],
                       rotation=x[3:7],
                       init_frame=step,
                       init_measurement=uv,
                       uv_wi=uv,
                       y=xf,
                       half_size_wi=20,
                       half_size_wm=6,
                       patch_wm=np.zeros(im.shape, dtype=np.float32),
                       patch_wi=im,
                       R=np.identity(3, dtype=np.float32),
                       type=2,
                       h=None,
                       z=None,
                       H=None,
                       S=None,
                       begin=begin)
        return feature

    def manage(self, x, cov, inparams, frame, step, min_num_of_features=10):
        """

        :param x:
        :param cov: EKF coviance
        :param inparams:
        :param frame:
        :param step:
        :param min_num_of_features:
        :return:
        """
        print step
        self.delete_features(x, cov)
        measured = 0
        for feature in self.features:
            if feature.low_innovation_inlier or feature.high_innovation_inlier:
                measured += 1
        self.update_features_info()
        x, cov = self.inverse_depth2xyz(x, cov)
        if measured < min_num_of_features:
            x, cov = self.initialize_features(x, cov, inparams, frame, inparams, min_num_of_features - measured)
            return x, cov

    def delete_features(self, x, cov):
        """

        :param x:
        :param cov:
        :return:
        """
        if len(self.features) == 0:
            return x, cov
        delete_move = 0
        for i in range(len(self.features)):
            self.features[i]['begin'] -= delete_move
            if self.features[i]['times_measured'] < 0.5 * self.features[i]['times_predicted']:
                delete_move += self.features[i]['type'] * 2
                x, cov = self.delete_feature(x, cov, i)
                self.features.pop(i)
            return x, cov

    def delete_feature(self, x, cov, f_id):
        """

        :param x:
        :param cov:
        :param f_id:
        :return:
        """
        rows2delete = self.features[f_id]['type'] * 3
        begin = self.features[f_id]['begin']
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

    def update_features_info(self):
        """

        :return:
        """
        i = 0
        for i in range(len(self.features)):
            if self.features[i]['h'] is not None:
                self.features[i]['times_predicted'] += 1
            if self.features[i]['low_innovation_inlier'] or self.features[i]['high_innovation_inlier']:
                self.features[i]['times_measured'] += 1
        self.features[i]['individually_compatible'] = 0
        self.features[i]['low_innovation_inlier'] = 0
        self.features[i]['high_innovation_inlier'] = 0
        self.features[i]['h'] = None
        self.features[i]['z'] = None
        self.features[i]['H'] = None
        self.features[i]['S'] = None
        pass

    def inverse_depth2xyz(self, x, cov):
        """

        :param x:
        :param cov:
        :return:
        """
        lin_index_thresh = 0.1
        convert = 0

        for i in range(len(self.features)):
            if convert == 1:
                self.features[i]['begin'] -= 3
                continue

            if self.features[i]['type'] == 2:
                begin = self.features[i]['begin']
                std_rho = np.sqrt(cov[begin + 5, begin + 5])
                rho = x[begin + 5]
                std_d = std_rho / (rho ** 2)
                theta = x[begin + 3]
                phi = x[begin + 4]
                mi = mathematics.m(theta, phi)
                x_c1 = x[begin:begin + 3]
                x_c2 = x[0:3]
                xyz = x_c2 + (1 / rho) * mi
                temp = np.zeros(3, dtype=np.float32)
                temp[:] = xyz - x_c2
                temp2 = xyz - x_c1
                d_c2p = np.linalg.norm(temp)
                cos_alpha = (np.dot(temp.T, temp2) / (d_c2p * np.linalg.norm(temp2)))
                linearity_index = 4 * std_d * cos_alpha / d_c2p
                if linearity_index < lin_index_thresh:
                    x2 = np.zeros(x.shape[0] - 3, dtype=np.float32)
                    x2[0:begin] = x[0:begin]
                    x2[begin:begin + 3] = x[begin:begin + 6]
                    if x.shape[0] > begin + 6:
                        x2[begin + 3:] = x[begin + 6:]
                    mat_j = np.zeros([3, 6], dtype=np.float32)
                    mat_j[0:3, 0:3] = np.identity(3, dtype=np.float32)
                    mat_j[3, :] = (1 / rho) * [np.cos(phi) * np.cos(theta), 0, -np.cos(phi) * np.sin(theta)]
                    mat_j[4, :] = (1 / rho) * [-np.sin(phi) * np.sin(theta), -np.cos(phi), -np.sin(phi) * np.cos(theta)]
                    mat_j[5, :] = mi / rho ** 2
                    mat_j_all = np.zeros([cov.shape[0], cov.shape[1] - 3], dtype=np.float32)
                    mat_j_all[0:begin, 0:begin] = np.identity(begin, dtype=np.float32)
                    mat_j_all[begin:begin + mat_j.shape[0], begin:begin + mat_j.shape[1]] = mat_j
                    if x.shape[0] > begin + 6:
                        mat_j_all[begin + mat_j.shape[0]:, begin + mat_j.shape[1]] = np.identity(
                            mat_j_all[begin + mat_j.shape[0]:, begin + mat_j.shape[1]].shape[0], dtype=np.float32)
                    cov = np.dot(np.dot(mat_j_all, cov), mat_j_all.T)
                    convert = 1

        return x, cov

    def initialize_features(self, x, cov, inparams, frame, step, num_of_features):
        """

        :param x:
        :param cov:
        :param inparams:
        :param frame:
        :param step:
        :param num_of_features:
        :return:
        """
        max_attempts = 50
        attempts = 0
        initialized = 0

        while (initialized < num_of_features) and (attempts < max_attempts):
            size = x.shape[0]
            attempts += 1
            x, cov = self.initialize_feature(x, cov, inparams, frame, step)
            if size < x.shape[0]:
                initialized += 1
        return x, cov

    def initialize_feature(self, x, mat_p, inparams, frame, step):
        """

        :param x:
        :param mat_p:
        :param inparams:
        :param frame:
        :param step:
        :return:
        """
        half_patch_wi = 20
        excluded_band = half_patch_wi + 1
        max_init_attempts = 1
        init_box_semisize = [30, 20]
        init_rho = 1
        std_rho = 1
        detected_new = 0
        # newFeature, newFeatureY
        n_f = np.ones(2, dtype=np.float32)
        self.predict_camera_measurements(inparams, x)

        for i in range(max_init_attempts):
            if detected_new == 1:
                return 1
            are_there_corners = 0
            are_there_features = 0
            region_center = np.random.random_sample(2)

            region_center[0] = np.floor(region_center[0] * (
                inparams['width'] - 2 * excluded_band -
                2 * init_box_semisize[0] + 0.5) + 
                excluded_band + init_box_semisize[0])

            region_center[1] = np.floor(region_center[1] * (
                inparams['height'] - 2 * excluded_band -
                2 * init_box_semisize[1] + 0.5) +
                excluded_band + init_box_semisize[1])

            for j in range(len(self.features)):
                if ((self.features[j]['h'] > region_center[0] - init_box_semisize[0]) and
                        (self.features[j]['h'] < region_center[0] + init_box_semisize[0]) and
                        (self.features[j]['h'] > region_center[1] - init_box_semisize[1]) and
                        (self.features[j]['h'] < region_center[1] + init_box_semisize[1])):
                    are_there_features = 1
                    break
            if are_there_features == 1:
                continue
            frame_part = (frame[region_center[1] - init_box_semisize[1]:
                          region_center[1] + init_box_semisize[1],
                          region_center[0] - init_box_semisize[0]:
                          region_center[0] + init_box_semisize[0]])

            kp = detectors.detect("FAST", frame_part)
            kp_mat = np.zeros([len(kp), 2], dtype=np.float32)
            for j in range(len(kp)):
                (xp, yp) = kp[j].pt
                kp_mat[j, :] = [xp, yp]
            if len(kp) > 0:
                temp = np.ones(kp_mat.shape, dtype=np.float32)
                temp[:, 0] *= (- init_box_semisize[0] + region_center[0] - 1)
                temp[:, 1] *= (- init_box_semisize[1] + region_center[1] - 1)
                kp_mat += temp
                are_there_corners = 1

            if are_there_corners == 1:
                n_f = kp_mat[0, :].T
                detected_new = 1

            if n_f[0] * n_f[1] >= 0:
                temp = n_f
                x, mat_p, xf = self.add_features_id(temp, x, mat_p, inparams, init_rho, std_rho)
                self.add_feature_info(n_f, frame, x, step, xf)
        pass

    def predict_camera_measurements(self, inparams, x):
        """

        :param inparams:
        :param x:
        :return:
        """
        r = x[0:3]
        mat_r = mathematics.q2r(x[3:7])
        for i in range(len(self.features)):
            begin = self.features[i]['begin']
            if self.features[i]['type'] == 1:
                yi = x[begin:begin + 3]
                hi = self.hi(yi, r, mat_r, inparams)
            else:
                yi = x[begin:begin + 6]
                hi = self.hi(yi, r, mat_r, inparams)
            self.features[i]['h'] = hi
        pass

    # ID = inverse depth
    @staticmethod
    def hi(yi, r, mat_r, inparams):
        """

        :param yi:
        :param r:
        :param mat_r:
        :param inparams:
        :return:
        """
        if yi.shape[0] > 3:
            theta, phi, rho = yi[[3, 4, 5]]
            mv = mathematics.m
            hrl = np.dot(mat_r.T, (yi[0:3] - r)) * rho + mv
        else:
            hrl = np.dot(mat_r.T, (yi - r))
        if ((math.atan2(hrl[0], hrl[2]) * 180 / math.pi < -60) or
                (math.atan2(hrl[0], hrl[2]) * 180 / math.pi > 60) or
                (math.atan2(hrl[1], hrl[2]) * 180 / math.pi < -60) or
                (math.atan2(hrl[1], hrl[2]) * 180 / math.pi > 60)):
            return None
        yi2 = yi(2)
        if yi2 == 0:
            yi2 += 1
        uv_u1 = inparams['u0'] + (yi(0) / yi2) * inparams['fku']
        uv_u2 = inparams['v0'] + (yi(1) / yi2) * inparams['fkv']
        uv_u = np.zeros(2, dtype=np.float32)
        uv_u[:] = [uv_u1, uv_u2]
        uv_d = mathematics.distort_fm(uv_u, inparams)

        # is feature visible ?
        if ((uv_d[0] > 0) and (uv_d[0] < inparams['w']) and
                (uv_d[1] > 0) and (uv_d[1] < inparams['h'])):
            return uv_d
        else:
            return None

    def predict_features_appearance(self, inparams, x):
        """

        :param inparams:
        :param x:
        :return:
        """
        r = x[0:3]
        mat_r = mathematics.q2r(x[3:7])
        for i in range(len(self.features)):
            if self.features[i]['h'] is not None:
                begin = self.features[i]['begin']
                if self.features[i]['type'] == 1:
                    xyz_w = x[begin:begin + 3]
                else:
                    xyz_w = mathematics.id2cartesian(x[begin:begin + 6])
                self.features[i]['patch'] = self.pred_patch_fc(self.features[i], r, mat_r, xyz_w, inparams)

    @staticmethod
    def pred_patch_fc(f, r, mat_r, xyz, inparams):
        """

        :param f:
        :param r:
        :param mat_r:
        :param xyz:
        :param inparams:
        :return:
        """
        uv_p = f['h']
        half_patch_wm = f['half_size_wm']

        if ((uv_p[0] > half_patch_wm) and
                (uv_p[0] < inparams['width'] - half_patch_wm) and
                (uv_p[1] > half_patch_wm) and
                (uv_p[1] < inparams['height'] - half_patch_wm)):
            uv_p_f = f['init_measurement']
            mat_r_wk_p_f = f['rotation']
            vec_r_wk_p_f = f['position']
            temp1 = np.zeros([4, 4], dtype=np.float32)
            temp2 = np.zeros([4, 4], dtype=np.float32)
            temp1[0:3, 0:3] = mat_r_wk_p_f
            temp2[0:3, 3] = vec_r_wk_p_f
            mat_h_wk_p_f = np.dot(temp1, temp2)
            temp1[0:3, 0:3] = mat_r
            temp2[0:3, 3] = r
            mat_h_wk = np.dot(temp1, temp2)
            mat_h_kpf_k = np.dot(mat_h_wk_p_f.T, mat_h_wk)
            patch_p_f = f['patch_wi']
            half_patch_wi = f['half_size_wi']
            n1 = np.zeros(3, dtype=np.float32)
            n2 = np.zeros(3, dtype=np.float32)
            n1[:] = [uv_p_f[0] - inparams['u0'], uv_p_f[1] - inparams['v0'], - inparams['fku']]
            n2[:] = [uv_p[0] - inparams['u0'], uv_p[1] - inparams['v0'], - inparams['fku']]
            temp = np.zeros(4, dtype=np.float32)
            temp[0:3] = n2
            temp[3] = 1
            temp = np.dot(mat_h_kpf_k, temp)
            temp = temp / temp[3]
            n2 = temp[0:3]
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)
            n = n1 + n2
            n = n / np.linalg.norm(n)
            temp[0:3] = xyz
            temp[3] = 1
            xyz_kpf = np.dot(np.linalg.inv(mat_h_wk_p_f), temp)
            xyz_kpf = xyz_kpf / xyz_kpf[3]
            d = n[0] * xyz_kpf[0] - n[1] * xyz_kpf[1] - n[2] * xyz_kpf[2]
            h3x3 = mat_h_kpf_k[0:3, 0:3]
            h3x1 = mat_h_kpf_k[0:3, 3]
            uv_p_pred_patch = mathematics.rotate_with_dist_fc_c2c1(uv_p_f, h3x3, h3x1, n, d, inparams)
            uv_c1 = uv_p_pred_patch - half_patch_wm
            uv_c2 = mathematics.rotate_with_dist_fc_c1c2(uv_c1, h3x3, h3x1, n, d, inparams)
            uv_c2[:] = np.floor(uv_c2 - uv_p_f - half_patch_wi - 1) - 1
            uv_c2[0] = max(uv_c2[0], 0)
            uv_c2[1] = max(uv_c2[1], 0)
            if uv_c2[0] >= patch_p_f.shape[1] - 2 * half_patch_wm:
                uv_c2[0] -= (2 * half_patch_wm + 2)
            if uv_c2[1] >= patch_p_f.shape[0] - 2 * half_patch_wm:
                uv_c2[1] -= (2 * half_patch_wm + 2)
            patch = patch_p_f[uv_c2[1], uv_c2[0], 2 * half_patch_wm - 1, 2 * half_patch_wm - 1]
        else:
            patch = None
        return patch

    def add_features_id(self, uvd, x, mat_p, inparams, init_rho, std_rho):
        """

        :param uvd:
        :param x:
        :param mat_p:
        :param inparams:
        :param init_rho:
        :param std_rho:
        :return:
        """
        xv = x[0:13]
        xf = mathematics.hinv(uvd, xv, inparams, init_rho)
        x = np.concatenate(x, xf)
        mat_p = self.add_feature_covariance_id(uvd, x, mat_p, inparams, std_rho)
        return x, mat_p, xf

    @staticmethod
    def add_feature_covariance_id(uvd, x, mat_p, inparams, std_rho):
        """

        :param uvd:
        :param x:
        :param mat_p:
        :param inparams:
        :param std_rho:
        :return:
        """
        mat_r = mathematics.q2r(x[3:7])
        uv_u = mathematics.undistort_fm(uvd, inparams)
        xyz_c = np.zeros(3, np.float32)
        x_c = (-(inparams['u0'] - uv_u[0]) / inparams['fku'])
        y_c = (-(inparams['v0'] - uv_u[1]) / inparams['fkv'])
        xyz_c[:] = [x_c, y_c, 1]
        xyz_w = np.dot(mat_r, xyz_c)
        dtheta_dgw = np.zeros(3, np.float32)
        dtheta_dgw[:] = [(xyz_w[0] / (xyz_w[0] ** 2 + xyz_w[2] ** 2)), 0, (-xyz_w[0] / (xyz_w[0] ** 2 + xyz_w[2] ** 2))]
        dphi_dgw = np.zeros(3, np.float32)
        dphi_dgw[:] = [(xyz_w[0] * xyz_w[1]) / ((np.sum(xyz_w ** 2)) * np.sqrt(xyz_w[0] ** 2 + xyz_w[2] ** 2)),
                       -np.sqrt(xyz_w[0] ** 2 + xyz_w[2] ** 2) / (np.sum(xyz_w ** 2)),
                       (xyz_w[2] * xyz_w[1]) / ((np.sum(xyz_w ** 2)) * np.sqrt(xyz_w[0] ** 2 + xyz_w[2] ** 2))]
        dgw_dqwr = mathematics.d_r_q_times_a_by_dq(x[3:7], xyz_c)
        dtheta_dqwr = np.dot(dtheta_dgw.T, dgw_dqwr)
        dphi_dqwr = np.dot(dphi_dgw.T, dgw_dqwr)
        dy_dqwr = np.zeros([6, 4], dtype=np.float32)
        dy_dqwr[3, 0:4] = dtheta_dqwr.T
        dy_dqwr[4, 0:4] = dphi_dqwr.T
        dy_drw = np.zeros([6, 3], dtype=np.float32)
        dy_drw[0:3, 0:3] = np.identity(3, dtype=np.float32)
        dy_dxv = np.zeros([6, 13], dtype=np.float32)
        dy_dxv[0:6, 0:3] = dy_drw
        dy_dxv[0:6, 4:8] = dy_drw
        dyprima_dgw = np.zeros([5, 3], dtype=np.float32)
        dyprima_dgw[3, 0:3] = dtheta_dgw.T
        dyprima_dgw[4, 0:3] = dphi_dgw.T
        dgw_dgc = mat_r
        dgc_dhu = np.zeros([3, 2], dtype=np.float32)
        dgc_dhu[:] = [[1 / inparams['fku'], 0, 0], [1 / inparams['fkv'], 0, 0]]
        dhu_dhd = mathematics.jacob_undistort_fm(uvd, inparams)
        dyprima_dhd = np.dot(np.dot(np.dot(dyprima_dgw, dgw_dgc), dgc_dhu), dhu_dhd)
        dy_dhd = np.zeros([6, 3], dtype=np.float32)
        dy_dhd[0:5, 0:2] = dyprima_dhd
        dy_dhd[5, 2] = 1
        padd = np.zeros([3, 3], dtype=np.float32)
        padd[0:2, 0:2] = np.identity(2, dtype=np.float32) * inparams['sd'] ** 2
        padd[2, 2] = std_rho
        mat_p2 = np.zeros([mat_p.shape[0] + 6, mat_p.shape[1] + 6], dtype=np.float32)
        mat_p2[0:mat_p.shape[0], 0:mat_p.shape[1]] = mat_p
        mat_p2[mat_p.shape[0]:mat_p.shape[0] + 6, 0:13] = np.dot(dy_dxv, mat_p[0:13, 0:13])
        mat_p2[0:13, mat_p.shape[1]:mat_p.shape[1] + 6] = np.dot(mat_p[0:13, 0:13], dy_dxv.T)

        shp0 = mat_p.shape[0]
        shp1 = mat_p.shape[1]
        if mat_p.shape[0] > 13:
            mat_p2[shp0:shp0 + 6, 13:shp1 - 13] = np.dot(dy_dxv, mat_p[0:13, 13:])
            mat_p2[13:shp0 - 13, shp1:shp1 + 6] = np.dot(mat_p[13:, 0:13], dy_dxv.T)
            mat_p2[shp0:shp0 + 6, shp1:shp1 + 6] = (
                np.dot(np.dot(dy_dxv, mat_p[0:13, 0:13]), dy_dxv.T) +
                np.dot(np.dot(dy_dhd, padd), dy_dhd.T))
        return mat_p2

    def add_feature_info(self, uv, im, x_res, step, xf):
        """

        :param uv:
        :param im:
        :param x_res:
        :param step:
        :param xf:
        :return:
        """
        half_size_wi = 20
        x1 = min(max(uv[0] - half_size_wi, 0), im.shape[1])
        x2 = min(max(uv[0] - half_size_wi + 41, 0), im.shape[1])
        y1 = min(max(uv[1] - half_size_wi, 0), im.shape[0])
        y2 = min(max(uv[1] - half_size_wi + 41, 0), im.shape[0])
        im_patch = im[y1:y2, x1:x2]
        begin = 0
        if len(self.features) != 0:
            begin = self.features[-1]['begin'] + self.features[-1]['type'] * 3
        self.features.append(self.new_feature(im_patch, uv, x_res, step, xf, begin))

    def calculate_derivatives(self, inparams, x):
        """

        :param inparams:
        :param x:
        :return:
        """
        for i in range(len(self.features)):
            if self.features[i]['h'] is not None:
                begin = self.features[i]['begin']
                f = self.features[i]
                if f['type'] == 1:
                    y = x[begin:begin + 3]
                    self.features[i]['H'] = self.mat_h_xyz(x[0:13], y, inparams, f, x.shape[0])
                else:
                    y = x[begin:begin + 6]
                    self.features[i]['H'] = self.mat_h_id(x[0:13], y, inparams, f, x.shape[0])

    @staticmethod
    def mat_h_xyz(xv, y, inparams, f, size_x):
        """

        :param xv:
        :param y:
        :param inparams:
        :param f:
        :param size_x:
        :return:
        """
        zi = f['h']
        mat_h = np.zeros([2, size_x], dtype=np.float32)
        mat_h[:, 0:13] = mathematics.xyz_dh_dxv(inparams, xv, y, zi)
        mat_h[:, f['begin']:f['begin'] + 3] = mathematics.xyz_dh_dy(inparams, xv, y, zi)
        return mat_h

    @staticmethod
    def mat_h_id(xv, y, inparams, f, size_x):
        """

        :param xv:
        :param y:
        :param inparams:
        :param f:
        :param size_x:
        :return:
        """
        zi = f['h']
        mat_h = np.zeros([2, size_x], dtype=np.float32)
        mat_h[:, 0:13] = mathematics.id_dh_dxv(inparams, xv, y, zi)
        mat_h[:, f['begin']:f['begin'] + 6] = mathematics.id_dh_dy(inparams, xv, y, zi)
        return mat_h

    def rescue_hi_inliers(self, x, mat_p, map_obj, inparams):
        """

        :param x:
        :param mat_p:
        :param map_obj:
        :param inparams:
        :return:
        """
        chi2inv_2_95 = 5.9915
        map_obj.predictCameraMeasurements(inparams, x)
        map_obj.calculate_derivatives(x, inparams)
        for i in range(len(self.features)):
            f = map_obj.features[i]
            nui = f['z'] - f['h']
            si = np.dot(np.dot(f['H'], mat_p), f['H'].T)
            temp = np.dot(np.dot(nui.T, np.linalg.inv(si)), nui)

            if temp < chi2inv_2_95:
                self.features[i]['high_innovation_inlier'] = 1
        pass

    def update_hi_inliers(self, ekf_filter):
        """

        :param ekf_filter:
        :return:
        """
        z = None
        h = None
        mat_h = None
        for i, f in enumerate(self.features):
            if self.features[i]['high_innovation_inlier'] == 1:
                if z is not None:
                    z = f['z']
                    h = f['h']
                    mat_h = f['H']
                else:
                    z = np.concatenate([z, f['z']])
                    h = np.concatenate([h, f['h']])
                    mat_h = np.concatenate([mat_h, f['H']])
        mat_r = np.identity(mat_h.shape[0])
        ekf_filter.update(mat_h, mat_r, z, h)
        pass

    def update_li_inliers(self, ekf_filter):
        """

        :param ekf_filter:
        :return:
        """
        z = None
        h = None
        mat_h = None
        for i, f in enumerate(self.features):
            if self.features[i]['high_innovation_inlier'] == 1:
                if z is not None:
                    z = f['z']
                    h = f['h']
                    mat_h = f['H']
                else:
                    z = np.concatenate([z, f['z']])
                    h = np.concatenate([h, f['h']])
                    mat_h = np.concatenate([mat_h, f['H']])
        mat_r = np.identity(mat_h.shape[0])
        ekf_filter.update(mat_h, mat_r, z, h)
        pass
