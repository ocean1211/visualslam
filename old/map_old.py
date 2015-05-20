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
        self.delete_features(x, cov)
        measured = 0
        for feature in self.features:
            if feature['low_innovation_inlier'] or feature['high_innovation_inlier']:
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
        del_list = []
        for i in range(len(self.features)):
            if self.features[i]['times_predicted'] < 5:
                continue
            self.features[i]['begin'] -= delete_move
            if self.features[i]['times_measured'] < 0.5 * self.features[i]['times_predicted']:
                delete_move += self.features[i]['type'] * 2
                x, cov = self.delete_feature(x, cov, i)
                del_list.append(i)

        for i in del_list:
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
                temp = np.zeros(3, dtype=np.float64)
                temp[:] = xyz - x_c2
                temp2 = xyz - x_c1
                d_c2p = np.linalg.norm(temp)
                cos_alpha = (np.dot(temp.T, temp2) / (d_c2p * np.linalg.norm(temp2)))
                linearity_index = 4 * std_d * cos_alpha / d_c2p
                if linearity_index < lin_index_thresh:
                    x2 = np.zeros(x.shape[0] - 3, dtype=np.float64)
                    x2[0:begin] = x[0:begin]
                    x2[begin:begin + 6] = x[begin:begin + 6]
                    if x.shape[0] > begin + 6:
                        x2[begin + 3:] = x[begin + 6:]
                    mat_j = np.zeros([3, 6], dtype=np.float64)
                    mat_j[0:3, 0:3] = np.identity(3, dtype=np.float64)
                    mat_j[:, 3] = (1 / rho) * np.array([np.cos(phi) * np.cos(theta), 0.0, -np.cos(phi) * np.sin(theta)])
                    mat_j[:, 4] = (1 / rho) * np.array([-np.sin(phi) * np.sin(theta), -np.cos(phi), -np.sin(phi) * np.cos(theta)])
                    mat_j[:, 5] = mi / rho ** 2
                    mat_j_all = np.identity(cov.shape[0], dtype=np.float64)
                    mat_j_all[0:begin, 0:begin] = np.identity(begin, dtype=np.float64)
                    mat_j_all[begin:begin + mat_j.shape[0], begin:begin + mat_j.shape[1]] = mat_j
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
                self.features[i]['patch_wm'] = self.pred_patch_fc(self.features[i], r, mat_r, xyz_w, inparams)

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
            mat_r_wk_p_f = mathematics.q2r(f['rotation'])
            vec_r_wk_p_f = f['position']

            temp1 = np.identity(4, dtype=np.float64)
            temp2 = np.identity(4, dtype=np.float64)
            temp1[0:3, 0:3] = mat_r_wk_p_f
            temp2[0:3, 3] = vec_r_wk_p_f
            mat_h_wk_p_f = np.dot(temp1, temp2)
            temp1[0:3, 0:3] = mat_r
            temp2[0:3, 3] = r
            mat_h_wk = np.dot(temp1, temp2)
            mat_h_kpf_k = np.dot(mat_h_wk_p_f.T, mat_h_wk)
            patch_p_f = f['patch_wi']
            half_patch_wi = f['half_size_wi']
            n1 = np.zeros(3, dtype=np.float64)
            n2 = np.zeros(3, dtype=np.float64)
            n1[:] = [uv_p_f[0] - inparams['u0'], uv_p_f[1] - inparams['v0'], - inparams['fku']]
            n2[:] = [uv_p[0] - inparams['u0'], uv_p[1] - inparams['v0'], - inparams['fku']]
            temp = np.zeros(4, dtype=np.float64)
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
            patch = patch_p_f[uv_c2[1]:uv_c2[1] + 2 * half_patch_wm - 1,
                              uv_c2[0]:uv_c2[0] + 2 * half_patch_wm - 1]
        else:
            patch = None
        return patch


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
        mat_h = np.zeros([2, size_x], dtype=np.float64)
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
        mat_h = np.zeros([2, size_x], dtype=np.float64)
        mat_h[:, 0:13] = mathematics.id_dh_dxv(inparams, xv, y, zi)
        mat_h[:, f['begin']:f['begin'] + 6] = mathematics.id_dh_dy(inparams, xv, y, zi)
        return mat_h

    def rescue_hi_inliers(self, x, mat_p, inparams):
        """

        :param x:
        :param mat_p:
        :param map_obj:
        :param inparams:
        :return:
        """
        chi2inv_2_95 = 5.9915
        self.predict_camera_measurements(inparams, x)
        self.calculate_derivatives(inparams, x)
        for i in range(len(self.features)):
            f = self.features[i]
            if (f['individually_compatible'] == 1 and
               f['low_innovation_inlier'] == 0):
                nui = f['z'] - f['h']
                si = np.dot(np.dot(f['H'], mat_p), f['H'].T)
                temp = np.dot(np.dot(nui.T, np.linalg.inv(si)), nui)

                if temp < chi2inv_2_95:
                    self.features[i]['high_innovation_inlier'] = 1

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
                if z is None:
                    z = f['z']
                    h = f['h']
                    mat_h = f['H']
                else:
                    z = np.concatenate([z, f['z']])
                    h = np.concatenate([h, f['h']])
                    mat_h = np.concatenate([mat_h, f['H']])
        if mat_h is not None:
            mat_r = np.identity(mat_h.shape[0])
            ekf_filter.update(mat_h, mat_r, z, h)
        return ekf_filter

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
                if z is None:
                    z = f['z']
                    h = f['h']
                    mat_h = f['H']
                else:
                    z = np.concatenate([z, f['z']])
                    h = np.concatenate([h, f['h']])
                    mat_h = np.concatenate([mat_h, f['H']])
        if mat_h is not None:
            mat_r = np.identity(mat_h.shape[0])
            ekf_filter.update(mat_h, mat_r, z, h)
        return ekf_filter
