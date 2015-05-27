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
