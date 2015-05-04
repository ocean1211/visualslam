#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
RANSAC
---------------------

"""

# importy
import numpy as np
import map
import mathematics


class Ransac:
    # konstruktor
    def __init__(self):
        pass

    def hypotheses(self, x_p, mat_p_p, features, inparams):
        """

        :param x_p:
        :param mat_p_p:
        :param features:
        :param inparams:
        :return:
        """
        p_at_least_one_spurious_free = 0.99
        thresh = inparams['image_noise']
        n_hyp = 1000
        max_hypothesis_support = 0
        pattern = self.generate_state_pattern(x_p, features)
        n = 0
        while n < n_hyp:
            position, zi, num_ic_matches = self.select_random_match(features)
            print position, zi, num_ic_matches
            hi = features[position]['h']
            mat_hi = features[position]['H']
            mat_s = np.dot(np.dot(mat_hi, mat_p_p), mat_hi.T) + features[position]['R']
            mat_k = np.dot(np.dot(mat_p_p, mat_hi.T), np.linalg.inv(mat_s))
            xi = x_p + np.dot(mat_k, (zi-hi))
            hyp_support, pos_id, pos_xyz = (
                self.compute_hypothesis_support_fast(
                    xi, inparams, pattern, features, thresh))
            if hyp_support > max_hypothesis_support:
                max_hypothesis_support = hyp_support
                features = self.set_as_most_supported_hypothesis(pos_id, pos_xyz, features)
                epsilon = 1-(hyp_support/float(num_ic_matches))
                n_hyp = np.ceil(np.log(1-p_at_least_one_spurious_free) / np.log(1 - (1-epsilon)))
            n += 1
        return features

    @staticmethod
    def generate_state_pattern(x, features):
        """

        :param x:
        :param features:
        :return:
        """
        pattern = np.zeros([x.shape[0], 4], dtype=np.float32)

        for f in features:
            if f['z'] is not None:
                if f['type'] == 1:
                    pattern[f['begin']:f['begin'] + 3, 3] = 1
                else:
                    pattern[f['begin']:f['begin'] + 3, 0] = 1
                    pattern[f['begin'] + 3:f['begin'] + 5, 1] = 1

        return pattern

    @staticmethod
    def matching(frame, map_obj, inparams):
        """

        :param frame:
        :param map_obj:
        :param inparams:
        :return:
        """
        corr_thresh = 0.5
        chi_095_2 = 5.9915
        max_corr = 0

        for i in range(len(map_obj.features)):
            if map_obj.features[i]['h'] is not None:
                max_corr = 0
                mat_s = map_obj.features[i]['S']
                e, v = np.linalg.eig(mat_s)
                if np.all(e.all() < 100):
                    inv_s = np.linalg.inv(mat_s)
                    patch = map_obj.features[i]['patch_wm']
                    half_size_x = np.ceil(2*np.sqrt(mat_s[0, 0]))
                    half_size_y = np.ceil(2*np.sqrt(mat_s[1, 1]))
                    h = map_obj.features[i]['h']
                    size_wm = map_obj.features[i]['half_size_wm']
                    candidate = np.ndarray([0, 0], dtype=np.float32)

                    for x in range(np.int32(h[0] - half_size_x),
                                   np.int32(h[0] + half_size_x)):
                        for y in range(np.int32(h[1] - half_size_y),
                                       np.int32(h[1] + half_size_y)):
                            nu = np.zeros(2, dtype=np.float64)
                            nu[:] = [x-h[0], y-h[1]]
                            if np.dot(np.dot(nu.T, inv_s), nu) < chi_095_2:
                                if ((x > size_wm) and
                                        (x < inparams['width'] - size_wm) and
                                        (y > size_wm) and
                                        (y < inparams['height'] - size_wm)):
                                    y1 = y-size_wm
                                    x1 = x-size_wm
                                    im2 = frame[y1:y1 + patch.shape[0],
                                                x1:x1 + patch.shape[1]]
                                    m_im1 = np.mean(patch)
                                    m_im2 = np.mean(im2)
                                    temp = patch - m_im1
                                    temp2 = im2 - m_im2
                                    temp = temp * temp2
                                    sd1 = mathematics.std_dev(patch, m_im1)
                                    sd2 = mathematics.std_dev(im2, m_im2)
                                    if sd1*sd2 != 0:

                                        corr = np.sum(temp)/(sd1*sd2)
                                        if corr > max_corr:
                                            max_corr = corr
                                            candidate = np.array([x, y])

                    if max_corr > corr_thresh:
                        print candidate
                        map_obj.features[i]['individually_compatible'] = 1
                        candidate = candidate.astype(dtype=np.float32)
                        map_obj.features[i]['z'] = candidate

        return map_obj

    @staticmethod
    def select_random_match(features):
        """

        :param features:
        :return:
        """
        position = 0
        if len(features) == 0:
            return 0

        ind_compatible = 0
        for i, f in enumerate(features):
            if f['individually_compatible'] == 1:
                ind_compatible += 1
        if ind_compatible == 0:
            return 0, None, 0

        while True:
            position = np.random.randint(0, len(features))
            if features[position]['individually_compatible'] == 1:
                break

        zi = features[position]['z']
        print zi
        return position, zi, ind_compatible

    @staticmethod
    def set_as_most_supported_hypothesis(pos_id, pos_xyz, features):
        """

        :param pos_id:
        :param pos_xyz:
        :param features:
        :return:
        """
        jid = 0
        jxyz = 0

        for i in range(len(features)):
            features[i]['low_innovation_inlier'] = 0
            if features[i]['z'] is not None:
                if features[i]['type'] == 1:
                    if pos_xyz[jxyz] == 1:
                        features[i]['low_innovation_inlier'] = 1
                        jxyz += 1
                else:
                    if pos_id[jid] == 1:
                        features[i]['low_innovation_inlier'] = 1
                        jid += 1
        return features

    def search_ic_matches(self, x_p, mat_p_p, map_obj, inparams, frame):
        """

        :param x_p:
        :param mat_p_p:
        :param map_obj:
        :param inparams:
        :param frame:
        :return:
        """
        map_obj.predict_camera_measurements(inparams, x_p)
        map_obj.calculate_derivatives(inparams, x_p)
        for i in range(len(map_obj.features)):
            if map_obj.features[i]['h'] is not None:
                map_obj.features[i]['S'] = (
                    np.dot(
                        np.dot(
                            map_obj.features[i]['H'],
                            mat_p_p),
                        map_obj.features[i]['H'].T
                    )
                    + map_obj.features[i]['R']
                )
        map_obj.predict_features_appearance(inparams, x_p)
        map_obj = self.matching(frame, map_obj, inparams)
        return map_obj

    @staticmethod
    def count_matches_under_a_threshold(features):
        """

        :param features:
        :return:
        """
        hyp_support = 0
        thresh = 0.5
        for i, f in enumerate(features):
            if f['z'] is not None:
                nu = f['z'] - f['h']
                temp = np.sqrt(np.sum(nu**2))
                if temp < thresh:
                    hyp_support += 1
        return hyp_support

    @staticmethod
    def compute_hypothesis_support_fast(x, inparams, pattern,
                                        features, thresh):
        """

        :param x:
        :param inparams:
        :param pattern:
        :param features:
        :param thresh:
        :return:
        """
        hyp_support = 0
        pos_id = np.zeros(pattern.shape[0], dtype=np.float32)
        pos_xyz = np.zeros(pattern.shape[0], dtype=np.float32)

        for i, f in enumerate(features):
            if f['z'] is not None:
                if f['type'] == 1:
                    mat_r = mathematics.q2r(x[3:7])
                    ri_minus_rwc = x[f['begin']:f['begin'] + 3] - x[0:3]
                    hc = np.dot(mat_r.T, ri_minus_rwc).astype(dtype=np.float32)
                    h_norm = np.zeros(2, dtype=np.float32)
                else:
                    mat_r = mathematics.q2r(x[3:7])
                    ri_minus_rwc = x[f['begin']:f['begin'] + 3] - x[0:3]
                    ri_minus_rwc_by_rhoi = ri_minus_rwc*x[f['begin']+5]
                    mi = mathematics.m(f['begin']+3, f['begin']+4)
                    hc = np.dot(mat_r.T, (ri_minus_rwc_by_rhoi + mi))
                h_norm = np.array([hc[0]/hc[2], hc[1]/hc[2]])
                h_image = np.zeros(2, dtype=np.float32)
                h_image[0] = h_norm[0]+inparams['fku'] + inparams['u0']
                h_image[1] = h_norm[1]+inparams['fkv'] + inparams['v0']
                h_image = mathematics.distort_fm(h_image, inparams)
                nu = f['z'] - h_image
                residual = np.linalg.norm(nu)
                if f['type'] == 1:
                    pos_xyz[f['begin']] = np.float32(residual > thresh)
                    hyp_support += pos_xyz[f['begin']]
                else:
                    pos_id[f['begin']] = np.float32(residual > thresh)
                    hyp_support += pos_id[f['begin']]

        return hyp_support, pos_id, pos_xyz
