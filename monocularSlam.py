#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import ekf
import numpy as np
import map
import cv2
import ransac


class Slam:
    # konstruktor
    def __init__(self, inputobj, params):     
        self.input = inputobj
        self.counter = 0
        self.params = params
        self.stepOn = True
        self.ekf_filter = None
        self.m_map = map.Map()
        self.m_ransac = ransac.Ransac()
        self.end = False
        self.frame = None
        pass

    def init(self):
        
        # EKF filter init
        ekf_filter = ekf.Ekf(self.params["dt"], 0.007, 0.007, 1.0)
        
        v0 = 0
        w0 = 1.0e-15
        std_v = 0.025
        std_w = 0.025

        ekf_filter.x = np.array([0, 0, 0, 1, 0, 0, 0,
                                 v0, v0, v0, w0, w0, w0])
        ekf_filter.P[0:7, 0:7] = np.identity(7) * 2.2204e-016
        ekf_filter.P[7:10, 7:10] = np.identity(3) * (std_v ** 2)
        ekf_filter.P[10:13, 10:13] = np.identity(3) * (std_w ** 2)

        self.ekf_filter = ekf_filter

        cv2.namedWindow('vizualization', cv2.WINDOW_AUTOSIZE)

        pass
    
    def step(self):
        if self.counter == 0:
            frame = self.input.frame(gray=True)
            self.frame = frame
            self.counter += 1
            if frame is None:
                self.end = True
                return -1

        if not self.stepOn:
            return -1

        frame = self.frame
        # MAP.manage
        self.ekf_filter.x, self.ekf_filter.P = self.m_map.manage(
            self.ekf_filter.x, self.ekf_filter.P,
            self.params, frame, self.counter)

        # Filter prediction
        self.ekf_filter.predict()


        # Next frame
        frame = self.input.frame(gray=True)
        if frame is None:
            self.end = True
            return -1
        
        # Ransac
        self.m_map = self.m_ransac.search_ic_matches(
            self.ekf_filter.x_p,
            self.ekf_filter.P_p,
            self.m_map,
            self.params,
            frame
            )
        self.m_map.features = self.m_ransac.hypotheses(
            self.ekf_filter.x_p,
            self.ekf_filter.P_p,
            self.m_map.features,
            self.params
            )
        # Filter update
        self.ekf_filter = self.m_map.update_li_inliers(self.ekf_filter)
        self.m_map.resque_hi_inliers(
            self.ekf_filter.x,
            self.ekf_filter.P,
            self.params
            )
        self.ekf_filter = self.m_map.update_hi_inliers(self.ekf_filter)

        frame2 = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for f in self.m_map.features:
            if f['individually_compatible']:
                cv2.circle(frame2, (f['h'][1], f['h'][0]), 4, (0, 0, 255), 3)
            if f['low_innovation_inlier']:
                cv2.circle(frame2, (f['h'][1], f['h'][0]), 4, (0, 255, 0), 3)
            if f['high_innovation_inlier']:
                cv2.circle(frame2, (f['h'][1], f['h'][0]), 4, (255, 0, 0), 3)

        # show

        return frame2