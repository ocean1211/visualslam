#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Monoslam:
    # konstruktor

    def __init__(self, data_grabber, camera_params):
        self.input = data_grabber
        self.counter = 0
        self.params = camera_params
        pass
    
    def step(self):

        pass