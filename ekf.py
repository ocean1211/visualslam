#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
EKF 
---------------------

"""

# importy
import mathematics
import numpy as np


# trida rozsireneho kalmanova filtru
class Ekf:

    # konstruktor
    def __init__(self, dt, lin, ang, img): 
        self.dtime = dt
        self.x = np.zeros(13, dtype=np.float32)
        self.x_p = np.zeros(13, dtype=np.float32)
        self.P = np.zeros([13, 13], dtype=np.float32)
        self.P_p = np.zeros([13, 13], dtype=np.float32)    
        # LINEAR ACCELERATION DEV
        self.lin_dev = lin
        # ANGULAR ACCELERATION DEV
        self.ang_dev = ang  
        self.image_noise = img
        pass

    # krok predikce
    def predict(self):
        # Camera motion prediction fv and features prediction      
        self.predictCameraMotion()
        # Camera motion prediction jacobian  dfv by dxv
        jacobian_fv_xv = self.predictMotionJacobian()
        lincov = self.lin_dev ** 2 * self.dtime ** 2
        angcov = self.ang_dev ** 2 * self.dtime ** 2
        
        mat_pn = np.zeros([6, 6], dtype=np.float32)
        mat_pn[0:3, 0:3] = np.identity(3) * lincov
        mat_pn[3:6, 3:6] = np.identity(3) * angcov
        
        mat_g = np.zeros([13, 6], dtype=np.float32)
        mat_g[7:10, 0:3] = np.identity(3)
        mat_g[10:13, 3:6] = np.identity(3)
        mat_g[0:3, 0:3] = np.identity(3) * self.dtime
        
        temp4x4 = mathematics.dq3_by_dq1(self.x[3:7])
        temp4x3 = mathematics.dq_omega_dt(self.x[10:13], self.dtime)
        mat_g[3:7, 3:6] = np.dot(temp4x4, temp4x3)
        
        mat_q = np.dot(np.dot(mat_g, mat_pn), mat_g.T)
        
        self.P_p = np.zeros(self.P.shape, dtype=np.float32)
        self.P_p[0:13, 0:13] = np.dot(np.dot(jacobian_fv_xv, self.P), jacobian_fv_xv.T) + mat_q
        if self.P.shape[0] > 13:
            self.P_p[13:, 13:] = self.P[13:, 13:]
            self.P_p[0:13, 13:] = np.dot(jacobian_fv_xv, self.P_p[0:13, 13:])
            self.P_p[13:, 0:13] = np.dot(self.P_p[0:13, 13:], jacobian_fv_xv.T)
        pass

    def predictCameraMotion(self):
        r = self.x[0:3]
        q = self.x[3:7]
        v = self.x[7, 10]  
        omega = self.x[10:13]
        qwt = mathematics.QuaternionFromAngularVelocity(omega * self.dtime)    
        self.x_p[0:3] = r + v * self.dtime
        self.x_p[3:7] = q * mathematics.qprod(q, qwt)
    
    def predictMotionJacobian(self):
        jacobian_fv_xv = np.identity([13, 13], dtype=np.float32)        
        temp3x3A = np.identity([13, 13]) * self.dtime
        jacobian_fv_xv[0:3, 7:10] = temp3x3A
        qwt = mathematics.QuaternionFromAngularVelocity(self.x[10:13] * self.dtime)
        jacobian_fv_xv[3:7, 3:7] = mathematics.dq3_by_dq2(qwt)
        temp4x4 = mathematics.dq3_by_dq1(self.x[3:7])
        temp4x3 = mathematics.dq_omega_dt(self.x[10:13], self.dtime)
        jacobian_fv_xv[3:7, 10:13] = np.dot(temp4x4, temp4x3)
        return jacobian_fv_xv
        
#krok filtrace
    def update(self, H, R, z, h):
        if (z.shape[0] > 0):
            #S = model.dH() * model.P_p * model.dH().T + model.R()
            self.S = np.dot(np.dot(H, self.P_p), H.T) + R
            #K = model.P_p * model.dH().T * np.linalg.inv(S)
            self.K = np.dot(np.dot(self.P_p, H.T), np.linalg.inv(S))

            self.x = self.x_p + np.dot(K, (z-h))
            self.P = self.P_p - np.dot(np.dot(K, S), K.T)
            self.P = 0.5 * self.P + 0.5 * self.P.T

            jnorm = dqnorm_by_dq(self.x[3:7])
            
            self.P[0:3, 3:7] = np.dot(self.P[0:3, 3:7], jnorm.T)
            self.P[3:7, 0:3] = np.dot(jnorm, self.P[3:7, 0:3])
            self.P[3:7, 3:7] = np.dot(np.dot(jnorm, self.P[3:7, 0:3]), jnorm.T)
            self.P[3:7, 7:] = np.dot(jnorm, self.P[3:7, 7:])
            self.P[7:, 3:7] = np.dot(self.P[7:, 3:7], jnorm.T)    
            
            qnorm = np.sum(self.x[3:7] ** 2)            
            self.x[3:7] = self.x[3:7] / qnorm
          
        else:
            self.x = self.x_p
            self.P = self.P_p
            self.K[:, :] = 0
        pass

        
