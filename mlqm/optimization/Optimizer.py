import tensorflow as tf
import numpy

import logging
# Set up logging:
logger = logging.getLogger()

class Optimizer(object):

    def __init__(self,delta : float ,eps : float, npt : int):
        '''Create an optimizer to compute the gradients for the SR update.

        Arguments:
            delta {float} --
            eps {float} --
            npt {int} -- Number of parameters in the wave function
        '''

        self.eps    = tf.convert_to_tensor(eps, dtype=tf.float64)
        self.delta  = tf.convert_to_tensor(delta, dtype=tf.float64)
        self.npt    = npt

    def par_dist(self, dp_i, S_ij):
        # dist = 0
        # for i in range (self.npt):
        #     for j in range (self.npt):
        #         dist += S_ij[i,j]*dp_i[i]*dp_i[j]

        # This replaces the double for loop.  Don't do that in python.
        dp_i = tf.cast(dp_i, tf.float32)
        S_ij = tf.cast(S_ij, tf.float32)
        D_ij = S_ij * (dp_i * tf.transpose(dp_i))
        dist = tf.reduce_sum(D_ij)
        return dist

    def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):
        f_i= tf.cast(tf.cast(self.delta, tf.float32) * ( dpsi_i * energy - dpsi_i_EL ), tf.float64)
        # This also replaces the double for loop ... don't do that :)
        S_ij = dpsi_ij - dpsi_i * tf.transpose(dpsi_i)

#        print("dpsi_i", dpsi_i)
#        print("dpsi_ij", dpsi_ij)
#        print("S_ij=", S_ij)
#        print("dpsi_i_EL", dpsi_i_EL)
#        print("energy", energy)
        i = 0
        while True:
            S_ij_d = tf.cast(tf.identity(S_ij), tf.float64)
            S_ij_d += 2**i * self.eps * tf.eye(self.npt, dtype=tf.float64)
            i += 1
            det_test = tf.linalg.det(S_ij_d)
            # torch.set_printoptions(precision=8)
            try:
################################################################################
                # Note: TF doesn't support upper cholesky and cholesky_solve... 
                U_ij = tf.linalg.cholesky(S_ij_d)
################################################################################
                positive_definite = True
            except RuntimeError:
                positive_definite = False
                logger.error("Warning, Cholesky did not find a positive definite matrix")
            if (positive_definite):
                # Arguments to cholesky_solve are in a different order in TF
################################################################################
                dp_i = tf.linalg.cholesky_solve(U_ij, f_i)
################################################################################
                dp_0 = self.delta * tf.cast(energy, tf.float64)
                dp_0 = 1. - dp_0 - tf.reduce_sum(tf.cast(dpsi_i, tf.float64)*dp_i)
                # dp_0 = 1. - self.delta * energy - tf.reduce_sum(dpsi_i*dp_i)
                dp_i = dp_i / dp_0
                dist = self.par_dist(tf.cast(dp_i, tf.float32), S_ij)
                dist_reg = self.par_dist(dp_i, S_ij_d)
                dist_norm = self.par_dist(dp_i, dpsi_i * tf.transpose(dpsi_i))
                # Originally this accessed the [0] element but that's not necessary now
                # logger.debug("dist param", dist.numpy())
                # logger.debug("dist reg", dist_reg.numpy())
                # logger.debug("dist param norm", dist_norm.numpy())
                dp_i = tf.cast(dp_i, tf.float32)
                if (dist < 0.001 and dist_reg < 0.001 and dist_norm < 0.2):
                    break
        return dp_i
