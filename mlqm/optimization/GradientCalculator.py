import tensorflow as tf

class GradientCalculator(object):

    def __init__(self, dtype=tf.float64):
        self.dtype  = dtype

    @tf.function
    def f_i(self, dpsi_i, energy, dpsi_i_EL):
        return dpsi_i * energy - dpsi_i_EL

    @tf.function
    def S_ij(self, dpsi_ij, dpsi_i):
        return dpsi_ij - dpsi_i * tf.transpose(dpsi_i)


    # @tf.function
    # def par_dist(self, dp_i, S_ij):
    #     D_ij = S_ij * (dp_i * tf.transpose(dp_i))
    #     dist = tf.reduce_sum(D_ij)
    #     return dist

    @tf.function
    def par_dist(self, dp_i, S_ij):
        dist = tf.reduce_sum(dp_i*tf.linalg.matmul(S_ij, dp_i))
        return dist


    @tf.function
    def regularize_S_ij(self, S_ij, eps):
        dtype = S_ij.dtype
        npt   = S_ij.shape[0]
        S_ij_d = S_ij + eps * tf.eye(npt, dtype=dtype)
        return S_ij_d


    
    def cast(self, *args):

        return (tf.cast(a, self.dtype) if a.dtype != self.dtype else a for a in args)


    # @tf.function
    def pd_solve(self, S_ij, eps, f_i):


        # Regularize along the diagonal:
        S_ij_d = self.regularize_S_ij(S_ij, eps)

        # Next, we need S_ij to be positive definite.
        U_ij = tf.linalg.cholesky(S_ij_d)

        dp_i = tf.linalg.cholesky_solve(U_ij, f_i)

        return dp_i
