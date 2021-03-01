import tensorflow as tf
import numpy

from mlqm import DEFAULT_TENSOR_TYPE, MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd


class Estimator(dict):
    """ Accumulate block and totalk averages and errors
    """
    def __init__(self):
        dict.__init__(self)
        # This class holds accumulated measurements of various tensors and their square.
        # It also enables MPI reduction


    def __setitem__(self, key, value):
        # if the value is a tf tensor, set the item as normal
        if tf.is_tensor(value):
            dict.__setitem__(self, key, value)
        else:
            raise KeyError(f"Estimator only accepts tf tensors!  Received {type(value)}")

    # @tf.function
    def accumulate(self, key, value):
        if key in self.keys():
            self.__setitem__(key, value + self[key])
        else:
            self.__setitem__(key, value)

    # @tf.function
    def allreduce(self):

        for key in self.keys():
            self[key] = hvd.allreduce(self[key], op=hvd.Sum, device_dense="GPU")
        return

    # def accumulate(self, weight=1, ** kwargs):
    #     # energy, energy_jf, ke_jf, ke_direct, pe, acceptance,weight,r,dpsi_i,dpsi_i_EL,dpsi_ij,estim_wgt) :
    #     for key in kwargs:
    #         self.tensor_dict[key]      += kwargs[key] * weight
    #         if key == "energy" or key == "energy_jf":
    #             self.tensor_dict[key+"2"]  += (kwargs[key]* weight)**2

    #     self.tensor_dict['weight'] += weight


    def finalize(self, weight):

        for key in self.keys():
            if key == 'weight': continue
            self[key] /= weight

        return

            # error= tf.sqrt((self.tensor_dict["energy2"] - self.tensor_dict["energy"]**2) / (nav-1))
            # error_jf = tf.sqrt((self.tensor_dict["energy_jf2"] - self.tensor_dict["energy_jf"]**2) / (nav-1))
            # return error, error_jf
