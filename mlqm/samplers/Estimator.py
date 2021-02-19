import tensorflow as tf
import numpy

from mlqm import DEFAULT_TENSOR_TYPE, MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd


class Estimator(object):
    """ Accumulate block and totalk averages and errors
    """
    def __init__(self,*,info=None):
        if info is not None:
            print(f"Set the following estimators: E, E2,E_jf,E2_jf,acc,weight,Psi_i,H*Psi_i,Psi_ij ")

        # This class holds accumulated measurements of various tensors and their square.
        # It also enables MPI reduction

        self.reset()


    def reset(self):
        self.tensor_dict = {
            "energy"     : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "energy2"    : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "energy_jf"  : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "energy_jf2" : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "ke_jf"      : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "ke_direct"  : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "pe"         : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "acceptance" : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "weight"     : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "r"          : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "dpsi_i"     : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "dpsi_i_EL"  : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "dpsi_ij"    : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
        }

    # @tf.function
    def allreduce(self):

        for key in self.tensor_dict.keys():
            self.tensor_dict[key] = hvd.allreduce(self.tensor_dict[key], op=hvd.Sum, device_dense="GPU")
        return

    def accumulate(self, weight=1, ** kwargs):
        # energy, energy_jf, ke_jf, ke_direct, pe, acceptance,weight,r,dpsi_i,dpsi_i_EL,dpsi_ij,estim_wgt) :
        for key in kwargs:
            self.tensor_dict[key]      += kwargs[key] * weight
            if key == "energy" or key == "energy_jf":
                self.tensor_dict[key+"2"]  += (kwargs[key]* weight)**2

        self.tensor_dict['weight'] += weight


    def finalize(self,nav):

        for key in self.tensor_dict.keys():
            if key == 'weight': continue
            self.tensor_dict[key] /= self.tensor_dict['weight']


        error= tf.sqrt((self.tensor_dict["energy2"] - self.tensor_dict["energy"]**2) / (nav-1))
        error_jf = tf.sqrt((self.tensor_dict["energy_jf2"] - self.tensor_dict["energy_jf"]**2) / (nav-1))
        return error, error_jf
