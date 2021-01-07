import tensorflow as tf
import numpy

try:
    import horovod.tensorflow as hvd
    hvd.init()
    MPI_AVAILABLE=True
except:
    MPI_AVAILABLE=False

from mlqm import DEFAULT_TENSOR_TYPE


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
            "energy"     : 0,
            "energy2"    : 0,
            "energy_jf"  : 0,
            "energy2_jf" : 0,
            "acceptance" : 0,
            "weight"     : tf.convert_to_tensor(0., dtype=DEFAULT_TENSOR_TYPE),
            "r"          : 0,
            "dpsi_i"     : 0,
            "dpsi_i_EL"  : 0,
            "dpsi_ij"    : 0,
        }

    # @tf.function
    def allreduce(self):

        for key in self.tensor_dict.keys():
            self.tensor_dict[key] = hvd.allreduce(self.tensor_dict[key], op=hvd.Sum)
        return

    def accumulate(self,energy,energy_jf,acceptance,weight,r,dpsi_i,dpsi_i_EL,dpsi_ij,estim_wgt) :
        self.tensor_dict["energy"]     += energy/estim_wgt
        self.tensor_dict["energy2"]    += (energy/estim_wgt)**2
        self.tensor_dict["energy_jf"]  += energy_jf/estim_wgt
        self.tensor_dict["energy2_jf"] += (energy_jf/estim_wgt)**2
        self.tensor_dict["acceptance"] += acceptance/estim_wgt
        self.tensor_dict["weight"]     += weight/estim_wgt
        self.tensor_dict["r"]          += r/estim_wgt
        self.tensor_dict["dpsi_i"]     += dpsi_i/estim_wgt
        self.tensor_dict["dpsi_i_EL"]  += dpsi_i_EL/estim_wgt
        self.tensor_dict["dpsi_ij"]    += dpsi_ij/estim_wgt

    def finalize(self,nav):
        self.tensor_dict["energy"]     /= nav
        self.tensor_dict["energy2"]    /= nav
        self.tensor_dict["energy_jf"]  /= nav
        self.tensor_dict["energy2_jf"] /= nav
        self.tensor_dict["acceptance"] /= nav
        self.tensor_dict["r"]          /= nav
        self.tensor_dict["dpsi_i"]     /= nav
        self.tensor_dict["dpsi_i_EL"]  /= nav
        self.tensor_dict["dpsi_ij"]    /= nav
        error= tf.sqrt((self.tensor_dict["energy2"] - self.tensor_dict["energy"]**2) / (nav-1))
        error_jf = tf.sqrt((self.tensor_dict["energy2_jf"] - self.tensor_dict["energy_jf"]**2) / (nav-1))
        return error, error_jf
