import tensorflow as tf

class FlatOptimizer(object):

    def __init__(self, delta):
        '''
        
        Create a basic optimizer with a specified learning rate (time step) delta
        
        Arguments:
            delta {float} -- Imaginary time step for optimization (aka, learning rate)
        '''

        self.delta = tf.convert_to_tensor(delta, dtype=tf.float64)

    # @tf.function
    def apply_gradients(self, grads_and_vars):

        # Update the parameters:
        for grad, var in grads_and_vars:
           var.assign_add(self.delta * grad)

        return


# class EnergyAdaptiveOptimizer(object):
#     pass