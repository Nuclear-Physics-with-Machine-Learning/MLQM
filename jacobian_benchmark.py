import tensorflow as tf
import time, timeit

N_WALKERS = 1000
DIM = 3
N_PARTICLES = 4

x_input = tf.random.uniform(shape=(N_WALKERS, N_PARTICLES, DIM))

class DeepSetsWavefunction(tf.keras.models.Model):
    """Create a neural network eave function in N dimensions

    Boundary condition, if not supplied, is gaussian in every dimension

    Extends:
        tf.keras.models.Model
    """
    def __init__(self, ndim : int, nparticles: int, mean_subtract : bool, boundary_condition :tf.keras.layers.Layer = None):
        '''Deep Sets wavefunction for symmetric particle wavefunctions

        Implements a deep set network for multiple particles in the same system

        Arguments:
            ndim {int} -- Number of dimensions
            nparticles {int} -- Number of particls

        Keyword Arguments:
            boundary_condition {tf.keras.layers.Layer} -- [description] (default: {None})

        Raises:
            Exception -- [description]
        '''
        tf.keras.models.Model.__init__(self)

        self.ndim = ndim
        if self.ndim < 1 or self.ndim > 3:
           raise Exception("Dimension must be 1, 2, or 3 for DeepSetsWavefunction")

        self.nparticles = nparticles

        self.mean_subtract = mean_subtract


        n_filters_per_layer = 32
        n_layers            = 4
        bias                = True
        activation          = tf.keras.activations.tanh


        self.individual_net = tf.keras.models.Sequential()

        self.individual_net.add(
            tf.keras.layers.Dense(n_filters_per_layer,
                use_bias = bias)
            )

        for l in range(n_layers):
            self.individual_net.add(
                tf.keras.layers.Dense(n_filters_per_layer,
                    use_bias    = bias,
                    activation = activation)
                )


        self.aggregate_net = tf.keras.models.Sequential()

        for l in range(n_layers):
            self.individual_net.add(
                tf.keras.layers.Dense(n_filters_per_layer,
                    use_bias    = bias,
                    activation = activation)
                )
        self.aggregate_net.add(tf.keras.layers.Dense(1,
            use_bias = False))


    @tf.function(experimental_compile=False)
    def call(self, inputs, trainable=None):
        # Mean subtract for all particles:
        if self.nparticles > 1 and self.mean_subtract:
            mean = tf.reduce_mean(inputs, axis=1)
            xinputs = inputs - mean[:,None,:]
        else:
            xinputs = inputs

        x = []
        for p in range(self.nparticles):
            x.append(self.individual_net(xinputs[:,p,:]))

        x = tf.add_n(x)
        x = self.aggregate_net(x)

        # Compute the initial boundary condition, which the network will slowly overcome
        # boundary_condition = tf.math.abs(self.normalization_weight * tf.reduce_sum(xinputs**self.normalization_exponent, axis=(1,2))
        boundary_condition = -1. * tf.reduce_sum(xinputs**2, axis=(1,2))
        boundary_condition = tf.reshape(boundary_condition, [-1,1])


        return x + boundary_condition

    def n_parameters(self):
        return tf.reduce_sum( [ tf.reduce_prod(p.shape) for p in self.trainable_variables ])

wavefunction = DeepSetsWavefunction(ndim=DIM, nparticles=N_PARTICLES, mean_subtract=True)
output = wavefunction(x_input)


@tf.function
def jacobian_comp(inputs, _wavefunction):

    with tf.GradientTape() as tape:
        log_psiw = _wavefunction(inputs)

    # By default, this essentially SUMS over the dimension of log_psiw
    jac = tape.jacobian(log_psiw, _wavefunction.trainable_variables)

    return jac


start = time.time()
jc = jacobian_comp(x_input, wavefunction)
print("Jacobian Compilation time: ", time.time() - start)


start = time.time()
jacobian_comp(x_input, wavefunction)
print("Jacobian Execution time: ", time.time() - start)


@tf.function
def jacobian_grad(inputs, _wavefunction):
    
    n_walkers = inputs.shape[0]
    
    with tf.GradientTape(persistent=True) as tape:
        log_psiw = _wavefunction(inputs)

        split = tf.split(log_psiw, n_walkers)

    # print(split)
    # By default, this essentially SUMS over the dimension of log_psiw
    grad = [tape.gradient(s, _wavefunction.trainable_variables) for s in split]

    jac = []
    for i, l in enumerate(_wavefunction.trainable_variables):
        temp = tf.stack([g[i] for g in grad])
        temp = tf.reshape(temp,  log_psiw.shape + l.shape)
        jac.append(temp)
    
    return jac
                             

start = time.time()
jg = jacobian_grad(x_input, wavefunction)
print("Stacked Gradient Compilation time: ", time.time() - start)

start = time.time()
jacobian_comp(x_input, wavefunction)
print("Stacked Gradient Execution time: ", time.time() - start)

