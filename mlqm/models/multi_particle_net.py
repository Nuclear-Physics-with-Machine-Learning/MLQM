import os, sys, pathlib
from dataclasses import dataclass, field

# Add the mlqm path:
current_dir = pathlib.Path(__file__).parent.resolve()
init = pathlib.Path("__init__.py")

while current_dir != current_dir.root:

    if (current_dir / init).is_file():
        current_dir = current_dir.parent 
    else:
        break

# Here, we break and add the directory to the path:
sys.path.insert(0,str(current_dir))

import tensorflow as tf

from NeuralSpatialComponent import NeuralSpatialComponent
from mlqm import DEFAULT_TENSOR_TYPE
tf.keras.backend.set_floatx(DEFAULT_TENSOR_TYPE)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

config = AttrDict()

config['n_filters_per_layer'] = 2
config['n_layers'] = 2  
config['bias'] = False 
config['residual'] = False
config['activation'] = 'tanh'
config['confinement'] = 0.5

n_walkers = 4
n_particles = 2
n_dim = 3

_input = tf.Variable(tf.ones(shape=(n_walkers, n_particles, n_dim), dtype=DEFAULT_TENSOR_TYPE))


# This network needs to map spatial dimensions to a single output dimension.
# We perform an optimization since we need to do this for n particles.
# This maps 3 spatial dimensions to n output particles, which form part 
# of a column in the slater determinant.
#  

simple_net = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(n_particles),
            tf.keras.layers.Dense(n_particles),
        ]  
    )

# FIrst, capture the input shape:
input_shape = _input.shape

flat_input = tf.reshape(_input, (-1, n_dim)) # leave the spatial dimension


flat_output = simple_net(flat_input)
print(flat_output.shape)

# Reshape to the right shape:
output = tf.reshape(flat_output, (n_walkers, n_particles, n_particles))

print(output)
# spatial_nets = []
# # for i_particle in range(n_particles):
# #     # spatial_nets.append(NeuralSpatialComponent(
# #     #     ndim          = n_dim, 
# #     #     nparticles    = n_particles,
# #     #     configuration = config
# #     # ))
# #     spatial_nets.append(
# #         [
# #             tf.keras.layers.Conv2D(filters=5,
# #                 kernel_size=[1,3], 
# #                 data_format="channels_last"),
# #             tf.keras.layers.Conv2D(filters=1,
# #                 kernel_size=[1,1], 
# #                 data_format="channels_last")
# #             ]
# #         )
# #     )

# for i_particle in range(n_particles):
#     # spatial_nets.append(NeuralSpatialComponent(
#     #     ndim          = n_dim, 
#     #     nparticles    = n_particles,
#     #     configuration = config
#     # ))
#     spatial_nets.append(
#         tf.keras.models.Sequential([
#             tf.keras.layers.Conv1D(filters=5,
#                 kernel_size=[3], 
#                 data_format="channels_last"),
#             # tf.keras.layers.Conv1D(filters=1,
#             #     kernel_size=[1], 
#             #     data_format="channels_last")
#             ]
#         )
#     )


# shaped_input = tf.reshape(_input, (n_walkers, n_particles, 1, n_dim))
# output_stack = [s(shaped_input) for s in spatial_nets]

# print(output_stack[0])

# # print("_input.shape: ", _input.shape)
# # # We loop over the networks, which compute the states for each particle at once.
# # for i_particle in range(n_particles):
# #     # Treat the input list like a time series with [N_batch,N_timestamps,Channels], where the 
# #     # channels is starting as the spatial components (n_dim).
# #     # We use convoltions
# #     # print("this_input.shape: ", this_input.shape)
# #     i_output = spatial_nets[i_particle](this_input)
# #     print("Output: ", i_output)
# #     slater_determinant[:,i_particle,j_particle].assign(i_output)


