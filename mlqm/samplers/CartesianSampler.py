import tensorflow as tf
import numpy


class CartesianSampler(object):
    """Cartesian Sampler in N dimension
    
    Sample from 3D coordinates, with some granularity delta
    """
    def __init__(self, 
        n           : int, 
        nparticles  : int,
        delta       : float, 
        mins        : float, 
        maxes       : float):

        if n < 1: 
            raise Exception("Dimension must be at least 1 for ExponentialBoundaryCondition")

        # Use numpy to broadcast to the right dimension:
        delta = numpy.asarray(delta, dtype=numpy.float32)
        delta = numpy.broadcast_to(delta, (n,))

        self.n = n
        self.nparticles = nparticles
        self.delta = delta

        # Use numpy to broadcast to the right dimension:
        mins = numpy.asarray(mins, dtype=numpy.float32)
        mins = numpy.broadcast_to(mins, (n,))

        # Use numpy to broadcast to the right dimension:
        maxes = numpy.asarray(maxes, dtype=numpy.float32)
        maxes = numpy.broadcast_to(maxes, (n,))

        self.mins  = mins
        self.maxes = maxes


    def voxel_size(self):
        return numpy.prod(self.delta)

    def sample(self, device : str=None):
        """Sample points in N-d Space
        
        By default, samples points uniformly across all dimensions.
        Returns a torch tensor on the chosen device with gradients enabled.
        
        Keyword Arguments:
            device {str} -- What device to move data onto in torch (default: {None})
        """


        # First, we generate the coordinates across all axes:
        coords = [ numpy.arange(
                self.mins[d], self.maxes[d], self.delta[d], dtype=numpy.float32) 
            for d in range(self.n) ]


        # Second, we zip this all together into 
        mesh = numpy.meshgrid(*coords)

        # We want to arrange this into a flattened vector of coordinates
        # with the shape [n_coords, n_dims]

        # Flatten:
        mesh = [ m.flatten() for m in mesh]

        n_points = len(mesh[0])

        # Stack:
        mesh = numpy.stack(mesh, axis=-1)

        # Re-stack the mesh for the number of particles:
        mesh = numpy.tile(mesh, reps=self.nparticles)

        mesh = mesh.reshape((n_points, self.nparticles, self.n))

        mesh = tf.convert_to_tensor(mesh)


        return tf.Variable(mesh, trainable=True)
