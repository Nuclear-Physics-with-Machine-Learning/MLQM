import torch


n_points  = 10
dimension = 3
    

# This pulls points from a uniform distribution, I think:
x_original = torch.rand(size=[n_points,dimension])

# This draws points from a gaussian.  Each individual point is a
# sample from the gaussian, which may not be right for the metropolis algorithm:
kick       = torch.randn(size=x_original.shape)

x_updated  = x_original + kick

# This is just comparing the magnitudue of the coordinates

# Compute the x**2 + y**2 + z**2 scalars:
x_original_magnitude = torch.sum(x_original**2, axis=-1)
x_updated_magnitude  = torch.sum(x_updated**2, axis=-1)

# These two should be scalar vectors now:
assert len(x_original_magnitude.shape) == 1
assert x_original_magnitude.shape[0] == n_points

# Select the smaller points, just for fun:
condition = x_original_magnitude < x_updated_magnitude

# condition is now a boolean vector, it's true when x_original is less than x_updated

# See the "where" function here, which can select from two arrays:
# https://pytorch.org/docs/stable/torch.html


# In order to make the fucntion work, the condition needs to be 
# "broadcastable" to the original points.  
# This means it needs to be of shape [n_points, 1] rather than [n_points]
smallest_points = torch.where(condition.view([n_points,1]), x_original, x_updated)

#  Verify we have the smallest points in terms of norm:
smallest_magnitude = torch.sum(smallest_points**2, axis=-1)

assert (smallest_magnitude <= x_original_magnitude).all()
assert (smallest_magnitude <= x_updated_magnitude).all()

print("done")