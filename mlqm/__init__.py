# Natural Units:
# H_BAR = 1.0
# ELECTRON_CHARGE = 1.0

# Nuclear Units:
# H_BAR = 197.327
ELECTRON_CHARGE = 1.0


DEFAULT_TENSOR_TYPE="float64"
MAX_PARALLEL_ITERATIONS=4000

try:
    import horovod.tensorflow as hvd
    hvd.init()
    MPI_AVAILABLE=True
except:
    MPI_AVAILABLE=False

if MPI_AVAILABLE and hvd.size() == 1:
    # Turn off mpi if only 1 rank
    MPI_AVAILABLE = False
