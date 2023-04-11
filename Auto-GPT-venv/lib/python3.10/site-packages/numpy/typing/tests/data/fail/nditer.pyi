import numpy as np

class Test(np.nditer): ...  # E: Cannot inherit from final class

np.nditer([0, 1], flags=["test"])  # E: incompatible type
np.nditer([0, 1], op_flags=[["test"]])  # E: incompatible type
np.nditer([0, 1], itershape=(1.0,))  # E: incompatible type
np.nditer([0, 1], buffersize=1.0)  # E: incompatible type
