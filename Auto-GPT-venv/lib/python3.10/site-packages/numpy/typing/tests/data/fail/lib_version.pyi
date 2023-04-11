from numpy.lib import NumpyVersion

version: NumpyVersion

NumpyVersion(b"1.8.0")  # E: incompatible type
version >= b"1.8.0"  # E: Unsupported operand types
