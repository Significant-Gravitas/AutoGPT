import numpy as np

nd = np.array([[1, 2], [3, 4]])

# reshape
reveal_type(nd.reshape())  # E: ndarray
reveal_type(nd.reshape(4))  # E: ndarray
reveal_type(nd.reshape(2, 2))  # E: ndarray
reveal_type(nd.reshape((2, 2)))  # E: ndarray

reveal_type(nd.reshape((2, 2), order="C"))  # E: ndarray
reveal_type(nd.reshape(4, order="C"))  # E: ndarray

# resize does not return a value

# transpose
reveal_type(nd.transpose())  # E: ndarray
reveal_type(nd.transpose(1, 0))  # E: ndarray
reveal_type(nd.transpose((1, 0)))  # E: ndarray

# swapaxes
reveal_type(nd.swapaxes(0, 1))  # E: ndarray

# flatten
reveal_type(nd.flatten())  # E: ndarray
reveal_type(nd.flatten("C"))  # E: ndarray

# ravel
reveal_type(nd.ravel())  # E: ndarray
reveal_type(nd.ravel("C"))  # E: ndarray

# squeeze
reveal_type(nd.squeeze())  # E: ndarray
reveal_type(nd.squeeze(0))  # E: ndarray
reveal_type(nd.squeeze((0, 2)))  # E: ndarray
