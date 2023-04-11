import numpy as np

nd1 = np.array([[1, 2], [3, 4]])

# reshape
nd1.reshape(4)
nd1.reshape(2, 2)
nd1.reshape((2, 2))

nd1.reshape((2, 2), order="C")
nd1.reshape(4, order="C")

# resize
nd1.resize()
nd1.resize(4)
nd1.resize(2, 2)
nd1.resize((2, 2))

nd1.resize((2, 2), refcheck=True)
nd1.resize(4, refcheck=True)

nd2 = np.array([[1, 2], [3, 4]])

# transpose
nd2.transpose()
nd2.transpose(1, 0)
nd2.transpose((1, 0))

# swapaxes
nd2.swapaxes(0, 1)

# flatten
nd2.flatten()
nd2.flatten("C")

# ravel
nd2.ravel()
nd2.ravel("C")

# squeeze
nd2.squeeze()

nd3 = np.array([[1, 2]])
nd3.squeeze(0)

nd4 = np.array([[[1, 2]]])
nd4.squeeze((0, 1))
