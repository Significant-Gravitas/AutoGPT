import numpy as np

a: np.ndarray
generator = (i for i in range(10))

np.require(a, requirements=1)  # E: No overload variant
np.require(a, requirements="TEST")  # E: incompatible type

np.zeros("test")  # E: incompatible type
np.zeros()  # E: require at least one argument

np.ones("test")  # E: incompatible type
np.ones()  # E: require at least one argument

np.array(0, float, True)  # E: No overload variant

np.linspace(None, 'bob')  # E: No overload variant
np.linspace(0, 2, num=10.0)  # E: No overload variant
np.linspace(0, 2, endpoint='True')  # E: No overload variant
np.linspace(0, 2, retstep=b'False')  # E: No overload variant
np.linspace(0, 2, dtype=0)  # E: No overload variant
np.linspace(0, 2, axis=None)  # E: No overload variant

np.logspace(None, 'bob')  # E: No overload variant
np.logspace(0, 2, base=None)  # E: No overload variant

np.geomspace(None, 'bob')  # E: No overload variant

np.stack(generator)  # E: No overload variant
np.hstack({1, 2})  # E: No overload variant
np.vstack(1)  # E: No overload variant

np.array([1], like=1)  # E: No overload variant
