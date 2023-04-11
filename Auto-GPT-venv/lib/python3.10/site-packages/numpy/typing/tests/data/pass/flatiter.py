import numpy as np

a = np.empty((2, 2)).flat

a.base
a.copy()
a.coords
a.index
iter(a)
next(a)
a[0]
a[[0, 1, 2]]
a[...]
a[:]
a.__array__()
a.__array__(np.dtype(np.float64))
