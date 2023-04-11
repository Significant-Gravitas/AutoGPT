from operator import mul
from functools import reduce

import numpy as np
from numpy.random import randint
from numpy.lib import Arrayterator
from numpy.testing import assert_


def test():
    np.random.seed(np.arange(10))

    # Create a random array
    ndims = randint(5)+1
    shape = tuple(randint(10)+1 for dim in range(ndims))
    els = reduce(mul, shape)
    a = np.arange(els)
    a.shape = shape

    buf_size = randint(2*els)
    b = Arrayterator(a, buf_size)

    # Check that each block has at most ``buf_size`` elements
    for block in b:
        assert_(len(block.flat) <= (buf_size or els))

    # Check that all elements are iterated correctly
    assert_(list(b.flat) == list(a.flat))

    # Slice arrayterator
    start = [randint(dim) for dim in shape]
    stop = [randint(dim)+1 for dim in shape]
    step = [randint(dim)+1 for dim in shape]
    slice_ = tuple(slice(*t) for t in zip(start, stop, step))
    c = b[slice_]
    d = a[slice_]

    # Check that each block has at most ``buf_size`` elements
    for block in c:
        assert_(len(block.flat) <= (buf_size or els))

    # Check that the arrayterator is sliced correctly
    assert_(np.all(c.__array__() == d))

    # Check that all elements are iterated correctly
    assert_(list(c.flat) == list(d.flat))
