import numpy as np
from numpy.testing import (
        assert_raises, assert_raises_regex,
        )


class TestIndexErrors:
    '''Tests to exercise indexerrors not covered by other tests.'''

    def test_arraytypes_fasttake(self):
        'take from a 0-length dimension'
        x = np.empty((2, 3, 0, 4))
        assert_raises(IndexError, x.take, [0], axis=2)
        assert_raises(IndexError, x.take, [1], axis=2)
        assert_raises(IndexError, x.take, [0], axis=2, mode='wrap')
        assert_raises(IndexError, x.take, [0], axis=2, mode='clip')

    def test_take_from_object(self):
        # Check exception taking from object array
        d = np.zeros(5, dtype=object)
        assert_raises(IndexError, d.take, [6])

        # Check exception taking from 0-d array
        d = np.zeros((5, 0), dtype=object)
        assert_raises(IndexError, d.take, [1], axis=1)
        assert_raises(IndexError, d.take, [0], axis=1)
        assert_raises(IndexError, d.take, [0])
        assert_raises(IndexError, d.take, [0], mode='wrap')
        assert_raises(IndexError, d.take, [0], mode='clip')

    def test_multiindex_exceptions(self):
        a = np.empty(5, dtype=object)
        assert_raises(IndexError, a.item, 20)
        a = np.empty((5, 0), dtype=object)
        assert_raises(IndexError, a.item, (0, 0))

        a = np.empty(5, dtype=object)
        assert_raises(IndexError, a.itemset, 20, 0)
        a = np.empty((5, 0), dtype=object)
        assert_raises(IndexError, a.itemset, (0, 0), 0)

    def test_put_exceptions(self):
        a = np.zeros((5, 5))
        assert_raises(IndexError, a.put, 100, 0)
        a = np.zeros((5, 5), dtype=object)
        assert_raises(IndexError, a.put, 100, 0)
        a = np.zeros((5, 5, 0))
        assert_raises(IndexError, a.put, 100, 0)
        a = np.zeros((5, 5, 0), dtype=object)
        assert_raises(IndexError, a.put, 100, 0)

    def test_iterators_exceptions(self):
        "cases in iterators.c"
        def assign(obj, ind, val):
            obj[ind] = val

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a[0, 5, None, 2])
        assert_raises(IndexError, lambda: a[0, 5, 0, 2])
        assert_raises(IndexError, lambda: assign(a, (0, 5, None, 2), 1))
        assert_raises(IndexError, lambda: assign(a, (0, 5, 0, 2),  1))

        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a[0, 0, None, 2])
        assert_raises(IndexError, lambda: assign(a, (0, 0, None, 2), 1))

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a.flat[10])
        assert_raises(IndexError, lambda: assign(a.flat, 10, 5))
        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a.flat[10])
        assert_raises(IndexError, lambda: assign(a.flat, 10, 5))

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a.flat[np.array(10)])
        assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))
        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a.flat[np.array(10)])
        assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a.flat[np.array([10])])
        assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))
        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a.flat[np.array([10])])
        assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))

    def test_mapping(self):
        "cases from mapping.c"

        def assign(obj, ind, val):
            obj[ind] = val

        a = np.zeros((0, 10))
        assert_raises(IndexError, lambda: a[12])

        a = np.zeros((3, 5))
        assert_raises(IndexError, lambda: a[(10, 20)])
        assert_raises(IndexError, lambda: assign(a, (10, 20), 1))
        a = np.zeros((3, 0))
        assert_raises(IndexError, lambda: a[(1, 0)])
        assert_raises(IndexError, lambda: assign(a, (1, 0), 1))

        a = np.zeros((10,))
        assert_raises(IndexError, lambda: assign(a, 10, 1))
        a = np.zeros((0,))
        assert_raises(IndexError, lambda: assign(a, 10, 1))

        a = np.zeros((3, 5))
        assert_raises(IndexError, lambda: a[(1, [1, 20])])
        assert_raises(IndexError, lambda: assign(a, (1, [1, 20]), 1))
        a = np.zeros((3, 0))
        assert_raises(IndexError, lambda: a[(1, [0, 1])])
        assert_raises(IndexError, lambda: assign(a, (1, [0, 1]), 1))

    def test_mapping_error_message(self):
        a = np.zeros((3, 5))
        index = (1, 2, 3, 4, 5)
        assert_raises_regex(
                IndexError,
                "too many indices for array: "
                "array is 2-dimensional, but 5 were indexed",
                lambda: a[index])

    def test_methods(self):
        "cases from methods.c"

        a = np.zeros((3, 3))
        assert_raises(IndexError, lambda: a.item(100))
        assert_raises(IndexError, lambda: a.itemset(100, 1))
        a = np.zeros((0, 3))
        assert_raises(IndexError, lambda: a.item(100))
        assert_raises(IndexError, lambda: a.itemset(100, 1))
