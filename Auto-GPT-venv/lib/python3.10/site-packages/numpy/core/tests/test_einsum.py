import itertools

import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_raises, suppress_warnings, assert_raises_regex, assert_allclose
    )

# Setup for optimize einsum
chars = 'abcdefghij'
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
global_size_dict = dict(zip(chars, sizes))


class TestEinsum:
    def test_einsum_errors(self):
        for do_opt in [True, False]:
            # Need enough arguments
            assert_raises(ValueError, np.einsum, optimize=do_opt)
            assert_raises(ValueError, np.einsum, "", optimize=do_opt)

            # subscripts must be a string
            assert_raises(TypeError, np.einsum, 0, 0, optimize=do_opt)

            # out parameter must be an array
            assert_raises(TypeError, np.einsum, "", 0, out='test',
                          optimize=do_opt)

            # order parameter must be a valid order
            assert_raises(ValueError, np.einsum, "", 0, order='W',
                          optimize=do_opt)

            # casting parameter must be a valid casting
            assert_raises(ValueError, np.einsum, "", 0, casting='blah',
                          optimize=do_opt)

            # dtype parameter must be a valid dtype
            assert_raises(TypeError, np.einsum, "", 0, dtype='bad_data_type',
                          optimize=do_opt)

            # other keyword arguments are rejected
            assert_raises(TypeError, np.einsum, "", 0, bad_arg=0,
                          optimize=do_opt)

            # issue 4528 revealed a segfault with this call
            assert_raises(TypeError, np.einsum, *(None,)*63, optimize=do_opt)

            # number of operands must match count in subscripts string
            assert_raises(ValueError, np.einsum, "", 0, 0, optimize=do_opt)
            assert_raises(ValueError, np.einsum, ",", 0, [0], [0],
                          optimize=do_opt)
            assert_raises(ValueError, np.einsum, ",", [0], optimize=do_opt)

            # can't have more subscripts than dimensions in the operand
            assert_raises(ValueError, np.einsum, "i", 0, optimize=do_opt)
            assert_raises(ValueError, np.einsum, "ij", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, "...i", 0, optimize=do_opt)
            assert_raises(ValueError, np.einsum, "i...j", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, "i...", 0, optimize=do_opt)
            assert_raises(ValueError, np.einsum, "ij...", [0, 0], optimize=do_opt)

            # invalid ellipsis
            assert_raises(ValueError, np.einsum, "i..", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, ".i...", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, "j->..j", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, "j->.j...", [0, 0], optimize=do_opt)

            # invalid subscript character
            assert_raises(ValueError, np.einsum, "i%...", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, "...j$", [0, 0], optimize=do_opt)
            assert_raises(ValueError, np.einsum, "i->&", [0, 0], optimize=do_opt)

            # output subscripts must appear in input
            assert_raises(ValueError, np.einsum, "i->ij", [0, 0], optimize=do_opt)

            # output subscripts may only be specified once
            assert_raises(ValueError, np.einsum, "ij->jij", [[0, 0], [0, 0]],
                          optimize=do_opt)

            # dimensions much match when being collapsed
            assert_raises(ValueError, np.einsum, "ii",
                          np.arange(6).reshape(2, 3), optimize=do_opt)
            assert_raises(ValueError, np.einsum, "ii->i",
                          np.arange(6).reshape(2, 3), optimize=do_opt)

            # broadcasting to new dimensions must be enabled explicitly
            assert_raises(ValueError, np.einsum, "i", np.arange(6).reshape(2, 3),
                          optimize=do_opt)
            assert_raises(ValueError, np.einsum, "i->i", [[0, 1], [0, 1]],
                          out=np.arange(4).reshape(2, 2), optimize=do_opt)
            with assert_raises_regex(ValueError, "'b'"):
                # gh-11221 - 'c' erroneously appeared in the error message
                a = np.ones((3, 3, 4, 5, 6))
                b = np.ones((3, 4, 5))
                np.einsum('aabcb,abc', a, b)

            # Check order kwarg, asanyarray allows 1d to pass through
            assert_raises(ValueError, np.einsum, "i->i", np.arange(6).reshape(-1, 1),
                          optimize=do_opt, order='d')

    def test_einsum_views(self):
        # pass-through
        for do_opt in [True, False]:
            a = np.arange(6)
            a.shape = (2, 3)

            b = np.einsum("...", a, optimize=do_opt)
            assert_(b.base is a)

            b = np.einsum(a, [Ellipsis], optimize=do_opt)
            assert_(b.base is a)

            b = np.einsum("ij", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a)

            b = np.einsum(a, [0, 1], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a)

            # output is writeable whenever input is writeable
            b = np.einsum("...", a, optimize=do_opt)
            assert_(b.flags['WRITEABLE'])
            a.flags['WRITEABLE'] = False
            b = np.einsum("...", a, optimize=do_opt)
            assert_(not b.flags['WRITEABLE'])

            # transpose
            a = np.arange(6)
            a.shape = (2, 3)

            b = np.einsum("ji", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.T)

            b = np.einsum(a, [1, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.T)

            # diagonal
            a = np.arange(9)
            a.shape = (3, 3)

            b = np.einsum("ii->i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i] for i in range(3)])

            b = np.einsum(a, [0, 0], [0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i] for i in range(3)])

            # diagonal with various ways of broadcasting an additional dimension
            a = np.arange(27)
            a.shape = (3, 3, 3)

            b = np.einsum("...ii->...i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)] for x in a])

            b = np.einsum(a, [Ellipsis, 0, 0], [Ellipsis, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)] for x in a])

            b = np.einsum("ii...->...i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(2, 0, 1)])

            b = np.einsum(a, [0, 0, Ellipsis], [Ellipsis, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(2, 0, 1)])

            b = np.einsum("...ii->i...", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum(a, [Ellipsis, 0, 0], [0, Ellipsis], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum("jii->ij", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum(a, [1, 0, 0], [0, 1], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum("ii...->i...", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(2, 0, 1)[:, i, i] for i in range(3)])

            b = np.einsum(a, [0, 0, Ellipsis], [0, Ellipsis], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(2, 0, 1)[:, i, i] for i in range(3)])

            b = np.einsum("i...i->i...", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(1, 0, 2)[:, i, i] for i in range(3)])

            b = np.einsum(a, [0, Ellipsis, 0], [0, Ellipsis], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(1, 0, 2)[:, i, i] for i in range(3)])

            b = np.einsum("i...i->...i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(1, 0, 2)])

            b = np.einsum(a, [0, Ellipsis, 0], [Ellipsis, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(1, 0, 2)])

            # triple diagonal
            a = np.arange(27)
            a.shape = (3, 3, 3)

            b = np.einsum("iii->i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i, i] for i in range(3)])

            b = np.einsum(a, [0, 0, 0], [0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i, i] for i in range(3)])

            # swap axes
            a = np.arange(24)
            a.shape = (2, 3, 4)

            b = np.einsum("ijk->jik", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.swapaxes(0, 1))

            b = np.einsum(a, [0, 1, 2], [1, 0, 2], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.swapaxes(0, 1))

    @np._no_nep50_warning()
    def check_einsum_sums(self, dtype, do_opt=False):
        dtype = np.dtype(dtype)
        # Check various sums.  Does many sizes to exercise unrolled loops.

        # sum(a, axis=-1)
        for n in range(1, 17):
            a = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("i->", a, optimize=do_opt),
                         np.sum(a, axis=-1).astype(dtype))
            assert_equal(np.einsum(a, [0], [], optimize=do_opt),
                         np.sum(a, axis=-1).astype(dtype))

        for n in range(1, 17):
            a = np.arange(2*3*n, dtype=dtype).reshape(2, 3, n)
            assert_equal(np.einsum("...i->...", a, optimize=do_opt),
                         np.sum(a, axis=-1).astype(dtype))
            assert_equal(np.einsum(a, [Ellipsis, 0], [Ellipsis], optimize=do_opt),
                         np.sum(a, axis=-1).astype(dtype))

        # sum(a, axis=0)
        for n in range(1, 17):
            a = np.arange(2*n, dtype=dtype).reshape(2, n)
            assert_equal(np.einsum("i...->...", a, optimize=do_opt),
                         np.sum(a, axis=0).astype(dtype))
            assert_equal(np.einsum(a, [0, Ellipsis], [Ellipsis], optimize=do_opt),
                         np.sum(a, axis=0).astype(dtype))

        for n in range(1, 17):
            a = np.arange(2*3*n, dtype=dtype).reshape(2, 3, n)
            assert_equal(np.einsum("i...->...", a, optimize=do_opt),
                         np.sum(a, axis=0).astype(dtype))
            assert_equal(np.einsum(a, [0, Ellipsis], [Ellipsis], optimize=do_opt),
                         np.sum(a, axis=0).astype(dtype))

        # trace(a)
        for n in range(1, 17):
            a = np.arange(n*n, dtype=dtype).reshape(n, n)
            assert_equal(np.einsum("ii", a, optimize=do_opt),
                         np.trace(a).astype(dtype))
            assert_equal(np.einsum(a, [0, 0], optimize=do_opt),
                         np.trace(a).astype(dtype))

            # gh-15961: should accept numpy int64 type in subscript list
            np_array = np.asarray([0, 0])
            assert_equal(np.einsum(a, np_array, optimize=do_opt),
                         np.trace(a).astype(dtype))
            assert_equal(np.einsum(a, list(np_array), optimize=do_opt),
                         np.trace(a).astype(dtype))

        # multiply(a, b)
        assert_equal(np.einsum("..., ...", 3, 4), 12)  # scalar case
        for n in range(1, 17):
            a = np.arange(3 * n, dtype=dtype).reshape(3, n)
            b = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            assert_equal(np.einsum("..., ...", a, b, optimize=do_opt),
                         np.multiply(a, b))
            assert_equal(np.einsum(a, [Ellipsis], b, [Ellipsis], optimize=do_opt),
                         np.multiply(a, b))

        # inner(a,b)
        for n in range(1, 17):
            a = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            b = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("...i, ...i", a, b, optimize=do_opt), np.inner(a, b))
            assert_equal(np.einsum(a, [Ellipsis, 0], b, [Ellipsis, 0], optimize=do_opt),
                         np.inner(a, b))

        for n in range(1, 11):
            a = np.arange(n * 3 * 2, dtype=dtype).reshape(n, 3, 2)
            b = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("i..., i...", a, b, optimize=do_opt),
                         np.inner(a.T, b.T).T)
            assert_equal(np.einsum(a, [0, Ellipsis], b, [0, Ellipsis], optimize=do_opt),
                         np.inner(a.T, b.T).T)

        # outer(a,b)
        for n in range(1, 17):
            a = np.arange(3, dtype=dtype)+1
            b = np.arange(n, dtype=dtype)+1
            assert_equal(np.einsum("i,j", a, b, optimize=do_opt),
                         np.outer(a, b))
            assert_equal(np.einsum(a, [0], b, [1], optimize=do_opt),
                         np.outer(a, b))

        # Suppress the complex warnings for the 'as f8' tests
        with suppress_warnings() as sup:
            sup.filter(np.ComplexWarning)

            # matvec(a,b) / a.dot(b) where a is matrix, b is vector
            for n in range(1, 17):
                a = np.arange(4*n, dtype=dtype).reshape(4, n)
                b = np.arange(n, dtype=dtype)
                assert_equal(np.einsum("ij, j", a, b, optimize=do_opt),
                             np.dot(a, b))
                assert_equal(np.einsum(a, [0, 1], b, [1], optimize=do_opt),
                             np.dot(a, b))

                c = np.arange(4, dtype=dtype)
                np.einsum("ij,j", a, b, out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))
                c[...] = 0
                np.einsum(a, [0, 1], b, [1], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))

            for n in range(1, 17):
                a = np.arange(4*n, dtype=dtype).reshape(4, n)
                b = np.arange(n, dtype=dtype)
                assert_equal(np.einsum("ji,j", a.T, b.T, optimize=do_opt),
                             np.dot(b.T, a.T))
                assert_equal(np.einsum(a.T, [1, 0], b.T, [1], optimize=do_opt),
                             np.dot(b.T, a.T))

                c = np.arange(4, dtype=dtype)
                np.einsum("ji,j", a.T, b.T, out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(b.T.astype('f8'),
                                    a.T.astype('f8')).astype(dtype))
                c[...] = 0
                np.einsum(a.T, [1, 0], b.T, [1], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(b.T.astype('f8'),
                                    a.T.astype('f8')).astype(dtype))

            # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
            for n in range(1, 17):
                if n < 8 or dtype != 'f2':
                    a = np.arange(4*n, dtype=dtype).reshape(4, n)
                    b = np.arange(n*6, dtype=dtype).reshape(n, 6)
                    assert_equal(np.einsum("ij,jk", a, b, optimize=do_opt),
                                 np.dot(a, b))
                    assert_equal(np.einsum(a, [0, 1], b, [1, 2], optimize=do_opt),
                                 np.dot(a, b))

            for n in range(1, 17):
                a = np.arange(4*n, dtype=dtype).reshape(4, n)
                b = np.arange(n*6, dtype=dtype).reshape(n, 6)
                c = np.arange(24, dtype=dtype).reshape(4, 6)
                np.einsum("ij,jk", a, b, out=c, dtype='f8', casting='unsafe',
                          optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))
                c[...] = 0
                np.einsum(a, [0, 1], b, [1, 2], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))

            # matrix triple product (note this is not currently an efficient
            # way to multiply 3 matrices)
            a = np.arange(12, dtype=dtype).reshape(3, 4)
            b = np.arange(20, dtype=dtype).reshape(4, 5)
            c = np.arange(30, dtype=dtype).reshape(5, 6)
            if dtype != 'f2':
                assert_equal(np.einsum("ij,jk,kl", a, b, c, optimize=do_opt),
                             a.dot(b).dot(c))
                assert_equal(np.einsum(a, [0, 1], b, [1, 2], c, [2, 3],
                                       optimize=do_opt), a.dot(b).dot(c))

            d = np.arange(18, dtype=dtype).reshape(3, 6)
            np.einsum("ij,jk,kl", a, b, c, out=d,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            tgt = a.astype('f8').dot(b.astype('f8'))
            tgt = tgt.dot(c.astype('f8')).astype(dtype)
            assert_equal(d, tgt)

            d[...] = 0
            np.einsum(a, [0, 1], b, [1, 2], c, [2, 3], out=d,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            tgt = a.astype('f8').dot(b.astype('f8'))
            tgt = tgt.dot(c.astype('f8')).astype(dtype)
            assert_equal(d, tgt)

            # tensordot(a, b)
            if np.dtype(dtype) != np.dtype('f2'):
                a = np.arange(60, dtype=dtype).reshape(3, 4, 5)
                b = np.arange(24, dtype=dtype).reshape(4, 3, 2)
                assert_equal(np.einsum("ijk, jil -> kl", a, b),
                             np.tensordot(a, b, axes=([1, 0], [0, 1])))
                assert_equal(np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3]),
                             np.tensordot(a, b, axes=([1, 0], [0, 1])))

                c = np.arange(10, dtype=dtype).reshape(5, 2)
                np.einsum("ijk,jil->kl", a, b, out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                             axes=([1, 0], [0, 1])).astype(dtype))
                c[...] = 0
                np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                             axes=([1, 0], [0, 1])).astype(dtype))

        # logical_and(logical_and(a!=0, b!=0), c!=0)
        neg_val = -2 if dtype.kind != "u" else np.iinfo(dtype).max - 1
        a = np.array([1,   3,   neg_val, 0,  12,  13,   0,   1], dtype=dtype)
        b = np.array([0,   3.5, 0., neg_val,  0,   1,    3,   12], dtype=dtype)
        c = np.array([True, True, False, True, True, False, True, True])

        assert_equal(np.einsum("i,i,i->i", a, b, c,
                     dtype='?', casting='unsafe', optimize=do_opt),
                     np.logical_and(np.logical_and(a != 0, b != 0), c != 0))
        assert_equal(np.einsum(a, [0], b, [0], c, [0], [0],
                     dtype='?', casting='unsafe'),
                     np.logical_and(np.logical_and(a != 0, b != 0), c != 0))

        a = np.arange(9, dtype=dtype)
        assert_equal(np.einsum(",i->", 3, a), 3*np.sum(a))
        assert_equal(np.einsum(3, [], a, [0], []), 3*np.sum(a))
        assert_equal(np.einsum("i,->", a, 3), 3*np.sum(a))
        assert_equal(np.einsum(a, [0], 3, [], []), 3*np.sum(a))

        # Various stride0, contiguous, and SSE aligned variants
        for n in range(1, 25):
            a = np.arange(n, dtype=dtype)
            if np.dtype(dtype).itemsize > 1:
                assert_equal(np.einsum("...,...", a, a, optimize=do_opt),
                             np.multiply(a, a))
                assert_equal(np.einsum("i,i", a, a, optimize=do_opt), np.dot(a, a))
                assert_equal(np.einsum("i,->i", a, 2, optimize=do_opt), 2*a)
                assert_equal(np.einsum(",i->i", 2, a, optimize=do_opt), 2*a)
                assert_equal(np.einsum("i,->", a, 2, optimize=do_opt), 2*np.sum(a))
                assert_equal(np.einsum(",i->", 2, a, optimize=do_opt), 2*np.sum(a))

                assert_equal(np.einsum("...,...", a[1:], a[:-1], optimize=do_opt),
                             np.multiply(a[1:], a[:-1]))
                assert_equal(np.einsum("i,i", a[1:], a[:-1], optimize=do_opt),
                             np.dot(a[1:], a[:-1]))
                assert_equal(np.einsum("i,->i", a[1:], 2, optimize=do_opt), 2*a[1:])
                assert_equal(np.einsum(",i->i", 2, a[1:], optimize=do_opt), 2*a[1:])
                assert_equal(np.einsum("i,->", a[1:], 2, optimize=do_opt),
                             2*np.sum(a[1:]))
                assert_equal(np.einsum(",i->", 2, a[1:], optimize=do_opt),
                             2*np.sum(a[1:]))

        # An object array, summed as the data type
        a = np.arange(9, dtype=object)

        b = np.einsum("i->", a, dtype=dtype, casting='unsafe')
        assert_equal(b, np.sum(a))
        assert_equal(b.dtype, np.dtype(dtype))

        b = np.einsum(a, [0], [], dtype=dtype, casting='unsafe')
        assert_equal(b, np.sum(a))
        assert_equal(b.dtype, np.dtype(dtype))

        # A case which was failing (ticket #1885)
        p = np.arange(2) + 1
        q = np.arange(4).reshape(2, 2) + 3
        r = np.arange(4).reshape(2, 2) + 7
        assert_equal(np.einsum('z,mz,zm->', p, q, r), 253)

        # singleton dimensions broadcast (gh-10343)
        p = np.ones((10,2))
        q = np.ones((1,2))
        assert_array_equal(np.einsum('ij,ij->j', p, q, optimize=True),
                           np.einsum('ij,ij->j', p, q, optimize=False))
        assert_array_equal(np.einsum('ij,ij->j', p, q, optimize=True),
                           [10.] * 2)

        # a blas-compatible contraction broadcasting case which was failing
        # for optimize=True (ticket #10930)
        x = np.array([2., 3.])
        y = np.array([4.])
        assert_array_equal(np.einsum("i, i", x, y, optimize=False), 20.)
        assert_array_equal(np.einsum("i, i", x, y, optimize=True), 20.)

        # all-ones array was bypassing bug (ticket #10930)
        p = np.ones((1, 5)) / 2
        q = np.ones((5, 5)) / 2
        for optimize in (True, False):
            assert_array_equal(np.einsum("...ij,...jk->...ik", p, p,
                                         optimize=optimize),
                               np.einsum("...ij,...jk->...ik", p, q,
                                         optimize=optimize))
            assert_array_equal(np.einsum("...ij,...jk->...ik", p, q,
                                         optimize=optimize),
                               np.full((1, 5), 1.25))

        # Cases which were failing (gh-10899)
        x = np.eye(2, dtype=dtype)
        y = np.ones(2, dtype=dtype)
        assert_array_equal(np.einsum("ji,i->", x, y, optimize=optimize),
                           [2.])  # contig_contig_outstride0_two
        assert_array_equal(np.einsum("i,ij->", y, x, optimize=optimize),
                           [2.])  # stride0_contig_outstride0_two
        assert_array_equal(np.einsum("ij,i->", x, y, optimize=optimize),
                           [2.])  # contig_stride0_outstride0_two

    def test_einsum_sums_int8(self):
        self.check_einsum_sums('i1')

    def test_einsum_sums_uint8(self):
        self.check_einsum_sums('u1')

    def test_einsum_sums_int16(self):
        self.check_einsum_sums('i2')

    def test_einsum_sums_uint16(self):
        self.check_einsum_sums('u2')

    def test_einsum_sums_int32(self):
        self.check_einsum_sums('i4')
        self.check_einsum_sums('i4', True)

    def test_einsum_sums_uint32(self):
        self.check_einsum_sums('u4')
        self.check_einsum_sums('u4', True)

    def test_einsum_sums_int64(self):
        self.check_einsum_sums('i8')

    def test_einsum_sums_uint64(self):
        self.check_einsum_sums('u8')

    def test_einsum_sums_float16(self):
        self.check_einsum_sums('f2')

    def test_einsum_sums_float32(self):
        self.check_einsum_sums('f4')

    def test_einsum_sums_float64(self):
        self.check_einsum_sums('f8')
        self.check_einsum_sums('f8', True)

    def test_einsum_sums_longdouble(self):
        self.check_einsum_sums(np.longdouble)

    def test_einsum_sums_cfloat64(self):
        self.check_einsum_sums('c8')
        self.check_einsum_sums('c8', True)

    def test_einsum_sums_cfloat128(self):
        self.check_einsum_sums('c16')

    def test_einsum_sums_clongdouble(self):
        self.check_einsum_sums(np.clongdouble)

    def test_einsum_misc(self):
        # This call used to crash because of a bug in
        # PyArray_AssignZero
        a = np.ones((1, 2))
        b = np.ones((2, 2, 1))
        assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
        assert_equal(np.einsum('ij...,j...->i...', a, b, optimize=True), [[[2], [2]]])

        # Regression test for issue #10369 (test unicode inputs with Python 2)
        assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
        assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4]), 20)
        assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4],
                               optimize='greedy'), 20)

        # The iterator had an issue with buffering this reduction
        a = np.ones((5, 12, 4, 2, 3), np.int64)
        b = np.ones((5, 12, 11), np.int64)
        assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b),
                     np.einsum('ijklm,ijn->', a, b))
        assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b, optimize=True),
                     np.einsum('ijklm,ijn->', a, b, optimize=True))

        # Issue #2027, was a problem in the contiguous 3-argument
        # inner loop implementation
        a = np.arange(1, 3)
        b = np.arange(1, 5).reshape(2, 2)
        c = np.arange(1, 9).reshape(4, 2)
        assert_equal(np.einsum('x,yx,zx->xzy', a, b, c),
                     [[[1,  3], [3,  9], [5, 15], [7, 21]],
                     [[8, 16], [16, 32], [24, 48], [32, 64]]])
        assert_equal(np.einsum('x,yx,zx->xzy', a, b, c, optimize=True),
                     [[[1,  3], [3,  9], [5, 15], [7, 21]],
                     [[8, 16], [16, 32], [24, 48], [32, 64]]])

        # Ensure explicitly setting out=None does not cause an error
        # see issue gh-15776 and issue gh-15256
        assert_equal(np.einsum('i,j', [1], [2], out=None), [[2]])

    def test_subscript_range(self):
        # Issue #7741, make sure that all letters of Latin alphabet (both uppercase & lowercase) can be used
        # when creating a subscript from arrays
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        np.einsum(a, [0, 20], b, [20, 2], [0, 2], optimize=False)
        np.einsum(a, [0, 27], b, [27, 2], [0, 2], optimize=False)
        np.einsum(a, [0, 51], b, [51, 2], [0, 2], optimize=False)
        assert_raises(ValueError, lambda: np.einsum(a, [0, 52], b, [52, 2], [0, 2], optimize=False))
        assert_raises(ValueError, lambda: np.einsum(a, [-1, 5], b, [5, 2], [-1, 2], optimize=False))

    def test_einsum_broadcast(self):
        # Issue #2455 change in handling ellipsis
        # remove the 'middle broadcast' error
        # only use the 'RIGHT' iteration in prepare_op_axes
        # adds auto broadcast on left where it belongs
        # broadcast on right has to be explicit
        # We need to test the optimized parsing as well

        A = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        B = np.arange(3)
        ref = np.einsum('ijk,j->ijk', A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('ij...,j...->ij...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ij...,...j->ij...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ij...,j->ij...', A, B, optimize=opt), ref)  # used to raise error

        A = np.arange(12).reshape((4, 3))
        B = np.arange(6).reshape((3, 2))
        ref = np.einsum('ik,kj->ij', A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('ik...,k...->i...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ik...,...kj->i...j', A, B, optimize=opt), ref)
            assert_equal(np.einsum('...k,kj', A, B, optimize=opt), ref)  # used to raise error
            assert_equal(np.einsum('ik,k...->i...', A, B, optimize=opt), ref)  # used to raise error

        dims = [2, 3, 4, 5]
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        ref = np.einsum('ijkl,k->ijl', a, v, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('ijkl,k', a, v, optimize=opt), ref)
            assert_equal(np.einsum('...kl,k', a, v, optimize=opt), ref)  # used to raise error
            assert_equal(np.einsum('...kl,k...', a, v, optimize=opt), ref)

        J, K, M = 160, 160, 120
        A = np.arange(J * K * M).reshape(1, 1, 1, J, K, M)
        B = np.arange(J * K * M * 3).reshape(J, K, M, 3)
        ref = np.einsum('...lmn,...lmno->...o', A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('...lmn,lmno->...o', A, B,
                                   optimize=opt), ref)  # used to raise error

    def test_einsum_fixedstridebug(self):
        # Issue #4485 obscure einsum bug
        # This case revealed a bug in nditer where it reported a stride
        # as 'fixed' (0) when it was in fact not fixed during processing
        # (0 or 4). The reason for the bug was that the check for a fixed
        # stride was using the information from the 2D inner loop reuse
        # to restrict the iteration dimensions it had to validate to be
        # the same, but that 2D inner loop reuse logic is only triggered
        # during the buffer copying step, and hence it was invalid to
        # rely on those values. The fix is to check all the dimensions
        # of the stride in question, which in the test case reveals that
        # the stride is not fixed.
        #
        # NOTE: This test is triggered by the fact that the default buffersize,
        #       used by einsum, is 8192, and 3*2731 = 8193, is larger than that
        #       and results in a mismatch between the buffering and the
        #       striding for operand A.
        A = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
        B = np.arange(2 * 3 * 2731).reshape(2, 3, 2731).astype(np.int16)
        es = np.einsum('cl, cpx->lpx',  A,  B)
        tp = np.tensordot(A,  B,  axes=(0,  0))
        assert_equal(es,  tp)
        # The following is the original test case from the bug report,
        # made repeatable by changing random arrays to aranges.
        A = np.arange(3 * 3).reshape(3, 3).astype(np.float64)
        B = np.arange(3 * 3 * 64 * 64).reshape(3, 3, 64, 64).astype(np.float32)
        es = np.einsum('cl, cpxy->lpxy',  A, B)
        tp = np.tensordot(A, B,  axes=(0, 0))
        assert_equal(es, tp)

    def test_einsum_fixed_collapsingbug(self):
        # Issue #5147.
        # The bug only occurred when output argument of einssum was used.
        x = np.random.normal(0, 1, (5, 5, 5, 5))
        y1 = np.zeros((5, 5))
        np.einsum('aabb->ab', x, out=y1)
        idx = np.arange(5)
        y2 = x[idx[:, None], idx[:, None], idx, idx]
        assert_equal(y1, y2)

    def test_einsum_failed_on_p9_and_s390x(self):
        # Issues gh-14692 and gh-12689
        # Bug with signed vs unsigned char errored on power9 and s390x Linux
        tensor = np.random.random_sample((10, 10, 10, 10))
        x = np.einsum('ijij->', tensor)
        y = tensor.trace(axis1=0, axis2=2).trace()
        assert_allclose(x, y)

    def test_einsum_all_contig_non_contig_output(self):
        # Issue gh-5907, tests that the all contiguous special case
        # actually checks the contiguity of the output
        x = np.ones((5, 5))
        out = np.ones(10)[::2]
        correct_base = np.ones(10)
        correct_base[::2] = 5
        # Always worked (inner iteration is done with 0-stride):
        np.einsum('mi,mi,mi->m', x, x, x, out=out)
        assert_array_equal(out.base, correct_base)
        # Example 1:
        out = np.ones(10)[::2]
        np.einsum('im,im,im->m', x, x, x, out=out)
        assert_array_equal(out.base, correct_base)
        # Example 2, buffering causes x to be contiguous but
        # special cases do not catch the operation before:
        out = np.ones((2, 2, 2))[..., 0]
        correct_base = np.ones((2, 2, 2))
        correct_base[..., 0] = 2
        x = np.ones((2, 2), np.float32)
        np.einsum('ij,jk->ik', x, x, out=out)
        assert_array_equal(out.base, correct_base)

    @pytest.mark.parametrize("dtype",
             np.typecodes["AllFloat"] + np.typecodes["AllInteger"])
    def test_different_paths(self, dtype):
        # Test originally added to cover broken float16 path: gh-20305
        # Likely most are covered elsewhere, at least partially.
        dtype = np.dtype(dtype)
        # Simple test, designed to excersize most specialized code paths,
        # note the +0.5 for floats.  This makes sure we use a float value
        # where the results must be exact.
        arr = (np.arange(7) + 0.5).astype(dtype)
        scalar = np.array(2, dtype=dtype)

        # contig -> scalar:
        res = np.einsum('i->', arr)
        assert res == arr.sum()
        # contig, contig -> contig:
        res = np.einsum('i,i->i', arr, arr)
        assert_array_equal(res, arr * arr)
        # noncontig, noncontig -> contig:
        res = np.einsum('i,i->i', arr.repeat(2)[::2], arr.repeat(2)[::2])
        assert_array_equal(res, arr * arr)
        # contig + contig -> scalar
        assert np.einsum('i,i->', arr, arr) == (arr * arr).sum()
        # contig + scalar -> contig (with out)
        out = np.ones(7, dtype=dtype)
        res = np.einsum('i,->i', arr, dtype.type(2), out=out)
        assert_array_equal(res, arr * dtype.type(2))
        # scalar + contig -> contig (with out)
        res = np.einsum(',i->i', scalar, arr)
        assert_array_equal(res, arr * dtype.type(2))
        # scalar + contig -> scalar
        res = np.einsum(',i->', scalar, arr)
        # Use einsum to compare to not have difference due to sum round-offs:
        assert res == np.einsum('i->', scalar * arr)
        # contig + scalar -> scalar
        res = np.einsum('i,->', arr, scalar)
        # Use einsum to compare to not have difference due to sum round-offs:
        assert res == np.einsum('i->', scalar * arr)
        # contig + contig + contig -> scalar
        arr = np.array([0.5, 0.5, 0.25, 4.5, 3.], dtype=dtype)
        res = np.einsum('i,i,i->', arr, arr, arr)
        assert_array_equal(res, (arr * arr * arr).sum())
        # four arrays:
        res = np.einsum('i,i,i,i->', arr, arr, arr, arr)
        assert_array_equal(res, (arr * arr * arr * arr).sum())

    def test_small_boolean_arrays(self):
        # See gh-5946.
        # Use array of True embedded in False.
        a = np.zeros((16, 1, 1), dtype=np.bool_)[:2]
        a[...] = True
        out = np.zeros((16, 1, 1), dtype=np.bool_)[:2]
        tgt = np.ones((2, 1, 1), dtype=np.bool_)
        res = np.einsum('...ij,...jk->...ik', a, a, out=out)
        assert_equal(res, tgt)

    def test_out_is_res(self):
        a = np.arange(9).reshape(3, 3)
        res = np.einsum('...ij,...jk->...ik', a, a, out=a)
        assert res is a

    def optimize_compare(self, subscripts, operands=None):
        # Tests all paths of the optimization function against
        # conventional einsum
        if operands is None:
            args = [subscripts]
            terms = subscripts.split('->')[0].split(',')
            for term in terms:
                dims = [global_size_dict[x] for x in term]
                args.append(np.random.rand(*dims))
        else:
            args = [subscripts] + operands

        noopt = np.einsum(*args, optimize=False)
        opt = np.einsum(*args, optimize='greedy')
        assert_almost_equal(opt, noopt)
        opt = np.einsum(*args, optimize='optimal')
        assert_almost_equal(opt, noopt)

    def test_hadamard_like_products(self):
        # Hadamard outer products
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    def test_index_transformations(self):
        # Simple index transformation cases
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    def test_complex(self):
        # Long test cases
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def test_collapse(self):
        # Inner products
        self.optimize_compare('ab,ab,c->')
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->')
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,cd,cd->cd')
        self.optimize_compare('ab,ab,cd,cd,ef,ef->')

    def test_expand(self):
        # Outer products
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')

    def test_edge_cases(self):
        # Difficult edge cases for optimization
        self.optimize_compare('eb,cb,fb->cef')
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('bca,cdb,dbf,afc->')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('abcd,ad')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('fff,fae,bef,def->abd')
        self.optimize_compare('efc,dbc,acf,fd->abe')
        self.optimize_compare('ba,ac,da->bcd')

    def test_inner_product(self):
        # Inner products
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    def test_random_cases(self):
        # Randomly built test cases
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('bcf,bbb,fbf,fc->')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('bcb,bb,fc,fff->')
        self.optimize_compare('fbb,dfd,fc,fc->')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def test_combined_views_mapping(self):
        # gh-10792
        a = np.arange(9).reshape(1, 1, 3, 1, 3)
        b = np.einsum('bbcdc->d', a)
        assert_equal(b, [12])

    def test_broadcasting_dot_cases(self):
        # Ensures broadcasting cases are not mistaken for GEMM

        a = np.random.rand(1, 5, 4)
        b = np.random.rand(4, 6)
        c = np.random.rand(5, 6)
        d = np.random.rand(10)

        self.optimize_compare('ijk,kl,jl', operands=[a, b, c])
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])

        e = np.random.rand(1, 1, 5, 4)
        f = np.random.rand(7, 7)
        self.optimize_compare('abjk,kl,jl', operands=[e, b, c])
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

        # Edge case found in gh-11308
        g = np.arange(64).reshape(2, 4, 8)
        self.optimize_compare('obk,ijk->ioj', operands=[g, g])

    def test_output_order(self):
        # Ensure output order is respected for optimize cases, the below
        # conraction should yield a reshaped tensor view
        # gh-16415

        a = np.ones((2, 3, 5), order='F')
        b = np.ones((4, 3), order='F')

        for opt in [True, False]:
            tmp = np.einsum('...ft,mf->...mt', a, b, order='a', optimize=opt)
            assert_(tmp.flags.f_contiguous)

            tmp = np.einsum('...ft,mf->...mt', a, b, order='f', optimize=opt)
            assert_(tmp.flags.f_contiguous)

            tmp = np.einsum('...ft,mf->...mt', a, b, order='c', optimize=opt)
            assert_(tmp.flags.c_contiguous)

            tmp = np.einsum('...ft,mf->...mt', a, b, order='k', optimize=opt)
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

            tmp = np.einsum('...ft,mf->...mt', a, b, optimize=opt)
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

        c = np.ones((4, 3), order='C')
        for opt in [True, False]:
            tmp = np.einsum('...ft,mf->...mt', a, c, order='a', optimize=opt)
            assert_(tmp.flags.c_contiguous)

        d = np.ones((2, 3, 5), order='C')
        for opt in [True, False]:
            tmp = np.einsum('...ft,mf->...mt', d, c, order='a', optimize=opt)
            assert_(tmp.flags.c_contiguous)

class TestEinsumPath:
    def build_operands(self, string, size_dict=global_size_dict):

        # Builds views based off initial operands
        operands = [string]
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(np.random.rand(*dims))

        return operands

    def assert_path_equal(self, comp, benchmark):
        # Checks if list of tuples are equivalent
        ret = (len(comp) == len(benchmark))
        assert_(ret)
        for pos in range(len(comp) - 1):
            ret &= isinstance(comp[pos + 1], tuple)
            ret &= (comp[pos + 1] == benchmark[pos + 1])
        assert_(ret)

    def test_memory_contraints(self):
        # Ensure memory constraints are satisfied

        outer_test = self.build_operands('a,b,c->abc')

        path, path_str = np.einsum_path(*outer_test, optimize=('greedy', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        path, path_str = np.einsum_path(*outer_test, optimize=('optimal', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        long_test = self.build_operands('acdf,jbje,gihb,hfac')
        path, path_str = np.einsum_path(*long_test, optimize=('greedy', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        path, path_str = np.einsum_path(*long_test, optimize=('optimal', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

    def test_long_paths(self):
        # Long complex cases

        # Long test 1
        long_test1 = self.build_operands('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        path, path_str = np.einsum_path(*long_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        path, path_str = np.einsum_path(*long_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        # Long test 2
        long_test2 = self.build_operands('chd,bde,agbc,hiad,bdi,cgh,agdb')
        path, path_str = np.einsum_path(*long_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)])

        path, path_str = np.einsum_path(*long_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path',
                                      (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)])

    def test_edge_paths(self):
        # Difficult edge cases

        # Edge test1
        edge_test1 = self.build_operands('eb,cb,fb->cef')
        path, path_str = np.einsum_path(*edge_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        path, path_str = np.einsum_path(*edge_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        # Edge test2
        edge_test2 = self.build_operands('dd,fb,be,cdb->cef')
        path, path_str = np.einsum_path(*edge_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        path, path_str = np.einsum_path(*edge_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        # Edge test3
        edge_test3 = self.build_operands('bca,cdb,dbf,afc->')
        path, path_str = np.einsum_path(*edge_test3, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        path, path_str = np.einsum_path(*edge_test3, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test4
        edge_test4 = self.build_operands('dcc,fce,ea,dbf->ab')
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])

        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test5
        edge_test4 = self.build_operands('a,ac,ab,ad,cd,bd,bc->',
                                         size_dict={"a": 20, "b": 20, "c": 20, "d": 20})
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

    def test_path_type_input(self):
        # Test explicit path handling
        path_test = self.build_operands('dcc,fce,ea,dbf->ab')

        path, path_str = np.einsum_path(*path_test, optimize=False)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        path, path_str = np.einsum_path(*path_test, optimize=True)
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])

        exp_path = ['einsum_path', (0, 2), (0, 2), (0, 1)]
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        self.assert_path_equal(path, exp_path)

        # Double check einsum works on the input path
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)

    def test_path_type_input_internal_trace(self):
        #gh-20962
        path_test = self.build_operands('cab,cdd->ab')
        exp_path = ['einsum_path', (1,), (0, 1)]

        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        self.assert_path_equal(path, exp_path)

        # Double check einsum works on the input path
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)

    def test_path_type_input_invalid(self):
        path_test = self.build_operands('ab,bc,cd,de->ae')
        exp_path = ['einsum_path', (2, 3), (0, 1)]
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(
            RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

        path_test = self.build_operands('a,a,a->a')
        exp_path = ['einsum_path', (1,), (0, 1)]
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(
            RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

    def test_spaces(self):
        #gh-10794
        arr = np.array([[1]])
        for sp in itertools.product(['', ' '], repeat=4):
            # no error for any spacing
            np.einsum('{}...a{}->{}...a{}'.format(*sp), arr)

def test_overlap():
    a = np.arange(9, dtype=int).reshape(3, 3)
    b = np.arange(9, dtype=int).reshape(3, 3)
    d = np.dot(a, b)
    # sanity check
    c = np.einsum('ij,jk->ik', a, b)
    assert_equal(c, d)
    #gh-10080, out overlaps one of the operands
    c = np.einsum('ij,jk->ik', a, b, out=b)
    assert_equal(c, d)
