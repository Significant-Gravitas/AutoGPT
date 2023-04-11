import os

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_array_almost_equal,
    assert_raises, _assert_valid_refcount,
    )


class TestRegression:
    def test_poly1d(self):
        # Ticket #28
        assert_equal(np.poly1d([1]) - np.poly1d([1, 0]),
                     np.poly1d([-1, 1]))

    def test_cov_parameters(self):
        # Ticket #91
        x = np.random.random((3, 3))
        y = x.copy()
        np.cov(x, rowvar=True)
        np.cov(y, rowvar=False)
        assert_array_equal(x, y)

    def test_mem_digitize(self):
        # Ticket #95
        for i in range(100):
            np.digitize([1, 2, 3, 4], [1, 3])
            np.digitize([0, 1, 2, 3, 4], [1, 3])

    def test_unique_zero_sized(self):
        # Ticket #205
        assert_array_equal([], np.unique(np.array([])))

    def test_mem_vectorise(self):
        # Ticket #325
        vt = np.vectorize(lambda *args: args)
        vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1, 1, 2)))
        vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1,
           1, 2)), np.zeros((2, 2)))

    def test_mgrid_single_element(self):
        # Ticket #339
        assert_array_equal(np.mgrid[0:0:1j], [0])
        assert_array_equal(np.mgrid[0:0], [])

    def test_refcount_vectorize(self):
        # Ticket #378
        def p(x, y):
            return 123
        v = np.vectorize(p)
        _assert_valid_refcount(v)

    def test_poly1d_nan_roots(self):
        # Ticket #396
        p = np.poly1d([np.nan, np.nan, 1], r=False)
        assert_raises(np.linalg.LinAlgError, getattr, p, "r")

    def test_mem_polymul(self):
        # Ticket #448
        np.polymul([], [1.])

    def test_mem_string_concat(self):
        # Ticket #469
        x = np.array([])
        np.append(x, 'asdasd\tasdasd')

    def test_poly_div(self):
        # Ticket #553
        u = np.poly1d([1, 2, 3])
        v = np.poly1d([1, 2, 3, 4, 5])
        q, r = np.polydiv(u, v)
        assert_equal(q*v + r, u)

    def test_poly_eq(self):
        # Ticket #554
        x = np.poly1d([1, 2, 3])
        y = np.poly1d([3, 4])
        assert_(x != y)
        assert_(x == x)

    def test_polyfit_build(self):
        # Ticket #628
        ref = [-1.06123820e-06, 5.70886914e-04, -1.13822012e-01,
               9.95368241e+00, -3.14526520e+02]
        x = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
             104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
             116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 129,
             130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
             146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
             158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
             170, 171, 172, 173, 174, 175, 176]
        y = [9.0, 3.0, 7.0, 4.0, 4.0, 8.0, 6.0, 11.0, 9.0, 8.0, 11.0, 5.0,
             6.0, 5.0, 9.0, 8.0, 6.0, 10.0, 6.0, 10.0, 7.0, 6.0, 6.0, 6.0,
             13.0, 4.0, 9.0, 11.0, 4.0, 5.0, 8.0, 5.0, 7.0, 7.0, 6.0, 12.0,
             7.0, 7.0, 9.0, 4.0, 12.0, 6.0, 6.0, 4.0, 3.0, 9.0, 8.0, 8.0,
             6.0, 7.0, 9.0, 10.0, 6.0, 8.0, 4.0, 7.0, 7.0, 10.0, 8.0, 8.0,
             6.0, 3.0, 8.0, 4.0, 5.0, 7.0, 8.0, 6.0, 6.0, 4.0, 12.0, 9.0,
             8.0, 8.0, 8.0, 6.0, 7.0, 4.0, 4.0, 5.0, 7.0]
        tested = np.polyfit(x, y, 4)
        assert_array_almost_equal(ref, tested)

    def test_polydiv_type(self):
        # Make polydiv work for complex types
        msg = "Wrong type, should be complex"
        x = np.ones(3, dtype=complex)
        q, r = np.polydiv(x, x)
        assert_(q.dtype == complex, msg)
        msg = "Wrong type, should be float"
        x = np.ones(3, dtype=int)
        q, r = np.polydiv(x, x)
        assert_(q.dtype == float, msg)

    def test_histogramdd_too_many_bins(self):
        # Ticket 928.
        assert_raises(ValueError, np.histogramdd, np.ones((1, 10)), bins=2**10)

    def test_polyint_type(self):
        # Ticket #944
        msg = "Wrong type, should be complex"
        x = np.ones(3, dtype=complex)
        assert_(np.polyint(x).dtype == complex, msg)
        msg = "Wrong type, should be float"
        x = np.ones(3, dtype=int)
        assert_(np.polyint(x).dtype == float, msg)

    def test_ndenumerate_crash(self):
        # Ticket 1140
        # Shouldn't crash:
        list(np.ndenumerate(np.array([[]])))

    def test_asfarray_none(self):
        # Test for changeset r5065
        assert_array_equal(np.array([np.nan]), np.asfarray([None]))

    def test_large_fancy_indexing(self):
        # Large enough to fail on 64-bit.
        nbits = np.dtype(np.intp).itemsize * 8
        thesize = int((2**nbits)**(1.0/5.0)+1)

        def dp():
            n = 3
            a = np.ones((n,)*5)
            i = np.random.randint(0, n, size=thesize)
            a[np.ix_(i, i, i, i, i)] = 0

        def dp2():
            n = 3
            a = np.ones((n,)*5)
            i = np.random.randint(0, n, size=thesize)
            a[np.ix_(i, i, i, i, i)]

        assert_raises(ValueError, dp)
        assert_raises(ValueError, dp2)

    def test_void_coercion(self):
        dt = np.dtype([('a', 'f4'), ('b', 'i4')])
        x = np.zeros((1,), dt)
        assert_(np.r_[x, x].dtype == dt)

    def test_who_with_0dim_array(self):
        # ticket #1243
        import os
        import sys

        oldstdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            try:
                np.who({'foo': np.array(1)})
            except Exception:
                raise AssertionError("ticket #1243")
        finally:
            sys.stdout.close()
            sys.stdout = oldstdout

    def test_include_dirs(self):
        # As a sanity check, just test that get_include
        # includes something reasonable.  Somewhat
        # related to ticket #1405.
        include_dirs = [np.get_include()]
        for path in include_dirs:
            assert_(isinstance(path, str))
            assert_(path != '')

    def test_polyder_return_type(self):
        # Ticket #1249
        assert_(isinstance(np.polyder(np.poly1d([1]), 0), np.poly1d))
        assert_(isinstance(np.polyder([1], 0), np.ndarray))
        assert_(isinstance(np.polyder(np.poly1d([1]), 1), np.poly1d))
        assert_(isinstance(np.polyder([1], 1), np.ndarray))

    def test_append_fields_dtype_list(self):
        # Ticket #1676
        from numpy.lib.recfunctions import append_fields

        base = np.array([1, 2, 3], dtype=np.int32)
        names = ['a', 'b', 'c']
        data = np.eye(3).astype(np.int32)
        dlist = [np.float64, np.int32, np.int32]
        try:
            append_fields(base, names, data, dlist)
        except Exception:
            raise AssertionError()

    def test_loadtxt_fields_subarrays(self):
        # For ticket #1936
        from io import StringIO

        dt = [("a", 'u1', 2), ("b", 'u1', 2)]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))

        dt = [("a", [("a", 'u1', (1, 3)), ("b", 'u1')])]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([(((0, 1, 2), 3),)], dtype=dt))

        dt = [("a", 'u1', (2, 2))]
        x = np.loadtxt(StringIO("0 1 2 3"), dtype=dt)
        assert_equal(x, np.array([(((0, 1), (2, 3)),)], dtype=dt))

        dt = [("a", 'u1', (2, 3, 2))]
        x = np.loadtxt(StringIO("0 1 2 3 4 5 6 7 8 9 10 11"), dtype=dt)
        data = [((((0, 1), (2, 3), (4, 5)), ((6, 7), (8, 9), (10, 11))),)]
        assert_equal(x, np.array(data, dtype=dt))

    def test_nansum_with_boolean(self):
        # gh-2978
        a = np.zeros(2, dtype=bool)
        try:
            np.nansum(a)
        except Exception:
            raise AssertionError()

    def test_py3_compat(self):
        # gh-2561
        # Test if the oldstyle class test is bypassed in python3
        class C():
            """Old-style class in python2, normal class in python3"""
            pass

        out = open(os.devnull, 'w')
        try:
            np.info(C(), output=out)
        except AttributeError:
            raise AssertionError()
        finally:
            out.close()
