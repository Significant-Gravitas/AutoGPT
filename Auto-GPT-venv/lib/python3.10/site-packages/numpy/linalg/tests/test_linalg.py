""" Test functions for linalg module

"""
import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest

import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, suppress_warnings,
    assert_raises_regex, HAS_LAPACK64, IS_WASM
    )


def consistent_subclass(out, in_):
    # For ndarray subclass input, our output should have the same subclass
    # (non-ndarray input gets converted to ndarray).
    return type(out) is (type(in_) if isinstance(in_, np.ndarray)
                         else np.ndarray)


old_assert_almost_equal = assert_almost_equal


def assert_almost_equal(a, b, single_decimal=6, double_decimal=12, **kw):
    if asarray(a).dtype.type in (single, csingle):
        decimal = single_decimal
    else:
        decimal = double_decimal
    old_assert_almost_equal(a, b, decimal=decimal, **kw)


def get_real_dtype(dtype):
    return {single: single, double: double,
            csingle: single, cdouble: double}[dtype]


def get_complex_dtype(dtype):
    return {single: csingle, double: cdouble,
            csingle: csingle, cdouble: cdouble}[dtype]


def get_rtol(dtype):
    # Choose a safe rtol
    if dtype in (single, csingle):
        return 1e-5
    else:
        return 1e-11


# used to categorize tests
all_tags = {
  'square', 'nonsquare', 'hermitian',  # mutually exclusive
  'generalized', 'size-0', 'strided' # optional additions
}


class LinalgCase:
    def __init__(self, name, a, b, tags=set()):
        """
        A bundle of arguments to be passed to a test case, with an identifying
        name, the operands a and b, and a set of tags to filter the tests
        """
        assert_(isinstance(name, str))
        self.name = name
        self.a = a
        self.b = b
        self.tags = frozenset(tags)  # prevent shared tags

    def check(self, do):
        """
        Run the function `do` on this test case, expanding arguments
        """
        do(self.a, self.b, tags=self.tags)

    def __repr__(self):
        return f'<LinalgCase: {self.name}>'


def apply_tag(tag, cases):
    """
    Add the given tag (a string) to each of the cases (a list of LinalgCase
    objects)
    """
    assert tag in all_tags, "Invalid tag"
    for case in cases:
        case.tags = case.tags | {tag}
    return cases


#
# Base test cases
#

np.random.seed(1234)

CASES = []

# square test cases
CASES += apply_tag('square', [
    LinalgCase("single",
               array([[1., 2.], [3., 4.]], dtype=single),
               array([2., 1.], dtype=single)),
    LinalgCase("double",
               array([[1., 2.], [3., 4.]], dtype=double),
               array([2., 1.], dtype=double)),
    LinalgCase("double_2",
               array([[1., 2.], [3., 4.]], dtype=double),
               array([[2., 1., 4.], [3., 4., 6.]], dtype=double)),
    LinalgCase("csingle",
               array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=csingle),
               array([2. + 1j, 1. + 2j], dtype=csingle)),
    LinalgCase("cdouble",
               array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
               array([2. + 1j, 1. + 2j], dtype=cdouble)),
    LinalgCase("cdouble_2",
               array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
               array([[2. + 1j, 1. + 2j, 1 + 3j], [1 - 2j, 1 - 3j, 1 - 6j]], dtype=cdouble)),
    LinalgCase("0x0",
               np.empty((0, 0), dtype=double),
               np.empty((0,), dtype=double),
               tags={'size-0'}),
    LinalgCase("8x8",
               np.random.rand(8, 8),
               np.random.rand(8)),
    LinalgCase("1x1",
               np.random.rand(1, 1),
               np.random.rand(1)),
    LinalgCase("nonarray",
               [[1, 2], [3, 4]],
               [2, 1]),
])

# non-square test-cases
CASES += apply_tag('nonsquare', [
    LinalgCase("single_nsq_1",
               array([[1., 2., 3.], [3., 4., 6.]], dtype=single),
               array([2., 1.], dtype=single)),
    LinalgCase("single_nsq_2",
               array([[1., 2.], [3., 4.], [5., 6.]], dtype=single),
               array([2., 1., 3.], dtype=single)),
    LinalgCase("double_nsq_1",
               array([[1., 2., 3.], [3., 4., 6.]], dtype=double),
               array([2., 1.], dtype=double)),
    LinalgCase("double_nsq_2",
               array([[1., 2.], [3., 4.], [5., 6.]], dtype=double),
               array([2., 1., 3.], dtype=double)),
    LinalgCase("csingle_nsq_1",
               array(
                   [[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]], dtype=csingle),
               array([2. + 1j, 1. + 2j], dtype=csingle)),
    LinalgCase("csingle_nsq_2",
               array(
                   [[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]], dtype=csingle),
               array([2. + 1j, 1. + 2j, 3. - 3j], dtype=csingle)),
    LinalgCase("cdouble_nsq_1",
               array(
                   [[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]], dtype=cdouble),
               array([2. + 1j, 1. + 2j], dtype=cdouble)),
    LinalgCase("cdouble_nsq_2",
               array(
                   [[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]], dtype=cdouble),
               array([2. + 1j, 1. + 2j, 3. - 3j], dtype=cdouble)),
    LinalgCase("cdouble_nsq_1_2",
               array(
                   [[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]], dtype=cdouble),
               array([[2. + 1j, 1. + 2j], [1 - 1j, 2 - 2j]], dtype=cdouble)),
    LinalgCase("cdouble_nsq_2_2",
               array(
                   [[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]], dtype=cdouble),
               array([[2. + 1j, 1. + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]], dtype=cdouble)),
    LinalgCase("8x11",
               np.random.rand(8, 11),
               np.random.rand(8)),
    LinalgCase("1x5",
               np.random.rand(1, 5),
               np.random.rand(1)),
    LinalgCase("5x1",
               np.random.rand(5, 1),
               np.random.rand(5)),
    LinalgCase("0x4",
               np.random.rand(0, 4),
               np.random.rand(0),
               tags={'size-0'}),
    LinalgCase("4x0",
               np.random.rand(4, 0),
               np.random.rand(4),
               tags={'size-0'}),
])

# hermitian test-cases
CASES += apply_tag('hermitian', [
    LinalgCase("hsingle",
               array([[1., 2.], [2., 1.]], dtype=single),
               None),
    LinalgCase("hdouble",
               array([[1., 2.], [2., 1.]], dtype=double),
               None),
    LinalgCase("hcsingle",
               array([[1., 2 + 3j], [2 - 3j, 1]], dtype=csingle),
               None),
    LinalgCase("hcdouble",
               array([[1., 2 + 3j], [2 - 3j, 1]], dtype=cdouble),
               None),
    LinalgCase("hempty",
               np.empty((0, 0), dtype=double),
               None,
               tags={'size-0'}),
    LinalgCase("hnonarray",
               [[1, 2], [2, 1]],
               None),
    LinalgCase("matrix_b_only",
               array([[1., 2.], [2., 1.]]),
               None),
    LinalgCase("hmatrix_1x1",
               np.random.rand(1, 1),
               None),
])


#
# Gufunc test cases
#
def _make_generalized_cases():
    new_cases = []

    for case in CASES:
        if not isinstance(case.a, np.ndarray):
            continue

        a = np.array([case.a, 2 * case.a, 3 * case.a])
        if case.b is None:
            b = None
        else:
            b = np.array([case.b, 7 * case.b, 6 * case.b])
        new_case = LinalgCase(case.name + "_tile3", a, b,
                              tags=case.tags | {'generalized'})
        new_cases.append(new_case)

        a = np.array([case.a] * 2 * 3).reshape((3, 2) + case.a.shape)
        if case.b is None:
            b = None
        else:
            b = np.array([case.b] * 2 * 3).reshape((3, 2) + case.b.shape)
        new_case = LinalgCase(case.name + "_tile213", a, b,
                              tags=case.tags | {'generalized'})
        new_cases.append(new_case)

    return new_cases


CASES += _make_generalized_cases()


#
# Generate stride combination variations of the above
#
def _stride_comb_iter(x):
    """
    Generate cartesian product of strides for all axes
    """

    if not isinstance(x, np.ndarray):
        yield x, "nop"
        return

    stride_set = [(1,)] * x.ndim
    stride_set[-1] = (1, 3, -4)
    if x.ndim > 1:
        stride_set[-2] = (1, 3, -4)
    if x.ndim > 2:
        stride_set[-3] = (1, -4)

    for repeats in itertools.product(*tuple(stride_set)):
        new_shape = [abs(a * b) for a, b in zip(x.shape, repeats)]
        slices = tuple([slice(None, None, repeat) for repeat in repeats])

        # new array with different strides, but same data
        xi = np.empty(new_shape, dtype=x.dtype)
        xi.view(np.uint32).fill(0xdeadbeef)
        xi = xi[slices]
        xi[...] = x
        xi = xi.view(x.__class__)
        assert_(np.all(xi == x))
        yield xi, "stride_" + "_".join(["%+d" % j for j in repeats])

        # generate also zero strides if possible
        if x.ndim >= 1 and x.shape[-1] == 1:
            s = list(x.strides)
            s[-1] = 0
            xi = np.lib.stride_tricks.as_strided(x, strides=s)
            yield xi, "stride_xxx_0"
        if x.ndim >= 2 and x.shape[-2] == 1:
            s = list(x.strides)
            s[-2] = 0
            xi = np.lib.stride_tricks.as_strided(x, strides=s)
            yield xi, "stride_xxx_0_x"
        if x.ndim >= 2 and x.shape[:-2] == (1, 1):
            s = list(x.strides)
            s[-1] = 0
            s[-2] = 0
            xi = np.lib.stride_tricks.as_strided(x, strides=s)
            yield xi, "stride_xxx_0_0"


def _make_strided_cases():
    new_cases = []
    for case in CASES:
        for a, a_label in _stride_comb_iter(case.a):
            for b, b_label in _stride_comb_iter(case.b):
                new_case = LinalgCase(case.name + "_" + a_label + "_" + b_label, a, b,
                                      tags=case.tags | {'strided'})
                new_cases.append(new_case)
    return new_cases


CASES += _make_strided_cases()


#
# Test different routines against the above cases
#
class LinalgTestCase:
    TEST_CASES = CASES

    def check_cases(self, require=set(), exclude=set()):
        """
        Run func on each of the cases with all of the tags in require, and none
        of the tags in exclude
        """
        for case in self.TEST_CASES:
            # filter by require and exclude
            if case.tags & require != require:
                continue
            if case.tags & exclude:
                continue

            try:
                case.check(self.do)
            except Exception as e:
                msg = f'In test case: {case!r}\n\n'
                msg += traceback.format_exc()
                raise AssertionError(msg) from e


class LinalgSquareTestCase(LinalgTestCase):

    def test_sq_cases(self):
        self.check_cases(require={'square'},
                         exclude={'generalized', 'size-0'})

    def test_empty_sq_cases(self):
        self.check_cases(require={'square', 'size-0'},
                         exclude={'generalized'})


class LinalgNonsquareTestCase(LinalgTestCase):

    def test_nonsq_cases(self):
        self.check_cases(require={'nonsquare'},
                         exclude={'generalized', 'size-0'})

    def test_empty_nonsq_cases(self):
        self.check_cases(require={'nonsquare', 'size-0'},
                         exclude={'generalized'})


class HermitianTestCase(LinalgTestCase):

    def test_herm_cases(self):
        self.check_cases(require={'hermitian'},
                         exclude={'generalized', 'size-0'})

    def test_empty_herm_cases(self):
        self.check_cases(require={'hermitian', 'size-0'},
                         exclude={'generalized'})


class LinalgGeneralizedSquareTestCase(LinalgTestCase):

    @pytest.mark.slow
    def test_generalized_sq_cases(self):
        self.check_cases(require={'generalized', 'square'},
                         exclude={'size-0'})

    @pytest.mark.slow
    def test_generalized_empty_sq_cases(self):
        self.check_cases(require={'generalized', 'square', 'size-0'})


class LinalgGeneralizedNonsquareTestCase(LinalgTestCase):

    @pytest.mark.slow
    def test_generalized_nonsq_cases(self):
        self.check_cases(require={'generalized', 'nonsquare'},
                         exclude={'size-0'})

    @pytest.mark.slow
    def test_generalized_empty_nonsq_cases(self):
        self.check_cases(require={'generalized', 'nonsquare', 'size-0'})


class HermitianGeneralizedTestCase(LinalgTestCase):

    @pytest.mark.slow
    def test_generalized_herm_cases(self):
        self.check_cases(require={'generalized', 'hermitian'},
                         exclude={'size-0'})

    @pytest.mark.slow
    def test_generalized_empty_herm_cases(self):
        self.check_cases(require={'generalized', 'hermitian', 'size-0'},
                         exclude={'none'})


def dot_generalized(a, b):
    a = asarray(a)
    if a.ndim >= 3:
        if a.ndim == b.ndim:
            # matrix x matrix
            new_shape = a.shape[:-1] + b.shape[-1:]
        elif a.ndim == b.ndim + 1:
            # matrix x vector
            new_shape = a.shape[:-1]
        else:
            raise ValueError("Not implemented...")
        r = np.empty(new_shape, dtype=np.common_type(a, b))
        for c in itertools.product(*map(range, a.shape[:-2])):
            r[c] = dot(a[c], b[c])
        return r
    else:
        return dot(a, b)


def identity_like_generalized(a):
    a = asarray(a)
    if a.ndim >= 3:
        r = np.empty(a.shape, dtype=a.dtype)
        r[...] = identity(a.shape[-2])
        return r
    else:
        return identity(a.shape[0])


class SolveCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # kept apart from TestSolve for use for testing with matrices.
    def do(self, a, b, tags):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot_generalized(a, x))
        assert_(consistent_subclass(x, b))


class TestSolve(SolveCases):
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.solve(x, x).dtype, dtype)

    def test_0_size(self):
        class ArraySubclass(np.ndarray):
            pass
        # Test system of 0x0 matrices
        a = np.arange(8).reshape(2, 2, 2)
        b = np.arange(6).reshape(1, 2, 3).view(ArraySubclass)

        expected = linalg.solve(a, b)[:, 0:0, :]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, :])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # Test errors for non-square and only b's dimension being 0
        assert_raises(linalg.LinAlgError, linalg.solve, a[:, 0:0, 0:1], b)
        assert_raises(ValueError, linalg.solve, a, b[:, 0:0, :])

        # Test broadcasting error
        b = np.arange(6).reshape(1, 3, 2)  # broadcasting error
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])

        # Test zero "single equations" with 0x0 matrices.
        b = np.arange(2).reshape(1, 2).view(ArraySubclass)
        expected = linalg.solve(a, b)[:, 0:0]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        b = np.arange(3).reshape(1, 3)
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])
        assert_raises(ValueError, linalg.solve, a[:, 0:0, 0:0], b)

    def test_0_size_k(self):
        # test zero multiple equation (K=0) case.
        class ArraySubclass(np.ndarray):
            pass
        a = np.arange(4).reshape(1, 2, 2)
        b = np.arange(6).reshape(3, 2, 1).view(ArraySubclass)

        expected = linalg.solve(a, b)[:, :, 0:0]
        result = linalg.solve(a, b[:, :, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # test both zero.
        expected = linalg.solve(a, b)[:, 0:0, 0:0]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))


class InvCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        a_inv = linalg.inv(a)
        assert_almost_equal(dot_generalized(a, a_inv),
                            identity_like_generalized(a))
        assert_(consistent_subclass(a_inv, a))


class TestInv(InvCases):
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.inv(x).dtype, dtype)

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.inv(a)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res.shape)
        assert_(isinstance(res, ArraySubclass))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.inv(a)
        assert_(res.dtype.type is np.complex64)
        assert_equal(a.shape, res.shape)
        assert_(isinstance(res, ArraySubclass))


class EigvalsCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        ev = linalg.eigvals(a)
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(ev, evalues)


class TestEigvals(EigvalsCases):
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.eigvals(x).dtype, dtype)
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        assert_equal(linalg.eigvals(x).dtype, get_complex_dtype(dtype))

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.eigvals(a)
        assert_(res.dtype.type is np.float64)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.eigvals(a)
        assert_(res.dtype.type is np.complex64)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))


class EigCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        evalues, evectors = linalg.eig(a)
        assert_allclose(dot_generalized(a, evectors),
                        np.asarray(evectors) * np.asarray(evalues)[..., None, :],
                        rtol=get_rtol(evalues.dtype))
        assert_(consistent_subclass(evectors, a))


class TestEig(EigCases):
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w, v = np.linalg.eig(x)
        assert_equal(w.dtype, dtype)
        assert_equal(v.dtype, dtype)

        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        w, v = np.linalg.eig(x)
        assert_equal(w.dtype, get_complex_dtype(dtype))
        assert_equal(v.dtype, get_complex_dtype(dtype))

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res, res_v = linalg.eig(a)
        assert_(res_v.dtype.type is np.float64)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res, res_v = linalg.eig(a)
        assert_(res_v.dtype.type is np.complex64)
        assert_(res.dtype.type is np.complex64)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))


class SVDBaseTests:
    hermitian = False

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        u, s, vh = linalg.svd(x)
        assert_equal(u.dtype, dtype)
        assert_equal(s.dtype, get_real_dtype(dtype))
        assert_equal(vh.dtype, dtype)
        s = linalg.svd(x, compute_uv=False, hermitian=self.hermitian)
        assert_equal(s.dtype, get_real_dtype(dtype))


class SVDCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False)
        assert_allclose(a, dot_generalized(np.asarray(u) * np.asarray(s)[..., None, :],
                                           np.asarray(vt)),
                        rtol=get_rtol(u.dtype))
        assert_(consistent_subclass(u, a))
        assert_(consistent_subclass(vt, a))


class TestSVD(SVDCases, SVDBaseTests):
    def test_empty_identity(self):
        """ Empty input should put an identity matrix in u or vh """
        x = np.empty((4, 0))
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        assert_equal(u.shape, (4, 4))
        assert_equal(vh.shape, (0, 0))
        assert_equal(u, np.eye(4))

        x = np.empty((0, 4))
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        assert_equal(u.shape, (0, 0))
        assert_equal(vh.shape, (4, 4))
        assert_equal(vh, np.eye(4))


class SVDHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False, hermitian=True)
        assert_allclose(a, dot_generalized(np.asarray(u) * np.asarray(s)[..., None, :],
                                           np.asarray(vt)),
                        rtol=get_rtol(u.dtype))
        def hermitian(mat):
            axes = list(range(mat.ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return np.conj(np.transpose(mat, axes=axes))

        assert_almost_equal(np.matmul(u, hermitian(u)), np.broadcast_to(np.eye(u.shape[-1]), u.shape))
        assert_almost_equal(np.matmul(vt, hermitian(vt)), np.broadcast_to(np.eye(vt.shape[-1]), vt.shape))
        assert_equal(np.sort(s)[..., ::-1], s)
        assert_(consistent_subclass(u, a))
        assert_(consistent_subclass(vt, a))


class TestSVDHermitian(SVDHermitianCases, SVDBaseTests):
    hermitian = True


class CondCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # cond(x, p) for p in (None, 2, -2)

    def do(self, a, b, tags):
        c = asarray(a)  # a might be a matrix
        if 'size-0' in tags:
            assert_raises(LinAlgError, linalg.cond, c)
            return

        # +-2 norms
        s = linalg.svd(c, compute_uv=False)
        assert_almost_equal(
            linalg.cond(a), s[..., 0] / s[..., -1],
            single_decimal=5, double_decimal=11)
        assert_almost_equal(
            linalg.cond(a, 2), s[..., 0] / s[..., -1],
            single_decimal=5, double_decimal=11)
        assert_almost_equal(
            linalg.cond(a, -2), s[..., -1] / s[..., 0],
            single_decimal=5, double_decimal=11)

        # Other norms
        cinv = np.linalg.inv(c)
        assert_almost_equal(
            linalg.cond(a, 1),
            abs(c).sum(-2).max(-1) * abs(cinv).sum(-2).max(-1),
            single_decimal=5, double_decimal=11)
        assert_almost_equal(
            linalg.cond(a, -1),
            abs(c).sum(-2).min(-1) * abs(cinv).sum(-2).min(-1),
            single_decimal=5, double_decimal=11)
        assert_almost_equal(
            linalg.cond(a, np.inf),
            abs(c).sum(-1).max(-1) * abs(cinv).sum(-1).max(-1),
            single_decimal=5, double_decimal=11)
        assert_almost_equal(
            linalg.cond(a, -np.inf),
            abs(c).sum(-1).min(-1) * abs(cinv).sum(-1).min(-1),
            single_decimal=5, double_decimal=11)
        assert_almost_equal(
            linalg.cond(a, 'fro'),
            np.sqrt((abs(c)**2).sum(-1).sum(-1)
                    * (abs(cinv)**2).sum(-1).sum(-1)),
            single_decimal=5, double_decimal=11)


class TestCond(CondCases):
    def test_basic_nonsvd(self):
        # Smoketest the non-svd norms
        A = array([[1., 0, 1], [0, -2., 0], [0, 0, 3.]])
        assert_almost_equal(linalg.cond(A, inf), 4)
        assert_almost_equal(linalg.cond(A, -inf), 2/3)
        assert_almost_equal(linalg.cond(A, 1), 4)
        assert_almost_equal(linalg.cond(A, -1), 0.5)
        assert_almost_equal(linalg.cond(A, 'fro'), np.sqrt(265 / 12))

    def test_singular(self):
        # Singular matrices have infinite condition number for
        # positive norms, and negative norms shouldn't raise
        # exceptions
        As = [np.zeros((2, 2)), np.ones((2, 2))]
        p_pos = [None, 1, 2, 'fro']
        p_neg = [-1, -2]
        for A, p in itertools.product(As, p_pos):
            # Inversion may not hit exact infinity, so just check the
            # number is large
            assert_(linalg.cond(A, p) > 1e15)
        for A, p in itertools.product(As, p_neg):
            linalg.cond(A, p)

    @pytest.mark.xfail(True, run=False,
                       reason="Platform/LAPACK-dependent failure, "
                              "see gh-18914")
    def test_nan(self):
        # nans should be passed through, not converted to infs
        ps = [None, 1, -1, 2, -2, 'fro']
        p_pos = [None, 1, 2, 'fro']

        A = np.ones((2, 2))
        A[0,1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(isinstance(c, np.float_))
            assert_(np.isnan(c))

        A = np.ones((3, 2, 2))
        A[1,0,1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(np.isnan(c[1]))
            if p in p_pos:
                assert_(c[0] > 1e15)
                assert_(c[2] > 1e15)
            else:
                assert_(not np.isnan(c[0]))
                assert_(not np.isnan(c[2]))

    def test_stacked_singular(self):
        # Check behavior when only some of the stacked matrices are
        # singular
        np.random.seed(1234)
        A = np.random.rand(2, 2, 2, 2)
        A[0,0] = 0
        A[1,1] = 0

        for p in (None, 1, 2, 'fro', -1, -2):
            c = linalg.cond(A, p)
            assert_equal(c[0,0], np.inf)
            assert_equal(c[1,1], np.inf)
            assert_(np.isfinite(c[0,1]))
            assert_(np.isfinite(c[1,0]))


class PinvCases(LinalgSquareTestCase,
                LinalgNonsquareTestCase,
                LinalgGeneralizedSquareTestCase,
                LinalgGeneralizedNonsquareTestCase):

    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a)
        # `a @ a_ginv == I` does not hold if a is singular
        dot = dot_generalized
        assert_almost_equal(dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11)
        assert_(consistent_subclass(a_ginv, a))


class TestPinv(PinvCases):
    pass


class PinvHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a, hermitian=True)
        # `a @ a_ginv == I` does not hold if a is singular
        dot = dot_generalized
        assert_almost_equal(dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11)
        assert_(consistent_subclass(a_ginv, a))


class TestPinvHermitian(PinvHermitianCases):
    pass


class DetCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        d = linalg.det(a)
        (s, ld) = linalg.slogdet(a)
        if asarray(a).dtype.type in (single, double):
            ad = asarray(a).astype(double)
        else:
            ad = asarray(a).astype(cdouble)
        ev = linalg.eigvals(ad)
        assert_almost_equal(d, multiply.reduce(ev, axis=-1))
        assert_almost_equal(s * np.exp(ld), multiply.reduce(ev, axis=-1))

        s = np.atleast_1d(s)
        ld = np.atleast_1d(ld)
        m = (s != 0)
        assert_almost_equal(np.abs(s[m]), 1)
        assert_equal(ld[~m], -inf)


class TestDet(DetCases):
    def test_zero(self):
        assert_equal(linalg.det([[0.0]]), 0.0)
        assert_equal(type(linalg.det([[0.0]])), double)
        assert_equal(linalg.det([[0.0j]]), 0.0)
        assert_equal(type(linalg.det([[0.0j]])), cdouble)

        assert_equal(linalg.slogdet([[0.0]]), (0.0, -inf))
        assert_equal(type(linalg.slogdet([[0.0]])[0]), double)
        assert_equal(type(linalg.slogdet([[0.0]])[1]), double)
        assert_equal(linalg.slogdet([[0.0j]]), (0.0j, -inf))
        assert_equal(type(linalg.slogdet([[0.0j]])[0]), cdouble)
        assert_equal(type(linalg.slogdet([[0.0j]])[1]), double)

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(np.linalg.det(x).dtype, dtype)
        ph, s = np.linalg.slogdet(x)
        assert_equal(s.dtype, get_real_dtype(dtype))
        assert_equal(ph.dtype, dtype)

    def test_0_size(self):
        a = np.zeros((0, 0), dtype=np.complex64)
        res = linalg.det(a)
        assert_equal(res, 1.)
        assert_(res.dtype.type is np.complex64)
        res = linalg.slogdet(a)
        assert_equal(res, (1, 0))
        assert_(res[0].dtype.type is np.complex64)
        assert_(res[1].dtype.type is np.float32)

        a = np.zeros((0, 0), dtype=np.float64)
        res = linalg.det(a)
        assert_equal(res, 1.)
        assert_(res.dtype.type is np.float64)
        res = linalg.slogdet(a)
        assert_equal(res, (1, 0))
        assert_(res[0].dtype.type is np.float64)
        assert_(res[1].dtype.type is np.float64)


class LstsqCases(LinalgSquareTestCase, LinalgNonsquareTestCase):

    def do(self, a, b, tags):
        arr = np.asarray(a)
        m, n = arr.shape
        u, s, vt = linalg.svd(a, False)
        x, residuals, rank, sv = linalg.lstsq(a, b, rcond=-1)
        if m == 0:
            assert_((x == 0).all())
        if m <= n:
            assert_almost_equal(b, dot(a, x))
            assert_equal(rank, m)
        else:
            assert_equal(rank, n)
        assert_almost_equal(sv, sv.__array_wrap__(s))
        if rank == n and m > n:
            expect_resids = (
                np.asarray(abs(np.dot(a, x) - b)) ** 2).sum(axis=0)
            expect_resids = np.asarray(expect_resids)
            if np.asarray(b).ndim == 1:
                expect_resids.shape = (1,)
                assert_equal(residuals.shape, expect_resids.shape)
        else:
            expect_resids = np.array([]).view(type(x))
        assert_almost_equal(residuals, expect_resids)
        assert_(np.issubdtype(residuals.dtype, np.floating))
        assert_(consistent_subclass(x, b))
        assert_(consistent_subclass(residuals, b))


class TestLstsq(LstsqCases):
    def test_future_rcond(self):
        a = np.array([[0., 1.,  0.,  1.,  2.,  0.],
                      [0., 2.,  0.,  0.,  1.,  0.],
                      [1., 0.,  1.,  0.,  0.,  4.],
                      [0., 0.,  0.,  2.,  3.,  0.]]).T

        b = np.array([1, 0, 0, 0, 0, 0])
        with suppress_warnings() as sup:
            w = sup.record(FutureWarning, "`rcond` parameter will change")
            x, residuals, rank, s = linalg.lstsq(a, b)
            assert_(rank == 4)
            x, residuals, rank, s = linalg.lstsq(a, b, rcond=-1)
            assert_(rank == 4)
            x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
            assert_(rank == 3)
            # Warning should be raised exactly once (first command)
            assert_(len(w) == 1)

    @pytest.mark.parametrize(["m", "n", "n_rhs"], [
        (4, 2, 2),
        (0, 4, 1),
        (0, 4, 2),
        (4, 0, 1),
        (4, 0, 2),
        (4, 2, 0),
        (0, 0, 0)
    ])
    def test_empty_a_b(self, m, n, n_rhs):
        a = np.arange(m * n).reshape(m, n)
        b = np.ones((m, n_rhs))
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
        if m == 0:
            assert_((x == 0).all())
        assert_equal(x.shape, (n, n_rhs))
        assert_equal(residuals.shape, ((n_rhs,) if m > n else (0,)))
        if m > n and n_rhs > 0:
            # residuals are exactly the squared norms of b's columns
            r = b - np.dot(a, x)
            assert_almost_equal(residuals, (r * r).sum(axis=-2))
        assert_equal(rank, min(m, n))
        assert_equal(s.shape, (min(m, n),))

    def test_incompatible_dims(self):
        # use modified version of docstring example
        x = np.array([0, 1, 2, 3])
        y = np.array([-1, 0.2, 0.9, 2.1, 3.3])
        A = np.vstack([x, np.ones(len(x))]).T
        with assert_raises_regex(LinAlgError, "Incompatible dimensions"):
            linalg.lstsq(A, y, rcond=None)


@pytest.mark.parametrize('dt', [np.dtype(c) for c in '?bBhHiIqQefdgFDGO'])
class TestMatrixPower:

    rshft_0 = np.eye(4)
    rshft_1 = rshft_0[[3, 0, 1, 2]]
    rshft_2 = rshft_0[[2, 3, 0, 1]]
    rshft_3 = rshft_0[[1, 2, 3, 0]]
    rshft_all = [rshft_0, rshft_1, rshft_2, rshft_3]
    noninv = array([[1, 0], [0, 0]])
    stacked = np.block([[[rshft_0]]]*2)
    #FIXME the 'e' dtype might work in future
    dtnoinv = [object, np.dtype('e'), np.dtype('g'), np.dtype('G')]

    def test_large_power(self, dt):
        rshft = self.rshft_1.astype(dt)
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 0), self.rshft_0)
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 1), self.rshft_1)
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 2), self.rshft_2)
        assert_equal(
            matrix_power(rshft, 2**100 + 2**10 + 2**5 + 3), self.rshft_3)

    def test_power_is_zero(self, dt):
        def tz(M):
            mz = matrix_power(M, 0)
            assert_equal(mz, identity_like_generalized(M))
            assert_equal(mz.dtype, M.dtype)

        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt != object:
                tz(self.stacked.astype(dt))

    def test_power_is_one(self, dt):
        def tz(mat):
            mz = matrix_power(mat, 1)
            assert_equal(mz, mat)
            assert_equal(mz.dtype, mat.dtype)

        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt != object:
                tz(self.stacked.astype(dt))

    def test_power_is_two(self, dt):
        def tz(mat):
            mz = matrix_power(mat, 2)
            mmul = matmul if mat.dtype != object else dot
            assert_equal(mz, mmul(mat, mat))
            assert_equal(mz.dtype, mat.dtype)

        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt != object:
                tz(self.stacked.astype(dt))

    def test_power_is_minus_one(self, dt):
        def tz(mat):
            invmat = matrix_power(mat, -1)
            mmul = matmul if mat.dtype != object else dot
            assert_almost_equal(
                mmul(invmat, mat), identity_like_generalized(mat))

        for mat in self.rshft_all:
            if dt not in self.dtnoinv:
                tz(mat.astype(dt))

    def test_exceptions_bad_power(self, dt):
        mat = self.rshft_0.astype(dt)
        assert_raises(TypeError, matrix_power, mat, 1.5)
        assert_raises(TypeError, matrix_power, mat, [1])

    def test_exceptions_non_square(self, dt):
        assert_raises(LinAlgError, matrix_power, np.array([1], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.array([[1], [2]], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.ones((4, 3, 2), dt), 1)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_exceptions_not_invertible(self, dt):
        if dt in self.dtnoinv:
            return
        mat = self.noninv.astype(dt)
        assert_raises(LinAlgError, matrix_power, mat, -1)


class TestEigvalshCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        # note that eigenvalue arrays returned by eig must be sorted since
        # their order isn't guaranteed.
        ev = linalg.eigvalsh(a, 'L')
        evalues, evectors = linalg.eig(a)
        evalues.sort(axis=-1)
        assert_allclose(ev, evalues, rtol=get_rtol(ev.dtype))

        ev2 = linalg.eigvalsh(a, 'U')
        assert_allclose(ev2, evalues, rtol=get_rtol(ev.dtype))


class TestEigvalsh:
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w = np.linalg.eigvalsh(x)
        assert_equal(w.dtype, get_real_dtype(dtype))

    def test_invalid(self):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        assert_raises(ValueError, np.linalg.eigvalsh, x, UPLO="lrong")
        assert_raises(ValueError, np.linalg.eigvalsh, x, "lower")
        assert_raises(ValueError, np.linalg.eigvalsh, x, "upper")

    def test_UPLO(self):
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)

        # Check default is 'L'
        w = np.linalg.eigvalsh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'L'
        w = np.linalg.eigvalsh(Klo, UPLO='L')
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'l'
        w = np.linalg.eigvalsh(Klo, UPLO='l')
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'U'
        w = np.linalg.eigvalsh(Kup, UPLO='U')
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'u'
        w = np.linalg.eigvalsh(Kup, UPLO='u')
        assert_allclose(w, tgt, rtol=rtol)

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.eigvalsh(a)
        assert_(res.dtype.type is np.float64)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.eigvalsh(a)
        assert_(res.dtype.type is np.float32)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))


class TestEighCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        # note that eigenvalue arrays returned by eig must be sorted since
        # their order isn't guaranteed.
        ev, evc = linalg.eigh(a)
        evalues, evectors = linalg.eig(a)
        evalues.sort(axis=-1)
        assert_almost_equal(ev, evalues)

        assert_allclose(dot_generalized(a, evc),
                        np.asarray(ev)[..., None, :] * np.asarray(evc),
                        rtol=get_rtol(ev.dtype))

        ev2, evc2 = linalg.eigh(a, 'U')
        assert_almost_equal(ev2, evalues)

        assert_allclose(dot_generalized(a, evc2),
                        np.asarray(ev2)[..., None, :] * np.asarray(evc2),
                        rtol=get_rtol(ev.dtype), err_msg=repr(a))


class TestEigh:
    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w, v = np.linalg.eigh(x)
        assert_equal(w.dtype, get_real_dtype(dtype))
        assert_equal(v.dtype, dtype)

    def test_invalid(self):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        assert_raises(ValueError, np.linalg.eigh, x, UPLO="lrong")
        assert_raises(ValueError, np.linalg.eigh, x, "lower")
        assert_raises(ValueError, np.linalg.eigh, x, "upper")

    def test_UPLO(self):
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)

        # Check default is 'L'
        w, v = np.linalg.eigh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'L'
        w, v = np.linalg.eigh(Klo, UPLO='L')
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'l'
        w, v = np.linalg.eigh(Klo, UPLO='l')
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'U'
        w, v = np.linalg.eigh(Kup, UPLO='U')
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'u'
        w, v = np.linalg.eigh(Kup, UPLO='u')
        assert_allclose(w, tgt, rtol=rtol)

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res, res_v = linalg.eigh(a)
        assert_(res_v.dtype.type is np.float64)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res, res_v = linalg.eigh(a)
        assert_(res_v.dtype.type is np.complex64)
        assert_(res.dtype.type is np.float32)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))


class _TestNormBase:
    dt = None
    dec = None

    @staticmethod
    def check_dtype(x, res):
        if issubclass(x.dtype.type, np.inexact):
            assert_equal(res.dtype, x.real.dtype)
        else:
            # For integer input, don't have to test float precision of output.
            assert_(issubclass(res.dtype.type, np.floating))


class _TestNormGeneral(_TestNormBase):

    def test_empty(self):
        assert_equal(norm([]), 0.0)
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)

    def test_vector_return_type(self):
        a = np.array([1, 0, 1])

        exact_types = np.typecodes['AllInteger']
        inexact_types = np.typecodes['AllFloat']

        all_types = exact_types + inexact_types

        for each_type in all_types:
            at = a.astype(each_type)

            an = norm(at, -np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 0.0)

            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                an = norm(at, -1)
                self.check_dtype(at, an)
                assert_almost_equal(an, 0.0)

            an = norm(at, 0)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2)

            an = norm(at, 1)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, 2)
            self.check_dtype(at, an)
            assert_almost_equal(an, an.dtype.type(2.0)**an.dtype.type(1.0/2.0))

            an = norm(at, 4)
            self.check_dtype(at, an)
            assert_almost_equal(an, an.dtype.type(2.0)**an.dtype.type(1.0/4.0))

            an = norm(at, np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 1.0)

    def test_vector(self):
        a = [1, 2, 3, 4]
        b = [-1, -2, -3, -4]
        c = [-1, 2, -3, 4]

        def _test(v):
            np.testing.assert_almost_equal(norm(v), 30 ** 0.5,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, inf), 4.0,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -inf), 1.0,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 1), 10.0,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -1), 12.0 / 25,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 2), 30 ** 0.5,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -2), ((205. / 144) ** -0.5),
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 0), 4,
                                           decimal=self.dec)

        for v in (a, b, c,):
            _test(v)

        for v in (array(a, dtype=self.dt), array(b, dtype=self.dt),
                  array(c, dtype=self.dt)):
            _test(v)

    def test_axis(self):
        # Vector norms.
        # Compare the use of `axis` with computing the norm of each row
        # or column separately.
        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        for order in [None, -1, 0, 1, 2, 3, np.Inf, -np.Inf]:
            expected0 = [norm(A[:, k], ord=order) for k in range(A.shape[1])]
            assert_almost_equal(norm(A, ord=order, axis=0), expected0)
            expected1 = [norm(A[k, :], ord=order) for k in range(A.shape[0])]
            assert_almost_equal(norm(A, ord=order, axis=1), expected1)

        # Matrix norms.
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)
        nd = B.ndim
        for order in [None, -2, 2, -1, 1, np.Inf, -np.Inf, 'fro']:
            for axis in itertools.combinations(range(-nd, nd), 2):
                row_axis, col_axis = axis
                if row_axis < 0:
                    row_axis += nd
                if col_axis < 0:
                    col_axis += nd
                if row_axis == col_axis:
                    assert_raises(ValueError, norm, B, ord=order, axis=axis)
                else:
                    n = norm(B, ord=order, axis=axis)

                    # The logic using k_index only works for nd = 3.
                    # This has to be changed if nd is increased.
                    k_index = nd - (row_axis + col_axis)
                    if row_axis < col_axis:
                        expected = [norm(B[:].take(k, axis=k_index), ord=order)
                                    for k in range(B.shape[k_index])]
                    else:
                        expected = [norm(B[:].take(k, axis=k_index).T, ord=order)
                                    for k in range(B.shape[k_index])]
                    assert_almost_equal(n, expected)

    def test_keepdims(self):
        A = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        allclose_err = 'order {0}, axis = {1}'
        shape_err = 'Shape mismatch found {0}, expected {1}, order={2}, axis={3}'

        # check the order=None, axis=None case
        expected = norm(A, ord=None, axis=None)
        found = norm(A, ord=None, axis=None, keepdims=True)
        assert_allclose(np.squeeze(found), expected,
                        err_msg=allclose_err.format(None, None))
        expected_shape = (1, 1, 1)
        assert_(found.shape == expected_shape,
                shape_err.format(found.shape, expected_shape, None, None))

        # Vector norms.
        for order in [None, -1, 0, 1, 2, 3, np.Inf, -np.Inf]:
            for k in range(A.ndim):
                expected = norm(A, ord=order, axis=k)
                found = norm(A, ord=order, axis=k, keepdims=True)
                assert_allclose(np.squeeze(found), expected,
                                err_msg=allclose_err.format(order, k))
                expected_shape = list(A.shape)
                expected_shape[k] = 1
                expected_shape = tuple(expected_shape)
                assert_(found.shape == expected_shape,
                        shape_err.format(found.shape, expected_shape, order, k))

        # Matrix norms.
        for order in [None, -2, 2, -1, 1, np.Inf, -np.Inf, 'fro', 'nuc']:
            for k in itertools.permutations(range(A.ndim), 2):
                expected = norm(A, ord=order, axis=k)
                found = norm(A, ord=order, axis=k, keepdims=True)
                assert_allclose(np.squeeze(found), expected,
                                err_msg=allclose_err.format(order, k))
                expected_shape = list(A.shape)
                expected_shape[k[0]] = 1
                expected_shape[k[1]] = 1
                expected_shape = tuple(expected_shape)
                assert_(found.shape == expected_shape,
                        shape_err.format(found.shape, expected_shape, order, k))


class _TestNorm2D(_TestNormBase):
    # Define the part for 2d arrays separately, so we can subclass this
    # and run the tests using np.matrix in matrixlib.tests.test_matrix_linalg.
    array = np.array

    def test_matrix_empty(self):
        assert_equal(norm(self.array([[]], dtype=self.dt)), 0.0)

    def test_matrix_return_type(self):
        a = self.array([[1, 0, 1], [0, 1, 1]])

        exact_types = np.typecodes['AllInteger']

        # float32, complex64, float64, complex128 types are the only types
        # allowed by `linalg`, which performs the matrix operations used
        # within `norm`.
        inexact_types = 'fdFD'

        all_types = exact_types + inexact_types

        for each_type in all_types:
            at = a.astype(each_type)

            an = norm(at, -np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                an = norm(at, -1)
                self.check_dtype(at, an)
                assert_almost_equal(an, 1.0)

            an = norm(at, 1)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, 2)
            self.check_dtype(at, an)
            assert_almost_equal(an, 3.0**(1.0/2.0))

            an = norm(at, -2)
            self.check_dtype(at, an)
            assert_almost_equal(an, 1.0)

            an = norm(at, np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, 'fro')
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, 'nuc')
            self.check_dtype(at, an)
            # Lower bar needed to support low precision floats.
            # They end up being off by 1 in the 7th place.
            np.testing.assert_almost_equal(an, 2.7320508075688772, decimal=6)

    def test_matrix_2x2(self):
        A = self.array([[1, 3], [5, 7]], dtype=self.dt)
        assert_almost_equal(norm(A), 84 ** 0.5)
        assert_almost_equal(norm(A, 'fro'), 84 ** 0.5)
        assert_almost_equal(norm(A, 'nuc'), 10.0)
        assert_almost_equal(norm(A, inf), 12.0)
        assert_almost_equal(norm(A, -inf), 4.0)
        assert_almost_equal(norm(A, 1), 10.0)
        assert_almost_equal(norm(A, -1), 6.0)
        assert_almost_equal(norm(A, 2), 9.1231056256176615)
        assert_almost_equal(norm(A, -2), 0.87689437438234041)

        assert_raises(ValueError, norm, A, 'nofro')
        assert_raises(ValueError, norm, A, -3)
        assert_raises(ValueError, norm, A, 0)

    def test_matrix_3x3(self):
        # This test has been added because the 2x2 example
        # happened to have equal nuclear norm and induced 1-norm.
        # The 1/10 scaling factor accommodates the absolute tolerance
        # used in assert_almost_equal.
        A = (1 / 10) * \
            self.array([[1, 2, 3], [6, 0, 5], [3, 2, 1]], dtype=self.dt)
        assert_almost_equal(norm(A), (1 / 10) * 89 ** 0.5)
        assert_almost_equal(norm(A, 'fro'), (1 / 10) * 89 ** 0.5)
        assert_almost_equal(norm(A, 'nuc'), 1.3366836911774836)
        assert_almost_equal(norm(A, inf), 1.1)
        assert_almost_equal(norm(A, -inf), 0.6)
        assert_almost_equal(norm(A, 1), 1.0)
        assert_almost_equal(norm(A, -1), 0.4)
        assert_almost_equal(norm(A, 2), 0.88722940323461277)
        assert_almost_equal(norm(A, -2), 0.19456584790481812)

    def test_bad_args(self):
        # Check that bad arguments raise the appropriate exceptions.

        A = self.array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        # Using `axis=<integer>` or passing in a 1-D array implies vector
        # norms are being computed, so also using `ord='fro'`
        # or `ord='nuc'` or any other string raises a ValueError.
        assert_raises(ValueError, norm, A, 'fro', 0)
        assert_raises(ValueError, norm, A, 'nuc', 0)
        assert_raises(ValueError, norm, [3, 4], 'fro', None)
        assert_raises(ValueError, norm, [3, 4], 'nuc', None)
        assert_raises(ValueError, norm, [3, 4], 'test', None)

        # Similarly, norm should raise an exception when ord is any finite
        # number other than 1, 2, -1 or -2 when computing matrix norms.
        for order in [0, 3]:
            assert_raises(ValueError, norm, A, order, None)
            assert_raises(ValueError, norm, A, order, (0, 1))
            assert_raises(ValueError, norm, B, order, (1, 2))

        # Invalid axis
        assert_raises(np.AxisError, norm, B, None, 3)
        assert_raises(np.AxisError, norm, B, None, (2, 3))
        assert_raises(ValueError, norm, B, None, (0, 1, 2))


class _TestNorm(_TestNorm2D, _TestNormGeneral):
    pass


class TestNorm_NonSystematic:

    def test_longdouble_norm(self):
        # Non-regression test: p-norm of longdouble would previously raise
        # UnboundLocalError.
        x = np.arange(10, dtype=np.longdouble)
        old_assert_almost_equal(norm(x, ord=3), 12.65, decimal=2)

    def test_intmin(self):
        # Non-regression test: p-norm of signed integer would previously do
        # float cast and abs in the wrong order.
        x = np.array([-2 ** 31], dtype=np.int32)
        old_assert_almost_equal(norm(x, ord=3), 2 ** 31, decimal=5)

    def test_complex_high_ord(self):
        # gh-4156
        d = np.empty((2,), dtype=np.clongdouble)
        d[0] = 6 + 7j
        d[1] = -6 + 7j
        res = 11.615898132184
        old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=10)
        d = d.astype(np.complex128)
        old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=9)
        d = d.astype(np.complex64)
        old_assert_almost_equal(np.linalg.norm(d, ord=3), res, decimal=5)


# Separate definitions so we can use them for matrix tests.
class _TestNormDoubleBase(_TestNormBase):
    dt = np.double
    dec = 12


class _TestNormSingleBase(_TestNormBase):
    dt = np.float32
    dec = 6


class _TestNormInt64Base(_TestNormBase):
    dt = np.int64
    dec = 12


class TestNormDouble(_TestNorm, _TestNormDoubleBase):
    pass


class TestNormSingle(_TestNorm, _TestNormSingleBase):
    pass


class TestNormInt64(_TestNorm, _TestNormInt64Base):
    pass


class TestMatrixRank:

    def test_matrix_rank(self):
        # Full rank matrix
        assert_equal(4, matrix_rank(np.eye(4)))
        # rank deficient matrix
        I = np.eye(4)
        I[-1, -1] = 0.
        assert_equal(matrix_rank(I), 3)
        # All zeros - zero rank
        assert_equal(matrix_rank(np.zeros((4, 4))), 0)
        # 1 dimension - rank 1 unless all 0
        assert_equal(matrix_rank([1, 0, 0, 0]), 1)
        assert_equal(matrix_rank(np.zeros((4,))), 0)
        # accepts array-like
        assert_equal(matrix_rank([1]), 1)
        # greater than 2 dimensions treated as stacked matrices
        ms = np.array([I, np.eye(4), np.zeros((4,4))])
        assert_equal(matrix_rank(ms), np.array([3, 4, 0]))
        # works on scalar
        assert_equal(matrix_rank(1), 1)

    def test_symmetric_rank(self):
        assert_equal(4, matrix_rank(np.eye(4), hermitian=True))
        assert_equal(1, matrix_rank(np.ones((4, 4)), hermitian=True))
        assert_equal(0, matrix_rank(np.zeros((4, 4)), hermitian=True))
        # rank deficient matrix
        I = np.eye(4)
        I[-1, -1] = 0.
        assert_equal(3, matrix_rank(I, hermitian=True))
        # manually supplied tolerance
        I[-1, -1] = 1e-8
        assert_equal(4, matrix_rank(I, hermitian=True, tol=0.99e-8))
        assert_equal(3, matrix_rank(I, hermitian=True, tol=1.01e-8))


def test_reduced_rank():
    # Test matrices with reduced rank
    rng = np.random.RandomState(20120714)
    for i in range(100):
        # Make a rank deficient matrix
        X = rng.normal(size=(40, 10))
        X[:, 0] = X[:, 1] + X[:, 2]
        # Assert that matrix_rank detected deficiency
        assert_equal(matrix_rank(X), 9)
        X[:, 3] = X[:, 4] + X[:, 5]
        assert_equal(matrix_rank(X), 8)


class TestQR:
    # Define the array class here, so run this on matrices elsewhere.
    array = np.array

    def check_qr(self, a):
        # This test expects the argument `a` to be an ndarray or
        # a subclass of an ndarray of inexact type.
        a_type = type(a)
        a_dtype = a.dtype
        m, n = a.shape
        k = min(m, n)

        # mode == 'complete'
        q, r = linalg.qr(a, mode='complete')
        assert_(q.dtype == a_dtype)
        assert_(r.dtype == a_dtype)
        assert_(isinstance(q, a_type))
        assert_(isinstance(r, a_type))
        assert_(q.shape == (m, m))
        assert_(r.shape == (m, n))
        assert_almost_equal(dot(q, r), a)
        assert_almost_equal(dot(q.T.conj(), q), np.eye(m))
        assert_almost_equal(np.triu(r), r)

        # mode == 'reduced'
        q1, r1 = linalg.qr(a, mode='reduced')
        assert_(q1.dtype == a_dtype)
        assert_(r1.dtype == a_dtype)
        assert_(isinstance(q1, a_type))
        assert_(isinstance(r1, a_type))
        assert_(q1.shape == (m, k))
        assert_(r1.shape == (k, n))
        assert_almost_equal(dot(q1, r1), a)
        assert_almost_equal(dot(q1.T.conj(), q1), np.eye(k))
        assert_almost_equal(np.triu(r1), r1)

        # mode == 'r'
        r2 = linalg.qr(a, mode='r')
        assert_(r2.dtype == a_dtype)
        assert_(isinstance(r2, a_type))
        assert_almost_equal(r2, r1)


    @pytest.mark.parametrize(["m", "n"], [
        (3, 0),
        (0, 3),
        (0, 0)
    ])
    def test_qr_empty(self, m, n):
        k = min(m, n)
        a = np.empty((m, n))

        self.check_qr(a)

        h, tau = np.linalg.qr(a, mode='raw')
        assert_equal(h.dtype, np.double)
        assert_equal(tau.dtype, np.double)
        assert_equal(h.shape, (n, m))
        assert_equal(tau.shape, (k,))

    def test_mode_raw(self):
        # The factorization is not unique and varies between libraries,
        # so it is not possible to check against known values. Functional
        # testing is a possibility, but awaits the exposure of more
        # of the functions in lapack_lite. Consequently, this test is
        # very limited in scope. Note that the results are in FORTRAN
        # order, hence the h arrays are transposed.
        a = self.array([[1, 2], [3, 4], [5, 6]], dtype=np.double)

        # Test double
        h, tau = linalg.qr(a, mode='raw')
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        assert_(h.shape == (2, 3))
        assert_(tau.shape == (2,))

        h, tau = linalg.qr(a.T, mode='raw')
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        assert_(h.shape == (3, 2))
        assert_(tau.shape == (2,))

    def test_mode_all_but_economic(self):
        a = self.array([[1, 2], [3, 4]])
        b = self.array([[1, 2], [3, 4], [5, 6]])
        for dt in "fd":
            m1 = a.astype(dt)
            m2 = b.astype(dt)
            self.check_qr(m1)
            self.check_qr(m2)
            self.check_qr(m2.T)

        for dt in "fd":
            m1 = 1 + 1j * a.astype(dt)
            m2 = 1 + 1j * b.astype(dt)
            self.check_qr(m1)
            self.check_qr(m2)
            self.check_qr(m2.T)

    def check_qr_stacked(self, a):
        # This test expects the argument `a` to be an ndarray or
        # a subclass of an ndarray of inexact type.
        a_type = type(a)
        a_dtype = a.dtype
        m, n = a.shape[-2:]
        k = min(m, n)

        # mode == 'complete'
        q, r = linalg.qr(a, mode='complete')
        assert_(q.dtype == a_dtype)
        assert_(r.dtype == a_dtype)
        assert_(isinstance(q, a_type))
        assert_(isinstance(r, a_type))
        assert_(q.shape[-2:] == (m, m))
        assert_(r.shape[-2:] == (m, n))
        assert_almost_equal(matmul(q, r), a)
        I_mat = np.identity(q.shape[-1])
        stack_I_mat = np.broadcast_to(I_mat, 
                        q.shape[:-2] + (q.shape[-1],)*2)
        assert_almost_equal(matmul(swapaxes(q, -1, -2).conj(), q), stack_I_mat)
        assert_almost_equal(np.triu(r[..., :, :]), r)

        # mode == 'reduced'
        q1, r1 = linalg.qr(a, mode='reduced')
        assert_(q1.dtype == a_dtype)
        assert_(r1.dtype == a_dtype)
        assert_(isinstance(q1, a_type))
        assert_(isinstance(r1, a_type))
        assert_(q1.shape[-2:] == (m, k))
        assert_(r1.shape[-2:] == (k, n))
        assert_almost_equal(matmul(q1, r1), a)
        I_mat = np.identity(q1.shape[-1])
        stack_I_mat = np.broadcast_to(I_mat, 
                        q1.shape[:-2] + (q1.shape[-1],)*2)
        assert_almost_equal(matmul(swapaxes(q1, -1, -2).conj(), q1), 
                            stack_I_mat)
        assert_almost_equal(np.triu(r1[..., :, :]), r1)

        # mode == 'r'
        r2 = linalg.qr(a, mode='r')
        assert_(r2.dtype == a_dtype)
        assert_(isinstance(r2, a_type))
        assert_almost_equal(r2, r1)

    @pytest.mark.parametrize("size", [
        (3, 4), (4, 3), (4, 4), 
        (3, 0), (0, 3)])
    @pytest.mark.parametrize("outer_size", [
        (2, 2), (2,), (2, 3, 4)])
    @pytest.mark.parametrize("dt", [
        np.single, np.double, 
        np.csingle, np.cdouble])
    def test_stacked_inputs(self, outer_size, size, dt):

        A = np.random.normal(size=outer_size + size).astype(dt)
        B = np.random.normal(size=outer_size + size).astype(dt)
        self.check_qr_stacked(A)
        self.check_qr_stacked(A + 1.j*B)


class TestCholesky:
    # TODO: are there no other tests for cholesky?

    @pytest.mark.parametrize(
        'shape', [(1, 1), (2, 2), (3, 3), (50, 50), (3, 10, 10)]
    )
    @pytest.mark.parametrize(
        'dtype', (np.float32, np.float64, np.complex64, np.complex128)
    )
    def test_basic_property(self, shape, dtype):
        # Check A = L L^H
        np.random.seed(1)
        a = np.random.randn(*shape)
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j*np.random.randn(*shape)

        t = list(range(len(shape)))
        t[-2:] = -1, -2

        a = np.matmul(a.transpose(t).conj(), a)
        a = np.asarray(a, dtype=dtype)

        c = np.linalg.cholesky(a)

        b = np.matmul(c, c.transpose(t).conj())
        with np._no_nep50_warning():
            atol = 500 * a.shape[0] * np.finfo(dtype).eps
        assert_allclose(b, a, atol=atol, err_msg=f'{shape} {dtype}\n{a}\n{c}')

    def test_0_size(self):
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.cholesky(a)
        assert_equal(a.shape, res.shape)
        assert_(res.dtype.type is np.float64)
        # for documentation purpose:
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((1, 0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.cholesky(a)
        assert_equal(a.shape, res.shape)
        assert_(res.dtype.type is np.complex64)
        assert_(isinstance(res, np.ndarray))


def test_byteorder_check():
    # Byte order check should pass for native order
    if sys.byteorder == 'little':
        native = '<'
    else:
        native = '>'

    for dtt in (np.float32, np.float64):
        arr = np.eye(4, dtype=dtt)
        n_arr = arr.newbyteorder(native)
        sw_arr = arr.newbyteorder('S').byteswap()
        assert_equal(arr.dtype.byteorder, '=')
        for routine in (linalg.inv, linalg.det, linalg.pinv):
            # Normal call
            res = routine(arr)
            # Native but not '='
            assert_array_equal(res, routine(n_arr))
            # Swapped
            assert_array_equal(res, routine(sw_arr))


@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
def test_generalized_raise_multiloop():
    # It should raise an error even if the error doesn't occur in the
    # last iteration of the ufunc inner loop

    invertible = np.array([[1, 2], [3, 4]])
    non_invertible = np.array([[1, 1], [1, 1]])

    x = np.zeros([4, 4, 2, 2])[1::2]
    x[...] = invertible
    x[0, 0] = non_invertible

    assert_raises(np.linalg.LinAlgError, np.linalg.inv, x)


def test_xerbla_override():
    # Check that our xerbla has been successfully linked in. If it is not,
    # the default xerbla routine is called, which prints a message to stdout
    # and may, or may not, abort the process depending on the LAPACK package.

    XERBLA_OK = 255

    try:
        pid = os.fork()
    except (OSError, AttributeError):
        # fork failed, or not running on POSIX
        pytest.skip("Not POSIX or fork failed.")

    if pid == 0:
        # child; close i/o file handles
        os.close(1)
        os.close(0)
        # Avoid producing core files.
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # These calls may abort.
        try:
            np.linalg.lapack_lite.xerbla()
        except ValueError:
            pass
        except Exception:
            os._exit(os.EX_CONFIG)

        try:
            a = np.array([[1.]])
            np.linalg.lapack_lite.dorgqr(
                1, 1, 1, a,
                0,  # <- invalid value
                a, a, 0, 0)
        except ValueError as e:
            if "DORGQR parameter number 5" in str(e):
                # success, reuse error code to mark success as
                # FORTRAN STOP returns as success.
                os._exit(XERBLA_OK)

        # Did not abort, but our xerbla was not linked in.
        os._exit(os.EX_CONFIG)
    else:
        # parent
        pid, status = os.wait()
        if os.WEXITSTATUS(status) != XERBLA_OK:
            pytest.skip('Numpy xerbla not linked in.')


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.slow
def test_sdot_bug_8577():
    # Regression test that loading certain other libraries does not
    # result to wrong results in float32 linear algebra.
    #
    # There's a bug gh-8577 on OSX that can trigger this, and perhaps
    # there are also other situations in which it occurs.
    #
    # Do the check in a separate process.

    bad_libs = ['PyQt5.QtWidgets', 'IPython']

    template = textwrap.dedent("""
    import sys
    {before}
    try:
        import {bad_lib}
    except ImportError:
        sys.exit(0)
    {after}
    x = np.ones(2, dtype=np.float32)
    sys.exit(0 if np.allclose(x.dot(x), 2.0) else 1)
    """)

    for bad_lib in bad_libs:
        code = template.format(before="import numpy as np", after="",
                               bad_lib=bad_lib)
        subprocess.check_call([sys.executable, "-c", code])

        # Swapped import order
        code = template.format(after="import numpy as np", before="",
                               bad_lib=bad_lib)
        subprocess.check_call([sys.executable, "-c", code])


class TestMultiDot:

    def test_basic_function_with_three_arguments(self):
        # multi_dot with three arguments uses a fast hand coded algorithm to
        # determine the optimal order. Therefore test it separately.
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))

        assert_almost_equal(multi_dot([A, B, C]), A.dot(B).dot(C))
        assert_almost_equal(multi_dot([A, B, C]), np.dot(A, np.dot(B, C)))

    def test_basic_function_with_two_arguments(self):
        # separate code path with two arguments
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))

        assert_almost_equal(multi_dot([A, B]), A.dot(B))
        assert_almost_equal(multi_dot([A, B]), np.dot(A, B))

    def test_basic_function_with_dynamic_programming_optimization(self):
        # multi_dot with four or more arguments uses the dynamic programming
        # optimization and therefore deserve a separate
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D = np.random.random((2, 1))
        assert_almost_equal(multi_dot([A, B, C, D]), A.dot(B).dot(C).dot(D))

    def test_vector_as_first_argument(self):
        # The first argument can be 1-D
        A1d = np.random.random(2)  # 1-D
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D = np.random.random((2, 2))

        # the result should be 1-D
        assert_equal(multi_dot([A1d, B, C, D]).shape, (2,))

    def test_vector_as_last_argument(self):
        # The last argument can be 1-D
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D1d = np.random.random(2)  # 1-D

        # the result should be 1-D
        assert_equal(multi_dot([A, B, C, D1d]).shape, (6,))

    def test_vector_as_first_and_last_argument(self):
        # The first and last arguments can be 1-D
        A1d = np.random.random(2)  # 1-D
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D1d = np.random.random(2)  # 1-D

        # the result should be a scalar
        assert_equal(multi_dot([A1d, B, C, D1d]).shape, ())

    def test_three_arguments_and_out(self):
        # multi_dot with three arguments uses a fast hand coded algorithm to
        # determine the optimal order. Therefore test it separately.
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))

        out = np.zeros((6, 2))
        ret = multi_dot([A, B, C], out=out)
        assert out is ret
        assert_almost_equal(out, A.dot(B).dot(C))
        assert_almost_equal(out, np.dot(A, np.dot(B, C)))

    def test_two_arguments_and_out(self):
        # separate code path with two arguments
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        out = np.zeros((6, 6))
        ret = multi_dot([A, B], out=out)
        assert out is ret
        assert_almost_equal(out, A.dot(B))
        assert_almost_equal(out, np.dot(A, B))

    def test_dynamic_programming_optimization_and_out(self):
        # multi_dot with four or more arguments uses the dynamic programming
        # optimization and therefore deserve a separate test
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D = np.random.random((2, 1))
        out = np.zeros((6, 1))
        ret = multi_dot([A, B, C, D], out=out)
        assert out is ret
        assert_almost_equal(out, A.dot(B).dot(C).dot(D))

    def test_dynamic_programming_logic(self):
        # Test for the dynamic programming part
        # This test is directly taken from Cormen page 376.
        arrays = [np.random.random((30, 35)),
                  np.random.random((35, 15)),
                  np.random.random((15, 5)),
                  np.random.random((5, 10)),
                  np.random.random((10, 20)),
                  np.random.random((20, 25))]
        m_expected = np.array([[0., 15750., 7875., 9375., 11875., 15125.],
                               [0.,     0., 2625., 4375.,  7125., 10500.],
                               [0.,     0.,    0.,  750.,  2500.,  5375.],
                               [0.,     0.,    0.,    0.,  1000.,  3500.],
                               [0.,     0.,    0.,    0.,     0.,  5000.],
                               [0.,     0.,    0.,    0.,     0.,     0.]])
        s_expected = np.array([[0,  1,  1,  3,  3,  3],
                               [0,  0,  2,  3,  3,  3],
                               [0,  0,  0,  3,  3,  3],
                               [0,  0,  0,  0,  4,  5],
                               [0,  0,  0,  0,  0,  5],
                               [0,  0,  0,  0,  0,  0]], dtype=int)
        s_expected -= 1  # Cormen uses 1-based index, python does not.

        s, m = _multi_dot_matrix_chain_order(arrays, return_costs=True)

        # Only the upper triangular part (without the diagonal) is interesting.
        assert_almost_equal(np.triu(s[:-1, 1:]),
                            np.triu(s_expected[:-1, 1:]))
        assert_almost_equal(np.triu(m), np.triu(m_expected))

    def test_too_few_input_arrays(self):
        assert_raises(ValueError, multi_dot, [])
        assert_raises(ValueError, multi_dot, [np.random.random((3, 3))])


class TestTensorinv:

    @pytest.mark.parametrize("arr, ind", [
        (np.ones((4, 6, 8, 2)), 2),
        (np.ones((3, 3, 2)), 1),
        ])
    def test_non_square_handling(self, arr, ind):
        with assert_raises(LinAlgError):
            linalg.tensorinv(arr, ind=ind)

    @pytest.mark.parametrize("shape, ind", [
        # examples from docstring
        ((4, 6, 8, 3), 2),
        ((24, 8, 3), 1),
        ])
    def test_tensorinv_shape(self, shape, ind):
        a = np.eye(24)
        a.shape = shape
        ainv = linalg.tensorinv(a=a, ind=ind)
        expected = a.shape[ind:] + a.shape[:ind]
        actual = ainv.shape
        assert_equal(actual, expected)

    @pytest.mark.parametrize("ind", [
        0, -2,
        ])
    def test_tensorinv_ind_limit(self, ind):
        a = np.eye(24)
        a.shape = (4, 6, 8, 3)
        with assert_raises(ValueError):
            linalg.tensorinv(a=a, ind=ind)

    def test_tensorinv_result(self):
        # mimic a docstring example
        a = np.eye(24)
        a.shape = (24, 8, 3)
        ainv = linalg.tensorinv(a, ind=1)
        b = np.ones(24)
        assert_allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))


class TestTensorsolve:

    @pytest.mark.parametrize("a, axes", [
        (np.ones((4, 6, 8, 2)), None),
        (np.ones((3, 3, 2)), (0, 2)),
        ])
    def test_non_square_handling(self, a, axes):
        with assert_raises(LinAlgError):
            b = np.ones(a.shape[:2])
            linalg.tensorsolve(a, b, axes=axes)

    @pytest.mark.parametrize("shape",
        [(2, 3, 6), (3, 4, 4, 3), (0, 3, 3, 0)],
    )
    def test_tensorsolve_result(self, shape):
        a = np.random.randn(*shape)
        b = np.ones(a.shape[:2])
        x = np.linalg.tensorsolve(a, b)
        assert_allclose(np.tensordot(a, x, axes=len(x.shape)), b)


def test_unsupported_commontype():
    # linalg gracefully handles unsupported type
    arr = np.array([[1, -2], [2, 5]], dtype='float16')
    with assert_raises_regex(TypeError, "unsupported in linalg"):
        linalg.cholesky(arr)


#@pytest.mark.slow
#@pytest.mark.xfail(not HAS_LAPACK64, run=False,
#                   reason="Numpy not compiled with 64-bit BLAS/LAPACK")
#@requires_memory(free_bytes=16e9)
@pytest.mark.skip(reason="Bad memory reports lead to OOM in ci testing")
def test_blas64_dot():
    n = 2**32
    a = np.zeros([1, n], dtype=np.float32)
    b = np.ones([1, 1], dtype=np.float32)
    a[0,-1] = 1
    c = np.dot(b, a)
    assert_equal(c[0,-1], 1)


@pytest.mark.xfail(not HAS_LAPACK64,
                   reason="Numpy not compiled with 64-bit BLAS/LAPACK")
def test_blas64_geqrf_lwork_smoketest():
    # Smoke test LAPACK geqrf lwork call with 64-bit integers
    dtype = np.float64
    lapack_routine = np.linalg.lapack_lite.dgeqrf

    m = 2**32 + 1
    n = 2**32 + 1
    lda = m

    # Dummy arrays, not referenced by the lapack routine, so don't
    # need to be of the right size
    a = np.zeros([1, 1], dtype=dtype)
    work = np.zeros([1], dtype=dtype)
    tau = np.zeros([1], dtype=dtype)

    # Size query
    results = lapack_routine(m, n, a, lda, tau, work, -1, 0)
    assert_equal(results['info'], 0)
    assert_equal(results['m'], m)
    assert_equal(results['n'], m)

    # Should result to an integer of a reasonable size
    lwork = int(work.item())
    assert_(2**32 < lwork < 2**42)
