import warnings
import itertools
import sys
import ctypes as ct

import pytest
from pytest import param

import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_array_almost_equal, assert_no_warnings,
    assert_allclose, HAS_REFCOUNT, suppress_warnings, IS_WASM
    )
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle


UNARY_UFUNCS = [obj for obj in np.core.umath.__dict__.values()
                    if isinstance(obj, np.ufunc)]
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]


class TestUfuncKwargs:
    def test_kwarg_exact(self):
        assert_raises(TypeError, np.add, 1, 2, castingx='safe')
        assert_raises(TypeError, np.add, 1, 2, dtypex=int)
        assert_raises(TypeError, np.add, 1, 2, extobjx=[4096])
        assert_raises(TypeError, np.add, 1, 2, outx=None)
        assert_raises(TypeError, np.add, 1, 2, sigx='ii->i')
        assert_raises(TypeError, np.add, 1, 2, signaturex='ii->i')
        assert_raises(TypeError, np.add, 1, 2, subokx=False)
        assert_raises(TypeError, np.add, 1, 2, wherex=[True])

    def test_sig_signature(self):
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      signature='ii->i')

    def test_sig_dtype(self):
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      dtype=int)
        assert_raises(TypeError, np.add, 1, 2, signature='ii->i',
                      dtype=int)

    def test_extobj_refcount(self):
        # Should not segfault with USE_DEBUG.
        assert_raises(TypeError, np.add, 1, 2, extobj=[4096], parrot=True)


class TestUfuncGenericLoops:
    """Test generic loops.

    The loops to be tested are:

        PyUFunc_ff_f_As_dd_d
        PyUFunc_ff_f
        PyUFunc_dd_d
        PyUFunc_gg_g
        PyUFunc_FF_F_As_DD_D
        PyUFunc_DD_D
        PyUFunc_FF_F
        PyUFunc_GG_G
        PyUFunc_OO_O
        PyUFunc_OO_O_method
        PyUFunc_f_f_As_d_d
        PyUFunc_d_d
        PyUFunc_f_f
        PyUFunc_g_g
        PyUFunc_F_F_As_D_D
        PyUFunc_F_F
        PyUFunc_D_D
        PyUFunc_G_G
        PyUFunc_O_O
        PyUFunc_O_O_method
        PyUFunc_On_Om

    Where:

        f -- float
        d -- double
        g -- long double
        F -- complex float
        D -- complex double
        G -- complex long double
        O -- python object

    It is difficult to assure that each of these loops is entered from the
    Python level as the special cased loops are a moving target and the
    corresponding types are architecture dependent. We probably need to
    define C level testing ufuncs to get at them. For the time being, I've
    just looked at the signatures registered in the build directory to find
    relevant functions.

    """
    np_dtypes = [
        (np.single, np.single), (np.single, np.double),
        (np.csingle, np.csingle), (np.csingle, np.cdouble),
        (np.double, np.double), (np.longdouble, np.longdouble),
        (np.cdouble, np.cdouble), (np.clongdouble, np.clongdouble)]

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_unary_PyUFunc(self, input_dtype, output_dtype, f=np.exp, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    def f2(x, y):
        return x**y

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_binary_PyUFunc(self, input_dtype, output_dtype, f=f2, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs, xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    # class to use in testing object method loops
    class foo:
        def conjugate(self):
            return np.bool_(1)

        def logical_xor(self, obj):
            return np.bool_(1)

    def test_unary_PyUFunc_O_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.abs(x) == 1))

    def test_unary_PyUFunc_O_O_method_simple(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.conjugate(x) == True))

    def test_binary_PyUFunc_OO_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.add(x, x) == 2))

    def test_binary_PyUFunc_OO_O_method(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_binary_PyUFunc_On_Om_method(self, foo=foo):
        x = np.full((10, 2, 3), foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_python_complex_conjugate(self):
        # The conjugate ufunc should fall back to calling the method:
        arr = np.array([1+2j, 3-4j], dtype="O")
        assert isinstance(arr[0], complex)
        res = np.conjugate(arr)
        assert res.dtype == np.dtype("O")
        assert_array_equal(res, np.array([1-2j, 3+4j], dtype="O"))

    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    def test_unary_PyUFunc_O_O_method_full(self, ufunc):
        """Compare the result of the object loop with non-object one"""
        val = np.float64(np.pi/4)

        class MyFloat(np.float64):
            def __getattr__(self, attr):
                try:
                    return super().__getattr__(attr)
                except AttributeError:
                    return lambda: getattr(np.core.umath, attr)(val)

        # Use 0-D arrays, to ensure the same element call
        num_arr = np.array(val, dtype=np.float64)
        obj_arr = np.array(MyFloat(val), dtype="O")

        with np.errstate(all="raise"):
            try:
                res_num = ufunc(num_arr)
            except Exception as exc:
                with assert_raises(type(exc)):
                    ufunc(obj_arr)
            else:
                res_obj = ufunc(obj_arr)
                assert_array_almost_equal(res_num.astype("O"), res_obj)


def _pickleable_module_global():
    pass


class TestUfunc:
    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_(pickle.loads(pickle.dumps(np.sin,
                                              protocol=proto)) is np.sin)

            # Check that ufunc not defined in the top level numpy namespace
            # such as numpy.core._rational_tests.test_add can also be pickled
            res = pickle.loads(pickle.dumps(_rational_tests.test_add,
                                            protocol=proto))
            assert_(res is _rational_tests.test_add)

    def test_pickle_withstring(self):
        astring = (b"cnumpy.core\n_ufunc_reconstruct\np0\n"
                   b"(S'numpy.core.umath'\np1\nS'cos'\np2\ntp3\nRp4\n.")
        assert_(pickle.loads(astring) is np.cos)

    def test_pickle_name_is_qualname(self):
        # This tests that a simplification of our ufunc pickle code will
        # lead to allowing qualnames as names.  Future ufuncs should
        # possible add a specific qualname, or a hook into pickling instead
        # (dask+numba may benefit).
        _pickleable_module_global.ufunc = umt._pickleable_module_global_ufunc
        obj = pickle.loads(pickle.dumps(_pickleable_module_global.ufunc))
        assert obj is umt._pickleable_module_global_ufunc

    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()
        assert_array_equal(np.add.reduceat(x, idx)[::2], [1, 3, 5, 7])

    def test_all_ufunc(self):
        """Try to check presence and results of all ufuncs.

        The list of ufuncs comes from generate_umath.py and is as follows:

        =====  ====  =============  ===============  ========================
        done   args   function        types                notes
        =====  ====  =============  ===============  ========================
        n      1     conjugate      nums + O
        n      1     absolute       nums + O         complex -> real
        n      1     negative       nums + O
        n      1     sign           nums + O         -> int
        n      1     invert         bool + ints + O  flts raise an error
        n      1     degrees        real + M         cmplx raise an error
        n      1     radians        real + M         cmplx raise an error
        n      1     arccos         flts + M
        n      1     arccosh        flts + M
        n      1     arcsin         flts + M
        n      1     arcsinh        flts + M
        n      1     arctan         flts + M
        n      1     arctanh        flts + M
        n      1     cos            flts + M
        n      1     sin            flts + M
        n      1     tan            flts + M
        n      1     cosh           flts + M
        n      1     sinh           flts + M
        n      1     tanh           flts + M
        n      1     exp            flts + M
        n      1     expm1          flts + M
        n      1     log            flts + M
        n      1     log10          flts + M
        n      1     log1p          flts + M
        n      1     sqrt           flts + M         real x < 0 raises error
        n      1     ceil           real + M
        n      1     trunc          real + M
        n      1     floor          real + M
        n      1     fabs           real + M
        n      1     rint           flts + M
        n      1     isnan          flts             -> bool
        n      1     isinf          flts             -> bool
        n      1     isfinite       flts             -> bool
        n      1     signbit        real             -> bool
        n      1     modf           real             -> (frac, int)
        n      1     logical_not    bool + nums + M  -> bool
        n      2     left_shift     ints + O         flts raise an error
        n      2     right_shift    ints + O         flts raise an error
        n      2     add            bool + nums + O  boolean + is ||
        n      2     subtract       bool + nums + O  boolean - is ^
        n      2     multiply       bool + nums + O  boolean * is &
        n      2     divide         nums + O
        n      2     floor_divide   nums + O
        n      2     true_divide    nums + O         bBhH -> f, iIlLqQ -> d
        n      2     fmod           nums + M
        n      2     power          nums + O
        n      2     greater        bool + nums + O  -> bool
        n      2     greater_equal  bool + nums + O  -> bool
        n      2     less           bool + nums + O  -> bool
        n      2     less_equal     bool + nums + O  -> bool
        n      2     equal          bool + nums + O  -> bool
        n      2     not_equal      bool + nums + O  -> bool
        n      2     logical_and    bool + nums + M  -> bool
        n      2     logical_or     bool + nums + M  -> bool
        n      2     logical_xor    bool + nums + M  -> bool
        n      2     maximum        bool + nums + O
        n      2     minimum        bool + nums + O
        n      2     bitwise_and    bool + ints + O  flts raise an error
        n      2     bitwise_or     bool + ints + O  flts raise an error
        n      2     bitwise_xor    bool + ints + O  flts raise an error
        n      2     arctan2        real + M
        n      2     remainder      ints + real + O
        n      2     hypot          real + M
        =====  ====  =============  ===============  ========================

        Types other than those listed will be accepted, but they are cast to
        the smallest compatible type for which the function is defined. The
        casting rules are:

        bool -> int8 -> float32
        ints -> double

        """
        pass

    # from include/numpy/ufuncobject.h
    size_inferred = 2
    can_ignore = 4
    def test_signature0(self):
        # the arguments to test_signature are: nin, nout, core_signature
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i),(i)->()")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1,  1,  0))
        assert_equal(ixs, (0, 0))
        assert_equal(flags, (self.size_inferred,))
        assert_equal(sizes, (-1,))

    def test_signature1(self):
        # empty core signature; treat as plain ufunc (with trivial core)
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(),()->()")
        assert_equal(enabled, 0)
        assert_equal(num_dims, (0,  0,  0))
        assert_equal(ixs, ())
        assert_equal(flags, ())
        assert_equal(sizes, ())

    def test_signature2(self):
        # more complicated names for variables
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1,i2),(J_1)->(_kAB)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 1))
        assert_equal(ixs, (0, 1, 2, 3))
        assert_equal(flags, (self.size_inferred,)*4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature3(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1, i12),   (J_1)->(i12, i2)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 2))
        assert_equal(ixs, (0, 1, 2, 1, 3))
        assert_equal(flags, (self.size_inferred,)*4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature4(self):
        # matrix_multiply signature from _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n,k),(k,m)->(n,m)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred,)*3)
        assert_equal(sizes, (-1, -1, -1))

    def test_signature5(self):
        # matmul signature from _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n?,k),(k,m?)->(n?,m?)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred | self.can_ignore,
                             self.size_inferred,
                             self.size_inferred | self.can_ignore))
        assert_equal(sizes, (-1, -1, -1))

    def test_signature6(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(3)->()")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature7(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3),(03,3),(n)->(9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (0, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature8(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3?),(3?,3?),(n)->(9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature9(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(  3)  -> ( )")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature10(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "( 3? ) , (3? ,  3?) ,(n )-> ( 9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature_failure_extra_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "((i)),(i)->()")

    def test_signature_failure_mismatching_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),)i(->()")

    def test_signature_failure_signature_missing_input_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),->()")

    def test_signature_failure_signature_missing_output_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 2, "(i),(i)->()")

    def test_get_signature(self):
        assert_equal(umt.inner1d.signature, "(i),(i)->()")

    def test_forced_sig(self):
        a = 0.5*np.arange(3, dtype='f8')
        assert_equal(np.add(a, 0.5), [0.5, 1, 1.5])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig='i', casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig='ii->i', casting='unsafe'), [0, 0, 1])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig=('i4',), casting='unsafe'),
                         [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig=('i4', 'i4', 'i4'),
                                            casting='unsafe'), [0, 0, 1])

        b = np.zeros((3,), dtype='f8')
        np.add(a, 0.5, out=b)
        assert_equal(b, [0.5, 1, 1.5])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig='i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig='ii->i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig=('i4',), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig=('i4', 'i4', 'i4'), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])

    def test_signature_all_None(self):
        # signature all None, is an acceptable alternative (since 1.21)
        # to not providing a signature.
        res1 = np.add([3], [4], sig=(None, None, None))
        res2 = np.add([3], [4])
        assert_array_equal(res1, res2)
        res1 = np.maximum([3], [4], sig=(None, None, None))
        res2 = np.maximum([3], [4])
        assert_array_equal(res1, res2)

        with pytest.raises(TypeError):
            # special case, that would be deprecated anyway, so errors:
            np.add(3, 4, signature=(None,))

    def test_signature_dtype_type(self):
        # Since that will be the normal behaviour (past NumPy 1.21)
        # we do support the types already:
        float_dtype = type(np.dtype(np.float64))
        np.add(3, 4, signature=(float_dtype, float_dtype, None))

    @pytest.mark.parametrize("get_kwarg", [
            lambda dt: dict(dtype=x),
            lambda dt: dict(signature=(x, None, None))])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        # We allow certain dtype instances when there is a clear singleton
        # and the given one is equivalent; mainly for backcompat.
        int64 = np.dtype("int64")
        int64_2 = pickle.loads(pickle.dumps(int64))
        # Relies on pickling behavior, if assert fails just remove test...
        assert int64 is not int64_2

        assert np.add(1, 2, **get_kwarg(int64_2)).dtype == int64
        td = np.timedelta(2, "s")
        assert np.add(td, td, **get_kwarg("m8")).dtype == "m8[s]"

    @pytest.mark.parametrize("get_kwarg", [
            param(lambda x: dict(dtype=x), id="dtype"),
            param(lambda x: dict(signature=(x, None, None)), id="signature")])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        msg = "The `dtype` and `signature` arguments to ufuncs"

        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype("int64").newbyteorder()))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype("m8[ns]")))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg("m8[ns]"))

    @pytest.mark.parametrize("casting", ["unsafe", "same_kind", "safe"])
    def test_partial_signature_mismatch(self, casting):
        # If the second argument matches already, no need to specify it:
        res = np.ldexp(np.float32(1.), np.int_(2), dtype="d")
        assert res.dtype == "d"
        res = np.ldexp(np.float32(1.), np.int_(2), signature=(None, None, "d"))
        assert res.dtype == "d"

        # ldexp only has a loop for long input as second argument, overriding
        # the output cannot help with that (no matter the casting)
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), dtype="d")
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), signature=(None, None, "d"))

    def test_partial_signature_mismatch_with_cache(self):
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=("e", "d", None))
        # Ensure e,d->None is in the dispatching cache (double loop)
        np.add(np.float16(1), np.float64(2))
        # The error must still be raised:
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=("e", "d", None))

    def test_use_output_signature_for_all_arguments(self):
        # Test that providing only `dtype=` or `signature=(None, None, dtype)`
        # is sufficient if falling back to a homogeneous signature works.
        # In this case, the `intp, intp -> intp` loop is chosen.
        res = np.power(1.5, 2.8, dtype=np.intp, casting="unsafe")
        assert res == 1  # the cast happens first.
        res = np.power(1.5, 2.8, signature=(None, None, np.intp),
                       casting="unsafe")
        assert res == 1
        with pytest.raises(TypeError):
            # the unsafe casting would normally cause errors though:
            np.power(1.5, 2.8, dtype=np.intp)

    def test_signature_errors(self):
        with pytest.raises(TypeError,
                    match="the signature object to ufunc must be a string or"):
            np.add(3, 4, signature=123.)  # neither a string nor a tuple

        with pytest.raises(ValueError):
            # bad symbols that do not translate to dtypes
            np.add(3, 4, signature="%^->#")

        with pytest.raises(ValueError):
            np.add(3, 4, signature=b"ii-i")  # incomplete and byte string

        with pytest.raises(ValueError):
            np.add(3, 4, signature="ii>i")  # incomplete string

        with pytest.raises(ValueError):
            np.add(3, 4, signature=(None, "f8"))  # bad length

        with pytest.raises(UnicodeDecodeError):
            np.add(3, 4, signature=b"\xff\xff->i")

    def test_forced_dtype_times(self):
        # Signatures only set the type numbers (not the actual loop dtypes)
        # so using `M` in a signature/dtype should generally work:
        a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='>M8[D]')
        np.maximum(a, a, dtype="M")
        np.maximum.reduce(a, dtype="M")

        arr = np.arange(10, dtype="m8[s]")
        np.add(arr, arr, dtype="m")
        np.maximum(arr, arr, dtype="m")

    @pytest.mark.parametrize("ufunc", [np.add, np.sqrt])
    def test_cast_safety(self, ufunc):
        """Basic test for the safest casts, because ufuncs inner loops can
        indicate a cast-safety as well (which is normally always "no").
        """
        def call_ufunc(arr, **kwargs):
            return ufunc(*(arr,) * ufunc.nin, **kwargs)

        arr = np.array([1., 2., 3.], dtype=np.float32)
        arr_bs = arr.astype(arr.dtype.newbyteorder())
        expected = call_ufunc(arr)
        # Normally, a "no" cast:
        res = call_ufunc(arr, casting="no")
        assert_array_equal(expected, res)
        # Byte-swapping is not allowed with "no" though:
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, casting="no")

        # But is allowed with "equiv":
        res = call_ufunc(arr_bs, casting="equiv")
        assert_array_equal(expected, res)

        # Casting to float64 is safe, but not equiv:
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, dtype=np.float64, casting="equiv")

        # but it is safe cast:
        res = call_ufunc(arr_bs, dtype=np.float64, casting="safe")
        expected = call_ufunc(arr.astype(np.float64))  # upcast
        assert_array_equal(expected, res)

    def test_true_divide(self):
        a = np.array(10)
        b = np.array(20)
        tgt = np.array(0.5)

        for tc in 'bhilqBHILQefdgFDG':
            dt = np.dtype(tc)
            aa = a.astype(dt)
            bb = b.astype(dt)

            # Check result value and dtype.
            for x, y in itertools.product([aa, -aa], [bb, -bb]):

                # Check with no output type specified
                if tc in 'FDG':
                    tgt = complex(x)/complex(y)
                else:
                    tgt = float(x)/float(y)

                res = np.true_divide(x, y)
                rtol = max(np.finfo(res).resolution, 1e-15)
                assert_allclose(res, tgt, rtol=rtol)

                if tc in 'bhilqBHILQ':
                    assert_(res.dtype.name == 'float64')
                else:
                    assert_(res.dtype.name == dt.name )

                # Check with output type specified.  This also checks for the
                # incorrect casts in issue gh-3484 because the unary '-' does
                # not change types, even for unsigned types, Hence casts in the
                # ufunc from signed to unsigned and vice versa will lead to
                # errors in the values.
                for tcout in 'bhilqBHILQ':
                    dtout = np.dtype(tcout)
                    assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)

                for tcout in 'efdg':
                    dtout = np.dtype(tcout)
                    if tc in 'FDG':
                        # Casting complex to float is not allowed
                        assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)
                    else:
                        tgt = float(x)/float(y)
                        rtol = max(np.finfo(dtout).resolution, 1e-15)
                        # The value of tiny for double double is NaN
                        with suppress_warnings() as sup:
                            sup.filter(UserWarning)
                            if not np.isnan(np.finfo(dtout).tiny):
                                atol = max(np.finfo(dtout).tiny, 3e-308)
                            else:
                                atol = 3e-308
                        # Some test values result in invalid for float16
                        # and the cast to it may overflow to inf.
                        with np.errstate(invalid='ignore', over='ignore'):
                            res = np.true_divide(x, y, dtype=dtout)
                        if not np.isfinite(res) and tcout == 'e':
                            continue
                        assert_allclose(res, tgt, rtol=rtol, atol=atol)
                        assert_(res.dtype.name == dtout.name)

                for tcout in 'FDG':
                    dtout = np.dtype(tcout)
                    tgt = complex(x)/complex(y)
                    rtol = max(np.finfo(dtout).resolution, 1e-15)
                    # The value of tiny for double double is NaN
                    with suppress_warnings() as sup:
                        sup.filter(UserWarning)
                        if not np.isnan(np.finfo(dtout).tiny):
                            atol = max(np.finfo(dtout).tiny, 3e-308)
                        else:
                            atol = 3e-308
                    res = np.true_divide(x, y, dtype=dtout)
                    if not np.isfinite(res):
                        continue
                    assert_allclose(res, tgt, rtol=rtol, atol=atol)
                    assert_(res.dtype.name == dtout.name)

        # Check booleans
        a = np.ones((), dtype=np.bool_)
        res = np.true_divide(a, a)
        assert_(res == 1.0)
        assert_(res.dtype.name == 'float64')
        res = np.true_divide(~a, a)
        assert_(res == 0.0)
        assert_(res.dtype.name == 'float64')

    def test_sum_stability(self):
        a = np.ones(500, dtype=np.float32)
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 4)

        a = np.ones(500, dtype=np.float64)
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 13)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sum(self):
        for dt in (int, np.float16, np.float32, np.float64, np.longdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                # warning if sum overflows, which it does in float16
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    assert_equal(len(w), 1 * overflow)

                    d = np.arange(1, v + 1, dtype=dt)

                    assert_almost_equal(np.sum(d), tgt)
                    assert_equal(len(w), 2 * overflow)

                    assert_almost_equal(np.sum(d[::-1]), tgt)
                    assert_equal(len(w), 3 * overflow)

            d = np.ones(500, dtype=dt)
            assert_almost_equal(np.sum(d[::2]), 250.)
            assert_almost_equal(np.sum(d[1::2]), 250.)
            assert_almost_equal(np.sum(d[::3]), 167.)
            assert_almost_equal(np.sum(d[1::3]), 167.)
            assert_almost_equal(np.sum(d[::-2]), 250.)
            assert_almost_equal(np.sum(d[-1::-2]), 250.)
            assert_almost_equal(np.sum(d[::-3]), 167.)
            assert_almost_equal(np.sum(d[-1::-3]), 167.)
            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.)

    def test_sum_complex(self):
        for dt in (np.complex64, np.complex128, np.clongdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
                d = np.empty(v, dtype=dt)
                d.real = np.arange(1, v + 1)
                d.imag = -np.arange(1, v + 1)
                assert_almost_equal(np.sum(d), tgt)
                assert_almost_equal(np.sum(d[::-1]), tgt)

            d = np.ones(500, dtype=dt) + 1j
            assert_almost_equal(np.sum(d[::2]), 250. + 250j)
            assert_almost_equal(np.sum(d[1::2]), 250. + 250j)
            assert_almost_equal(np.sum(d[::3]), 167. + 167j)
            assert_almost_equal(np.sum(d[1::3]), 167. + 167j)
            assert_almost_equal(np.sum(d[::-2]), 250. + 250j)
            assert_almost_equal(np.sum(d[-1::-2]), 250. + 250j)
            assert_almost_equal(np.sum(d[::-3]), 167. + 167j)
            assert_almost_equal(np.sum(d[-1::-3]), 167. + 167j)
            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt) + 1j
            d += d
            assert_almost_equal(d, 2. + 2j)

    def test_sum_initial(self):
        # Integer, single axis
        assert_equal(np.sum([3], initial=2), 5)

        # Floating point
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # Multiple non-adjacent axes
        assert_equal(np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
                     [12, 12, 12])

    def test_sum_where(self):
        # More extensive tests done in test_reduction_with_where.
        assert_equal(np.sum([[1., 2.], [3., 4.]], where=[True, False]), 4.)
        assert_equal(np.sum([[1., 2.], [3., 4.]], axis=0, initial=5.,
                            where=[True, False]), [9., 5.])

    def test_inner1d(self):
        a = np.arange(6).reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1))
        a = np.arange(6)
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a))

    def test_broadcast(self):
        msg = "broadcast"
        a = np.arange(4).reshape((2, 1, 2))
        b = np.arange(4).reshape((1, 2, 2))
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        msg = "extend & broadcast loop dimensions"
        b = np.arange(4).reshape((2, 2))
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        # Broadcast in core dimensions should fail
        a = np.arange(8).reshape((4, 2))
        b = np.arange(4).reshape((4, 1))
        assert_raises(ValueError, umt.inner1d, a, b)
        # Extend core dimensions should fail
        a = np.arange(8).reshape((4, 2))
        b = np.array(7)
        assert_raises(ValueError, umt.inner1d, a, b)
        # Broadcast should fail
        a = np.arange(2).reshape((2, 1, 1))
        b = np.arange(3).reshape((3, 1, 1))
        assert_raises(ValueError, umt.inner1d, a, b)

        # Writing to a broadcasted array with overlap should warn, gh-2705
        a = np.arange(2)
        b = np.arange(4).reshape((2, 2))
        u, v = np.broadcast_arrays(a, b)
        assert_equal(u.strides[0], 0)
        x = u + v
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            u += v
            assert_equal(len(w), 1)
            assert_(x[0, 0] != u[0, 0])

        # Output reduction should not be allowed.
        # See gh-15139
        a = np.arange(6).reshape(3, 2)
        b = np.ones(2)
        out = np.empty(())
        assert_raises(ValueError, umt.inner1d, a, b, out)
        out2 = np.empty(3)
        c = umt.inner1d(a, b, out2)
        assert_(c is out2)

    def test_out_broadcasts(self):
        # For ufuncs and gufuncs (not for reductions), we currently allow
        # the output to cause broadcasting of the input arrays.
        # both along dimensions with shape 1 and dimensions which do not
        # exist at all in the inputs.
        arr = np.arange(3).reshape(1, 3)
        out = np.empty((5, 4, 3))
        np.add(arr, arr, out=out)
        assert (out == np.arange(3) * 2).all()

        # The same holds for gufuncs (gh-16484)
        umt.inner1d(arr, arr, out=out)
        # the result would be just a scalar `5`, but is broadcast fully:
        assert (out == 5).all()

    @pytest.mark.parametrize(["arr", "out"], [
                ([2], np.empty(())),
                ([1, 2], np.empty(1)),
                (np.ones((4, 3)), np.empty((4, 1)))],
            ids=["(1,)->()", "(2,)->(1,)", "(4, 3)->(4, 1)"])
    def test_out_broadcast_errors(self, arr, out):
        # Output is (currently) allowed to broadcast inputs, but it cannot be
        # smaller than the actual result.
        with pytest.raises(ValueError, match="non-broadcastable"):
            np.positive(arr, out=out)

        with pytest.raises(ValueError, match="non-broadcastable"):
            np.add(np.ones(()), arr, out=out)

    def test_type_cast(self):
        msg = "type cast"
        a = np.arange(6, dtype='short').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)
        msg = "type cast on one argument"
        a = np.arange(6).reshape((2, 3))
        b = a + 0.1
        assert_array_almost_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1),
                                  err_msg=msg)

    def test_endian(self):
        msg = "big endian"
        a = np.arange(6, dtype='>i4').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)
        msg = "little endian"
        a = np.arange(6, dtype='<i4').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)

        # Output should always be native-endian
        Ba = np.arange(1, dtype='>f8')
        La = np.arange(1, dtype='<f8')
        assert_equal((Ba+Ba).dtype, np.dtype('f8'))
        assert_equal((Ba+La).dtype, np.dtype('f8'))
        assert_equal((La+Ba).dtype, np.dtype('f8'))
        assert_equal((La+La).dtype, np.dtype('f8'))

        assert_equal(np.absolute(La).dtype, np.dtype('f8'))
        assert_equal(np.absolute(Ba).dtype, np.dtype('f8'))
        assert_equal(np.negative(La).dtype, np.dtype('f8'))
        assert_equal(np.negative(Ba).dtype, np.dtype('f8'))

    def test_incontiguous_array(self):
        msg = "incontiguous memory layout of array"
        x = np.arange(64).reshape((2, 2, 2, 2, 2, 2))
        a = x[:, 0,:, 0,:, 0]
        b = x[:, 1,:, 1,:, 1]
        a[0, 0, 0] = -1
        msg2 = "make sure it references to the original array"
        assert_equal(x[0, 0, 0, 0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        x = np.arange(24).reshape(2, 3, 4)
        a = x.T
        b = x.T
        a[0, 0, 0] = -1
        assert_equal(x[0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)

    def test_output_argument(self):
        msg = "output argument"
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(4).reshape((2, 1, 2)) + 1
        c = np.zeros((2, 3), dtype='int')
        umt.inner1d(a, b, c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)

        msg = "output argument with type cast"
        c = np.zeros((2, 3), dtype='int16')
        umt.inner1d(a, b, c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)

        msg = "output argument with incontiguous layout"
        c = np.zeros((2, 3, 4), dtype='int16')
        umt.inner1d(a, b, c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a*b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a*b, axis=-1), err_msg=msg)

    def test_axes_argument(self):
        # inner1d signature: '(i),(i)->()'
        inner1d = umt.inner1d
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        # basic tests on inputs (outputs tested below with matrix_multiply).
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        # default
        c = inner1d(a, b, axes=[(-1,), (-1,), ()])
        assert_array_equal(c, (a * b).sum(-1))
        # integers ok for single axis.
        c = inner1d(a, b, axes=[-1, -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        # mix fine
        c = inner1d(a, b, axes=[(-1,), -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        # can omit last axis.
        c = inner1d(a, b, axes=[-1, -1])
        assert_array_equal(c, (a * b).sum(-1))
        # can pass in other types of integer (with __index__ protocol)
        c = inner1d(a, b, axes=[np.int8(-1), np.array(-1, dtype=np.int32)])
        assert_array_equal(c, (a * b).sum(-1))
        # swap some axes
        c = inner1d(a, b, axes=[0, 0])
        assert_array_equal(c, (a * b).sum(0))
        c = inner1d(a, b, axes=[0, 2])
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        # Check errors for improperly constructed axes arguments.
        # should have list.
        assert_raises(TypeError, inner1d, a, b, axes=-1)
        # needs enough elements
        assert_raises(ValueError, inner1d, a, b, axes=[-1])
        # should pass in indices.
        assert_raises(TypeError, inner1d, a, b, axes=[-1.0, -1.0])
        assert_raises(TypeError, inner1d, a, b, axes=[(-1.0,), -1])
        assert_raises(TypeError, inner1d, a, b, axes=[None, 1])
        # cannot pass an index unless there is only one dimension
        # (output is wrong in this case)
        assert_raises(TypeError, inner1d, a, b, axes=[-1, -1, -1])
        # or pass in generally the wrong number of axes
        assert_raises(ValueError, inner1d, a, b, axes=[-1, -1, (-1,)])
        assert_raises(ValueError, inner1d, a, b, axes=[-1, (-2, -1), ()])
        # axes need to have same length.
        assert_raises(ValueError, inner1d, a, b, axes=[0, 1])

        # matrix_multiply signature: '(m,n),(n,p)->(m,p)'
        mm = umt.matrix_multiply
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(8).reshape((2, 2, 2, 1)) + 1
        # Sanity check.
        c = mm(a, b)
        assert_array_equal(c, np.matmul(a, b))
        # Default axes.
        c = mm(a, b, axes=[(-2, -1), (-2, -1), (-2, -1)])
        assert_array_equal(c, np.matmul(a, b))
        # Default with explicit axes.
        c = mm(a, b, axes=[(1, 2), (2, 3), (2, 3)])
        assert_array_equal(c, np.matmul(a, b))
        # swap some axes.
        c = mm(a, b, axes=[(0, -1), (1, 2), (-2, -1)])
        assert_array_equal(c, np.matmul(a.transpose(1, 0, 2),
                                        b.transpose(0, 3, 1, 2)))
        # Default with output array.
        c = np.empty((2, 2, 3, 1))
        d = mm(a, b, out=c, axes=[(1, 2), (2, 3), (2, 3)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b))
        # Transposed output array
        c = np.empty((1, 2, 2, 3))
        d = mm(a, b, out=c, axes=[(-2, -1), (-2, -1), (3, 0)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b).transpose(3, 0, 1, 2))
        # Check errors for improperly constructed axes arguments.
        # wrong argument
        assert_raises(TypeError, mm, a, b, axis=1)
        # axes should be list
        assert_raises(TypeError, mm, a, b, axes=1)
        assert_raises(TypeError, mm, a, b, axes=((-2, -1), (-2, -1), (-2, -1)))
        # list needs to have right length
        assert_raises(ValueError, mm, a, b, axes=[])
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1)])
        # list should contain tuples for multiple axes
        assert_raises(TypeError, mm, a, b, axes=[-1, -1, -1])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), -1])
        assert_raises(TypeError,
                      mm, a, b, axes=[[-2, -1], [-2, -1], [-2, -1]])
        assert_raises(TypeError,
                      mm, a, b, axes=[(-2, -1), (-2, -1), [-2, -1]])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), None])
        # tuples should not have duplicated values
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1), (-2, -1), (-2, -2)])
        # arrays should have enough axes.
        z = np.zeros((2, 2))
        assert_raises(ValueError, mm, z, z[0])
        assert_raises(ValueError, mm, z, z, out=z[:, 0])
        assert_raises(ValueError, mm, z[1], z, axes=[0, 1])
        assert_raises(ValueError, mm, z, z, out=z[0], axes=[0, 1])
        # Regular ufuncs should not accept axes.
        assert_raises(TypeError, np.add, 1., 1., axes=[0])
        # should be able to deal with bad unrelated kwargs.
        assert_raises(TypeError, mm, z, z, axes=[0, 1], parrot=True)

    def test_axis_argument(self):
        # inner1d signature: '(i),(i)->()'
        inner1d = umt.inner1d
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axis=-1)
        assert_array_equal(c, (a * b).sum(-1))
        out = np.zeros_like(c)
        d = inner1d(a, b, axis=-1, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        c = inner1d(a, b, axis=0)
        assert_array_equal(c, (a * b).sum(0))
        # Sanity checks on innerwt and cumsum.
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, axis=0),
                           np.sum(a * b * w, axis=0))
        assert_array_equal(umt.cumsum(a, axis=0), np.cumsum(a, axis=0))
        assert_array_equal(umt.cumsum(a, axis=-1), np.cumsum(a, axis=-1))
        out = np.empty_like(a)
        b = umt.cumsum(a, out=out, axis=0)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=0))
        b = umt.cumsum(a, out=out, axis=1)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=-1))
        # Check errors.
        # Cannot pass in both axis and axes.
        assert_raises(TypeError, inner1d, a, b, axis=0, axes=[0, 0])
        # Not an integer.
        assert_raises(TypeError, inner1d, a, b, axis=[0])
        # more than 1 core dimensions.
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, axis=1)
        # Output wrong size in axis.
        out = np.empty((1, 2, 3), dtype=a.dtype)
        assert_raises(ValueError, umt.cumsum, a, out=out, axis=0)
        # Regular ufuncs should not accept axis.
        assert_raises(TypeError, np.add, 1., 1., axis=0)

    def test_keepdims_argument(self):
        # inner1d signature: '(i),(i)->()'
        inner1d = umt.inner1d
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        out = np.zeros_like(c)
        d = inner1d(a, b, keepdims=True, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        # Now combined with axis and axes.
        c = inner1d(a, b, axis=-1, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1, keepdims=False))
        c = inner1d(a, b, axis=-1, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axis=0, keepdims=False)
        assert_array_equal(c, (a * b).sum(0, keepdims=False))
        c = inner1d(a, b, axis=0, keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[(-1,), (-1,), ()], keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[(-1,), (-1,), (-1,)], keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axes=[0, 0], keepdims=False)
        assert_array_equal(c, (a * b).sum(0))
        c = inner1d(a, b, axes=[0, 0, 0], keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[0, 2], keepdims=False)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        c = inner1d(a, b, axes=[0, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1,
                                                             keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1,
                                                             keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 0], keepdims=True)
        assert_array_equal(c, (a * b.transpose(2, 0, 1)).sum(0, keepdims=True))
        # Hardly useful, but should work.
        c = inner1d(a, b, axes=[0, 2, 1], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 0, 2) * b.transpose(0, 2, 1))
                           .sum(1, keepdims=True))
        # Check with two core dimensions.
        a = np.eye(3) * np.arange(4.)[:, np.newaxis, np.newaxis]
        expected = uml.det(a)
        c = uml.det(a, keepdims=False)
        assert_array_equal(c, expected)
        c = uml.det(a, keepdims=True)
        assert_array_equal(c, expected[:, np.newaxis, np.newaxis])
        a = np.eye(3) * np.arange(4.)[:, np.newaxis, np.newaxis]
        expected_s, expected_l = uml.slogdet(a)
        cs, cl = uml.slogdet(a, keepdims=False)
        assert_array_equal(cs, expected_s)
        assert_array_equal(cl, expected_l)
        cs, cl = uml.slogdet(a, keepdims=True)
        assert_array_equal(cs, expected_s[:, np.newaxis, np.newaxis])
        assert_array_equal(cl, expected_l[:, np.newaxis, np.newaxis])
        # Sanity check on innerwt.
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, keepdims=True),
                           np.sum(a * b * w, axis=-1, keepdims=True))
        assert_array_equal(umt.innerwt(a, b, w, axis=0, keepdims=True),
                           np.sum(a * b * w, axis=0, keepdims=True))
        # Check errors.
        # Not a boolean
        assert_raises(TypeError, inner1d, a, b, keepdims='true')
        # More than 1 core dimension, and core output dimensions.
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, keepdims=True)
        assert_raises(TypeError, mm, a, b, keepdims=False)
        # Regular ufuncs should not accept keepdims.
        assert_raises(TypeError, np.add, 1., 1., keepdims=False)

    def test_innerwt(self):
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))
        a = np.arange(100, 124).reshape((2, 3, 4))
        b = np.arange(200, 224).reshape((2, 3, 4))
        w = np.arange(300, 324).reshape((2, 3, 4))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))

    def test_innerwt_empty(self):
        """Test generalized ufunc with zero-sized operands"""
        a = np.array([], dtype='f8')
        b = np.array([], dtype='f8')
        w = np.array([], dtype='f8')
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))

    def test_cross1d(self):
        """Test with fixed-sized signature."""
        a = np.eye(3)
        assert_array_equal(umt.cross1d(a, a), np.zeros((3, 3)))
        out = np.zeros((3, 3))
        result = umt.cross1d(a[0], a, out)
        assert_(result is out)
        assert_array_equal(result, np.vstack((np.zeros(3), a[2], -a[1])))
        assert_raises(ValueError, umt.cross1d, np.eye(4), np.eye(4))
        assert_raises(ValueError, umt.cross1d, a, np.arange(4.))
        # Wrong output core dimension.
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros((3, 4)))
        # Wrong output broadcast dimension (see gh-15139).
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros(3))

    def test_can_ignore_signature(self):
        # Comparing the effects of ? in signature:
        # matrix_multiply: (m,n),(n,p)->(m,p)    # all must be there.
        # matmul:        (m?,n),(n,p?)->(m?,p?)  # allow missing m, p.
        mat = np.arange(12).reshape((2, 3, 2))
        single_vec = np.arange(2)
        col_vec = single_vec[:, np.newaxis]
        col_vec_array = np.arange(8).reshape((2, 2, 2, 1)) + 1
        # matrix @ single column vector with proper dimension
        mm_col_vec = umt.matrix_multiply(mat, col_vec)
        # matmul does the same thing
        matmul_col_vec = umt.matmul(mat, col_vec)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        # matrix @ vector without dimension making it a column vector.
        # matrix multiply fails -> missing core dim.
        assert_raises(ValueError, umt.matrix_multiply, mat, single_vec)
        # matmul mimicker passes, and returns a vector.
        matmul_col = umt.matmul(mat, single_vec)
        assert_array_equal(matmul_col, mm_col_vec.squeeze())
        # Now with a column array: same as for column vector,
        # broadcasting sensibly.
        mm_col_vec = umt.matrix_multiply(mat, col_vec_array)
        matmul_col_vec = umt.matmul(mat, col_vec_array)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        # As above, but for row vector
        single_vec = np.arange(3)
        row_vec = single_vec[np.newaxis, :]
        row_vec_array = np.arange(24).reshape((4, 2, 1, 1, 3)) + 1
        # row vector @ matrix
        mm_row_vec = umt.matrix_multiply(row_vec, mat)
        matmul_row_vec = umt.matmul(row_vec, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        # single row vector @ matrix
        assert_raises(ValueError, umt.matrix_multiply, single_vec, mat)
        matmul_row = umt.matmul(single_vec, mat)
        assert_array_equal(matmul_row, mm_row_vec.squeeze())
        # row vector array @ matrix
        mm_row_vec = umt.matrix_multiply(row_vec_array, mat)
        matmul_row_vec = umt.matmul(row_vec_array, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        # Now for vector combinations
        # row vector @ column vector
        col_vec = row_vec.T
        col_vec_array = row_vec_array.swapaxes(-2, -1)
        mm_row_col_vec = umt.matrix_multiply(row_vec, col_vec)
        matmul_row_col_vec = umt.matmul(row_vec, col_vec)
        assert_array_equal(matmul_row_col_vec, mm_row_col_vec)
        # single row vector @ single col vector
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec)
        matmul_row_col = umt.matmul(single_vec, single_vec)
        assert_array_equal(matmul_row_col, mm_row_col_vec.squeeze())
        # row vector array @ matrix
        mm_row_col_array = umt.matrix_multiply(row_vec_array, col_vec_array)
        matmul_row_col_array = umt.matmul(row_vec_array, col_vec_array)
        assert_array_equal(matmul_row_col_array, mm_row_col_array)
        # Finally, check that things are *not* squeezed if one gives an
        # output.
        out = np.zeros_like(mm_row_col_array)
        out = umt.matrix_multiply(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        out[:] = 0
        out = umt.matmul(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        # And check one cannot put missing dimensions back.
        out = np.zeros_like(mm_row_col_vec)
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec,
                      out)
        # But fine for matmul, since it is just a broadcast.
        out = umt.matmul(single_vec, single_vec, out)
        assert_array_equal(out, mm_row_col_vec.squeeze())

    def test_matrix_multiply(self):
        self.compare_matrix_multiply_results(np.int64)
        self.compare_matrix_multiply_results(np.double)

    def test_matrix_multiply_umath_empty(self):
        res = umt.matrix_multiply(np.ones((0, 10)), np.ones((10, 0)))
        assert_array_equal(res, np.zeros((0, 0)))
        res = umt.matrix_multiply(np.ones((10, 0)), np.ones((0, 10)))
        assert_array_equal(res, np.zeros((10, 10)))

    def compare_matrix_multiply_results(self, tp):
        d1 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        d2 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        msg = "matrix multiply on type %s" % d1.dtype.name

        def permute_n(n):
            if n == 1:
                return ([0],)
            ret = ()
            base = permute_n(n-1)
            for perm in base:
                for i in range(n):
                    new = perm + [n-1]
                    new[n-1] = new[i]
                    new[i] = n-1
                    ret += (new,)
            return ret

        def slice_n(n):
            if n == 0:
                return ((),)
            ret = ()
            base = slice_n(n-1)
            for sl in base:
                ret += (sl+(slice(None),),)
                ret += (sl+(slice(0, 1),),)
            return ret

        def broadcastable(s1, s2):
            return s1 == s2 or s1 == 1 or s2 == 1

        permute_3 = permute_n(3)
        slice_3 = slice_n(3) + ((slice(None, None, -1),)*3,)

        ref = True
        for p1 in permute_3:
            for p2 in permute_3:
                for s1 in slice_3:
                    for s2 in slice_3:
                        a1 = d1.transpose(p1)[s1]
                        a2 = d2.transpose(p2)[s2]
                        ref = ref and a1.base is not None
                        ref = ref and a2.base is not None
                        if (a1.shape[-1] == a2.shape[-2] and
                                broadcastable(a1.shape[0], a2.shape[0])):
                            assert_array_almost_equal(
                                umt.matrix_multiply(a1, a2),
                                np.sum(a2[..., np.newaxis].swapaxes(-3, -1) *
                                       a1[..., np.newaxis,:], axis=-1),
                                err_msg=msg + ' %s %s' % (str(a1.shape),
                                                          str(a2.shape)))

        assert_equal(ref, True, err_msg="reference check")

    def test_euclidean_pdist(self):
        a = np.arange(12, dtype=float).reshape(4, 3)
        out = np.empty((a.shape[0] * (a.shape[0] - 1) // 2,), dtype=a.dtype)
        umt.euclidean_pdist(a, out)
        b = np.sqrt(np.sum((a[:, None] - a)**2, axis=-1))
        b = b[~np.tri(a.shape[0], dtype=bool)]
        assert_almost_equal(out, b)
        # An output array is required to determine p with signature (n,d)->(p)
        assert_raises(ValueError, umt.euclidean_pdist, a)

    def test_cumsum(self):
        a = np.arange(10)
        result = umt.cumsum(a)
        assert_array_equal(result, a.cumsum())

    def test_object_logical(self):
        a = np.array([3, None, True, False, "test", ""], dtype=object)
        assert_equal(np.logical_or(a, None),
                        np.array([x or None for x in a], dtype=object))
        assert_equal(np.logical_or(a, True),
                        np.array([x or True for x in a], dtype=object))
        assert_equal(np.logical_or(a, 12),
                        np.array([x or 12 for x in a], dtype=object))
        assert_equal(np.logical_or(a, "blah"),
                        np.array([x or "blah" for x in a], dtype=object))

        assert_equal(np.logical_and(a, None),
                        np.array([x and None for x in a], dtype=object))
        assert_equal(np.logical_and(a, True),
                        np.array([x and True for x in a], dtype=object))
        assert_equal(np.logical_and(a, 12),
                        np.array([x and 12 for x in a], dtype=object))
        assert_equal(np.logical_and(a, "blah"),
                        np.array([x and "blah" for x in a], dtype=object))

        assert_equal(np.logical_not(a),
                        np.array([not x for x in a], dtype=object))

        assert_equal(np.logical_or.reduce(a), 3)
        assert_equal(np.logical_and.reduce(a), None)

    def test_object_comparison(self):
        class HasComparisons:
            def __eq__(self, other):
                return '=='

        arr0d = np.array(HasComparisons())
        assert_equal(arr0d == arr0d, True)
        assert_equal(np.equal(arr0d, arr0d), True)  # normal behavior is a cast

        arr1d = np.array([HasComparisons()])
        assert_equal(arr1d == arr1d, np.array([True]))
        assert_equal(np.equal(arr1d, arr1d), np.array([True]))  # normal behavior is a cast
        assert_equal(np.equal(arr1d, arr1d, dtype=object), np.array(['==']))

    def test_object_array_reduction(self):
        # Reductions on object arrays
        a = np.array(['a', 'b', 'c'], dtype=object)
        assert_equal(np.sum(a), 'abc')
        assert_equal(np.max(a), 'c')
        assert_equal(np.min(a), 'a')
        a = np.array([True, False, True], dtype=object)
        assert_equal(np.sum(a), 2)
        assert_equal(np.prod(a), 0)
        assert_equal(np.any(a), True)
        assert_equal(np.all(a), False)
        assert_equal(np.max(a), True)
        assert_equal(np.min(a), False)
        assert_equal(np.array([[1]], dtype=object).sum(), 1)
        assert_equal(np.array([[[1, 2]]], dtype=object).sum((0, 1)), [1, 2])
        assert_equal(np.array([1], dtype=object).sum(initial=1), 2)
        assert_equal(np.array([[1], [2, 3]], dtype=object)
                     .sum(initial=[0], where=[False, True]), [0, 2, 3])

    def test_object_array_accumulate_inplace(self):
        # Checks that in-place accumulates work, see also gh-7402
        arr = np.ones(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        # Twice reproduced also for tuples:
        np.add.accumulate(arr, out=arr)
        np.add.accumulate(arr, out=arr)
        assert_array_equal(arr,
                           np.array([[1]*i for i in [1, 3, 6, 10]], dtype=object),
                          )

        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        np.add.accumulate(arr, out=arr, axis=-1)
        np.add.accumulate(arr, out=arr, axis=-1)
        assert_array_equal(arr[0, :],
                           np.array([[2]*i for i in [1, 3, 6, 10]], dtype=object),
                          )

    def test_object_array_accumulate_failure(self):
        # Typical accumulation on object works as expected:
        res = np.add.accumulate(np.array([1, 0, 2], dtype=object))
        assert_array_equal(res, np.array([1, 1, 3], dtype=object))
        # But errors are propagated from the inner-loop if they occur:
        with pytest.raises(TypeError):
            np.add.accumulate([1, None, 2])

    def test_object_array_reduceat_inplace(self):
        # Checks that in-place reduceats work, see also gh-7465
        arr = np.empty(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        out = np.empty(4, dtype=object)
        out[:] = [[1] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr)
        np.add.reduceat(arr, np.arange(4), out=arr)
        assert_array_equal(arr, out)

        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        out = np.ones((2, 4), dtype=object)
        out[0, :] = [[2] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        assert_array_equal(arr, out)

    def test_object_array_reduceat_failure(self):
        # Reduceat works as expected when no invalid operation occurs (None is
        # not involved in an operation here)
        res = np.add.reduceat(np.array([1, None, 2], dtype=object), [1, 2])
        assert_array_equal(res, np.array([None, 2], dtype=object))
        # But errors when None would be involved in an operation:
        with pytest.raises(TypeError):
            np.add.reduceat([1, None, 2], [0, 2])

    def test_zerosize_reduction(self):
        # Test with default dtype and object dtype
        for a in [[], np.array([], dtype=object)]:
            assert_equal(np.sum(a), 0)
            assert_equal(np.prod(a), 1)
            assert_equal(np.any(a), False)
            assert_equal(np.all(a), True)
            assert_raises(ValueError, np.max, a)
            assert_raises(ValueError, np.min, a)

    def test_axis_out_of_bounds(self):
        a = np.array([False, False])
        assert_raises(np.AxisError, a.all, axis=1)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.all, axis=-2)

        a = np.array([False, False])
        assert_raises(np.AxisError, a.any, axis=1)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.any, axis=-2)

    def test_scalar_reduction(self):
        # The functions 'sum', 'prod', etc allow specifying axis=0
        # even for scalars
        assert_equal(np.sum(3, axis=0), 3)
        assert_equal(np.prod(3.5, axis=0), 3.5)
        assert_equal(np.any(True, axis=0), True)
        assert_equal(np.all(False, axis=0), False)
        assert_equal(np.max(3, axis=0), 3)
        assert_equal(np.min(2.5, axis=0), 2.5)

        # Check scalar behaviour for ufuncs without an identity
        assert_equal(np.power.reduce(3), 3)

        # Make sure that scalars are coming out from this operation
        assert_(type(np.prod(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.sum(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.max(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.min(np.float32(2.5), axis=0)) is np.float32)

        # check if scalars/0-d arrays get cast
        assert_(type(np.any(0, axis=0)) is np.bool_)

        # assert that 0-d arrays get wrapped
        class MyArray(np.ndarray):
            pass
        a = np.array(1).view(MyArray)
        assert_(type(np.any(a)) is MyArray)

    def test_casting_out_param(self):
        # Test that it's possible to do casts on output
        a = np.ones((200, 100), np.int64)
        b = np.ones((200, 100), np.int64)
        c = np.ones((200, 100), np.float64)
        np.add(a, b, out=c)
        assert_equal(c, 2)

        a = np.zeros(65536)
        b = np.zeros(65536, dtype=np.float32)
        np.subtract(a, 0, out=b)
        assert_equal(b, 0)

    def test_where_param(self):
        # Test that the where= ufunc parameter works with regular arrays
        a = np.arange(7)
        b = np.ones(7)
        c = np.zeros(7)
        np.add(a, b, out=c, where=(a % 2 == 1))
        assert_equal(c, [0, 2, 0, 4, 0, 6, 0])

        a = np.arange(4).reshape(2, 2) + 2
        np.power(a, [2, 3], out=a, where=[[0, 1], [1, 0]])
        assert_equal(a, [[2, 27], [16, 5]])
        # Broadcasting the where= parameter
        np.subtract(a, 2, out=a, where=[True, False])
        assert_equal(a, [[0, 27], [14, 5]])

    def test_where_param_buffer_output(self):
        # This test is temporarily skipped because it requires
        # adding masking features to the nditer to work properly

        # With casting on output
        a = np.ones(10, np.int64)
        b = np.ones(10, np.int64)
        c = 1.5 * np.ones(10, np.float64)
        np.add(a, b, out=c, where=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        assert_equal(c, [2, 1.5, 1.5, 2, 1.5, 1.5, 2, 2, 2, 1.5])

    def test_where_param_alloc(self):
        # With casting and allocated output
        a = np.array([1], dtype=np.int64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

        # No casting and allocated output
        a = np.array([1], dtype=np.float64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

    def test_where_with_broadcasting(self):
        # See gh-17198
        a = np.random.random((5000, 4))
        b = np.random.random((5000, 1))

        where = a > 0.3
        out = np.full_like(a, 0)
        np.less(a, b, where=where, out=out)
        b_where = np.broadcast_to(b, a.shape)[where]
        assert_array_equal((a[where] < b_where), out[where].astype(bool))
        assert not out[~where].any()  # outside mask, out remains all 0

    def check_identityless_reduction(self, a):
        # np.minimum.reduce is an identityless reduction

        # Verify that it sees the zero at various positions
        a[...] = 1
        a[1, 0, 0] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [1, 0])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[1, 1, 1, 1], [0, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[1, 1, 1], [0, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

        a[...] = 1
        a[0, 1, 0] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [1, 0, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[1, 0, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

        a[...] = 1
        a[0, 0, 1] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [1, 0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[1, 0, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[0, 1, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

    @requires_memory(6 * 1024**3)
    def test_identityless_reduction_huge_array(self):
        # Regression test for gh-20921 (copying identity incorrectly failed)
        arr = np.zeros((2, 2**31), 'uint8')
        arr[:, 0] = [1, 3]
        arr[:, -1] = [4, 1]
        res = np.maximum.reduce(arr, axis=0)
        del arr
        assert res[0] == 3
        assert res[-1] == 4

    def test_identityless_reduction_corder(self):
        a = np.empty((2, 3, 4), order='C')
        self.check_identityless_reduction(a)

    def test_identityless_reduction_forder(self):
        a = np.empty((2, 3, 4), order='F')
        self.check_identityless_reduction(a)

    def test_identityless_reduction_otherorder(self):
        a = np.empty((2, 4, 3), order='C').swapaxes(1, 2)
        self.check_identityless_reduction(a)

    def test_identityless_reduction_noncontig(self):
        a = np.empty((3, 5, 4), order='C').swapaxes(1, 2)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_identityless_reduction_noncontig_unaligned(self):
        a = np.empty((3*4*5*8 + 1,), dtype='i1')
        a = a[1:].view(dtype='f8')
        a.shape = (3, 4, 5)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_initial_reduction(self):
        # np.minimum.reduce is an identityless reduction

        # For cases like np.maximum(np.abs(...), initial=0)
        # More generally, a supremum over non-negative numbers.
        assert_equal(np.maximum.reduce([], initial=0), 0)

        # For cases like reduction of an empty array over the reals.
        assert_equal(np.minimum.reduce([], initial=np.inf), np.inf)
        assert_equal(np.maximum.reduce([], initial=-np.inf), -np.inf)

        # Random tests
        assert_equal(np.minimum.reduce([5], initial=4), 4)
        assert_equal(np.maximum.reduce([4], initial=5), 5)
        assert_equal(np.maximum.reduce([5], initial=4), 5)
        assert_equal(np.minimum.reduce([4], initial=5), 4)

        # Check initial=None raises ValueError for both types of ufunc reductions
        assert_raises(ValueError, np.minimum.reduce, [], initial=None)
        assert_raises(ValueError, np.add.reduce, [], initial=None)

        # Check that np._NoValue gives default behavior.
        assert_equal(np.add.reduce([], initial=np._NoValue), 0)

        # Check that initial kwarg behaves as intended for dtype=object
        a = np.array([10], dtype=object)
        res = np.add.reduce(a, initial=5)
        assert_equal(res, 15)

    @pytest.mark.parametrize('axis', (0, 1, None))
    @pytest.mark.parametrize('where', (np.array([False, True, True]),
                                       np.array([[True], [False], [True]]),
                                       np.array([[True, False, False],
                                                 [False, True, False],
                                                 [False, True, True]])))
    def test_reduction_with_where(self, axis, where):
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.zeros_like(a)
        np.positive(a, out=a_check, where=where)

        res = np.add.reduce(a, axis=axis, where=where)
        check = a_check.sum(axis)
        assert_equal(res, check)
        # Check we do not overwrite elements of a internally.
        assert_array_equal(a, a_copy)

    @pytest.mark.parametrize(('axis', 'where'),
                             ((0, np.array([True, False, True])),
                              (1, [True, True, False]),
                              (None, True)))
    @pytest.mark.parametrize('initial', (-np.inf, 5.))
    def test_reduction_with_where_and_initial(self, axis, where, initial):
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.full(a.shape, -np.inf)
        np.positive(a, out=a_check, where=where)

        res = np.maximum.reduce(a, axis=axis, where=where, initial=initial)
        check = a_check.max(axis, initial=initial)
        assert_equal(res, check)

    def test_reduction_where_initial_needed(self):
        a = np.arange(9.).reshape(3, 3)
        m = [False, True, False]
        assert_raises(ValueError, np.maximum.reduce, a, where=m)

    def test_identityless_reduction_nonreorderable(self):
        a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])

        res = np.divide.reduce(a, axis=0)
        assert_equal(res, [8.0, 4.0, 8.0])

        res = np.divide.reduce(a, axis=1)
        assert_equal(res, [2.0, 8.0])

        res = np.divide.reduce(a, axis=())
        assert_equal(res, a)

        assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))

    def test_reduce_zero_axis(self):
        # If we have a n x m array and do a reduction with axis=1, then we are
        # doing n reductions, and each reduction takes an m-element array. For
        # a reduction operation without an identity, then:
        #   n > 0, m > 0: fine
        #   n = 0, m > 0: fine, doing 0 reductions of m-element arrays
        #   n > 0, m = 0: can't reduce a 0-element array, ValueError
        #   n = 0, m = 0: can't reduce a 0-element array, ValueError (for
        #     consistency with the above case)
        # This test doesn't actually look at return values, it just checks to
        # make sure that error we get an error in exactly those cases where we
        # expect one, and assumes the calculations themselves are done
        # correctly.

        def ok(f, *args, **kwargs):
            f(*args, **kwargs)

        def err(f, *args, **kwargs):
            assert_raises(ValueError, f, *args, **kwargs)

        def t(expect, func, n, m):
            expect(func, np.zeros((n, m)), axis=1)
            expect(func, np.zeros((m, n)), axis=0)
            expect(func, np.zeros((n // 2, n // 2, m)), axis=2)
            expect(func, np.zeros((n // 2, m, n // 2)), axis=1)
            expect(func, np.zeros((n, m // 2, m // 2)), axis=(1, 2))
            expect(func, np.zeros((m // 2, n, m // 2)), axis=(0, 2))
            expect(func, np.zeros((m // 3, m // 3, m // 3,
                                  n // 2, n // 2)),
                                 axis=(0, 1, 2))
            # Check what happens if the inner (resp. outer) dimensions are a
            # mix of zero and non-zero:
            expect(func, np.zeros((10, m, n)), axis=(0, 1))
            expect(func, np.zeros((10, n, m)), axis=(0, 2))
            expect(func, np.zeros((m, 10, n)), axis=0)
            expect(func, np.zeros((10, m, n)), axis=1)
            expect(func, np.zeros((10, n, m)), axis=2)

        # np.maximum is just an arbitrary ufunc with no reduction identity
        assert_equal(np.maximum.identity, None)
        t(ok, np.maximum.reduce, 30, 30)
        t(ok, np.maximum.reduce, 0, 30)
        t(err, np.maximum.reduce, 30, 0)
        t(err, np.maximum.reduce, 0, 0)
        err(np.maximum.reduce, [])
        np.maximum.reduce(np.zeros((0, 0)), axis=())

        # all of the combinations are fine for a reduction that has an
        # identity
        t(ok, np.add.reduce, 30, 30)
        t(ok, np.add.reduce, 0, 30)
        t(ok, np.add.reduce, 30, 0)
        t(ok, np.add.reduce, 0, 0)
        np.add.reduce([])
        np.add.reduce(np.zeros((0, 0)), axis=())

        # OTOH, accumulate always makes sense for any combination of n and m,
        # because it maps an m-element array to an m-element array. These
        # tests are simpler because accumulate doesn't accept multiple axes.
        for uf in (np.maximum, np.add):
            uf.accumulate(np.zeros((30, 0)), axis=0)
            uf.accumulate(np.zeros((0, 30)), axis=0)
            uf.accumulate(np.zeros((30, 30)), axis=0)
            uf.accumulate(np.zeros((0, 0)), axis=0)

    def test_safe_casting(self):
        # In old versions of numpy, in-place operations used the 'unsafe'
        # casting rules. In versions >= 1.10, 'same_kind' is the
        # default and an exception is raised instead of a warning.
        # when 'same_kind' is not satisfied.
        a = np.array([1, 2, 3], dtype=int)
        # Non-in-place addition is fine
        assert_array_equal(assert_no_warnings(np.add, a, 1.1),
                           [2.1, 3.1, 4.1])
        assert_raises(TypeError, np.add, a, 1.1, out=a)

        def add_inplace(a, b):
            a += b

        assert_raises(TypeError, add_inplace, a, 1.1)
        # Make sure that explicitly overriding the exception is allowed:
        assert_no_warnings(np.add, a, 1.1, out=a, casting="unsafe")
        assert_array_equal(a, [2, 3, 4])

    def test_ufunc_custom_out(self):
        # Test ufunc with built in input types and custom output type

        a = np.array([0, 1, 2], dtype='i8')
        b = np.array([0, 1, 2], dtype='i8')
        c = np.empty(3, dtype=_rational_tests.rational)

        # Output must be specified so numpy knows what
        # ufunc signature to look for
        result = _rational_tests.test_add(a, b, c)
        target = np.array([0, 2, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)

        # The new resolution means that we can (usually) find custom loops
        # as long as they match exactly:
        result = _rational_tests.test_add(a, b)
        assert_equal(result, target)

        # This works even more generally, so long the default common-dtype
        # promoter works out:
        result = _rational_tests.test_add(a, b.astype(np.uint16), out=c)
        assert_equal(result, target)

        # But, it can be fooled, e.g. (use scalars, which forces legacy
        # type resolution to kick in, which then fails):
        with assert_raises(TypeError):
            _rational_tests.test_add(a, np.uint16(2))

    def test_operand_flags(self):
        a = np.arange(16, dtype='l').reshape(4, 4)
        b = np.arange(9, dtype='l').reshape(3, 3)
        opflag_tests.inplace_add(a[:-1, :-1], b)
        assert_equal(a, np.array([[0, 2, 4, 3], [7, 9, 11, 7],
            [14, 16, 18, 11], [12, 13, 14, 15]], dtype='l'))

        a = np.array(0)
        opflag_tests.inplace_add(a, 3)
        assert_equal(a, 3)
        opflag_tests.inplace_add(a, [3, 4])
        assert_equal(a, 10)

    def test_struct_ufunc(self):
        import numpy.core._struct_ufunc_tests as struct_ufunc

        a = np.array([(1, 2, 3)], dtype='u8,u8,u8')
        b = np.array([(1, 2, 3)], dtype='u8,u8,u8')

        result = struct_ufunc.add_triplet(a, b)
        assert_equal(result, np.array([(2, 4, 6)], dtype='u8,u8,u8'))
        assert_raises(RuntimeError, struct_ufunc.register_fail)

    def test_custom_ufunc(self):
        a = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)
        b = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)

        result = _rational_tests.test_add_rationals(a, b)
        expected = np.array(
            [_rational_tests.rational(1),
             _rational_tests.rational(2, 3),
             _rational_tests.rational(1, 2)],
            dtype=_rational_tests.rational)
        assert_equal(result, expected)

    def test_custom_ufunc_forced_sig(self):
        # gh-9351 - looking for a non-first userloop would previously hang
        with assert_raises(TypeError):
            np.multiply(_rational_tests.rational(1), 1,
                        signature=(_rational_tests.rational, int, None))

    def test_custom_array_like(self):

        class MyThing:
            __array_priority__ = 1000

            rmul_count = 0
            getitem_count = 0

            def __init__(self, shape):
                self.shape = shape

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, i):
                MyThing.getitem_count += 1
                if not isinstance(i, tuple):
                    i = (i,)
                if len(i) > self.ndim:
                    raise IndexError("boo")

                return MyThing(self.shape[len(i):])

            def __rmul__(self, other):
                MyThing.rmul_count += 1
                return self

        np.float64(5)*MyThing((3, 3))
        assert_(MyThing.rmul_count == 1, MyThing.rmul_count)
        assert_(MyThing.getitem_count <= 2, MyThing.getitem_count)

    def test_inplace_fancy_indexing(self):

        a = np.arange(10)
        np.add.at(a, [2, 5, 2], 1)
        assert_equal(a, [0, 1, 4, 3, 4, 6, 6, 7, 8, 9])

        a = np.arange(10)
        b = np.array([100, 100, 100])
        np.add.at(a, [2, 5, 2], b)
        assert_equal(a, [0, 1, 202, 3, 4, 105, 6, 7, 8, 9])

        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, (slice(None), [1, 2, 1]), b)
        assert_equal(a, [[0, 201, 102], [3, 404, 205], [6, 607, 308]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), [1, 2, 1]), b)
        assert_equal(a,
            [[[0, 401, 202],
              [3, 404, 205],
              [6, 407, 208]],

             [[9, 410, 211],
              [12, 413, 214],
              [15, 416, 217]],

             [[18, 419, 220],
              [21, 422, 223],
              [24, 425, 226]]])

        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, ([1, 2, 1], slice(None)), b)
        assert_equal(a, [[0, 1, 2], [403, 404, 405], [206, 207, 208]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), [1, 2, 1], slice(None)), b)
        assert_equal(a,
            [[[0,  1,  2],
              [203, 404, 605],
              [106, 207, 308]],

             [[9,  10, 11],
              [212, 413, 614],
              [115, 216, 317]],

             [[18, 19, 20],
              [221, 422, 623],
              [124, 225, 326]]])

        a = np.arange(9).reshape(3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (0, [1, 2, 1]), b)
        assert_equal(a, [[0, 401, 202], [3, 4, 5], [6, 7, 8]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, ([1, 2, 1], 0, slice(None)), b)
        assert_equal(a,
            [[[0,  1,  2],
              [3,  4,  5],
              [6,  7,  8]],

             [[209, 410, 611],
              [12,  13, 14],
              [15,  16, 17]],

             [[118, 219, 320],
              [21,  22, 23],
              [24,  25, 26]]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), slice(None)), b)
        assert_equal(a,
            [[[100, 201, 302],
              [103, 204, 305],
              [106, 207, 308]],

             [[109, 210, 311],
              [112, 213, 314],
              [115, 216, 317]],

             [[118, 219, 320],
              [121, 222, 323],
              [124, 225, 326]]])

        a = np.arange(10)
        np.negative.at(a, [2, 5, 2])
        assert_equal(a, [0, 1, 2, 3, 4, -5, 6, 7, 8, 9])

        # Test 0-dim array
        a = np.array(0)
        np.add.at(a, (), 1)
        assert_equal(a, 1)

        assert_raises(IndexError, np.add.at, a, 0, 1)
        assert_raises(IndexError, np.add.at, a, [], 1)

        # Test mixed dtypes
        a = np.arange(10)
        np.power.at(a, [1, 2, 3, 2], 3.5)
        assert_equal(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))

        # Test boolean indexing and boolean ufuncs
        a = np.arange(10)
        index = a % 2 == 0
        np.equal.at(a, index, [0, 2, 4, 6, 8])
        assert_equal(a, [1, 1, 1, 3, 1, 5, 1, 7, 1, 9])

        # Test unary operator
        a = np.arange(10, dtype='u4')
        np.invert.at(a, [2, 5, 2])
        assert_equal(a, [0, 1, 2, 3, 4, 5 ^ 0xffffffff, 6, 7, 8, 9])

        # Test empty subspace
        orig = np.arange(4)
        a = orig[:, None][:, 0:0]
        np.add.at(a, [0, 1], 3)
        assert_array_equal(orig, np.arange(4))

        # Test with swapped byte order
        index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
        values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
        np.add.at(values, index, 3)
        assert_array_equal(values, [1, 8, 6, 4])

        # Test exception thrown
        values = np.array(['a', 1], dtype=object)
        assert_raises(TypeError, np.add.at, values, [0, 1], 1)
        assert_array_equal(values, np.array(['a', 1], dtype=object))

        # Test multiple output ufuncs raise error, gh-5665
        assert_raises(ValueError, np.modf.at, np.arange(10), [1])

        # Test maximum
        a = np.array([1, 2, 3])
        np.maximum.at(a, [0], 0)
        assert_equal(np.array([1, 2, 3]), a)

    def test_at_not_none_signature(self):
        # Test ufuncs with non-trivial signature raise a TypeError
        a = np.ones((2, 2, 2))
        b = np.ones((1, 2, 2))
        assert_raises(TypeError, np.matmul.at, a, [0], b)

        a = np.array([[[1, 2], [3, 4]]])
        assert_raises(TypeError, np.linalg._umath_linalg.det.at, a, [0])

    def test_reduce_arguments(self):
        f = np.add.reduce
        d = np.ones((5,2), dtype=int)
        o = np.ones((2,), dtype=d.dtype)
        r = o * 5
        assert_equal(f(d), r)
        # a, axis=0, dtype=None, out=None, keepdims=False
        assert_equal(f(d, axis=0), r)
        assert_equal(f(d, 0), r)
        assert_equal(f(d, 0, dtype=None), r)
        assert_equal(f(d, 0, dtype='i'), r)
        assert_equal(f(d, 0, 'i'), r)
        assert_equal(f(d, 0, None), r)
        assert_equal(f(d, 0, None, out=None), r)
        assert_equal(f(d, 0, None, out=o), r)
        assert_equal(f(d, 0, None, o), r)
        assert_equal(f(d, 0, None, None), r)
        assert_equal(f(d, 0, None, None, keepdims=False), r)
        assert_equal(f(d, 0, None, None, True), r.reshape((1,) + r.shape))
        assert_equal(f(d, 0, None, None, False, 0), r)
        assert_equal(f(d, 0, None, None, False, initial=0), r)
        assert_equal(f(d, 0, None, None, False, 0, True), r)
        assert_equal(f(d, 0, None, None, False, 0, where=True), r)
        # multiple keywords
        assert_equal(f(d, axis=0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False, initial=0,
                       where=True), r)

        # too little
        assert_raises(TypeError, f)
        # too much
        assert_raises(TypeError, f, d, 0, None, None, False, 0, True, 1)
        # invalid axis
        assert_raises(TypeError, f, d, "invalid")
        assert_raises(TypeError, f, d, axis="invalid")
        assert_raises(TypeError, f, d, axis="invalid", dtype=None,
                      keepdims=True)
        # invalid dtype
        assert_raises(TypeError, f, d, 0, "invalid")
        assert_raises(TypeError, f, d, dtype="invalid")
        assert_raises(TypeError, f, d, dtype="invalid", out=None)
        # invalid out
        assert_raises(TypeError, f, d, 0, None, "invalid")
        assert_raises(TypeError, f, d, out="invalid")
        assert_raises(TypeError, f, d, out="invalid", dtype=None)
        # keepdims boolean, no invalid value
        # assert_raises(TypeError, f, d, 0, None, None, "invalid")
        # assert_raises(TypeError, f, d, keepdims="invalid", axis=0, dtype=None)
        # invalid mix
        assert_raises(TypeError, f, d, 0, keepdims="invalid", dtype="invalid",
                     out=None)

        # invalid keyword
        assert_raises(TypeError, f, d, axis=0, dtype=None, invalid=0)
        assert_raises(TypeError, f, d, invalid=0)
        assert_raises(TypeError, f, d, 0, keepdims=True, invalid="invalid",
                      out=None)
        assert_raises(TypeError, f, d, axis=0, dtype=None, keepdims=True,
                      out=None, invalid=0)
        assert_raises(TypeError, f, d, axis=0, dtype=None,
                      out=None, invalid=0)

    def test_structured_equal(self):
        # https://github.com/numpy/numpy/issues/4855

        class MyA(np.ndarray):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return getattr(ufunc, method)(*(input.view(np.ndarray)
                                              for input in inputs), **kwargs)
        a = np.arange(12.).reshape(4,3)
        ra = a.view(dtype=('f8,f8,f8')).squeeze()
        mra = ra.view(MyA)

        target = np.array([ True, False, False, False], dtype=bool)
        assert_equal(np.all(target == (mra == ra[0])), True)

    def test_scalar_equal(self):
        # Scalar comparisons should always work, without deprecation warnings.
        # even when the ufunc fails.
        a = np.array(0.)
        b = np.array('a')
        assert_(a != b)
        assert_(b != a)
        assert_(not (a == b))
        assert_(not (b == a))

    def test_NotImplemented_not_returned(self):
        # See gh-5964 and gh-2091. Some of these functions are not operator
        # related and were fixed for other reasons in the past.
        binary_funcs = [
            np.power, np.add, np.subtract, np.multiply, np.divide,
            np.true_divide, np.floor_divide, np.bitwise_and, np.bitwise_or,
            np.bitwise_xor, np.left_shift, np.right_shift, np.fmax,
            np.fmin, np.fmod, np.hypot, np.logaddexp, np.logaddexp2,
            np.maximum, np.minimum, np.mod,
            np.greater, np.greater_equal, np.less, np.less_equal,
            np.equal, np.not_equal]

        a = np.array('1')
        b = 1
        c = np.array([1., 2.])
        for f in binary_funcs:
            assert_raises(TypeError, f, a, b)
            assert_raises(TypeError, f, c, a)

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or])  # logical_xor object loop is bad
    @pytest.mark.parametrize("signature",
             [(None, None, object), (object, None, None),
              (None, object, None)])
    def test_logical_ufuncs_object_signatures(self, ufunc, signature):
        a = np.array([True, None, False], dtype=object)
        res = ufunc(a, a, signature=signature)
        assert res.dtype == object

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize("signature",
                 [(bool, None, object), (object, None, bool),
                  (None, object, bool)])
    def test_logical_ufuncs_mixed_object_signatures(self, ufunc, signature):
        # Most mixed signatures fail (except those with bool out, e.g. `OO->?`)
        a = np.array([True, None, False])
        with pytest.raises(TypeError):
            ufunc(a, a, signature=signature)

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_support_anything(self, ufunc):
        # The logical ufuncs support even input that can't be promoted:
        a = np.array(b'1', dtype="V3")
        c = np.array([1., 2.])
        assert_array_equal(ufunc(a, c), ufunc([True, True], True))
        assert ufunc.reduce(a) == True
        # check that the output has no effect:
        out = np.zeros(2, dtype=np.int32)
        expected = ufunc([True, True], True).astype(out.dtype)
        assert_array_equal(ufunc(a, c, out=out), expected)
        out = np.zeros((), dtype=np.int32)
        assert ufunc.reduce(a, out=out) == True
        # Last check, test reduction when out and a match (the complexity here
        # is that the "i,i->?" may seem right, but should not match.
        a = np.array([3], dtype="i")
        out = np.zeros((), dtype=a.dtype)
        assert ufunc.reduce(a, out=out) == 1

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_reject_string(self, ufunc):
        """
        Logical ufuncs are normally well defined by working with the boolean
        equivalent, i.e. casting all inputs to bools should work.

        However, casting strings to bools is *currently* weird, because it
        actually uses `bool(int(str))`.  Thus we explicitly reject strings.
        This test should succeed (and can probably just be removed) as soon as
        string to bool casts are well defined in NumPy.
        """
        with pytest.raises(TypeError, match="contain a loop with signature"):
            ufunc(["1"], ["3"])
        with pytest.raises(TypeError, match="contain a loop with signature"):
            ufunc.reduce(["1", "2", "0"])

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_out_cast_check(self, ufunc):
        a = np.array('1')
        c = np.array([1., 2.])
        out = a.copy()
        with pytest.raises(TypeError):
            # It would be safe, but not equiv casting:
            ufunc(a, c, out=out, casting="equiv")

    def test_reducelike_byteorder_resolution(self):
        # See gh-20699, byte-order changes need some extra care in the type
        # resolution to make the following succeed:
        arr_be = np.arange(10, dtype=">i8")
        arr_le = np.arange(10, dtype="<i8")

        assert np.add.reduce(arr_be) == np.add.reduce(arr_le)
        assert_array_equal(np.add.accumulate(arr_be), np.add.accumulate(arr_le))
        assert_array_equal(
            np.add.reduceat(arr_be, [1]), np.add.reduceat(arr_le, [1]))

    def test_reducelike_out_promotes(self):
        # Check that the out argument to reductions is considered for
        # promotion.  See also gh-20455.
        # Note that these paths could prefer `initial=` in the future and
        # do not up-cast to the default integer for add and prod
        arr = np.ones(1000, dtype=np.uint8)
        out = np.zeros((), dtype=np.uint16)
        assert np.add.reduce(arr, out=out) == 1000
        arr[:10] = 2
        assert np.multiply.reduce(arr, out=out) == 2**10

        # For legacy dtypes, the signature currently has to be forced if `out=`
        # is passed.  The two paths below should differ, without `dtype=` the
        # expected result should be: `np.prod(arr.astype("f8")).astype("f4")`!
        arr = np.full(5, 2**25-1, dtype=np.int64)

        # float32 and int64 promote to float64:
        res = np.zeros((), dtype=np.float32)
        # If `dtype=` is passed, the calculation is forced to float32:
        single_res = np.zeros((), dtype=np.float32)
        np.multiply.reduce(arr, out=single_res, dtype=np.float32)
        assert single_res != res

    def test_reducelike_output_needs_identical_cast(self):
        # Checks the case where the we have a simple byte-swap works, maily
        # tests that this is not rejected directly.
        # (interesting because we require descriptor identity in reducelikes).
        arr = np.ones(20, dtype="f8")
        out = np.empty((), dtype=arr.dtype.newbyteorder())
        expected = np.add.reduce(arr)
        np.add.reduce(arr, out=out)
        assert_array_equal(expected, out)
        # Check reduceat:
        out = np.empty(2, dtype=arr.dtype.newbyteorder())
        expected = np.add.reduceat(arr, [0, 1])
        np.add.reduceat(arr, [0, 1], out=out)
        assert_array_equal(expected, out)
        # And accumulate:
        out = np.empty(arr.shape, dtype=arr.dtype.newbyteorder())
        expected = np.add.accumulate(arr)
        np.add.accumulate(arr, out=out)
        assert_array_equal(expected, out)

    def test_reduce_noncontig_output(self):
        # Check that reduction deals with non-contiguous output arrays
        # appropriately.
        #
        # gh-8036

        x = np.arange(7*13*8, dtype=np.int16).reshape(7, 13, 8)
        x = x[4:6,1:11:6,1:5].transpose(1, 2, 0)
        y_base = np.arange(4*4, dtype=np.int16).reshape(4, 4)
        y = y_base[::2,:]

        y_base_copy = y_base.copy()

        r0 = np.add.reduce(x, out=y.copy(), axis=2)
        r1 = np.add.reduce(x, out=y, axis=2)

        # The results should match, and y_base shouldn't get clobbered
        assert_equal(r0, r1)
        assert_equal(y_base[1,:], y_base_copy[1,:])
        assert_equal(y_base[3,:], y_base_copy[3,:])

    @pytest.mark.parametrize("with_cast", [True, False])
    def test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast):
        # Should raise an error mentioning "shape" or "size"
        arr = np.arange(5)
        out = np.arange(3)  # definitely wrong shape
        if with_cast:
            # If a cast is necessary on the output, we can be sure to use
            # the generic NpyIter (non-fast) path.
            out = out.astype(np.float64)

        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.reduceat(arr, [0, 3], out=out)

        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.accumulate(arr, out=out)

    @pytest.mark.parametrize('out_shape',
                             [(), (1,), (3,), (1, 1), (1, 3), (4, 3)])
    @pytest.mark.parametrize('keepdims', [True, False])
    @pytest.mark.parametrize('f_reduce', [np.add.reduce, np.minimum.reduce])
    def test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape):
        # Test that we're not incorrectly broadcasting dimensions.
        # See gh-15144 (failed for np.add.reduce previously).
        a = np.arange(12.).reshape(4, 3)
        out = np.empty(out_shape, a.dtype)

        correct_out = f_reduce(a, axis=0, keepdims=keepdims)
        if out_shape != correct_out.shape:
            with assert_raises(ValueError):
                f_reduce(a, axis=0, out=out, keepdims=keepdims)
        else:
            check = f_reduce(a, axis=0, out=out, keepdims=keepdims)
            assert_(check is out)
            assert_array_equal(check, correct_out)

    def test_reduce_output_does_not_broadcast_input(self):
        # Test that the output shape cannot broadcast an input dimension
        # (it never can add dimensions, but it might expand an existing one)
        a = np.ones((1, 10))
        out_correct = (np.empty((1, 1)))
        out_incorrect = np.empty((3, 1))
        np.add.reduce(a, axis=-1, out=out_correct, keepdims=True)
        np.add.reduce(a, axis=-1, out=out_correct[:, 0], keepdims=False)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect, keepdims=True)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect[:, 0], keepdims=False)

    def test_reduce_output_subclass_ok(self):
        class MyArr(np.ndarray):
            pass

        out = np.empty(())
        np.add.reduce(np.ones(5), out=out)  # no subclass, all fine
        out = out.view(MyArr)
        assert np.add.reduce(np.ones(5), out=out) is out
        assert type(np.add.reduce(out)) is MyArr

    def test_no_doc_string(self):
        # gh-9337
        assert_('\n' not in umt.inner1d_no_doc.__doc__)

    def test_invalid_args(self):
        # gh-7961
        exc = pytest.raises(TypeError, np.sqrt, None)
        # minimally check the exception text
        assert exc.match('loop of ufunc does not support')

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_finite(self, nat):
        try:
            assert not np.isfinite(nat)
        except TypeError:
            pass  # ok, just not implemented

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_nan(self, nat):
        try:
            assert np.isnan(nat)
        except TypeError:
            pass  # ok, just not implemented

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_inf(self, nat):
        try:
            assert not np.isinf(nat)
        except TypeError:
            pass  # ok, just not implemented


@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_types(ufunc):
    '''
    Check all ufuncs that the correct type is returned. Avoid
    object and boolean types since many operations are not defined for
    for them.

    Choose the shape so even dot and matmul will succeed
    '''
    for typ in ufunc.types:
        # types is a list of strings like ii->i
        if 'O' in typ or '?' in typ:
            continue
        inp, out = typ.split('->')
        args = [np.ones((3, 3), t) for t in inp]
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res = ufunc(*args)
        if isinstance(res, tuple):
            outs = tuple(out)
            assert len(res) == len(outs)
            for r, t in zip(res, outs):
                assert r.dtype == np.dtype(t)
        else:
            assert res.dtype == np.dtype(out)

@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
@np._no_nep50_warning()
def test_ufunc_noncontiguous(ufunc):
    '''
    Check that contiguous and non-contiguous calls to ufuncs
    have the same results for values in range(9)
    '''
    for typ in ufunc.types:
        # types is a list of strings like ii->i
        if any(set('O?mM') & set(typ)):
            # bool, object, datetime are too irregular for this simple test
            continue
        inp, out = typ.split('->')
        args_c = [np.empty(6, t) for t in inp]
        args_n = [np.empty(18, t)[::3] for t in inp]
        for a in args_c:
            a.flat = range(1,7)
        for a in args_n:
            a.flat = range(1,7)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res_c = ufunc(*args_c)
            res_n = ufunc(*args_n)
        if len(out) == 1:
            res_c = (res_c,)
            res_n = (res_n,)
        for c_ar, n_ar in zip(res_c, res_n):
            dt = c_ar.dtype
            if np.issubdtype(dt, np.floating):
                # for floating point results allow a small fuss in comparisons
                # since different algorithms (libm vs. intrinsics) can be used
                # for different input strides
                res_eps = np.finfo(dt).eps
                tol = 2*res_eps
                assert_allclose(res_c, res_n, atol=tol, rtol=tol)
            else:
                assert_equal(c_ar, n_ar)


@pytest.mark.parametrize('ufunc', [np.sign, np.equal])
def test_ufunc_warn_with_nan(ufunc):
    # issue gh-15127
    # test that calling certain ufuncs with a non-standard `nan` value does not
    # emit a warning
    # `b` holds a 64 bit signaling nan: the most significant bit of the
    # significand is zero.
    b = np.array([0x7ff0000000000001], 'i8').view('f8')
    assert np.isnan(b)
    if ufunc.nin == 1:
        ufunc(b)
    elif ufunc.nin == 2:
        ufunc(b, b.copy())
    else:
        raise ValueError('ufunc with more than 2 inputs')


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_ufunc_out_casterrors():
    # Tests that casting errors are correctly reported and buffers are
    # cleared.
    # The following array can be added to itself as an object array, but
    # the result cannot be cast to an integer output:
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.array([value] * int(np.BUFSIZE * 1.5) +
                   ["string"] +
                   [value] * int(1.5 * np.BUFSIZE), dtype=object)
    out = np.ones(len(arr), dtype=np.intp)

    count = sys.getrefcount(value)
    with pytest.raises(ValueError):
        # Output casting failure:
        np.add(arr, arr, out=out, casting="unsafe")

    assert count == sys.getrefcount(value)
    # output is unchanged after the error, this shows that the iteration
    # was aborted (this is not necessarily defined behaviour)
    assert out[-1] == 1

    with pytest.raises(ValueError):
        # Input casting failure:
        np.add(arr, arr, out=out, dtype=np.intp, casting="unsafe")

    assert count == sys.getrefcount(value)
    # output is unchanged after the error, this shows that the iteration
    # was aborted (this is not necessarily defined behaviour)
    assert out[-1] == 1


@pytest.mark.parametrize("bad_offset", [0, int(np.BUFSIZE * 1.5)])
def test_ufunc_input_casterrors(bad_offset):
    value = 123
    arr = np.array([value] * bad_offset +
                   ["string"] +
                   [value] * int(1.5 * np.BUFSIZE), dtype=object)
    with pytest.raises(ValueError):
        # Force cast inputs, but the buffered cast of `arr` to intp fails:
        np.add(arr, arr, dtype=np.intp, casting="unsafe")


@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize("bad_offset", [0, int(np.BUFSIZE * 1.5)])
def test_ufunc_input_floatingpoint_error(bad_offset):
    value = 123
    arr = np.array([value] * bad_offset +
                   [np.nan] +
                   [value] * int(1.5 * np.BUFSIZE))
    with np.errstate(invalid="raise"), pytest.raises(FloatingPointError):
        # Force cast inputs, but the buffered cast of `arr` to intp fails:
        np.add(arr, arr, dtype=np.intp, casting="unsafe")


def test_trivial_loop_invalid_cast():
    # This tests the fast-path "invalid cast", see gh-19904.
    with pytest.raises(TypeError,
            match="cast ufunc 'add' input 0"):
        # the void dtype definitely cannot cast to double:
        np.add(np.array(1, "i,i"), 3, signature="dd->d")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
@pytest.mark.parametrize("offset",
        [0, np.BUFSIZE//2, int(1.5*np.BUFSIZE)])
def test_reduce_casterrors(offset):
    # Test reporting of casting errors in reductions, we test various
    # offsets to where the casting error will occur, since these may occur
    # at different places during the reduction procedure. For example
    # the first item may be special.
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.array([value] * offset +
                   ["string"] +
                   [value] * int(1.5 * np.BUFSIZE), dtype=object)
    out = np.array(-1, dtype=np.intp)

    count = sys.getrefcount(value)
    with pytest.raises(ValueError, match="invalid literal"):
        # This is an unsafe cast, but we currently always allow that.
        # Note that the double loop is picked, but the cast fails.
        np.add.reduce(arr, dtype=np.intp, out=out)
    assert count == sys.getrefcount(value)
    # If an error occurred during casting, the operation is done at most until
    # the error occurs (the result of which would be `value * offset`) and -1
    # if the error happened immediately.
    # This does not define behaviour, the output is invalid and thus undefined
    assert out[()] < value * offset


@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize("method",
        [np.add.accumulate, np.add.reduce,
         pytest.param(lambda x: np.add.reduceat(x, [0]), id="reduceat"),
         pytest.param(lambda x: np.log.at(x, [2]), id="at")])
def test_ufunc_methods_floaterrors(method):
    # adding inf and -inf (or log(-inf) creates an invalid float and warns
    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all="warn"):
        with pytest.warns(RuntimeWarning, match="invalid value"):
            method(arr)

    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError):
            method(arr)


def _check_neg_zero(value):
    if value != 0.0:
        return False
    if not np.signbit(value.real):
        return False
    if value.dtype.kind == "c":
        return np.signbit(value.imag)
    return True

@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_addition_negative_zero(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)

    arr = np.array(neg_zero)
    arr2 = np.array(neg_zero)

    assert _check_neg_zero(arr + arr2)
    # In-place ops may end up on a different path (reduce path) see gh-21211
    arr += arr2
    assert _check_neg_zero(arr)


@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
@pytest.mark.parametrize("use_initial", [True, False])
def test_addition_reduce_negative_zero(dtype, use_initial):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)

    kwargs = {}
    if use_initial:
        kwargs["initial"] = neg_zero
    else:
        pytest.xfail("-0. propagation in sum currently requires initial")

    # Test various length, in case SIMD paths or chunking play a role.
    # 150 extends beyond the pairwise blocksize; probably not important.
    for i in range(0, 150):
        arr = np.array([neg_zero] * i, dtype=dtype)
        res = np.sum(arr, **kwargs)
        if i > 0 or use_initial:
            assert _check_neg_zero(res)
        else:
            # `sum([])` should probably be 0.0 and not -0.0 like `sum([-0.0])`
            assert not np.signbit(res.real)
            assert not np.signbit(res.imag)

class TestLowlevelAPIAccess:
    def test_resolve_dtypes_basic(self):
        # Basic test for dtype resolution:
        i4 = np.dtype("i4")
        f4 = np.dtype("f4")
        f8 = np.dtype("f8")

        r = np.add.resolve_dtypes((i4, f4, None))
        assert r == (f8, f8, f8)

        # Signature uses the same logic to parse as ufunc (less strict)
        # the following is "same-kind" casting so works:
        r = np.add.resolve_dtypes((
                i4, i4, None), signature=(None, None, "f4"))
        assert r == (f4, f4, f4)

        # Check NEP 50 "weak" promotion also:
        r = np.add.resolve_dtypes((f4, int, None))
        assert r == (f4, f4, f4)

        with pytest.raises(TypeError):
            np.add.resolve_dtypes((i4, f4, None), casting="no")

    def test_weird_dtypes(self):
        S0 = np.dtype("S0")
        # S0 is often converted by NumPy to S1, but not here:
        r = np.equal.resolve_dtypes((S0, S0, None))
        assert r == (S0, S0, np.dtype(bool))

        # Subarray dtypes are weird and only really exist nested, they need
        # the shift to full NEP 50 to be implemented nicely:
        dts = np.dtype("10i")
        with pytest.raises(NotImplementedError):
            np.equal.resolve_dtypes((dts, dts, None))

    def test_resolve_dtypes_reduction(self):
        i4 = np.dtype("i4")
        with pytest.raises(NotImplementedError):
            np.add.resolve_dtypes((i4, i4, i4), reduction=True)

    @pytest.mark.parametrize("dtypes", [
            (np.dtype("i"), np.dtype("i")),
            (None, np.dtype("i"), np.dtype("f")),
            (np.dtype("i"), None, np.dtype("f")),
            ("i4", "i4", None)])
    def test_resolve_dtypes_errors(self, dtypes):
        with pytest.raises(TypeError):
            np.add.resolve_dtypes(dtypes)

    def test_resolve_dtypes_reduction(self):
        i2 = np.dtype("i2")
        long_ = np.dtype("long")
        # Check special addition resolution:
        res = np.add.resolve_dtypes((None, i2, None), reduction=True)
        assert res == (long_, long_, long_)

    def test_resolve_dtypes_reduction_errors(self):
        i2 = np.dtype("i2")

        with pytest.raises(TypeError):
            np.add.resolve_dtypes((None, i2, i2))

        with pytest.raises(TypeError):
            np.add.signature((None, None, "i4"))

    @pytest.mark.skipif(not hasattr(ct, "pythonapi"),
            reason="`ctypes.pythonapi` required for capsule unpacking.")
    def test_loop_access(self):
        # This is a basic test for the full strided loop access
        data_t = ct.ARRAY(ct.c_char_p, 2)
        dim_t = ct.ARRAY(ct.c_ssize_t, 1)
        strides_t = ct.ARRAY(ct.c_ssize_t, 2)
        strided_loop_t = ct.CFUNCTYPE(
                ct.c_int, ct.c_void_p, data_t, dim_t, strides_t, ct.c_void_p)

        class call_info_t(ct.Structure):
            _fields_ = [
                ("strided_loop", strided_loop_t),
                ("context", ct.c_void_p),
                ("auxdata", ct.c_void_p),
                ("requires_pyapi", ct.c_byte),
                ("no_floatingpoint_errors", ct.c_byte),
            ]

        i4 = np.dtype("i4")
        dt, call_info_obj = np.negative._resolve_dtypes_and_context((i4, i4))
        assert dt == (i4, i4)  # can be used without casting

        # Fill in the rest of the information:
        np.negative._get_strided_loop(call_info_obj)

        ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
        call_info = ct.pythonapi.PyCapsule_GetPointer(
                ct.py_object(call_info_obj),
                ct.c_char_p(b"numpy_1.24_ufunc_call_info"))

        call_info = ct.cast(call_info, ct.POINTER(call_info_t)).contents

        arr = np.arange(10, dtype=i4)
        call_info.strided_loop(
                call_info.context,
                data_t(arr.ctypes.data, arr.ctypes.data),
                arr.ctypes.shape,  # is a C-array with 10 here
                strides_t(arr.ctypes.strides[0], arr.ctypes.strides[0]),
                call_info.auxdata)

        # We just directly called the negative inner-loop in-place:
        assert_array_equal(arr, -np.arange(10, dtype=i4))

    @pytest.mark.parametrize("strides", [1, (1, 2, 3), (1, "2")])
    def test__get_strided_loop_errors_bad_strides(self, strides):
        i4 = np.dtype("i4")
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))

        with pytest.raises(TypeError, match="fixed_strides.*tuple.*or None"):
            np.negative._get_strided_loop(call_info, fixed_strides=strides)

    def test__get_strided_loop_errors_bad_call_info(self):
        i4 = np.dtype("i4")
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))

        with pytest.raises(ValueError, match="PyCapsule"):
            np.negative._get_strided_loop("not the capsule!")

        with pytest.raises(TypeError, match=".*incompatible context"):
            np.add._get_strided_loop(call_info)

        np.negative._get_strided_loop(call_info)
        with pytest.raises(TypeError):
            # cannot call it a second time:
            np.negative._get_strided_loop(call_info)
