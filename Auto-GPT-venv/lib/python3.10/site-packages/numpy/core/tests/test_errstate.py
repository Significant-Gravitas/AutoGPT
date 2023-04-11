import pytest
import sysconfig

import numpy as np
from numpy.testing import assert_, assert_raises, IS_WASM

# The floating point emulation on ARM EABI systems lacking a hardware FPU is
# known to be buggy. This is an attempt to identify these hosts. It may not
# catch all possible cases, but it catches the known cases of gh-413 and
# gh-15562.
hosttype = sysconfig.get_config_var('HOST_GNU_TYPE')
arm_softfloat = False if hosttype is None else hosttype.endswith('gnueabi')

class TestErrstate:
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-413,-15562)')
    def test_invalid(self):
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            # This should work
            with np.errstate(invalid='ignore'):
                np.sqrt(a)
            # While this should fail!
            with assert_raises(FloatingPointError):
                np.sqrt(a)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-15562)')
    def test_divide(self):
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            # This should work
            with np.errstate(divide='ignore'):
                a // 0
            # While this should fail!
            with assert_raises(FloatingPointError):
                a // 0
            # As should this, see gh-15562
            with assert_raises(FloatingPointError):
                a // a

    def test_errcall(self):
        def foo(*args):
            print(args)

        olderrcall = np.geterrcall()
        with np.errstate(call=foo):
            assert_(np.geterrcall() is foo, 'call is not foo')
            with np.errstate(call=None):
                assert_(np.geterrcall() is None, 'call is not None')
        assert_(np.geterrcall() is olderrcall, 'call is not olderrcall')

    def test_errstate_decorator(self):
        @np.errstate(all='ignore')
        def foo():
            a = -np.arange(3)
            a // 0
            
        foo()
