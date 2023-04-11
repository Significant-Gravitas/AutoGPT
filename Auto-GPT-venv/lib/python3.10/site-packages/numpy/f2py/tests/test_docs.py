import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from . import util


def get_docdir():
    # assuming that documentation tests are run from a source
    # directory
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..',
        'doc', 'source', 'f2py', 'code'))


pytestmark = pytest.mark.skipif(
    not os.path.isdir(get_docdir()),
    reason=('Could not find f2py documentation sources'
            f' ({get_docdir()} does not exists)'))


def _path(*a):
    return os.path.join(*((get_docdir(),) + a))


class TestDocAdvanced(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/build-f2py']
    sources = [_path('asterisk1.f90'), _path('asterisk2.f90'),
               _path('ftype.f')]

    def test_asterisk1(self):
        foo = getattr(self.module, 'foo1')
        assert_equal(foo(), b'123456789A12')

    def test_asterisk2(self):
        foo = getattr(self.module, 'foo2')
        assert_equal(foo(2), b'12')
        assert_equal(foo(12), b'123456789A12')
        assert_equal(foo(24), b'123456789A123456789B')

    def test_ftype(self):
        ftype = self.module
        ftype.foo()
        assert_equal(ftype.data.a, 0)
        ftype.data.a = 3
        ftype.data.x = [1, 2, 3]
        assert_equal(ftype.data.a, 3)
        assert_array_equal(ftype.data.x,
                           np.array([1, 2, 3], dtype=np.float32))
        ftype.data.x[1] = 45
        assert_array_equal(ftype.data.x,
                           np.array([1, 45, 3], dtype=np.float32))

    # TODO: implement test methods for other example Fortran codes
