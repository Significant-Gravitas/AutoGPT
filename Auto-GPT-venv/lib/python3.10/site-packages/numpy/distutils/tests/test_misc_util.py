from os.path import join, sep, dirname

from numpy.distutils.misc_util import (
    appendpath, minrelpath, gpaths, get_shared_lib_extension, get_info
    )
from numpy.testing import (
    assert_, assert_equal
    )

ajoin = lambda *paths: join(*((sep,)+paths))

class TestAppendpath:

    def test_1(self):
        assert_equal(appendpath('prefix', 'name'), join('prefix', 'name'))
        assert_equal(appendpath('/prefix', 'name'), ajoin('prefix', 'name'))
        assert_equal(appendpath('/prefix', '/name'), ajoin('prefix', 'name'))
        assert_equal(appendpath('prefix', '/name'), join('prefix', 'name'))

    def test_2(self):
        assert_equal(appendpath('prefix/sub', 'name'),
                     join('prefix', 'sub', 'name'))
        assert_equal(appendpath('prefix/sub', 'sup/name'),
                     join('prefix', 'sub', 'sup', 'name'))
        assert_equal(appendpath('/prefix/sub', '/prefix/name'),
                     ajoin('prefix', 'sub', 'name'))

    def test_3(self):
        assert_equal(appendpath('/prefix/sub', '/prefix/sup/name'),
                     ajoin('prefix', 'sub', 'sup', 'name'))
        assert_equal(appendpath('/prefix/sub/sub2', '/prefix/sup/sup2/name'),
                     ajoin('prefix', 'sub', 'sub2', 'sup', 'sup2', 'name'))
        assert_equal(appendpath('/prefix/sub/sub2', '/prefix/sub/sup/name'),
                     ajoin('prefix', 'sub', 'sub2', 'sup', 'name'))

class TestMinrelpath:

    def test_1(self):
        n = lambda path: path.replace('/', sep)
        assert_equal(minrelpath(n('aa/bb')), n('aa/bb'))
        assert_equal(minrelpath('..'), '..')
        assert_equal(minrelpath(n('aa/..')), '')
        assert_equal(minrelpath(n('aa/../bb')), 'bb')
        assert_equal(minrelpath(n('aa/bb/..')), 'aa')
        assert_equal(minrelpath(n('aa/bb/../..')), '')
        assert_equal(minrelpath(n('aa/bb/../cc/../dd')), n('aa/dd'))
        assert_equal(minrelpath(n('.././..')), n('../..'))
        assert_equal(minrelpath(n('aa/bb/.././../dd')), n('dd'))

class TestGpaths:

    def test_gpaths(self):
        local_path = minrelpath(join(dirname(__file__), '..'))
        ls = gpaths('command/*.py', local_path)
        assert_(join(local_path, 'command', 'build_src.py') in ls, repr(ls))
        f = gpaths('system_info.py', local_path)
        assert_(join(local_path, 'system_info.py') == f[0], repr(f))

class TestSharedExtension:

    def test_get_shared_lib_extension(self):
        import sys
        ext = get_shared_lib_extension(is_python_ext=False)
        if sys.platform.startswith('linux'):
            assert_equal(ext, '.so')
        elif sys.platform.startswith('gnukfreebsd'):
            assert_equal(ext, '.so')
        elif sys.platform.startswith('darwin'):
            assert_equal(ext, '.dylib')
        elif sys.platform.startswith('win'):
            assert_equal(ext, '.dll')
        # just check for no crash
        assert_(get_shared_lib_extension(is_python_ext=True))


def test_installed_npymath_ini():
    # Regression test for gh-7707.  If npymath.ini wasn't installed, then this
    # will give an error.
    info = get_info('npymath')

    assert isinstance(info, dict)
    assert "define_macros" in info
