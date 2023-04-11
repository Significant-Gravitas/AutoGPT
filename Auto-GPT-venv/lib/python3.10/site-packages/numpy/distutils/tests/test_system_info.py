import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
from distutils.errors import DistutilsError

from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils


def get_class(name, notfound_action=1):
    """
    notfound_action:
      0 - do nothing
      1 - display warning message
      2 - raise error
    """
    cl = {'temp1': Temp1Info,
          'temp2': Temp2Info,
          'duplicate_options': DuplicateOptionInfo,
          }.get(name.lower(), _system_info)
    return cl()

simple_site = """
[ALL]
library_dirs = {dir1:s}{pathsep:s}{dir2:s}
libraries = {lib1:s},{lib2:s}
extra_compile_args = -I/fake/directory -I"/path with/spaces" -Os
runtime_library_dirs = {dir1:s}

[temp1]
library_dirs = {dir1:s}
libraries = {lib1:s}
runtime_library_dirs = {dir1:s}

[temp2]
library_dirs = {dir2:s}
libraries = {lib2:s}
extra_link_args = -Wl,-rpath={lib2_escaped:s}
rpath = {dir2:s}

[duplicate_options]
mylib_libs = {lib1:s}
libraries = {lib2:s}
"""
site_cfg = simple_site

fakelib_c_text = """
/* This file is generated from numpy/distutils/testing/test_system_info.py */
#include<stdio.h>
void foo(void) {
   printf("Hello foo");
}
void bar(void) {
   printf("Hello bar");
}
"""

def have_compiler():
    """ Return True if there appears to be an executable compiler
    """
    compiler = customized_ccompiler()
    try:
        cmd = compiler.compiler  # Unix compilers
    except AttributeError:
        try:
            if not compiler.initialized:
                compiler.initialize()  # MSVC is different
        except (DistutilsError, ValueError):
            return False
        cmd = [compiler.cc]
    try:
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        p.stdout.close()
        p.stderr.close()
        p.wait()
    except OSError:
        return False
    return True


HAVE_COMPILER = have_compiler()


class _system_info(system_info):

    def __init__(self,
                 default_lib_dirs=default_lib_dirs,
                 default_include_dirs=default_include_dirs,
                 verbosity=1,
                 ):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {'library_dirs': '',
                    'include_dirs': '',
                    'runtime_library_dirs': '',
                    'rpath': '',
                    'src_dirs': '',
                    'search_static_first': "0",
                    'extra_compile_args': '',
                    'extra_link_args': ''}
        self.cp = ConfigParser(defaults)
        # We have to parse the config files afterwards
        # to have a consistent temporary filepath

    def _check_libs(self, lib_dirs, libs, opt_libs, exts):
        """Override _check_libs to return with all dirs """
        info = {'libraries': libs, 'library_dirs': lib_dirs}
        return info


class Temp1Info(_system_info):
    """For testing purposes"""
    section = 'temp1'


class Temp2Info(_system_info):
    """For testing purposes"""
    section = 'temp2'

class DuplicateOptionInfo(_system_info):
    """For testing purposes"""
    section = 'duplicate_options'


class TestSystemInfoReading:

    def setup_method(self):
        """ Create the libraries """
        # Create 2 sources and 2 libraries
        self._dir1 = mkdtemp()
        self._src1 = os.path.join(self._dir1, 'foo.c')
        self._lib1 = os.path.join(self._dir1, 'libfoo.so')
        self._dir2 = mkdtemp()
        self._src2 = os.path.join(self._dir2, 'bar.c')
        self._lib2 = os.path.join(self._dir2, 'libbar.so')
        # Update local site.cfg
        global simple_site, site_cfg
        site_cfg = simple_site.format(**{
            'dir1': self._dir1,
            'lib1': self._lib1,
            'dir2': self._dir2,
            'lib2': self._lib2,
            'pathsep': os.pathsep,
            'lib2_escaped': _shell_utils.NativeParser.join([self._lib2])
        })
        # Write site.cfg
        fd, self._sitecfg = mkstemp()
        os.close(fd)
        with open(self._sitecfg, 'w') as fd:
            fd.write(site_cfg)
        # Write the sources
        with open(self._src1, 'w') as fd:
            fd.write(fakelib_c_text)
        with open(self._src2, 'w') as fd:
            fd.write(fakelib_c_text)
        # We create all class-instances

        def site_and_parse(c, site_cfg):
            c.files = [site_cfg]
            c.parse_config_files()
            return c
        self.c_default = site_and_parse(get_class('default'), self._sitecfg)
        self.c_temp1 = site_and_parse(get_class('temp1'), self._sitecfg)
        self.c_temp2 = site_and_parse(get_class('temp2'), self._sitecfg)
        self.c_dup_options = site_and_parse(get_class('duplicate_options'),
                                            self._sitecfg)

    def teardown_method(self):
        # Do each removal separately
        try:
            shutil.rmtree(self._dir1)
        except Exception:
            pass
        try:
            shutil.rmtree(self._dir2)
        except Exception:
            pass
        try:
            os.remove(self._sitecfg)
        except Exception:
            pass

    def test_all(self):
        # Read in all information in the ALL block
        tsi = self.c_default
        assert_equal(tsi.get_lib_dirs(), [self._dir1, self._dir2])
        assert_equal(tsi.get_libraries(), [self._lib1, self._lib2])
        assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])
        extra = tsi.calc_extra_info()
        assert_equal(extra['extra_compile_args'], ['-I/fake/directory', '-I/path with/spaces', '-Os'])

    def test_temp1(self):
        # Read in all information in the temp1 block
        tsi = self.c_temp1
        assert_equal(tsi.get_lib_dirs(), [self._dir1])
        assert_equal(tsi.get_libraries(), [self._lib1])
        assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])

    def test_temp2(self):
        # Read in all information in the temp2 block
        tsi = self.c_temp2
        assert_equal(tsi.get_lib_dirs(), [self._dir2])
        assert_equal(tsi.get_libraries(), [self._lib2])
        # Now from rpath and not runtime_library_dirs
        assert_equal(tsi.get_runtime_lib_dirs(key='rpath'), [self._dir2])
        extra = tsi.calc_extra_info()
        assert_equal(extra['extra_link_args'], ['-Wl,-rpath=' + self._lib2])

    def test_duplicate_options(self):
        # Ensure that duplicates are raising an AliasedOptionError
        tsi = self.c_dup_options
        assert_raises(AliasedOptionError, tsi.get_option_single, "mylib_libs", "libraries")
        assert_equal(tsi.get_libs("mylib_libs", [self._lib1]), [self._lib1])
        assert_equal(tsi.get_libs("libraries", [self._lib2]), [self._lib2])

    @pytest.mark.skipif(not HAVE_COMPILER, reason="Missing compiler")
    def test_compile1(self):
        # Compile source and link the first source
        c = customized_ccompiler()
        previousDir = os.getcwd()
        try:
            # Change directory to not screw up directories
            os.chdir(self._dir1)
            c.compile([os.path.basename(self._src1)], output_dir=self._dir1)
            # Ensure that the object exists
            assert_(os.path.isfile(self._src1.replace('.c', '.o')) or
                    os.path.isfile(self._src1.replace('.c', '.obj')))
        finally:
            os.chdir(previousDir)

    @pytest.mark.skipif(not HAVE_COMPILER, reason="Missing compiler")
    @pytest.mark.skipif('msvc' in repr(ccompiler.new_compiler()),
                         reason="Fails with MSVC compiler ")
    def test_compile2(self):
        # Compile source and link the second source
        tsi = self.c_temp2
        c = customized_ccompiler()
        extra_link_args = tsi.calc_extra_info()['extra_link_args']
        previousDir = os.getcwd()
        try:
            # Change directory to not screw up directories
            os.chdir(self._dir2)
            c.compile([os.path.basename(self._src2)], output_dir=self._dir2,
                      extra_postargs=extra_link_args)
            # Ensure that the object exists
            assert_(os.path.isfile(self._src2.replace('.c', '.o')))
        finally:
            os.chdir(previousDir)

    HAS_MKL = "mkl_rt" in mkl_info().calc_libraries_info().get("libraries", [])

    @pytest.mark.xfail(HAS_MKL, reason=("`[DEFAULT]` override doesn't work if "
                                        "numpy is built with MKL support"))
    def test_overrides(self):
        previousDir = os.getcwd()
        cfg = os.path.join(self._dir1, 'site.cfg')
        shutil.copy(self._sitecfg, cfg)
        try:
            os.chdir(self._dir1)
            # Check that the '[ALL]' section does not override
            # missing values from other sections
            info = mkl_info()
            lib_dirs = info.cp['ALL']['library_dirs'].split(os.pathsep)
            assert info.get_lib_dirs() != lib_dirs

            # But if we copy the values to a '[mkl]' section the value
            # is correct
            with open(cfg, 'r') as fid:
                mkl = fid.read().replace('[ALL]', '[mkl]', 1)
            with open(cfg, 'w') as fid:
                fid.write(mkl)
            info = mkl_info()
            assert info.get_lib_dirs() == lib_dirs

            # Also, the values will be taken from a section named '[DEFAULT]'
            with open(cfg, 'r') as fid:
                dflt = fid.read().replace('[mkl]', '[DEFAULT]', 1)
            with open(cfg, 'w') as fid:
                fid.write(dflt)
            info = mkl_info()
            assert info.get_lib_dirs() == lib_dirs
        finally:
            os.chdir(previousDir)


def test_distutils_parse_env_order(monkeypatch):
    from numpy.distutils.system_info import _parse_env_order
    env = 'NPY_TESTS_DISTUTILS_PARSE_ENV_ORDER'

    base_order = list('abcdef')

    monkeypatch.setenv(env, 'b,i,e,f')
    order, unknown = _parse_env_order(base_order, env)
    assert len(order) == 3
    assert order == list('bef')
    assert len(unknown) == 1

    # For when LAPACK/BLAS optimization is disabled
    monkeypatch.setenv(env, '')
    order, unknown = _parse_env_order(base_order, env)
    assert len(order) == 0
    assert len(unknown) == 0

    for prefix in '^!':
        monkeypatch.setenv(env, f'{prefix}b,i,e')
        order, unknown = _parse_env_order(base_order, env)
        assert len(order) == 4
        assert order == list('acdf')
        assert len(unknown) == 1

    with pytest.raises(ValueError):
        monkeypatch.setenv(env, 'b,^e,i')
        _parse_env_order(base_order, env)

    with pytest.raises(ValueError):
        monkeypatch.setenv(env, '!b,^e,i')
        _parse_env_order(base_order, env)
