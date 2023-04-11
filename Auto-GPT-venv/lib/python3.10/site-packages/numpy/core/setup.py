import os
import sys
import sysconfig
import pickle
import copy
import warnings
import textwrap
import glob
from os.path import join

from numpy.distutils import log
from numpy.distutils.msvccompiler import lib_opts_if_msvc
from distutils.dep_util import newer
from sysconfig import get_config_var
from numpy.compat import npy_load_module
from setup_common import *  # noqa: F403

# Set to True to enable relaxed strides checking. This (mostly) means
# that `strides[dim]` is ignored if `shape[dim] == 1` when setting flags.
NPY_RELAXED_STRIDES_CHECKING = (os.environ.get('NPY_RELAXED_STRIDES_CHECKING', "1") != "0")
if not NPY_RELAXED_STRIDES_CHECKING:
    raise SystemError(
        "Support for NPY_RELAXED_STRIDES_CHECKING=0 has been remove as of "
        "NumPy 1.23.  This error will eventually be removed entirely.")

# Put NPY_RELAXED_STRIDES_DEBUG=1 in the environment if you want numpy to use a
# bogus value for affected strides in order to help smoke out bad stride usage
# when relaxed stride checking is enabled.
NPY_RELAXED_STRIDES_DEBUG = (os.environ.get('NPY_RELAXED_STRIDES_DEBUG', "0") != "0")
NPY_RELAXED_STRIDES_DEBUG = NPY_RELAXED_STRIDES_DEBUG and NPY_RELAXED_STRIDES_CHECKING

# Set NPY_DISABLE_SVML=1 in the environment to disable the vendored SVML
# library. This option only has significance on a Linux x86_64 host and is most
# useful to avoid improperly requiring SVML when cross compiling.
NPY_DISABLE_SVML = (os.environ.get('NPY_DISABLE_SVML', "0") == "1")

# XXX: ugly, we use a class to avoid calling twice some expensive functions in
# config.h/numpyconfig.h. I don't see a better way because distutils force
# config.h generation inside an Extension class, and as such sharing
# configuration information between extensions is not easy.
# Using a pickled-based memoize does not work because config_cmd is an instance
# method, which cPickle does not like.
#
# Use pickle in all cases, as cPickle is gone in python3 and the difference
# in time is only in build. -- Charles Harris, 2013-03-30

class CallOnceOnly:
    def __init__(self):
        self._check_types = None
        self._check_ieee_macros = None
        self._check_complex = None

    def check_types(self, *a, **kw):
        if self._check_types is None:
            out = check_types(*a, **kw)
            self._check_types = pickle.dumps(out)
        else:
            out = copy.deepcopy(pickle.loads(self._check_types))
        return out

    def check_ieee_macros(self, *a, **kw):
        if self._check_ieee_macros is None:
            out = check_ieee_macros(*a, **kw)
            self._check_ieee_macros = pickle.dumps(out)
        else:
            out = copy.deepcopy(pickle.loads(self._check_ieee_macros))
        return out

    def check_complex(self, *a, **kw):
        if self._check_complex is None:
            out = check_complex(*a, **kw)
            self._check_complex = pickle.dumps(out)
        else:
            out = copy.deepcopy(pickle.loads(self._check_complex))
        return out

def can_link_svml():
    """SVML library is supported only on x86_64 architecture and currently
    only on linux
    """
    if NPY_DISABLE_SVML:
        return False
    platform = sysconfig.get_platform()
    return ("x86_64" in platform
            and "linux" in platform
            and sys.maxsize > 2**31)

def check_svml_submodule(svmlpath):
    if not os.path.exists(svmlpath + "/README.md"):
        raise RuntimeError("Missing `SVML` submodule! Run `git submodule "
                           "update --init` to fix this.")
    return True

def pythonlib_dir():
    """return path where libpython* is."""
    if sys.platform == 'win32':
        return os.path.join(sys.prefix, "libs")
    else:
        return get_config_var('LIBDIR')

def is_npy_no_signal():
    """Return True if the NPY_NO_SIGNAL symbol must be defined in configuration
    header."""
    return sys.platform == 'win32'

def is_npy_no_smp():
    """Return True if the NPY_NO_SMP symbol must be defined in public
    header (when SMP support cannot be reliably enabled)."""
    # Perhaps a fancier check is in order here.
    #  so that threads are only enabled if there
    #  are actually multiple CPUS? -- but
    #  threaded code can be nice even on a single
    #  CPU so that long-calculating code doesn't
    #  block.
    return 'NPY_NOSMP' in os.environ

def win32_checks(deflist):
    from numpy.distutils.misc_util import get_build_architecture
    a = get_build_architecture()

    # Distutils hack on AMD64 on windows
    print('BUILD_ARCHITECTURE: %r, os.name=%r, sys.platform=%r' %
          (a, os.name, sys.platform))
    if a == 'AMD64':
        deflist.append('DISTUTILS_USE_SDK')

    # On win32, force long double format string to be 'g', not
    # 'Lg', since the MS runtime does not support long double whose
    # size is > sizeof(double)
    if a == "Intel" or a == "AMD64":
        deflist.append('FORCE_NO_LONG_DOUBLE_FORMATTING')

def check_math_capabilities(config, ext, moredefs, mathlibs):
    def check_func(
        func_name,
        decl=False,
        headers=["feature_detection_math.h", "feature_detection_cmath.h"],
    ):
        return config.check_func(
            func_name,
            libraries=mathlibs,
            decl=decl,
            call=True,
            call_args=FUNC_CALL_ARGS[func_name],
            headers=headers,
        )

    def check_funcs_once(
            funcs_name,
            headers=["feature_detection_math.h", "feature_detection_cmath.h"],
            add_to_moredefs=True):
        call = dict([(f, True) for f in funcs_name])
        call_args = dict([(f, FUNC_CALL_ARGS[f]) for f in funcs_name])
        st = config.check_funcs_once(
            funcs_name,
            libraries=mathlibs,
            decl=False,
            call=call,
            call_args=call_args,
            headers=headers,
        )
        if st and add_to_moredefs:
            moredefs.extend([(fname2def(f), 1) for f in funcs_name])
        return st

    def check_funcs(
            funcs_name,
            headers=["feature_detection_math.h", "feature_detection_cmath.h"]):
        # Use check_funcs_once first, and if it does not work, test func per
        # func. Return success only if all the functions are available
        if not check_funcs_once(funcs_name, headers=headers):
            # Global check failed, check func per func
            for f in funcs_name:
                if check_func(f, headers=headers):
                    moredefs.append((fname2def(f), 1))
            return 0
        else:
            return 1

    # GH-14787: Work around GCC<8.4 bug when compiling with AVX512
    # support on Windows-based platforms
    def check_gh14787(fn):
        if fn == 'attribute_target_avx512f':
            if (sys.platform in ('win32', 'cygwin') and
                    config.check_compiler_gcc() and
                    not config.check_gcc_version_at_least(8, 4)):
                ext.extra_compile_args.extend(
                        ['-ffixed-xmm%s' % n for n in range(16, 32)])

    #use_msvc = config.check_decl("_MSC_VER")
    if not check_funcs_once(MANDATORY_FUNCS, add_to_moredefs=False):
        raise SystemError("One of the required function to build numpy is not"
                " available (the list is %s)." % str(MANDATORY_FUNCS))

    # Standard functions which may not be available and for which we have a
    # replacement implementation. Note that some of these are C99 functions.

    # XXX: hack to circumvent cpp pollution from python: python put its
    # config.h in the public namespace, so we have a clash for the common
    # functions we test. We remove every function tested by python's
    # autoconf, hoping their own test are correct
    for f in OPTIONAL_FUNCS_MAYBE:
        if config.check_decl(fname2def(f), headers=["Python.h"]):
            OPTIONAL_FILE_FUNCS.remove(f)

    check_funcs(OPTIONAL_FILE_FUNCS, headers=["feature_detection_stdio.h"])
    check_funcs(OPTIONAL_MISC_FUNCS, headers=["feature_detection_misc.h"])

    for h in OPTIONAL_HEADERS:
        if config.check_func("", decl=False, call=False, headers=[h]):
            h = h.replace(".", "_").replace(os.path.sep, "_")
            moredefs.append((fname2def(h), 1))

    # Try with both "locale.h" and "xlocale.h"
    locale_headers = [
        "stdlib.h",
        "xlocale.h",
        "feature_detection_locale.h",
    ]
    if not check_funcs(OPTIONAL_LOCALE_FUNCS, headers=locale_headers):
        # It didn't work with xlocale.h, maybe it will work with locale.h?
        locale_headers[1] = "locale.h"
        check_funcs(OPTIONAL_LOCALE_FUNCS, headers=locale_headers)

    for tup in OPTIONAL_INTRINSICS:
        headers = None
        if len(tup) == 2:
            f, args, m = tup[0], tup[1], fname2def(tup[0])
        elif len(tup) == 3:
            f, args, headers, m = tup[0], tup[1], [tup[2]], fname2def(tup[0])
        else:
            f, args, headers, m = tup[0], tup[1], [tup[2]], fname2def(tup[3])
        if config.check_func(f, decl=False, call=True, call_args=args,
                             headers=headers):
            moredefs.append((m, 1))

    for dec, fn in OPTIONAL_FUNCTION_ATTRIBUTES:
        if config.check_gcc_function_attribute(dec, fn):
            moredefs.append((fname2def(fn), 1))
            check_gh14787(fn)

    platform = sysconfig.get_platform()
    if ("x86_64" in platform):
        for dec, fn in OPTIONAL_FUNCTION_ATTRIBUTES_AVX:
            if config.check_gcc_function_attribute(dec, fn):
                moredefs.append((fname2def(fn), 1))
                check_gh14787(fn)
        for dec, fn, code, header in (
        OPTIONAL_FUNCTION_ATTRIBUTES_WITH_INTRINSICS_AVX):
            if config.check_gcc_function_attribute_with_intrinsics(
                    dec, fn, code, header):
                moredefs.append((fname2def(fn), 1))

    for fn in OPTIONAL_VARIABLE_ATTRIBUTES:
        if config.check_gcc_variable_attribute(fn):
            m = fn.replace("(", "_").replace(")", "_")
            moredefs.append((fname2def(m), 1))

def check_complex(config, mathlibs):
    priv = []
    pub = []

    # Check for complex support
    st = config.check_header('complex.h')
    if st:
        priv.append(('HAVE_COMPLEX_H', 1))
        pub.append(('NPY_USE_C99_COMPLEX', 1))

        for t in C99_COMPLEX_TYPES:
            st = config.check_type(t, headers=["complex.h"])
            if st:
                pub.append(('NPY_HAVE_%s' % type2def(t), 1))

        def check_prec(prec):
            flist = [f + prec for f in C99_COMPLEX_FUNCS]
            decl = dict([(f, True) for f in flist])
            if not config.check_funcs_once(flist, call=decl, decl=decl,
                                           libraries=mathlibs):
                for f in flist:
                    if config.check_func(f, call=True, decl=True,
                                         libraries=mathlibs):
                        priv.append((fname2def(f), 1))
            else:
                priv.extend([(fname2def(f), 1) for f in flist])

        check_prec('')
        check_prec('f')
        check_prec('l')

    return priv, pub

def check_ieee_macros(config):
    priv = []
    pub = []

    macros = []

    def _add_decl(f):
        priv.append(fname2def("decl_%s" % f))
        pub.append('NPY_%s' % fname2def("decl_%s" % f))

    # XXX: hack to circumvent cpp pollution from python: python put its
    # config.h in the public namespace, so we have a clash for the common
    # functions we test. We remove every function tested by python's
    # autoconf, hoping their own test are correct
    _macros = ["isnan", "isinf", "signbit", "isfinite"]
    for f in _macros:
        py_symbol = fname2def("decl_%s" % f)
        already_declared = config.check_decl(py_symbol,
                headers=["Python.h", "math.h"])
        if already_declared:
            if config.check_macro_true(py_symbol,
                    headers=["Python.h", "math.h"]):
                pub.append('NPY_%s' % fname2def("decl_%s" % f))
        else:
            macros.append(f)
    # Normally, isnan and isinf are macro (C99), but some platforms only have
    # func, or both func and macro version. Check for macro only, and define
    # replacement ones if not found.
    # Note: including Python.h is necessary because it modifies some math.h
    # definitions
    for f in macros:
        st = config.check_decl(f, headers=["Python.h", "math.h"])
        if st:
            _add_decl(f)

    return priv, pub

def check_types(config_cmd, ext, build_dir):
    private_defines = []
    public_defines = []

    # Expected size (in number of bytes) for each type. This is an
    # optimization: those are only hints, and an exhaustive search for the size
    # is done if the hints are wrong.
    expected = {'short': [2], 'int': [4], 'long': [8, 4],
                'float': [4], 'double': [8], 'long double': [16, 12, 8],
                'Py_intptr_t': [8, 4], 'PY_LONG_LONG': [8], 'long long': [8],
                'off_t': [8, 4]}

    # Check we have the python header (-dev* packages on Linux)
    result = config_cmd.check_header('Python.h')
    if not result:
        python = 'python'
        if '__pypy__' in sys.builtin_module_names:
            python = 'pypy'
        raise SystemError(
                "Cannot compile 'Python.h'. Perhaps you need to "
                "install {0}-dev|{0}-devel.".format(python))
    res = config_cmd.check_header("endian.h")
    if res:
        private_defines.append(('HAVE_ENDIAN_H', 1))
        public_defines.append(('NPY_HAVE_ENDIAN_H', 1))
    res = config_cmd.check_header("sys/endian.h")
    if res:
        private_defines.append(('HAVE_SYS_ENDIAN_H', 1))
        public_defines.append(('NPY_HAVE_SYS_ENDIAN_H', 1))

    # Check basic types sizes
    for type in ('short', 'int', 'long'):
        res = config_cmd.check_decl("SIZEOF_%s" % sym2def(type), headers=["Python.h"])
        if res:
            public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), "SIZEOF_%s" % sym2def(type)))
        else:
            res = config_cmd.check_type_size(type, expected=expected[type])
            if res >= 0:
                public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
            else:
                raise SystemError("Checking sizeof (%s) failed !" % type)

    for type in ('float', 'double', 'long double'):
        already_declared = config_cmd.check_decl("SIZEOF_%s" % sym2def(type),
                                                 headers=["Python.h"])
        res = config_cmd.check_type_size(type, expected=expected[type])
        if res >= 0:
            public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
            if not already_declared and not type == 'long double':
                private_defines.append(('SIZEOF_%s' % sym2def(type), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % type)

        # Compute size of corresponding complex type: used to check that our
        # definition is binary compatible with C99 complex type (check done at
        # build time in npy_common.h)
        complex_def = "struct {%s __x; %s __y;}" % (type, type)
        res = config_cmd.check_type_size(complex_def,
                                         expected=[2 * x for x in expected[type]])
        if res >= 0:
            public_defines.append(('NPY_SIZEOF_COMPLEX_%s' % sym2def(type), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % complex_def)

    for type in ('Py_intptr_t', 'off_t'):
        res = config_cmd.check_type_size(type, headers=["Python.h"],
                library_dirs=[pythonlib_dir()],
                expected=expected[type])

        if res >= 0:
            private_defines.append(('SIZEOF_%s' % sym2def(type), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % type)

    # We check declaration AND type because that's how distutils does it.
    if config_cmd.check_decl('PY_LONG_LONG', headers=['Python.h']):
        res = config_cmd.check_type_size('PY_LONG_LONG',  headers=['Python.h'],
                library_dirs=[pythonlib_dir()],
                expected=expected['PY_LONG_LONG'])
        if res >= 0:
            private_defines.append(('SIZEOF_%s' % sym2def('PY_LONG_LONG'), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def('PY_LONG_LONG'), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % 'PY_LONG_LONG')

        res = config_cmd.check_type_size('long long',
                expected=expected['long long'])
        if res >= 0:
            #private_defines.append(('SIZEOF_%s' % sym2def('long long'), '%d' % res))
            public_defines.append(('NPY_SIZEOF_%s' % sym2def('long long'), '%d' % res))
        else:
            raise SystemError("Checking sizeof (%s) failed !" % 'long long')

    if not config_cmd.check_decl('CHAR_BIT', headers=['Python.h']):
        raise RuntimeError(
            "Config wo CHAR_BIT is not supported"
            ", please contact the maintainers")

    return private_defines, public_defines

def check_mathlib(config_cmd):
    # Testing the C math library
    mathlibs = []
    mathlibs_choices = [[], ["m"], ["cpml"]]
    mathlib = os.environ.get("MATHLIB")
    if mathlib:
        mathlibs_choices.insert(0, mathlib.split(","))
    for libs in mathlibs_choices:
        if config_cmd.check_func(
            "log",
            libraries=libs,
            call_args="0",
            decl="double log(double);",
            call=True
        ):
            mathlibs = libs
            break
    else:
        raise RuntimeError(
            "math library missing; rerun setup.py after setting the "
            "MATHLIB env variable"
        )
    return mathlibs


def visibility_define(config):
    """Return the define value to use for NPY_VISIBILITY_HIDDEN (may be empty
    string)."""
    hide = '__attribute__((visibility("hidden")))'
    if config.check_gcc_function_attribute(hide, 'hideme'):
        return hide
    else:
        return ''

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import (Configuration, dot_join,
                                           exec_mod_from_location)
    from numpy.distutils.system_info import (get_info, blas_opt_info,
                                             lapack_opt_info)
    from numpy.distutils.ccompiler_opt import NPY_CXX_FLAGS
    from numpy.version import release as is_released

    config = Configuration('core', parent_package, top_path)
    local_dir = config.local_path
    codegen_dir = join(local_dir, 'code_generators')

    # Check whether we have a mismatch between the set C API VERSION and the
    # actual C API VERSION. Will raise a MismatchCAPIError if so.
    check_api_version(C_API_VERSION, codegen_dir)

    generate_umath_py = join(codegen_dir, 'generate_umath.py')
    n = dot_join(config.name, 'generate_umath')
    generate_umath = exec_mod_from_location('_'.join(n.split('.')),
                                            generate_umath_py)

    header_dir = 'include/numpy'  # this is relative to config.path_in_package

    cocache = CallOnceOnly()

    def generate_config_h(ext, build_dir):
        target = join(build_dir, header_dir, 'config.h')
        d = os.path.dirname(target)
        if not os.path.exists(d):
            os.makedirs(d)

        if newer(__file__, target):
            config_cmd = config.get_config_cmd()
            log.info('Generating %s', target)

            # Check sizeof
            moredefs, ignored = cocache.check_types(config_cmd, ext, build_dir)

            # Check math library and C99 math funcs availability
            mathlibs = check_mathlib(config_cmd)
            moredefs.append(('MATHLIB', ','.join(mathlibs)))

            check_math_capabilities(config_cmd, ext, moredefs, mathlibs)
            moredefs.extend(cocache.check_ieee_macros(config_cmd)[0])
            moredefs.extend(cocache.check_complex(config_cmd, mathlibs)[0])

            # Signal check
            if is_npy_no_signal():
                moredefs.append('__NPY_PRIVATE_NO_SIGNAL')

            # Windows checks
            if sys.platform == 'win32' or os.name == 'nt':
                win32_checks(moredefs)

            # C99 restrict keyword
            moredefs.append(('NPY_RESTRICT', config_cmd.check_restrict()))

            # Inline check
            inline = config_cmd.check_inline()

            if can_link_svml():
                moredefs.append(('NPY_CAN_LINK_SVML', 1))

            # Use bogus stride debug aid to flush out bugs where users use
            # strides of dimensions with length 1 to index a full contiguous
            # array.
            if NPY_RELAXED_STRIDES_DEBUG:
                moredefs.append(('NPY_RELAXED_STRIDES_DEBUG', 1))
            else:
                moredefs.append(('NPY_RELAXED_STRIDES_DEBUG', 0))

            # Get long double representation
            rep = check_long_double_representation(config_cmd)
            moredefs.append(('HAVE_LDOUBLE_%s' % rep, 1))

            if check_for_right_shift_internal_compiler_error(config_cmd):
                moredefs.append('NPY_DO_NOT_OPTIMIZE_LONG_right_shift')
                moredefs.append('NPY_DO_NOT_OPTIMIZE_ULONG_right_shift')
                moredefs.append('NPY_DO_NOT_OPTIMIZE_LONGLONG_right_shift')
                moredefs.append('NPY_DO_NOT_OPTIMIZE_ULONGLONG_right_shift')

            # Generate the config.h file from moredefs
            with open(target, 'w') as target_f:
                if sys.platform == 'darwin':
                    target_f.write(
                        "/* may be overridden by numpyconfig.h on darwin */\n"
                    )
                for d in moredefs:
                    if isinstance(d, str):
                        target_f.write('#define %s\n' % (d))
                    else:
                        target_f.write('#define %s %s\n' % (d[0], d[1]))

                # define inline to our keyword, or nothing
                target_f.write('#ifndef __cplusplus\n')
                if inline == 'inline':
                    target_f.write('/* #undef inline */\n')
                else:
                    target_f.write('#define inline %s\n' % inline)
                target_f.write('#endif\n')

                # add the guard to make sure config.h is never included directly,
                # but always through npy_config.h
                target_f.write(textwrap.dedent("""
                    #ifndef NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_
                    #error config.h should never be included directly, include npy_config.h instead
                    #endif
                    """))

            log.info('File: %s' % target)
            with open(target) as target_f:
                log.info(target_f.read())
            log.info('EOF')
        else:
            mathlibs = []
            with open(target) as target_f:
                for line in target_f:
                    s = '#define MATHLIB'
                    if line.startswith(s):
                        value = line[len(s):].strip()
                        if value:
                            mathlibs.extend(value.split(','))

        # Ugly: this can be called within a library and not an extension,
        # in which case there is no libraries attributes (and none is
        # needed).
        if hasattr(ext, 'libraries'):
            ext.libraries.extend(mathlibs)

        incl_dir = os.path.dirname(target)
        if incl_dir not in config.numpy_include_dirs:
            config.numpy_include_dirs.append(incl_dir)

        return target

    def generate_numpyconfig_h(ext, build_dir):
        """Depends on config.h: generate_config_h has to be called before !"""
        # put common include directory in build_dir on search path
        # allows using code generation in headers
        config.add_include_dirs(join(build_dir, "src", "common"))
        config.add_include_dirs(join(build_dir, "src", "npymath"))

        target = join(build_dir, header_dir, '_numpyconfig.h')
        d = os.path.dirname(target)
        if not os.path.exists(d):
            os.makedirs(d)
        if newer(__file__, target):
            config_cmd = config.get_config_cmd()
            log.info('Generating %s', target)

            # Check sizeof
            ignored, moredefs = cocache.check_types(config_cmd, ext, build_dir)

            if is_npy_no_signal():
                moredefs.append(('NPY_NO_SIGNAL', 1))

            if is_npy_no_smp():
                moredefs.append(('NPY_NO_SMP', 1))
            else:
                moredefs.append(('NPY_NO_SMP', 0))

            mathlibs = check_mathlib(config_cmd)
            moredefs.extend(cocache.check_ieee_macros(config_cmd)[1])
            moredefs.extend(cocache.check_complex(config_cmd, mathlibs)[1])

            if NPY_RELAXED_STRIDES_DEBUG:
                moredefs.append(('NPY_RELAXED_STRIDES_DEBUG', 1))

            # Check whether we can use inttypes (C99) formats
            if config_cmd.check_decl('PRIdPTR', headers=['inttypes.h']):
                moredefs.append(('NPY_USE_C99_FORMATS', 1))

            # visibility check
            hidden_visibility = visibility_define(config_cmd)
            moredefs.append(('NPY_VISIBILITY_HIDDEN', hidden_visibility))

            # Add the C API/ABI versions
            moredefs.append(('NPY_ABI_VERSION', '0x%.8X' % C_ABI_VERSION))
            moredefs.append(('NPY_API_VERSION', '0x%.8X' % C_API_VERSION))

            # Add moredefs to header
            with open(target, 'w') as target_f:
                for d in moredefs:
                    if isinstance(d, str):
                        target_f.write('#define %s\n' % (d))
                    else:
                        target_f.write('#define %s %s\n' % (d[0], d[1]))

                # Define __STDC_FORMAT_MACROS
                target_f.write(textwrap.dedent("""
                    #ifndef __STDC_FORMAT_MACROS
                    #define __STDC_FORMAT_MACROS 1
                    #endif
                    """))

            # Dump the numpyconfig.h header to stdout
            log.info('File: %s' % target)
            with open(target) as target_f:
                log.info(target_f.read())
            log.info('EOF')
        config.add_data_files((header_dir, target))
        return target

    def generate_api_func(module_name):
        def generate_api(ext, build_dir):
            script = join(codegen_dir, module_name + '.py')
            sys.path.insert(0, codegen_dir)
            try:
                m = __import__(module_name)
                log.info('executing %s', script)
                h_file, c_file, doc_file = m.generate_api(os.path.join(build_dir, header_dir))
            finally:
                del sys.path[0]
            config.add_data_files((header_dir, h_file),
                                  (header_dir, doc_file))
            return (h_file,)
        return generate_api

    generate_numpy_api = generate_api_func('generate_numpy_api')
    generate_ufunc_api = generate_api_func('generate_ufunc_api')

    config.add_include_dirs(join(local_dir, "src", "common"))
    config.add_include_dirs(join(local_dir, "src"))
    config.add_include_dirs(join(local_dir))

    config.add_data_dir('include/numpy')
    config.add_include_dirs(join('src', 'npymath'))
    config.add_include_dirs(join('src', 'multiarray'))
    config.add_include_dirs(join('src', 'umath'))
    config.add_include_dirs(join('src', 'npysort'))
    config.add_include_dirs(join('src', '_simd'))

    config.add_define_macros([("NPY_INTERNAL_BUILD", "1")]) # this macro indicates that Numpy build is in process
    config.add_define_macros([("HAVE_NPY_CONFIG_H", "1")])
    if sys.platform[:3] == "aix":
        config.add_define_macros([("_LARGE_FILES", None)])
    else:
        config.add_define_macros([("_FILE_OFFSET_BITS", "64")])
        config.add_define_macros([('_LARGEFILE_SOURCE', '1')])
        config.add_define_macros([('_LARGEFILE64_SOURCE', '1')])

    config.numpy_include_dirs.extend(config.paths('include'))

    deps = [join('src', 'npymath', '_signbit.c'),
            join('include', 'numpy', '*object.h'),
            join(codegen_dir, 'genapi.py'),
            ]

    #######################################################################
    #                          npymath library                            #
    #######################################################################

    subst_dict = dict([("sep", os.path.sep), ("pkgname", "numpy.core")])

    def get_mathlib_info(*args):
        # Another ugly hack: the mathlib info is known once build_src is run,
        # but we cannot use add_installed_pkg_config here either, so we only
        # update the substitution dictionary during npymath build
        config_cmd = config.get_config_cmd()
        # Check that the toolchain works, to fail early if it doesn't
        # (avoid late errors with MATHLIB which are confusing if the
        # compiler does not work).
        for lang, test_code, note in (
            ('c', 'int main(void) { return 0;}', ''),
            ('c++', (
                    'int main(void)'
                    '{ auto x = 0.0; return static_cast<int>(x); }'
                ), (
                    'note: A compiler with support for C++11 language '
                    'features is required.'
                )
             ),
        ):
            is_cpp = lang == 'c++'
            if is_cpp:
                # this a workaround to get rid of invalid c++ flags
                # without doing big changes to config.
                # c tested first, compiler should be here
                bk_c = config_cmd.compiler
                config_cmd.compiler = bk_c.cxx_compiler()

                # Check that Linux compiler actually support the default flags
                if hasattr(config_cmd.compiler, 'compiler'):
                    config_cmd.compiler.compiler.extend(NPY_CXX_FLAGS)
                    config_cmd.compiler.compiler_so.extend(NPY_CXX_FLAGS)

            st = config_cmd.try_link(test_code, lang=lang)
            if not st:
                # rerun the failing command in verbose mode
                config_cmd.compiler.verbose = True
                config_cmd.try_link(test_code, lang=lang)
                raise RuntimeError(
                    f"Broken toolchain: cannot link a simple {lang.upper()} "
                    f"program. {note}"
                )
            if is_cpp:
                config_cmd.compiler = bk_c
        mlibs = check_mathlib(config_cmd)

        posix_mlib = ' '.join(['-l%s' % l for l in mlibs])
        msvc_mlib = ' '.join(['%s.lib' % l for l in mlibs])
        subst_dict["posix_mathlib"] = posix_mlib
        subst_dict["msvc_mathlib"] = msvc_mlib

    npymath_sources = [join('src', 'npymath', 'npy_math_internal.h.src'),
                       join('src', 'npymath', 'npy_math.c'),
                       # join('src', 'npymath', 'ieee754.cpp'),
                       join('src', 'npymath', 'ieee754.c.src'),
                       join('src', 'npymath', 'npy_math_complex.c.src'),
                       join('src', 'npymath', 'halffloat.c'),
                       ]

    config.add_installed_library('npymath',
            sources=npymath_sources + [get_mathlib_info],
            install_dir='lib',
            build_info={
                'include_dirs' : [],  # empty list required for creating npy_math_internal.h
                'extra_compiler_args': [lib_opts_if_msvc],
            })
    config.add_npy_pkg_config("npymath.ini.in", "lib/npy-pkg-config",
            subst_dict)
    config.add_npy_pkg_config("mlib.ini.in", "lib/npy-pkg-config",
            subst_dict)

    #######################################################################
    #                     multiarray_tests module                         #
    #######################################################################

    config.add_extension('_multiarray_tests',
                    sources=[join('src', 'multiarray', '_multiarray_tests.c.src'),
                             join('src', 'common', 'mem_overlap.c'),
                             join('src', 'common', 'npy_argparse.c'),
                             join('src', 'common', 'npy_hashtable.c')],
                    depends=[join('src', 'common', 'mem_overlap.h'),
                             join('src', 'common', 'npy_argparse.h'),
                             join('src', 'common', 'npy_hashtable.h'),
                             join('src', 'common', 'npy_extint128.h')],
                    libraries=['npymath'])

    #######################################################################
    #             _multiarray_umath module - common part                  #
    #######################################################################

    common_deps = [
            join('src', 'common', 'dlpack', 'dlpack.h'),
            join('src', 'common', 'array_assign.h'),
            join('src', 'common', 'binop_override.h'),
            join('src', 'common', 'cblasfuncs.h'),
            join('src', 'common', 'lowlevel_strided_loops.h'),
            join('src', 'common', 'mem_overlap.h'),
            join('src', 'common', 'npy_argparse.h'),
            join('src', 'common', 'npy_cblas.h'),
            join('src', 'common', 'npy_config.h'),
            join('src', 'common', 'npy_ctypes.h'),
            join('src', 'common', 'npy_dlpack.h'),
            join('src', 'common', 'npy_extint128.h'),
            join('src', 'common', 'npy_import.h'),
            join('src', 'common', 'npy_hashtable.h'),
            join('src', 'common', 'npy_longdouble.h'),
            join('src', 'common', 'npy_svml.h'),
            join('src', 'common', 'templ_common.h.src'),
            join('src', 'common', 'ucsnarrow.h'),
            join('src', 'common', 'ufunc_override.h'),
            join('src', 'common', 'umathmodule.h'),
            join('src', 'common', 'numpyos.h'),
            join('src', 'common', 'npy_cpu_dispatch.h'),
            join('src', 'common', 'simd', 'simd.h'),
            ]

    common_src = [
            join('src', 'common', 'array_assign.c'),
            join('src', 'common', 'mem_overlap.c'),
            join('src', 'common', 'npy_argparse.c'),
            join('src', 'common', 'npy_hashtable.c'),
            join('src', 'common', 'npy_longdouble.c'),
            join('src', 'common', 'templ_common.h.src'),
            join('src', 'common', 'ucsnarrow.c'),
            join('src', 'common', 'ufunc_override.c'),
            join('src', 'common', 'numpyos.c'),
            join('src', 'common', 'npy_cpu_features.c'),
            ]

    if os.environ.get('NPY_USE_BLAS_ILP64', "0") != "0":
        blas_info = get_info('blas_ilp64_opt', 2)
    else:
        blas_info = get_info('blas_opt', 0)

    have_blas = blas_info and ('HAVE_CBLAS', None) in blas_info.get('define_macros', [])

    if have_blas:
        extra_info = blas_info
        # These files are also in MANIFEST.in so that they are always in
        # the source distribution independently of HAVE_CBLAS.
        common_src.extend([join('src', 'common', 'cblasfuncs.c'),
                           join('src', 'common', 'python_xerbla.c'),
                          ])
    else:
        extra_info = {}

    #######################################################################
    #             _multiarray_umath module - multiarray part              #
    #######################################################################

    multiarray_deps = [
            join('src', 'multiarray', 'abstractdtypes.h'),
            join('src', 'multiarray', 'arrayobject.h'),
            join('src', 'multiarray', 'arraytypes.h.src'),
            join('src', 'multiarray', 'arrayfunction_override.h'),
            join('src', 'multiarray', 'array_coercion.h'),
            join('src', 'multiarray', 'array_method.h'),
            join('src', 'multiarray', 'npy_buffer.h'),
            join('src', 'multiarray', 'calculation.h'),
            join('src', 'multiarray', 'common.h'),
            join('src', 'multiarray', 'common_dtype.h'),
            join('src', 'multiarray', 'convert_datatype.h'),
            join('src', 'multiarray', 'convert.h'),
            join('src', 'multiarray', 'conversion_utils.h'),
            join('src', 'multiarray', 'ctors.h'),
            join('src', 'multiarray', 'descriptor.h'),
            join('src', 'multiarray', 'dtypemeta.h'),
            join('src', 'multiarray', 'dtype_transfer.h'),
            join('src', 'multiarray', 'dragon4.h'),
            join('src', 'multiarray', 'einsum_debug.h'),
            join('src', 'multiarray', 'einsum_sumprod.h'),
            join('src', 'multiarray', 'experimental_public_dtype_api.h'),
            join('src', 'multiarray', 'getset.h'),
            join('src', 'multiarray', 'hashdescr.h'),
            join('src', 'multiarray', 'iterators.h'),
            join('src', 'multiarray', 'legacy_dtype_implementation.h'),
            join('src', 'multiarray', 'mapping.h'),
            join('src', 'multiarray', 'methods.h'),
            join('src', 'multiarray', 'multiarraymodule.h'),
            join('src', 'multiarray', 'nditer_impl.h'),
            join('src', 'multiarray', 'number.h'),
            join('src', 'multiarray', 'refcount.h'),
            join('src', 'multiarray', 'scalartypes.h'),
            join('src', 'multiarray', 'sequence.h'),
            join('src', 'multiarray', 'shape.h'),
            join('src', 'multiarray', 'strfuncs.h'),
            join('src', 'multiarray', 'typeinfo.h'),
            join('src', 'multiarray', 'usertypes.h'),
            join('src', 'multiarray', 'vdot.h'),
            join('src', 'multiarray', 'textreading', 'readtext.h'),
            join('include', 'numpy', 'arrayobject.h'),
            join('include', 'numpy', '_neighborhood_iterator_imp.h'),
            join('include', 'numpy', 'npy_endian.h'),
            join('include', 'numpy', 'arrayscalars.h'),
            join('include', 'numpy', 'noprefix.h'),
            join('include', 'numpy', 'npy_interrupt.h'),
            join('include', 'numpy', 'npy_3kcompat.h'),
            join('include', 'numpy', 'npy_math.h'),
            join('include', 'numpy', 'halffloat.h'),
            join('include', 'numpy', 'npy_common.h'),
            join('include', 'numpy', 'npy_os.h'),
            join('include', 'numpy', 'utils.h'),
            join('include', 'numpy', 'ndarrayobject.h'),
            join('include', 'numpy', 'npy_cpu.h'),
            join('include', 'numpy', 'numpyconfig.h'),
            join('include', 'numpy', 'ndarraytypes.h'),
            join('include', 'numpy', 'npy_1_7_deprecated_api.h'),
            # add library sources as distuils does not consider libraries
            # dependencies
            ] + npymath_sources

    multiarray_src = [
            join('src', 'multiarray', 'abstractdtypes.c'),
            join('src', 'multiarray', 'alloc.c'),
            join('src', 'multiarray', 'arrayobject.c'),
            join('src', 'multiarray', 'arraytypes.h.src'),
            join('src', 'multiarray', 'arraytypes.c.src'),
            join('src', 'multiarray', 'argfunc.dispatch.c.src'),
            join('src', 'multiarray', 'array_coercion.c'),
            join('src', 'multiarray', 'array_method.c'),
            join('src', 'multiarray', 'array_assign_scalar.c'),
            join('src', 'multiarray', 'array_assign_array.c'),
            join('src', 'multiarray', 'arrayfunction_override.c'),
            join('src', 'multiarray', 'buffer.c'),
            join('src', 'multiarray', 'calculation.c'),
            join('src', 'multiarray', 'compiled_base.c'),
            join('src', 'multiarray', 'common.c'),
            join('src', 'multiarray', 'common_dtype.c'),
            join('src', 'multiarray', 'convert.c'),
            join('src', 'multiarray', 'convert_datatype.c'),
            join('src', 'multiarray', 'conversion_utils.c'),
            join('src', 'multiarray', 'ctors.c'),
            join('src', 'multiarray', 'datetime.c'),
            join('src', 'multiarray', 'datetime_strings.c'),
            join('src', 'multiarray', 'datetime_busday.c'),
            join('src', 'multiarray', 'datetime_busdaycal.c'),
            join('src', 'multiarray', 'descriptor.c'),
            join('src', 'multiarray', 'dlpack.c'),
            join('src', 'multiarray', 'dtypemeta.c'),
            join('src', 'multiarray', 'dragon4.c'),
            join('src', 'multiarray', 'dtype_transfer.c'),
            join('src', 'multiarray', 'einsum.c.src'),
            join('src', 'multiarray', 'einsum_sumprod.c.src'),
            join('src', 'multiarray', 'experimental_public_dtype_api.c'),
            join('src', 'multiarray', 'flagsobject.c'),
            join('src', 'multiarray', 'getset.c'),
            join('src', 'multiarray', 'hashdescr.c'),
            join('src', 'multiarray', 'item_selection.c'),
            join('src', 'multiarray', 'iterators.c'),
            join('src', 'multiarray', 'legacy_dtype_implementation.c'),
            join('src', 'multiarray', 'lowlevel_strided_loops.c.src'),
            join('src', 'multiarray', 'mapping.c'),
            join('src', 'multiarray', 'methods.c'),
            join('src', 'multiarray', 'multiarraymodule.c'),
            join('src', 'multiarray', 'nditer_templ.c.src'),
            join('src', 'multiarray', 'nditer_api.c'),
            join('src', 'multiarray', 'nditer_constr.c'),
            join('src', 'multiarray', 'nditer_pywrap.c'),
            join('src', 'multiarray', 'number.c'),
            join('src', 'multiarray', 'refcount.c'),
            join('src', 'multiarray', 'sequence.c'),
            join('src', 'multiarray', 'shape.c'),
            join('src', 'multiarray', 'scalarapi.c'),
            join('src', 'multiarray', 'scalartypes.c.src'),
            join('src', 'multiarray', 'strfuncs.c'),
            join('src', 'multiarray', 'temp_elide.c'),
            join('src', 'multiarray', 'typeinfo.c'),
            join('src', 'multiarray', 'usertypes.c'),
            join('src', 'multiarray', 'vdot.c'),
            join('src', 'common', 'npy_sort.h.src'),
            join('src', 'npysort', 'x86-qsort.dispatch.cpp'),
            join('src', 'npysort', 'quicksort.cpp'),
            join('src', 'npysort', 'mergesort.cpp'),
            join('src', 'npysort', 'timsort.cpp'),
            join('src', 'npysort', 'heapsort.cpp'),
            join('src', 'npysort', 'radixsort.cpp'),
            join('src', 'common', 'npy_partition.h'),
            join('src', 'npysort', 'selection.cpp'),
            join('src', 'common', 'npy_binsearch.h'),
            join('src', 'npysort', 'binsearch.cpp'),
            join('src', 'multiarray', 'textreading', 'conversions.c'),
            join('src', 'multiarray', 'textreading', 'field_types.c'),
            join('src', 'multiarray', 'textreading', 'growth.c'),
            join('src', 'multiarray', 'textreading', 'readtext.c'),
            join('src', 'multiarray', 'textreading', 'rows.c'),
            join('src', 'multiarray', 'textreading', 'stream_pyobject.c'),
            join('src', 'multiarray', 'textreading', 'str_to_int.c'),
            join('src', 'multiarray', 'textreading', 'tokenize.cpp'),
            # Remove this once scipy macos arm64 build correctly
            # links to the arm64 npymath library,
            # see gh-22673
            join('src', 'npymath', 'arm64_exports.c'),
            ]

    #######################################################################
    #             _multiarray_umath module - umath part                   #
    #######################################################################

    def generate_umath_c(ext, build_dir):
        target = join(build_dir, header_dir, '__umath_generated.c')
        dir = os.path.dirname(target)
        if not os.path.exists(dir):
            os.makedirs(dir)
        script = generate_umath_py
        if newer(script, target):
            with open(target, 'w') as f:
                f.write(generate_umath.make_code(generate_umath.defdict,
                                                 generate_umath.__file__))
        return []

    def generate_umath_doc_header(ext, build_dir):
        from numpy.distutils.misc_util import exec_mod_from_location

        target = join(build_dir, header_dir, '_umath_doc_generated.h')
        dir = os.path.dirname(target)
        if not os.path.exists(dir):
            os.makedirs(dir)

        generate_umath_doc_py = join(codegen_dir, 'generate_umath_doc.py')
        if newer(generate_umath_doc_py, target):
            n = dot_join(config.name, 'generate_umath_doc')
            generate_umath_doc = exec_mod_from_location(
                '_'.join(n.split('.')), generate_umath_doc_py)
            generate_umath_doc.write_code(target)

    umath_src = [
            join('src', 'umath', 'umathmodule.c'),
            join('src', 'umath', 'reduction.c'),
            join('src', 'umath', 'funcs.inc.src'),
            join('src', 'umath', 'simd.inc.src'),
            join('src', 'umath', 'loops.h.src'),
            join('src', 'umath', 'loops_utils.h.src'),
            join('src', 'umath', 'loops.c.src'),
            join('src', 'umath', 'loops_unary_fp.dispatch.c.src'),
            join('src', 'umath', 'loops_arithm_fp.dispatch.c.src'),
            join('src', 'umath', 'loops_arithmetic.dispatch.c.src'),
            join('src', 'umath', 'loops_minmax.dispatch.c.src'),
            join('src', 'umath', 'loops_trigonometric.dispatch.c.src'),
            join('src', 'umath', 'loops_umath_fp.dispatch.c.src'),
            join('src', 'umath', 'loops_exponent_log.dispatch.c.src'),
            join('src', 'umath', 'loops_hyperbolic.dispatch.c.src'),
            join('src', 'umath', 'loops_modulo.dispatch.c.src'),
            join('src', 'umath', 'loops_comparison.dispatch.c.src'),
            join('src', 'umath', 'matmul.h.src'),
            join('src', 'umath', 'matmul.c.src'),
            join('src', 'umath', 'clip.h'),
            join('src', 'umath', 'clip.cpp'),
            join('src', 'umath', 'dispatching.c'),
            join('src', 'umath', 'legacy_array_method.c'),
            join('src', 'umath', 'wrapping_array_method.c'),
            join('src', 'umath', 'ufunc_object.c'),
            join('src', 'umath', 'extobj.c'),
            join('src', 'umath', 'scalarmath.c.src'),
            join('src', 'umath', 'ufunc_type_resolution.c'),
            join('src', 'umath', 'override.c'),
            join('src', 'umath', 'string_ufuncs.cpp'),
            # For testing. Eventually, should use public API and be separate:
            join('src', 'umath', '_scaled_float_dtype.c'),
            ]

    umath_deps = [
            generate_umath_py,
            join('include', 'numpy', 'npy_math.h'),
            join('include', 'numpy', 'halffloat.h'),
            join('src', 'multiarray', 'common.h'),
            join('src', 'multiarray', 'number.h'),
            join('src', 'common', 'templ_common.h.src'),
            join('src', 'umath', 'simd.inc.src'),
            join('src', 'umath', 'override.h'),
            join(codegen_dir, 'generate_ufunc_api.py'),
            join(codegen_dir, 'ufunc_docstrings.py'),
            ]

    svml_path = join('numpy', 'core', 'src', 'umath', 'svml')
    svml_objs = []
    # we have converted the following into universal intrinsics
    # so we can bring the benefits of performance for all platforms
    # not just for avx512 on linux without performance/accuracy regression,
    # actually the other way around, better performance and
    # after all maintainable code.
    svml_filter = (
    )
    if can_link_svml() and check_svml_submodule(svml_path):
        svml_objs = glob.glob(svml_path + '/**/*.s', recursive=True)
        svml_objs = [o for o in svml_objs if not o.endswith(svml_filter)]

        # The ordering of names returned by glob is undefined, so we sort
        # to make builds reproducible.
        svml_objs.sort()

    config.add_extension('_multiarray_umath',
                         # Forcing C language even though we have C++ sources.
                         # It forces the C linker and don't link C++ runtime.
                         language = 'c',
                         sources=multiarray_src + umath_src +
                                 common_src +
                                 [generate_config_h,
                                  generate_numpyconfig_h,
                                  generate_numpy_api,
                                  join(codegen_dir, 'generate_numpy_api.py'),
                                  join('*.py'),
                                  generate_umath_c,
                                  generate_umath_doc_header,
                                  generate_ufunc_api,
                                 ],
                         depends=deps + multiarray_deps + umath_deps +
                                common_deps,
                         libraries=['npymath'],
                         extra_objects=svml_objs,
                         extra_info=extra_info,
                         extra_cxx_compile_args=NPY_CXX_FLAGS)

    #######################################################################
    #                        umath_tests module                           #
    #######################################################################

    config.add_extension('_umath_tests', sources=[
        join('src', 'umath', '_umath_tests.c.src'),
        join('src', 'umath', '_umath_tests.dispatch.c'),
        join('src', 'common', 'npy_cpu_features.c'),
    ])

    #######################################################################
    #                   custom rational dtype module                      #
    #######################################################################

    config.add_extension('_rational_tests',
                    sources=[join('src', 'umath', '_rational_tests.c')])

    #######################################################################
    #                        struct_ufunc_test module                     #
    #######################################################################

    config.add_extension('_struct_ufunc_tests',
                    sources=[join('src', 'umath', '_struct_ufunc_tests.c')])


    #######################################################################
    #                        operand_flag_tests module                    #
    #######################################################################

    config.add_extension('_operand_flag_tests',
                    sources=[join('src', 'umath', '_operand_flag_tests.c')])

    #######################################################################
    #                        SIMD module                                  #
    #######################################################################

    config.add_extension('_simd',
        sources=[
            join('src', 'common', 'npy_cpu_features.c'),
            join('src', '_simd', '_simd.c'),
            join('src', '_simd', '_simd_inc.h.src'),
            join('src', '_simd', '_simd_data.inc.src'),
            join('src', '_simd', '_simd.dispatch.c.src'),
        ], depends=[
            join('src', 'common', 'npy_cpu_dispatch.h'),
            join('src', 'common', 'simd', 'simd.h'),
            join('src', '_simd', '_simd.h'),
            join('src', '_simd', '_simd_inc.h.src'),
            join('src', '_simd', '_simd_data.inc.src'),
            join('src', '_simd', '_simd_arg.inc'),
            join('src', '_simd', '_simd_convert.inc'),
            join('src', '_simd', '_simd_easyintrin.inc'),
            join('src', '_simd', '_simd_vector.inc'),
        ],
        libraries=['npymath']
    )

    config.add_subpackage('tests')
    config.add_data_dir('tests/data')
    config.add_data_dir('tests/examples')
    config.add_data_files('*.pyi')

    config.make_svn_version_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
