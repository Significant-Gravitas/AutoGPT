# Code common to build tools
import copy
import pathlib
import sys
import textwrap

from numpy.distutils.misc_util import mingw32


#-------------------
# Versioning support
#-------------------
# How to change C_API_VERSION ?
#   - increase C_API_VERSION value
#   - record the hash for the new C API with the cversions.py script
#   and add the hash to cversions.txt
# The hash values are used to remind developers when the C API number was not
# updated - generates a MismatchCAPIWarning warning which is turned into an
# exception for released version.

# Binary compatibility version number. This number is increased whenever the
# C-API is changed such that binary compatibility is broken, i.e. whenever a
# recompile of extension modules is needed.
C_ABI_VERSION = 0x01000009

# Minor API version.  This number is increased whenever a change is made to the
# C-API -- whether it breaks binary compatibility or not.  Some changes, such
# as adding a function pointer to the end of the function table, can be made
# without breaking binary compatibility.  In this case, only the C_API_VERSION
# (*not* C_ABI_VERSION) would be increased.  Whenever binary compatibility is
# broken, both C_API_VERSION and C_ABI_VERSION should be increased.
#
# The version needs to be kept in sync with that in cversions.txt.
#
# 0x00000008 - 1.7.x
# 0x00000009 - 1.8.x
# 0x00000009 - 1.9.x
# 0x0000000a - 1.10.x
# 0x0000000a - 1.11.x
# 0x0000000a - 1.12.x
# 0x0000000b - 1.13.x
# 0x0000000c - 1.14.x
# 0x0000000c - 1.15.x
# 0x0000000d - 1.16.x
# 0x0000000d - 1.19.x
# 0x0000000e - 1.20.x
# 0x0000000e - 1.21.x
# 0x0000000f - 1.22.x
# 0x00000010 - 1.23.x
# 0x00000010 - 1.24.x
C_API_VERSION = 0x00000010

class MismatchCAPIError(ValueError):
    pass


def get_api_versions(apiversion, codegen_dir):
    """
    Return current C API checksum and the recorded checksum.

    Return current C API checksum and the recorded checksum for the given
    version of the C API version.

    """
    # Compute the hash of the current API as defined in the .txt files in
    # code_generators
    sys.path.insert(0, codegen_dir)
    try:
        m = __import__('genapi')
        numpy_api = __import__('numpy_api')
        curapi_hash = m.fullapi_hash(numpy_api.full_api)
        apis_hash = m.get_versions_hash()
    finally:
        del sys.path[0]

    return curapi_hash, apis_hash[apiversion]

def check_api_version(apiversion, codegen_dir):
    """Emits a MismatchCAPIWarning if the C API version needs updating."""
    curapi_hash, api_hash = get_api_versions(apiversion, codegen_dir)

    # If different hash, it means that the api .txt files in
    # codegen_dir have been updated without the API version being
    # updated. Any modification in those .txt files should be reflected
    # in the api and eventually abi versions.
    # To compute the checksum of the current API, use numpy/core/cversions.py
    if not curapi_hash == api_hash:
        msg = ("API mismatch detected, the C API version "
               "numbers have to be updated. Current C api version is "
               f"{apiversion}, with checksum {curapi_hash}, but recorded "
               f"checksum in core/codegen_dir/cversions.txt is {api_hash}. If "
               "functions were added in the C API, you have to update "
               f"C_API_VERSION in {__file__}."
               )
        raise MismatchCAPIError(msg)


FUNC_CALL_ARGS = {}

def set_sig(sig):
    prefix, _, args = sig.partition("(")
    args = args.rpartition(")")[0]
    funcname = prefix.rpartition(" ")[-1]
    args = [arg.strip() for arg in args.split(",")]
    # We use {0} because 0 alone cannot be cast to complex on MSVC in C:
    FUNC_CALL_ARGS[funcname] = ", ".join("(%s){0}" % arg for arg in args)


for file in [
    "feature_detection_locale.h",
    "feature_detection_math.h",
    "feature_detection_cmath.h",
    "feature_detection_misc.h",
    "feature_detection_stdio.h",
]:
    with open(pathlib.Path(__file__).parent / file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if not line.strip():
                continue
            set_sig(line)

# Mandatory functions: if not found, fail the build
# Some of these can still be blocklisted if the C99 implementation
# is buggy, see numpy/core/src/common/npy_config.h
MANDATORY_FUNCS = [
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs",
    "floor", "ceil", "sqrt", "log10", "log", "exp", "asin",
    "acos", "atan", "fmod", 'modf', 'frexp', 'ldexp',
    "expm1", "log1p", "acosh", "asinh", "atanh",
    "rint", "trunc", "exp2",
    "copysign", "nextafter", "strtoll", "strtoull", "cbrt",
    "log2", "pow", "hypot", "atan2",
    "creal", "cimag", "conj"
]

OPTIONAL_LOCALE_FUNCS = ["strtold_l"]
OPTIONAL_FILE_FUNCS = ["ftello", "fseeko", "fallocate"]
OPTIONAL_MISC_FUNCS = ["backtrace", "madvise"]

# variable attributes tested via "int %s a" % attribute
OPTIONAL_VARIABLE_ATTRIBUTES = ["__thread", "__declspec(thread)"]

# Subset of OPTIONAL_*_FUNCS which may already have HAVE_* defined by Python.h
OPTIONAL_FUNCS_MAYBE = [
    "ftello", "fseeko"
    ]

C99_COMPLEX_TYPES = [
    'complex double', 'complex float', 'complex long double'
    ]
C99_COMPLEX_FUNCS = [
    "cabs", "cacos", "cacosh", "carg", "casin", "casinh", "catan",
    "catanh", "cexp", "clog", "cpow", "csqrt",
    # The long double variants (like csinl)  should be mandatory on C11,
    # but are missing in FreeBSD. Issue gh-22850
    "csin", "csinh", "ccos", "ccosh", "ctan", "ctanh",
    ]

OPTIONAL_HEADERS = [
# sse headers only enabled automatically on amd64/x32 builds
                "xmmintrin.h",  # SSE
                "emmintrin.h",  # SSE2
                "immintrin.h",  # AVX
                "features.h",  # for glibc version linux
                "xlocale.h",  # see GH#8367
                "dlfcn.h",  # dladdr
                "execinfo.h",  # backtrace
                "libunwind.h",  # backtrace for LLVM/Clang using libunwind
                "sys/mman.h", #madvise
]

# optional gcc compiler builtins and their call arguments and optional a
# required header and definition name (HAVE_ prepended)
# call arguments are required as the compiler will do strict signature checking
OPTIONAL_INTRINSICS = [("__builtin_isnan", '5.'),
                       ("__builtin_isinf", '5.'),
                       ("__builtin_isfinite", '5.'),
                       ("__builtin_bswap32", '5u'),
                       ("__builtin_bswap64", '5u'),
                       ("__builtin_expect", '5, 0'),
                       # Test `long long` for arm+clang 13 (gh-22811,
                       # but we use all versions of __builtin_mul_overflow):
                       ("__builtin_mul_overflow", '(long long)5, 5, (int*)5'),
                       # MMX only needed for icc, but some clangs don't have it
                       ("_m_from_int64", '0', "emmintrin.h"),
                       ("_mm_load_ps", '(float*)0', "xmmintrin.h"),  # SSE
                       ("_mm_prefetch", '(float*)0, _MM_HINT_NTA',
                        "xmmintrin.h"),  # SSE
                       ("_mm_load_pd", '(double*)0', "emmintrin.h"),  # SSE2
                       ("__builtin_prefetch", "(float*)0, 0, 3"),
                       # check that the linker can handle avx
                       ("__asm__ volatile", '"vpand %xmm1, %xmm2, %xmm3"',
                        "stdio.h", "LINK_AVX"),
                       ("__asm__ volatile", '"vpand %ymm1, %ymm2, %ymm3"',
                        "stdio.h", "LINK_AVX2"),
                       ("__asm__ volatile", '"vpaddd %zmm1, %zmm2, %zmm3"',
                        "stdio.h", "LINK_AVX512F"),
                       ("__asm__ volatile", '"vfpclasspd $0x40, %zmm15, %k6\\n"\
                                             "vmovdqu8 %xmm0, %xmm1\\n"\
                                             "vpbroadcastmb2q %k0, %xmm0\\n"',
                        "stdio.h", "LINK_AVX512_SKX"),
                       ("__asm__ volatile", '"xgetbv"', "stdio.h", "XGETBV"),
                       ]

# function attributes
# tested via "int %s %s(void *);" % (attribute, name)
# function name will be converted to HAVE_<upper-case-name> preprocessor macro
OPTIONAL_FUNCTION_ATTRIBUTES = [('__attribute__((optimize("unroll-loops")))',
                                'attribute_optimize_unroll_loops'),
                                ('__attribute__((optimize("O3")))',
                                 'attribute_optimize_opt_3'),
                                ('__attribute__((optimize("O2")))',
                                 'attribute_optimize_opt_2'),
                                ('__attribute__((nonnull (1)))',
                                 'attribute_nonnull'),
                                ]

OPTIONAL_FUNCTION_ATTRIBUTES_AVX = [('__attribute__((target ("avx")))',
    'attribute_target_avx'),
    ('__attribute__((target ("avx2")))',
    'attribute_target_avx2'),
    ('__attribute__((target ("avx512f")))',
    'attribute_target_avx512f'),
    ('__attribute__((target ("avx512f,avx512dq,avx512bw,avx512vl,avx512cd")))',
    'attribute_target_avx512_skx'),
    ]

# function attributes with intrinsics
# To ensure your compiler can compile avx intrinsics with just the attributes
# gcc 4.8.4 support attributes but not with intrisics
# tested via "#include<%s> int %s %s(void *){code; return 0;};" % (header, attribute, name, code)
# function name will be converted to HAVE_<upper-case-name> preprocessor macro
# The _mm512_castps_si512 instruction is specific check for AVX-512F support
# in gcc-4.9 which is missing a subset of intrinsics. See
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61878
OPTIONAL_FUNCTION_ATTRIBUTES_WITH_INTRINSICS_AVX = [
    ('__attribute__((target("avx2,fma")))',
    'attribute_target_avx2_with_intrinsics',
    '__m256 temp = _mm256_set1_ps(1.0); temp = \
    _mm256_fmadd_ps(temp, temp, temp)',
    'immintrin.h'),
    ('__attribute__((target("avx512f")))',
    'attribute_target_avx512f_with_intrinsics',
    '__m512i temp = _mm512_castps_si512(_mm512_set1_ps(1.0))',
    'immintrin.h'),
    ('__attribute__((target ("avx512f,avx512dq,avx512bw,avx512vl,avx512cd")))',
    'attribute_target_avx512_skx_with_intrinsics',
    '__mmask8 temp = _mm512_fpclass_pd_mask(_mm512_set1_pd(1.0), 0x01);\
    __m512i unused_temp = \
        _mm512_castps_si512(_mm512_set1_ps(1.0));\
    _mm_mask_storeu_epi8(NULL, 0xFF, _mm_broadcastmb_epi64(temp))',
    'immintrin.h'),
    ]

def fname2def(name):
    return "HAVE_%s" % name.upper()

def sym2def(symbol):
    define = symbol.replace(' ', '')
    return define.upper()

def type2def(symbol):
    define = symbol.replace(' ', '_')
    return define.upper()

# Code to detect long double representation taken from MPFR m4 macro
def check_long_double_representation(cmd):
    cmd._check_compiler()
    body = LONG_DOUBLE_REPRESENTATION_SRC % {'type': 'long double'}

    # Disable whole program optimization (the default on vs2015, with python 3.5+)
    # which generates intermediary object files and prevents checking the
    # float representation.
    if sys.platform == "win32" and not mingw32():
        try:
            cmd.compiler.compile_options.remove("/GL")
        except (AttributeError, ValueError):
            pass

    # Disable multi-file interprocedural optimization in the Intel compiler on Linux
    # which generates intermediary object files and prevents checking the
    # float representation.
    elif (sys.platform != "win32"
            and cmd.compiler.compiler_type.startswith('intel')
            and '-ipo' in cmd.compiler.cc_exe):
        newcompiler = cmd.compiler.cc_exe.replace(' -ipo', '')
        cmd.compiler.set_executables(
            compiler=newcompiler,
            compiler_so=newcompiler,
            compiler_cxx=newcompiler,
            linker_exe=newcompiler,
            linker_so=newcompiler + ' -shared'
        )

    # We need to use _compile because we need the object filename
    src, obj = cmd._compile(body, None, None, 'c')
    try:
        ltype = long_double_representation(pyod(obj))
        return ltype
    except ValueError:
        # try linking to support CC="gcc -flto" or icc -ipo
        # struct needs to be volatile so it isn't optimized away
        # additionally "clang -flto" requires the foo struct to be used
        body = body.replace('struct', 'volatile struct')
        body += "int main(void) { return foo.before[0]; }\n"
        src, obj = cmd._compile(body, None, None, 'c')
        cmd.temp_files.append("_configtest")
        cmd.compiler.link_executable([obj], "_configtest")
        ltype = long_double_representation(pyod("_configtest"))
        return ltype
    finally:
        cmd._clean()

LONG_DOUBLE_REPRESENTATION_SRC = r"""
/* "before" is 16 bytes to ensure there's no padding between it and "x".
 *    We're not expecting any "long double" bigger than 16 bytes or with
 *       alignment requirements stricter than 16 bytes.  */
typedef %(type)s test_type;

struct {
        char         before[16];
        test_type    x;
        char         after[8];
} foo = {
        { '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
          '\001', '\043', '\105', '\147', '\211', '\253', '\315', '\357' },
        -123456789.0,
        { '\376', '\334', '\272', '\230', '\166', '\124', '\062', '\020' }
};
"""

def pyod(filename):
    """Python implementation of the od UNIX utility (od -b, more exactly).

    Parameters
    ----------
    filename : str
        name of the file to get the dump from.

    Returns
    -------
    out : seq
        list of lines of od output

    Notes
    -----
    We only implement enough to get the necessary information for long double
    representation, this is not intended as a compatible replacement for od.
    """
    out = []
    with open(filename, 'rb') as fid:
        yo2 = [oct(o)[2:] for o in fid.read()]
    for i in range(0, len(yo2), 16):
        line = ['%07d' % int(oct(i)[2:])]
        line.extend(['%03d' % int(c) for c in yo2[i:i+16]])
        out.append(" ".join(line))
    return out


_BEFORE_SEQ = ['000', '000', '000', '000', '000', '000', '000', '000',
              '001', '043', '105', '147', '211', '253', '315', '357']
_AFTER_SEQ = ['376', '334', '272', '230', '166', '124', '062', '020']

_IEEE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000']
_IEEE_DOUBLE_LE = _IEEE_DOUBLE_BE[::-1]
_INTEL_EXTENDED_12B = ['000', '000', '000', '000', '240', '242', '171', '353',
                       '031', '300', '000', '000']
_INTEL_EXTENDED_16B = ['000', '000', '000', '000', '240', '242', '171', '353',
                       '031', '300', '000', '000', '000', '000', '000', '000']
_MOTOROLA_EXTENDED_12B = ['300', '031', '000', '000', '353', '171',
                          '242', '240', '000', '000', '000', '000']
_IEEE_QUAD_PREC_BE = ['300', '031', '326', '363', '105', '100', '000', '000',
                      '000', '000', '000', '000', '000', '000', '000', '000']
_IEEE_QUAD_PREC_LE = _IEEE_QUAD_PREC_BE[::-1]
_IBM_DOUBLE_DOUBLE_BE = (['301', '235', '157', '064', '124', '000', '000', '000'] +
                     ['000'] * 8)
_IBM_DOUBLE_DOUBLE_LE = (['000', '000', '000', '124', '064', '157', '235', '301'] +
                     ['000'] * 8)

def long_double_representation(lines):
    """Given a binary dump as given by GNU od -b, look for long double
    representation."""

    # Read contains a list of 32 items, each item is a byte (in octal
    # representation, as a string). We 'slide' over the output until read is of
    # the form before_seq + content + after_sequence, where content is the long double
    # representation:
    #  - content is 12 bytes: 80 bits Intel representation
    #  - content is 16 bytes: 80 bits Intel representation (64 bits) or quad precision
    #  - content is 8 bytes: same as double (not implemented yet)
    read = [''] * 32
    saw = None
    for line in lines:
        # we skip the first word, as od -b output an index at the beginning of
        # each line
        for w in line.split()[1:]:
            read.pop(0)
            read.append(w)

            # If the end of read is equal to the after_sequence, read contains
            # the long double
            if read[-8:] == _AFTER_SEQ:
                saw = copy.copy(read)
                # if the content was 12 bytes, we only have 32 - 8 - 12 = 12
                # "before" bytes. In other words the first 4 "before" bytes went
                # past the sliding window.
                if read[:12] == _BEFORE_SEQ[4:]:
                    if read[12:-8] == _INTEL_EXTENDED_12B:
                        return 'INTEL_EXTENDED_12_BYTES_LE'
                    if read[12:-8] == _MOTOROLA_EXTENDED_12B:
                        return 'MOTOROLA_EXTENDED_12_BYTES_BE'
                # if the content was 16 bytes, we are left with 32-8-16 = 16
                # "before" bytes, so 8 went past the sliding window.
                elif read[:8] == _BEFORE_SEQ[8:]:
                    if read[8:-8] == _INTEL_EXTENDED_16B:
                        return 'INTEL_EXTENDED_16_BYTES_LE'
                    elif read[8:-8] == _IEEE_QUAD_PREC_BE:
                        return 'IEEE_QUAD_BE'
                    elif read[8:-8] == _IEEE_QUAD_PREC_LE:
                        return 'IEEE_QUAD_LE'
                    elif read[8:-8] == _IBM_DOUBLE_DOUBLE_LE:
                        return 'IBM_DOUBLE_DOUBLE_LE'
                    elif read[8:-8] == _IBM_DOUBLE_DOUBLE_BE:
                        return 'IBM_DOUBLE_DOUBLE_BE'
                # if the content was 8 bytes, left with 32-8-8 = 16 bytes
                elif read[:16] == _BEFORE_SEQ:
                    if read[16:-8] == _IEEE_DOUBLE_LE:
                        return 'IEEE_DOUBLE_LE'
                    elif read[16:-8] == _IEEE_DOUBLE_BE:
                        return 'IEEE_DOUBLE_BE'

    if saw is not None:
        raise ValueError("Unrecognized format (%s)" % saw)
    else:
        # We never detected the after_sequence
        raise ValueError("Could not lock sequences (%s)" % saw)


def check_for_right_shift_internal_compiler_error(cmd):
    """
    On our arm CI, this fails with an internal compilation error

    The failure looks like the following, and can be reproduced on ARM64 GCC 5.4:

        <source>: In function 'right_shift':
        <source>:4:20: internal compiler error: in expand_shift_1, at expmed.c:2349
               ip1[i] = ip1[i] >> in2;
                      ^
        Please submit a full bug report,
        with preprocessed source if appropriate.
        See <http://gcc.gnu.org/bugs.html> for instructions.
        Compiler returned: 1

    This function returns True if this compiler bug is present, and we need to
    turn off optimization for the function
    """
    cmd._check_compiler()
    has_optimize = cmd.try_compile(textwrap.dedent("""\
        __attribute__((optimize("O3"))) void right_shift() {}
        """), None, None)
    if not has_optimize:
        return False

    no_err = cmd.try_compile(textwrap.dedent("""\
        typedef long the_type;  /* fails also for unsigned and long long */
        __attribute__((optimize("O3"))) void right_shift(the_type in2, the_type *ip1, int n) {
            for (int i = 0; i < n; i++) {
                if (in2 < (the_type)sizeof(the_type) * 8) {
                    ip1[i] = ip1[i] >> in2;
                }
            }
        }
        """), None, None)
    return not no_err
