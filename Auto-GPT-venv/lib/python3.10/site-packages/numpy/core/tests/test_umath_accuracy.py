import numpy as np
import os
from os import path
import sys
import pytest
from ctypes import c_longlong, c_double, c_float, c_int, cast, pointer, POINTER
from numpy.testing import assert_array_max_ulp
from numpy.testing._private.utils import _glibc_older_than
from numpy.core._multiarray_umath import __cpu_features__

UNARY_UFUNCS = [obj for obj in np.core.umath.__dict__.values() if
        isinstance(obj, np.ufunc)]
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]
UNARY_OBJECT_UFUNCS.remove(getattr(np, 'invert'))

IS_AVX = __cpu_features__.get('AVX512F', False) or \
        (__cpu_features__.get('FMA3', False) and __cpu_features__.get('AVX2', False))
# only run on linux with AVX, also avoid old glibc (numpy/numpy#20448).
runtest = (sys.platform.startswith('linux')
           and IS_AVX and not _glibc_older_than("2.17"))
platform_skip = pytest.mark.skipif(not runtest,
                                   reason="avoid testing inconsistent platform "
                                   "library implementations")

# convert string to hex function taken from:
# https://stackoverflow.com/questions/1592158/convert-hex-to-float #
def convert(s, datatype="np.float32"):
    i = int(s, 16)                   # convert from hex to a Python int
    if (datatype == "np.float64"):
        cp = pointer(c_longlong(i))           # make this into a c long long integer
        fp = cast(cp, POINTER(c_double))  # cast the int pointer to a double pointer
    else:
        cp = pointer(c_int(i))           # make this into a c integer
        fp = cast(cp, POINTER(c_float))  # cast the int pointer to a float pointer

    return fp.contents.value         # dereference the pointer, get the float

str_to_float = np.vectorize(convert)

class TestAccuracy:
    @platform_skip
    def test_validate_transcendentals(self):
        with np.errstate(all='ignore'):
            data_dir = path.join(path.dirname(__file__), 'data')
            files = os.listdir(data_dir)
            files = list(filter(lambda f: f.endswith('.csv'), files))
            for filename in files:
                filepath = path.join(data_dir, filename)
                with open(filepath) as fid:
                    file_without_comments = (r for r in fid if not r[0] in ('$', '#'))
                    data = np.genfromtxt(file_without_comments,
                                         dtype=('|S39','|S39','|S39',int),
                                         names=('type','input','output','ulperr'),
                                         delimiter=',',
                                         skip_header=1)
                    npname = path.splitext(filename)[0].split('-')[3]
                    npfunc = getattr(np, npname)
                    for datatype in np.unique(data['type']):
                        data_subset = data[data['type'] == datatype]
                        inval  = np.array(str_to_float(data_subset['input'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                        outval = np.array(str_to_float(data_subset['output'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                        perm = np.random.permutation(len(inval))
                        inval = inval[perm]
                        outval = outval[perm]
                        maxulperr = data_subset['ulperr'].max()
                        assert_array_max_ulp(npfunc(inval), outval, maxulperr)

    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    def test_validate_fp16_transcendentals(self, ufunc):
        with np.errstate(all='ignore'):
            arr = np.arange(65536, dtype=np.int16)
            datafp16 = np.frombuffer(arr.tobytes(), dtype=np.float16)
            datafp32 = datafp16.astype(np.float32)
            assert_array_max_ulp(ufunc(datafp16), ufunc(datafp32),
                    maxulp=1, dtype=np.float16)
