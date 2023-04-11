import os
import sys
import copy
import platform
import pytest

import numpy as np

from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util

wrap = None

# Extend core typeinfo with CHARACTER to test dtype('c')
_ti = _typeinfo['STRING']
typeinfo = dict(
    CHARACTER=type(_ti)(('c', _ti.num, 8, _ti.alignment, _ti.type)),
    **_typeinfo)


def setup_module():
    """
    Build the required testing extension module

    """
    global wrap

    # Check compiler availability first
    if not util.has_c_compiler():
        pytest.skip("No C compiler available")

    if wrap is None:
        config_code = """
        config.add_extension('test_array_from_pyobj_ext',
                             sources=['wrapmodule.c', 'fortranobject.c'],
                             define_macros=[])
        """
        d = os.path.dirname(__file__)
        src = [
            util.getpath("tests", "src", "array_from_pyobj", "wrapmodule.c"),
            util.getpath("src", "fortranobject.c"),
            util.getpath("src", "fortranobject.h"),
        ]
        wrap = util.build_module_distutils(src, config_code,
                                           "test_array_from_pyobj_ext")


def flags_info(arr):
    flags = wrap.array_attrs(arr)[6]
    return flags2names(flags)


def flags2names(flags):
    info = []
    for flagname in [
            "CONTIGUOUS",
            "FORTRAN",
            "OWNDATA",
            "ENSURECOPY",
            "ENSUREARRAY",
            "ALIGNED",
            "NOTSWAPPED",
            "WRITEABLE",
            "WRITEBACKIFCOPY",
            "UPDATEIFCOPY",
            "BEHAVED",
            "BEHAVED_RO",
            "CARRAY",
            "FARRAY",
    ]:
        if abs(flags) & getattr(wrap, flagname, 0):
            info.append(flagname)
    return info


class Intent:
    def __init__(self, intent_list=[]):
        self.intent_list = intent_list[:]
        flags = 0
        for i in intent_list:
            if i == "optional":
                flags |= wrap.F2PY_OPTIONAL
            else:
                flags |= getattr(wrap, "F2PY_INTENT_" + i.upper())
        self.flags = flags

    def __getattr__(self, name):
        name = name.lower()
        if name == "in_":
            name = "in"
        return self.__class__(self.intent_list + [name])

    def __str__(self):
        return "intent(%s)" % (",".join(self.intent_list))

    def __repr__(self):
        return "Intent(%r)" % (self.intent_list)

    def is_intent(self, *names):
        for name in names:
            if name not in self.intent_list:
                return False
        return True

    def is_intent_exact(self, *names):
        return len(self.intent_list) == len(names) and self.is_intent(*names)


intent = Intent()

_type_names = [
    "BOOL",
    "BYTE",
    "UBYTE",
    "SHORT",
    "USHORT",
    "INT",
    "UINT",
    "LONG",
    "ULONG",
    "LONGLONG",
    "ULONGLONG",
    "FLOAT",
    "DOUBLE",
    "CFLOAT",
    "STRING1",
    "STRING5",
    "CHARACTER",
]

_cast_dict = {"BOOL": ["BOOL"]}
_cast_dict["BYTE"] = _cast_dict["BOOL"] + ["BYTE"]
_cast_dict["UBYTE"] = _cast_dict["BOOL"] + ["UBYTE"]
_cast_dict["BYTE"] = ["BYTE"]
_cast_dict["UBYTE"] = ["UBYTE"]
_cast_dict["SHORT"] = _cast_dict["BYTE"] + ["UBYTE", "SHORT"]
_cast_dict["USHORT"] = _cast_dict["UBYTE"] + ["BYTE", "USHORT"]
_cast_dict["INT"] = _cast_dict["SHORT"] + ["USHORT", "INT"]
_cast_dict["UINT"] = _cast_dict["USHORT"] + ["SHORT", "UINT"]

_cast_dict["LONG"] = _cast_dict["INT"] + ["LONG"]
_cast_dict["ULONG"] = _cast_dict["UINT"] + ["ULONG"]

_cast_dict["LONGLONG"] = _cast_dict["LONG"] + ["LONGLONG"]
_cast_dict["ULONGLONG"] = _cast_dict["ULONG"] + ["ULONGLONG"]

_cast_dict["FLOAT"] = _cast_dict["SHORT"] + ["USHORT", "FLOAT"]
_cast_dict["DOUBLE"] = _cast_dict["INT"] + ["UINT", "FLOAT", "DOUBLE"]

_cast_dict["CFLOAT"] = _cast_dict["FLOAT"] + ["CFLOAT"]

_cast_dict['STRING1'] = ['STRING1']
_cast_dict['STRING5'] = ['STRING5']
_cast_dict['CHARACTER'] = ['CHARACTER']

# 32 bit system malloc typically does not provide the alignment required by
# 16 byte long double types this means the inout intent cannot be satisfied
# and several tests fail as the alignment flag can be randomly true or fals
# when numpy gains an aligned allocator the tests could be enabled again
#
# Furthermore, on macOS ARM64, LONGDOUBLE is an alias for DOUBLE.
if ((np.intp().dtype.itemsize != 4 or np.clongdouble().dtype.alignment <= 8)
        and sys.platform != "win32"
        and (platform.system(), platform.processor()) != ("Darwin", "arm")):
    _type_names.extend(["LONGDOUBLE", "CDOUBLE", "CLONGDOUBLE"])
    _cast_dict["LONGDOUBLE"] = _cast_dict["LONG"] + [
        "ULONG",
        "FLOAT",
        "DOUBLE",
        "LONGDOUBLE",
    ]
    _cast_dict["CLONGDOUBLE"] = _cast_dict["LONGDOUBLE"] + [
        "CFLOAT",
        "CDOUBLE",
        "CLONGDOUBLE",
    ]
    _cast_dict["CDOUBLE"] = _cast_dict["DOUBLE"] + ["CFLOAT", "CDOUBLE"]


class Type:
    _type_cache = {}

    def __new__(cls, name):
        if isinstance(name, np.dtype):
            dtype0 = name
            name = None
            for n, i in typeinfo.items():
                if not isinstance(i, type) and dtype0.type is i.type:
                    name = n
                    break
        obj = cls._type_cache.get(name.upper(), None)
        if obj is not None:
            return obj
        obj = object.__new__(cls)
        obj._init(name)
        cls._type_cache[name.upper()] = obj
        return obj

    def _init(self, name):
        self.NAME = name.upper()

        if self.NAME == 'CHARACTER':
            info = typeinfo[self.NAME]
            self.type_num = getattr(wrap, 'NPY_STRING')
            self.elsize = 1
            self.dtype = np.dtype('c')
        elif self.NAME.startswith('STRING'):
            info = typeinfo[self.NAME[:6]]
            self.type_num = getattr(wrap, 'NPY_STRING')
            self.elsize = int(self.NAME[6:] or 0)
            self.dtype = np.dtype(f'S{self.elsize}')
        else:
            info = typeinfo[self.NAME]
            self.type_num = getattr(wrap, 'NPY_' + self.NAME)
            self.elsize = info.bits // 8
            self.dtype = np.dtype(info.type)

        assert self.type_num == info.num
        self.type = info.type
        self.dtypechar = info.char

    def __repr__(self):
        return (f"Type({self.NAME})|type_num={self.type_num},"
                f" dtype={self.dtype},"
                f" type={self.type}, elsize={self.elsize},"
                f" dtypechar={self.dtypechar}")

    def cast_types(self):
        return [self.__class__(_m) for _m in _cast_dict[self.NAME]]

    def all_types(self):
        return [self.__class__(_m) for _m in _type_names]

    def smaller_types(self):
        bits = typeinfo[self.NAME].alignment
        types = []
        for name in _type_names:
            if typeinfo[name].alignment < bits:
                types.append(Type(name))
        return types

    def equal_types(self):
        bits = typeinfo[self.NAME].alignment
        types = []
        for name in _type_names:
            if name == self.NAME:
                continue
            if typeinfo[name].alignment == bits:
                types.append(Type(name))
        return types

    def larger_types(self):
        bits = typeinfo[self.NAME].alignment
        types = []
        for name in _type_names:
            if typeinfo[name].alignment > bits:
                types.append(Type(name))
        return types


class Array:

    def __repr__(self):
        return (f'Array({self.type}, {self.dims}, {self.intent},'
                f' {self.obj})|arr={self.arr}')

    def __init__(self, typ, dims, intent, obj):
        self.type = typ
        self.dims = dims
        self.intent = intent
        self.obj_copy = copy.deepcopy(obj)
        self.obj = obj

        # arr.dtypechar may be different from typ.dtypechar
        self.arr = wrap.call(typ.type_num,
                             typ.elsize,
                             dims, intent.flags, obj)

        assert isinstance(self.arr, np.ndarray)

        self.arr_attr = wrap.array_attrs(self.arr)

        if len(dims) > 1:
            if self.intent.is_intent("c"):
                assert (intent.flags & wrap.F2PY_INTENT_C)
                assert not self.arr.flags["FORTRAN"]
                assert self.arr.flags["CONTIGUOUS"]
                assert (not self.arr_attr[6] & wrap.FORTRAN)
            else:
                assert (not intent.flags & wrap.F2PY_INTENT_C)
                assert self.arr.flags["FORTRAN"]
                assert not self.arr.flags["CONTIGUOUS"]
                assert (self.arr_attr[6] & wrap.FORTRAN)

        if obj is None:
            self.pyarr = None
            self.pyarr_attr = None
            return

        if intent.is_intent("cache"):
            assert isinstance(obj, np.ndarray), repr(type(obj))
            self.pyarr = np.array(obj).reshape(*dims).copy()
        else:
            self.pyarr = np.array(
                np.array(obj, dtype=typ.dtypechar).reshape(*dims),
                order=self.intent.is_intent("c") and "C" or "F",
            )
            assert self.pyarr.dtype == typ
        self.pyarr.setflags(write=self.arr.flags["WRITEABLE"])
        assert self.pyarr.flags["OWNDATA"], (obj, intent)
        self.pyarr_attr = wrap.array_attrs(self.pyarr)

        if len(dims) > 1:
            if self.intent.is_intent("c"):
                assert not self.pyarr.flags["FORTRAN"]
                assert self.pyarr.flags["CONTIGUOUS"]
                assert (not self.pyarr_attr[6] & wrap.FORTRAN)
            else:
                assert self.pyarr.flags["FORTRAN"]
                assert not self.pyarr.flags["CONTIGUOUS"]
                assert (self.pyarr_attr[6] & wrap.FORTRAN)

        assert self.arr_attr[1] == self.pyarr_attr[1]  # nd
        assert self.arr_attr[2] == self.pyarr_attr[2]  # dimensions
        if self.arr_attr[1] <= 1:
            assert self.arr_attr[3] == self.pyarr_attr[3], repr((
                self.arr_attr[3],
                self.pyarr_attr[3],
                self.arr.tobytes(),
                self.pyarr.tobytes(),
            ))  # strides
        assert self.arr_attr[5][-2:] == self.pyarr_attr[5][-2:], repr((
            self.arr_attr[5], self.pyarr_attr[5]
            ))  # descr
        assert self.arr_attr[6] == self.pyarr_attr[6], repr((
            self.arr_attr[6],
            self.pyarr_attr[6],
            flags2names(0 * self.arr_attr[6] - self.pyarr_attr[6]),
            flags2names(self.arr_attr[6]),
            intent,
        ))  # flags

        if intent.is_intent("cache"):
            assert self.arr_attr[5][3] >= self.type.elsize
        else:
            assert self.arr_attr[5][3] == self.type.elsize
            assert (self.arr_equal(self.pyarr, self.arr))

        if isinstance(self.obj, np.ndarray):
            if typ.elsize == Type(obj.dtype).elsize:
                if not intent.is_intent("copy") and self.arr_attr[1] <= 1:
                    assert self.has_shared_memory()

    def arr_equal(self, arr1, arr2):
        if arr1.shape != arr2.shape:
            return False
        return (arr1 == arr2).all()

    def __str__(self):
        return str(self.arr)

    def has_shared_memory(self):
        """Check that created array shares data with input array."""
        if self.obj is self.arr:
            return True
        if not isinstance(self.obj, np.ndarray):
            return False
        obj_attr = wrap.array_attrs(self.obj)
        return obj_attr[0] == self.arr_attr[0]


class TestIntent:
    def test_in_out(self):
        assert str(intent.in_.out) == "intent(in,out)"
        assert intent.in_.c.is_intent("c")
        assert not intent.in_.c.is_intent_exact("c")
        assert intent.in_.c.is_intent_exact("c", "in")
        assert intent.in_.c.is_intent_exact("in", "c")
        assert not intent.in_.is_intent("c")


class TestSharedMemory:

    @pytest.fixture(autouse=True, scope="class", params=_type_names)
    def setup_type(self, request):
        request.cls.type = Type(request.param)
        request.cls.array = lambda self, dims, intent, obj: Array(
            Type(request.param), dims, intent, obj)

    @property
    def num2seq(self):
        if self.type.NAME.startswith('STRING'):
            elsize = self.type.elsize
            return ['1' * elsize, '2' * elsize]
        return [1, 2]

    @property
    def num23seq(self):
        if self.type.NAME.startswith('STRING'):
            elsize = self.type.elsize
            return [['1' * elsize, '2' * elsize, '3' * elsize],
                    ['4' * elsize, '5' * elsize, '6' * elsize]]
        return [[1, 2, 3], [4, 5, 6]]

    def test_in_from_2seq(self):
        a = self.array([2], intent.in_, self.num2seq)
        assert not a.has_shared_memory()

    def test_in_from_2casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num2seq, dtype=t.dtype)
            a = self.array([len(self.num2seq)], intent.in_, obj)
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory(), repr((self.type.dtype, t.dtype))
            else:
                assert not a.has_shared_memory()

    @pytest.mark.parametrize("write", ["w", "ro"])
    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("inp", ["2seq", "23seq"])
    def test_in_nocopy(self, write, order, inp):
        """Test if intent(in) array can be passed without copies"""
        seq = getattr(self, "num" + inp)
        obj = np.array(seq, dtype=self.type.dtype, order=order)
        obj.setflags(write=(write == 'w'))
        a = self.array(obj.shape,
                       ((order == 'C' and intent.in_.c) or intent.in_), obj)
        assert a.has_shared_memory()

    def test_inout_2seq(self):
        obj = np.array(self.num2seq, dtype=self.type.dtype)
        a = self.array([len(self.num2seq)], intent.inout, obj)
        assert a.has_shared_memory()

        try:
            a = self.array([2], intent.in_.inout, self.num2seq)
        except TypeError as msg:
            if not str(msg).startswith(
                    "failed to initialize intent(inout|inplace|cache) array"):
                raise
        else:
            raise SystemError("intent(inout) should have failed on sequence")

    def test_f_inout_23seq(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype, order="F")
        shape = (len(self.num23seq), len(self.num23seq[0]))
        a = self.array(shape, intent.in_.inout, obj)
        assert a.has_shared_memory()

        obj = np.array(self.num23seq, dtype=self.type.dtype, order="C")
        shape = (len(self.num23seq), len(self.num23seq[0]))
        try:
            a = self.array(shape, intent.in_.inout, obj)
        except ValueError as msg:
            if not str(msg).startswith(
                    "failed to initialize intent(inout) array"):
                raise
        else:
            raise SystemError(
                "intent(inout) should have failed on improper array")

    def test_c_inout_23seq(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype)
        shape = (len(self.num23seq), len(self.num23seq[0]))
        a = self.array(shape, intent.in_.c.inout, obj)
        assert a.has_shared_memory()

    def test_in_copy_from_2casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num2seq, dtype=t.dtype)
            a = self.array([len(self.num2seq)], intent.in_.copy, obj)
            assert not a.has_shared_memory()

    def test_c_in_from_23seq(self):
        a = self.array(
            [len(self.num23seq), len(self.num23seq[0])], intent.in_,
            self.num23seq)
        assert not a.has_shared_memory()

    def test_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype)
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_, obj)
            assert not a.has_shared_memory()

    def test_f_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype, order="F")
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_, obj)
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory()
            else:
                assert not a.has_shared_memory()

    def test_c_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype)
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.c, obj)
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory()
            else:
                assert not a.has_shared_memory()

    def test_f_copy_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype, order="F")
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.copy,
                obj)
            assert not a.has_shared_memory()

    def test_c_copy_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype)
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.c.copy,
                obj)
            assert not a.has_shared_memory()

    def test_in_cache_from_2casttype(self):
        for t in self.type.all_types():
            if t.elsize != self.type.elsize:
                continue
            obj = np.array(self.num2seq, dtype=t.dtype)
            shape = (len(self.num2seq), )
            a = self.array(shape, intent.in_.c.cache, obj)
            assert a.has_shared_memory()

            a = self.array(shape, intent.in_.cache, obj)
            assert a.has_shared_memory()

            obj = np.array(self.num2seq, dtype=t.dtype, order="F")
            a = self.array(shape, intent.in_.c.cache, obj)
            assert a.has_shared_memory()

            a = self.array(shape, intent.in_.cache, obj)
            assert a.has_shared_memory(), repr(t.dtype)

            try:
                a = self.array(shape, intent.in_.cache, obj[::-1])
            except ValueError as msg:
                if not str(msg).startswith(
                        "failed to initialize intent(cache) array"):
                    raise
            else:
                raise SystemError(
                    "intent(cache) should have failed on multisegmented array")

    def test_in_cache_from_2casttype_failure(self):
        for t in self.type.all_types():
            if t.NAME == 'STRING':
                # string elsize is 0, so skipping the test
                continue
            if t.elsize >= self.type.elsize:
                continue
            obj = np.array(self.num2seq, dtype=t.dtype)
            shape = (len(self.num2seq), )
            try:
                self.array(shape, intent.in_.cache, obj)  # Should succeed
            except ValueError as msg:
                if not str(msg).startswith(
                        "failed to initialize intent(cache) array"):
                    raise
            else:
                raise SystemError(
                    "intent(cache) should have failed on smaller array")

    def test_cache_hidden(self):
        shape = (2, )
        a = self.array(shape, intent.cache.hide, None)
        assert a.arr.shape == shape

        shape = (2, 3)
        a = self.array(shape, intent.cache.hide, None)
        assert a.arr.shape == shape

        shape = (-1, 3)
        try:
            a = self.array(shape, intent.cache.hide, None)
        except ValueError as msg:
            if not str(msg).startswith(
                    "failed to create intent(cache|hide)|optional array"):
                raise
        else:
            raise SystemError(
                "intent(cache) should have failed on undefined dimensions")

    def test_hidden(self):
        shape = (2, )
        a = self.array(shape, intent.hide, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))

        shape = (2, 3)
        a = self.array(shape, intent.hide, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert a.arr.flags["FORTRAN"] and not a.arr.flags["CONTIGUOUS"]

        shape = (2, 3)
        a = self.array(shape, intent.c.hide, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert not a.arr.flags["FORTRAN"] and a.arr.flags["CONTIGUOUS"]

        shape = (-1, 3)
        try:
            a = self.array(shape, intent.hide, None)
        except ValueError as msg:
            if not str(msg).startswith(
                    "failed to create intent(cache|hide)|optional array"):
                raise
        else:
            raise SystemError(
                "intent(hide) should have failed on undefined dimensions")

    def test_optional_none(self):
        shape = (2, )
        a = self.array(shape, intent.optional, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))

        shape = (2, 3)
        a = self.array(shape, intent.optional, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert a.arr.flags["FORTRAN"] and not a.arr.flags["CONTIGUOUS"]

        shape = (2, 3)
        a = self.array(shape, intent.c.optional, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert not a.arr.flags["FORTRAN"] and a.arr.flags["CONTIGUOUS"]

    def test_optional_from_2seq(self):
        obj = self.num2seq
        shape = (len(obj), )
        a = self.array(shape, intent.optional, obj)
        assert a.arr.shape == shape
        assert not a.has_shared_memory()

    def test_optional_from_23seq(self):
        obj = self.num23seq
        shape = (len(obj), len(obj[0]))
        a = self.array(shape, intent.optional, obj)
        assert a.arr.shape == shape
        assert not a.has_shared_memory()

        a = self.array(shape, intent.optional.c, obj)
        assert a.arr.shape == shape
        assert not a.has_shared_memory()

    def test_inplace(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype)
        assert not obj.flags["FORTRAN"] and obj.flags["CONTIGUOUS"]
        shape = obj.shape
        a = self.array(shape, intent.inplace, obj)
        assert obj[1][2] == a.arr[1][2], repr((obj, a.arr))
        a.arr[1][2] = 54
        assert obj[1][2] == a.arr[1][2] == np.array(54, dtype=self.type.dtype)
        assert a.arr is obj
        assert obj.flags["FORTRAN"]  # obj attributes are changed inplace!
        assert not obj.flags["CONTIGUOUS"]

    def test_inplace_from_casttype(self):
        for t in self.type.cast_types():
            if t is self.type:
                continue
            obj = np.array(self.num23seq, dtype=t.dtype)
            assert obj.dtype.type == t.type
            assert obj.dtype.type is not self.type.type
            assert not obj.flags["FORTRAN"] and obj.flags["CONTIGUOUS"]
            shape = obj.shape
            a = self.array(shape, intent.inplace, obj)
            assert obj[1][2] == a.arr[1][2], repr((obj, a.arr))
            a.arr[1][2] = 54
            assert obj[1][2] == a.arr[1][2] == np.array(54,
                                                        dtype=self.type.dtype)
            assert a.arr is obj
            assert obj.flags["FORTRAN"]  # obj attributes changed inplace!
            assert not obj.flags["CONTIGUOUS"]
            assert obj.dtype.type is self.type.type  # obj changed inplace!
