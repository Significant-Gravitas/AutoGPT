import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any

import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises, HAS_REFCOUNT,
    IS_PYSTON, _OLD_PROMOTION)
from numpy.compat import pickle
from itertools import permutations
import random

import hypothesis
from hypothesis.extra import numpy as hynp



def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(hash(a), hash(b),
                 "two equivalent types do not hash to the same value !")

def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b),
            "two different types hash to the same value !")

class TestBuiltin:
    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object,
                                   np.compat.unicode])
    def test_run(self, t):
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    @pytest.mark.parametrize('t', [int, float])
    def test_dtype(self, t):
        # Make sure equivalent byte order char hash the same (e.g. < and = on
        # little endian)
        dt = np.dtype(t)
        dt2 = dt.newbyteorder("<")
        dt3 = dt.newbyteorder(">")
        if dt == dt2:
            assert_(dt.byteorder != dt2.byteorder, "bogus test")
            assert_dtype_equal(dt, dt2)
        else:
            assert_(dt.byteorder != dt3.byteorder, "bogus test")
            assert_dtype_equal(dt, dt3)

    def test_equivalent_dtype_hashing(self):
        # Make sure equivalent dtypes with different type num hash equal
        uintp = np.dtype(np.uintp)
        if uintp.itemsize == 4:
            left = uintp
            right = np.dtype(np.uint32)
        else:
            left = uintp
            right = np.dtype(np.ulonglong)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    def test_invalid_types(self):
        # Make sure invalid type strings raise an error

        assert_raises(TypeError, np.dtype, 'O3')
        assert_raises(TypeError, np.dtype, 'O5')
        assert_raises(TypeError, np.dtype, 'O7')
        assert_raises(TypeError, np.dtype, 'b3')
        assert_raises(TypeError, np.dtype, 'h4')
        assert_raises(TypeError, np.dtype, 'I5')
        assert_raises(TypeError, np.dtype, 'e3')
        assert_raises(TypeError, np.dtype, 'f5')

        if np.dtype('g').itemsize == 8 or np.dtype('g').itemsize == 16:
            assert_raises(TypeError, np.dtype, 'g12')
        elif np.dtype('g').itemsize == 12:
            assert_raises(TypeError, np.dtype, 'g16')

        if np.dtype('l').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'l4')
            assert_raises(TypeError, np.dtype, 'L4')
        else:
            assert_raises(TypeError, np.dtype, 'l8')
            assert_raises(TypeError, np.dtype, 'L8')

        if np.dtype('q').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'q4')
            assert_raises(TypeError, np.dtype, 'Q4')
        else:
            assert_raises(TypeError, np.dtype, 'q8')
            assert_raises(TypeError, np.dtype, 'Q8')

    def test_richcompare_invalid_dtype_equality(self):
        # Make sure objects that cannot be converted to valid
        # dtypes results in False/True when compared to valid dtypes.
        # Here 7 cannot be converted to dtype. No exceptions should be raised

        assert not np.dtype(np.int32) == 7, "dtype richcompare failed for =="
        assert np.dtype(np.int32) != 7, "dtype richcompare failed for !="

    @pytest.mark.parametrize(
        'operation',
        [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # Make sure TypeError is raised for comparison operators
        # for invalid dtypes. Here 7 is an invalid dtype.

        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    @pytest.mark.parametrize("dtype",
             ['Bool', 'Bytes0', 'Complex32', 'Complex64',
              'Datetime64', 'Float16', 'Float32', 'Float64',
              'Int8', 'Int16', 'Int32', 'Int64',
              'Object0', 'Str0', 'Timedelta64',
              'UInt8', 'UInt16', 'Uint32', 'UInt32',
              'Uint64', 'UInt64', 'Void0',
              "Float128", "Complex128"])
    def test_numeric_style_types_are_invalid(self, dtype):
        with assert_raises(TypeError):
            np.dtype(dtype)

    def test_remaining_dtypes_with_bad_bytesize(self):
        # The np.<name> aliases were deprecated, these probably should be too 
        assert np.dtype("int0") is np.dtype("intp")
        assert np.dtype("uint0") is np.dtype("uintp")
        assert np.dtype("bool8") is np.dtype("bool")
        assert np.dtype("bytes0") is np.dtype("bytes")
        assert np.dtype("str0") is np.dtype("str")
        assert np.dtype("object0") is np.dtype("object")

    @pytest.mark.parametrize(
        'value',
        ['m8', 'M8', 'datetime64', 'timedelta64',
         'i4, (2,3)f8, f4', 'a3, 3u8, (3,4)a10',
         '>f', '<f', '=f', '|f',
        ])
    def test_dtype_bytes_str_equivalence(self, value):
        bytes_value = value.encode('ascii')
        from_bytes = np.dtype(bytes_value)
        from_str = np.dtype(value)
        assert_dtype_equal(from_bytes, from_str)

    def test_dtype_from_bytes(self):
        # Empty bytes object
        assert_raises(TypeError, np.dtype, b'')
        # Byte order indicator, but no type
        assert_raises(TypeError, np.dtype, b'|')

        # Single character with ordinal < NPY_NTYPES returns
        # type by index into _builtin_descrs
        assert_dtype_equal(np.dtype(bytes([0])), np.dtype('bool'))
        assert_dtype_equal(np.dtype(bytes([17])), np.dtype(object))

        # Single character where value is a valid type code
        assert_dtype_equal(np.dtype(b'f'), np.dtype('float32'))

        # Bytes with non-ascii values raise errors
        assert_raises(TypeError, np.dtype, b'\xff')
        assert_raises(TypeError, np.dtype, b's\xff')

    def test_bad_param(self):
        # Can't give a size that's too small
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i4', 'i1'],
                         'offsets':[0, 4],
                         'itemsize':4})
        # If alignment is enabled, the alignment (4) must divide the itemsize
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i4', 'i1'],
                         'offsets':[0, 4],
                         'itemsize':9}, align=True)
        # If alignment is enabled, the individual fields must be aligned
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i1', 'f4'],
                         'offsets':[0, 2]}, align=True)

    def test_field_order_equality(self):
        x = np.dtype({'names': ['A', 'B'],
                      'formats': ['i4', 'f4'],
                      'offsets': [0, 4]})
        y = np.dtype({'names': ['B', 'A'],
                      'formats': ['i4', 'f4'],
                      'offsets': [4, 0]})
        assert_equal(x == y, False)
        # This is an safe cast (not equiv) due to the different names:
        assert np.can_cast(x, y, casting="safe")


class TestRecord:
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        assert_dtype_equal(a, b)

    def test_different_names(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype([('yo', int)])
        b = np.dtype([('ye', int)])
        assert_dtype_not_equal(a, b)

    def test_different_titles(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['RRed pixel', 'Blue pixel']})
        assert_dtype_not_equal(a, b)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount_dictionary_setting(self):
        names = ["name1"]
        formats = ["f8"]
        titles = ["t1"]
        offsets = [0]
        d = dict(names=names, formats=formats, titles=titles, offsets=offsets)
        refcounts = {k: sys.getrefcount(i) for k, i in d.items()}
        np.dtype(d)
        refcounts_new = {k: sys.getrefcount(i) for k, i in d.items()}
        assert refcounts == refcounts_new

    def test_mutate(self):
        # Mutating a dtype should reset the cached hash value.
        # NOTE: Mutating should be deprecated, but new API added to replace it.
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        c = np.dtype([('ye', int)])
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)
        a.names = ['ye']
        assert_dtype_equal(a, c)
        assert_dtype_not_equal(a, b)
        state = b.__reduce__()[2]
        a.__setstate__(state)
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)

    def test_mutate_error(self):
        # NOTE: Mutating should be deprecated, but new API added to replace it.
        a = np.dtype("i,i")

        with pytest.raises(ValueError, match="must replace all names at once"):
            a.names = ["f0"]

        with pytest.raises(ValueError, match=".*and not string"):
            a.names = ["f0", b"not a unicode name"]

    def test_not_lists(self):
        """Test if an appropriate exception is raised when passing bad values to
        the dtype constructor.
        """
        assert_raises(TypeError, np.dtype,
                      dict(names={'A', 'B'}, formats=['f8', 'i4']))
        assert_raises(TypeError, np.dtype,
                      dict(names=['A', 'B'], formats={'f8', 'i4'}))

    def test_aligned_size(self):
        # Check that structured dtypes get padded to an aligned size
        dt = np.dtype('i4, i1', align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype([('f0', 'i4'), ('f1', 'i1')], align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'names':['f0', 'f1'],
                       'formats':['i4', 'u1'],
                       'offsets':[0, 4]}, align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'f0': ('i4', 0), 'f1':('u1', 4)}, align=True)
        assert_equal(dt.itemsize, 8)
        # Nesting should preserve that alignment
        dt1 = np.dtype([('f0', 'i4'),
                       ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                       ('f2', 'i1')], align=True)
        assert_equal(dt1.itemsize, 20)
        dt2 = np.dtype({'names':['f0', 'f1', 'f2'],
                       'formats':['i4',
                                  [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                  'i1'],
                       'offsets':[0, 4, 16]}, align=True)
        assert_equal(dt2.itemsize, 20)
        dt3 = np.dtype({'f0': ('i4', 0),
                       'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                       'f2': ('i1', 16)}, align=True)
        assert_equal(dt3.itemsize, 20)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        # Nesting should preserve packing
        dt1 = np.dtype([('f0', 'i4'),
                       ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                       ('f2', 'i1')], align=False)
        assert_equal(dt1.itemsize, 11)
        dt2 = np.dtype({'names':['f0', 'f1', 'f2'],
                       'formats':['i4',
                                  [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                  'i1'],
                       'offsets':[0, 4, 10]}, align=False)
        assert_equal(dt2.itemsize, 11)
        dt3 = np.dtype({'f0': ('i4', 0),
                       'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                       'f2': ('i1', 10)}, align=False)
        assert_equal(dt3.itemsize, 11)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        # Array of subtype should preserve alignment
        dt1 = np.dtype([('a', '|i1'),
                        ('b', [('f0', '<i2'),
                        ('f1', '<f4')], 2)], align=True)
        assert_equal(dt1.descr, [('a', '|i1'), ('', '|V3'),
                                 ('b', [('f0', '<i2'), ('', '|V2'),
                                 ('f1', '<f4')], (2,))])

    def test_union_struct(self):
        # Should be able to create union dtypes
        dt = np.dtype({'names':['f0', 'f1', 'f2'], 'formats':['<u4', '<u2', '<u2'],
                        'offsets':[0, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 4)
        a = np.array([3], dtype='<u4').view(dt)
        a['f1'] = 10
        a['f2'] = 36
        assert_equal(a['f0'], 10 + 36*256*256)
        # Should be able to specify fields out of order
        dt = np.dtype({'names':['f0', 'f1', 'f2'], 'formats':['<u4', '<u2', '<u2'],
                        'offsets':[4, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 8)
        # field name should not matter: assignment is by position
        dt2 = np.dtype({'names':['f2', 'f0', 'f1'],
                        'formats':['<u4', '<u2', '<u2'],
                        'offsets':[4, 0, 2]}, align=True)
        vals = [(0, 1, 2), (3, 2**15-1, 4)]
        vals2 = [(0, 1, 2), (3, 2**15-1, 4)]
        a = np.array(vals, dt)
        b = np.array(vals2, dt2)
        assert_equal(a.astype(dt2), b)
        assert_equal(b.astype(dt), a)
        assert_equal(a.view(dt2), b)
        assert_equal(b.view(dt), a)
        # Should not be able to overlap objects with other types
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['O', 'i1'],
                 'offsets':[0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['i4', 'O'],
                 'offsets':[0, 3]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':[[('a', 'O')], 'i1'],
                 'offsets':[0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['i4', [('a', 'O')]],
                 'offsets':[0, 3]})
        # Out of order should still be ok, however
        dt = np.dtype({'names':['f0', 'f1'],
                       'formats':['i1', 'O'],
                       'offsets':[np.dtype('intp').itemsize, 0]})

    @pytest.mark.parametrize(["obj", "dtype", "expected"],
        [([], ("(2)f4,"), np.empty((0, 2), dtype="f4")),
         (3, "(3)f4,", [3, 3, 3]),
         (np.float64(2), "(2)f4,", [2, 2]),
         ([((0, 1), (1, 2)), ((2,),)], '(2,2)f4', None),
         (["1", "2"], "(2)i,", None)])
    def test_subarray_list(self, obj, dtype, expected):
        dtype = np.dtype(dtype)
        res = np.array(obj, dtype=dtype)

        if expected is None:
            # iterate the 1-d list to fill the array
            expected = np.empty(len(obj), dtype=dtype)
            for i in range(len(expected)):
                expected[i] = obj[i]

        assert_array_equal(res, expected)

    def test_comma_datetime(self):
        dt = np.dtype('M8[D],datetime64[Y],i8')
        assert_equal(dt, np.dtype([('f0', 'M8[D]'),
                                   ('f1', 'datetime64[Y]'),
                                   ('f2', 'i8')]))

    def test_from_dictproxy(self):
        # Tests for PR #5920
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'f4']})
        assert_dtype_equal(dt, np.dtype(dt.fields))
        dt2 = np.dtype((np.void, dt.fields))
        assert_equal(dt2.fields, dt.fields)

    def test_from_dict_with_zero_width_field(self):
        # Regression test for #6430 / #2196
        dt = np.dtype([('val1', np.float32, (0,)), ('val2', int)])
        dt2 = np.dtype({'names': ['val1', 'val2'],
                        'formats': [(np.float32, (0,)), int]})

        assert_dtype_equal(dt, dt2)
        assert_equal(dt.fields['val1'][0].itemsize, 0)
        assert_equal(dt.itemsize, dt.fields['val2'][0].itemsize)

    def test_bool_commastring(self):
        d = np.dtype('?,?,?')  # raises?
        assert_equal(len(d.names), 3)
        for n in d.names:
            assert_equal(d.fields[n][0], np.dtype('?'))

    def test_nonint_offsets(self):
        # gh-8059
        def make_dtype(off):
            return np.dtype({'names': ['A'], 'formats': ['i4'],
                             'offsets': [off]})

        assert_raises(TypeError, make_dtype, 'ASD')
        assert_raises(OverflowError, make_dtype, 2**70)
        assert_raises(TypeError, make_dtype, 2.3)
        assert_raises(ValueError, make_dtype, -10)

        # no errors here:
        dt = make_dtype(np.uint32(0))
        np.zeros(1, dtype=dt)[0].item()

    def test_fields_by_index(self):
        dt = np.dtype([('a', np.int8), ('b', np.float32, 3)])
        assert_dtype_equal(dt[0], np.dtype(np.int8))
        assert_dtype_equal(dt[1], np.dtype((np.float32, 3)))
        assert_dtype_equal(dt[-1], dt[1])
        assert_dtype_equal(dt[-2], dt[0])
        assert_raises(IndexError, lambda: dt[-3])

        assert_raises(TypeError, operator.getitem, dt, 3.0)

        assert_equal(dt[1], dt[np.int8(1)])

    @pytest.mark.parametrize('align_flag',[False, True])
    def test_multifield_index(self, align_flag):
        # indexing with a list produces subfields
        # the align flag should be preserved
        dt = np.dtype([
            (('title', 'col1'), '<U20'), ('A', '<f8'), ('B', '<f8')
        ], align=align_flag)

        dt_sub = dt[['B', 'col1']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B', 'col1'],
                'formats': ['<f8', '<U20'],
                'offsets': [88, 0],
                'titles': [None, 'title'],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        dt_sub = dt[['B']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B'],
                'formats': ['<f8'],
                'offsets': [88],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        dt_sub = dt[[]]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': [],
                'formats': [],
                'offsets': [],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        assert_raises(TypeError, operator.getitem, dt, ())
        assert_raises(TypeError, operator.getitem, dt, [1, 2, 3])
        assert_raises(TypeError, operator.getitem, dt, ['col1', 2])
        assert_raises(KeyError, operator.getitem, dt, ['fake'])
        assert_raises(KeyError, operator.getitem, dt, ['title'])
        assert_raises(ValueError, operator.getitem, dt, ['col1', 'col1'])

    def test_partial_dict(self):
        # 'names' is missing
        assert_raises(ValueError, np.dtype,
                {'formats': ['i4', 'i4'], 'f0': ('i4', 0), 'f1':('i4', 4)})

    def test_fieldless_views(self):
        a = np.zeros(2, dtype={'names':[], 'formats':[], 'offsets':[],
                               'itemsize':8})
        assert_raises(ValueError, a.view, np.dtype([]))

        d = np.dtype((np.dtype([]), 10))
        assert_equal(d.shape, (10,))
        assert_equal(d.itemsize, 0)
        assert_equal(d.base, np.dtype([]))

        arr = np.fromiter((() for i in range(10)), [])
        assert_equal(arr.dtype, np.dtype([]))
        assert_raises(ValueError, np.frombuffer, b'', dtype=[])
        assert_equal(np.frombuffer(b'', dtype=[], count=2),
                     np.empty(2, dtype=[]))

        assert_raises(ValueError, np.dtype, ([], 'f8'))
        assert_raises(ValueError, np.zeros(1, dtype='i4').view, [])

        assert_equal(np.zeros(2, dtype=[]) == np.zeros(2, dtype=[]),
                     np.ones(2, dtype=bool))

        assert_equal(np.zeros((1, 2), dtype=[]) == a,
                     np.ones((1, 2), dtype=bool))


class TestSubarray:
    def test_single_subarray(self):
        a = np.dtype((int, (2)))
        b = np.dtype((int, (2,)))
        assert_dtype_equal(a, b)

        assert_equal(type(a.subdtype[1]), tuple)
        assert_equal(type(b.subdtype[1]), tuple)

    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (2, 3)))
        assert_dtype_equal(a, b)

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (3, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (2, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((int, (1, 2, 3)))
        b = np.dtype((int, (1, 2)))
        assert_dtype_not_equal(a, b)

    def test_shape_equal(self):
        """Test some data types that are equal"""
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', tuple())))
        # FutureWarning during deprecation period; after it is passed this
        # should instead check that "(1)f8" == "1f8" == ("f8", 1).
        with pytest.warns(FutureWarning):
            assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', 1)))
        assert_dtype_equal(np.dtype((int, 2)), np.dtype((int, (2,))))
        assert_dtype_equal(np.dtype(('<f4', (3, 2))), np.dtype(('<f4', (3, 2))))
        d = ([('a', 'f4', (1, 2)), ('b', 'f8', (3, 1))], (3, 2))
        assert_dtype_equal(np.dtype(d), np.dtype(d))

    def test_shape_simple(self):
        """Test some simple cases that shouldn't be equal"""
        assert_dtype_not_equal(np.dtype('f8'), np.dtype(('f8', (1,))))
        assert_dtype_not_equal(np.dtype(('f8', (1,))), np.dtype(('f8', (1, 1))))
        assert_dtype_not_equal(np.dtype(('f4', (3, 2))), np.dtype(('f4', (2, 3))))

    def test_shape_monster(self):
        """Test some more complicated cases that shouldn't be equal"""
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (1, 2)), ('b', 'f8', (1, 3))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'i8', (1, 3))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('e', 'f8', (1, 3)), ('d', 'f4', (2, 1))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', [('a', 'i4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', [('a', 'u4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))))

    def test_shape_sequence(self):
        # Any sequence of integers should work as shape, but the result
        # should be a tuple (immutable) of base type integers.
        a = np.array([1, 2, 3], dtype=np.int16)
        l = [1, 2, 3]
        # Array gets converted
        dt = np.dtype([('a', 'f4', a)])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))
        # List gets converted
        dt = np.dtype([('a', 'f4', l)])
        assert_(isinstance(dt['a'].shape, tuple))
        #

        class IntLike:
            def __index__(self):
                return 3

            def __int__(self):
                # (a PyNumber_Check fails without __int__)
                return 3

        dt = np.dtype([('a', 'f4', IntLike())])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))
        dt = np.dtype([('a', 'f4', (IntLike(),))])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))

    def test_shape_matches_ndim(self):
        dt = np.dtype([('a', 'f4', ())])
        assert_equal(dt['a'].shape, ())
        assert_equal(dt['a'].ndim, 0)

        dt = np.dtype([('a', 'f4')])
        assert_equal(dt['a'].shape, ())
        assert_equal(dt['a'].ndim, 0)

        dt = np.dtype([('a', 'f4', 4)])
        assert_equal(dt['a'].shape, (4,))
        assert_equal(dt['a'].ndim, 1)

        dt = np.dtype([('a', 'f4', (1, 2, 3))])
        assert_equal(dt['a'].shape, (1, 2, 3))
        assert_equal(dt['a'].ndim, 3)

    def test_shape_invalid(self):
        # Check that the shape is valid.
        max_int = np.iinfo(np.intc).max
        max_intp = np.iinfo(np.intp).max
        # Too large values (the datatype is part of this)
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int // 4 + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (max_int, 2))])
        # Takes a different code path (fails earlier:
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_intp + 1)])
        # Negative values
        assert_raises(ValueError, np.dtype, [('a', 'f4', -1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (-1, -1))])

    def test_alignment(self):
        #Check that subarrays are aligned
        t1 = np.dtype('(1,)i4', align=True)
        t2 = np.dtype('2i4', align=True)
        assert_equal(t1.alignment, t2.alignment)

    def test_aligned_empty(self):
        # Mainly regression test for gh-19696: construction failed completely
        dt = np.dtype([], align=True)
        assert dt == np.dtype([])
        dt = np.dtype({"names": [], "formats": [], "itemsize": 0}, align=True)
        assert dt == np.dtype([])

    def test_subarray_base_item(self):
        arr = np.ones(3, dtype=[("f", "i", 3)])
        # Extracting the field "absorbs" the subarray into a view:
        assert arr["f"].base is arr
        # Extract the structured item, and then check the tuple component:
        item = arr.item(0)
        assert type(item) is tuple and len(item) == 1
        assert item[0].base is arr

    def test_subarray_cast_copies(self):
        # Older versions of NumPy did NOT copy, but they got the ownership
        # wrong (not actually knowing the correct base!).  Versions since 1.21
        # (I think) crashed fairly reliable.  This defines the correct behavior
        # as a copy.  Keeping the ownership would be possible (but harder)
        arr = np.ones(3, dtype=[("f", "i", 3)])
        cast = arr.astype(object)
        for fields in cast:
            assert type(fields) == tuple and len(fields) == 1
            subarr = fields[0]
            assert subarr.base is None
            assert subarr.flags.owndata


def iter_struct_object_dtypes():
    """
    Iterates over a few complex dtypes and object pattern which
    fill the array with a given object (defaults to a singleton).

    Yields
    ------
    dtype : dtype
    pattern : tuple
        Structured tuple for use with `np.array`.
    count : int
        Number of objects stored in the dtype.
    singleton : object
        A singleton object. The returned pattern is constructed so that
        all objects inside the datatype are set to the singleton.
    """
    obj = object()

    dt = np.dtype([('b', 'O', (2, 3))])
    p = ([[obj] * 3] * 2,)
    yield pytest.param(dt, p, 6, obj, id="<subarray>")

    dt = np.dtype([('a', 'i4'), ('b', 'O', (2, 3))])
    p = (0, [[obj] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<subarray in field>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'i1')], (2, 3))])
    p = (0, [[(obj, 0)] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<structured subarray 1>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'O')], (2, 3))])
    p = (0, [[(obj, obj)] * 3] * 2)
    yield pytest.param(dt, p, 12, obj, id="<structured subarray 2>")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
class TestStructuredObjectRefcounting:
    """These tests cover various uses of complicated structured types which
    include objects and thus require reference counting.
    """
    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(["creation_func", "creation_obj"], [
        pytest.param(np.empty, None,
             # None is probably used for too many things
             marks=pytest.mark.skip("unreliable due to python's behaviour")),
        (np.ones, 1),
        (np.zeros, 0)])
    def test_structured_object_create_delete(self, dt, pat, count, singleton,
                                             creation_func, creation_obj):
        """Structured object reference counting in creation and deletion"""
        # The test assumes that 0, 1, and None are singletons.
        gc.collect()
        before = sys.getrefcount(creation_obj)
        arr = creation_func(3, dt)

        now = sys.getrefcount(creation_obj)
        assert now - before == count * 3
        del arr
        now = sys.getrefcount(creation_obj)
        assert now == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_item_setting(self, dt, pat, count, singleton):
        """Structured object reference counting for simple item setting"""
        one = 1

        gc.collect()
        before = sys.getrefcount(singleton)
        arr = np.array([pat] * 3, dt)
        assert sys.getrefcount(singleton) - before == count * 3
        # Fill with `1` and check that it was replaced correctly:
        before2 = sys.getrefcount(one)
        arr[...] = one
        after2 = sys.getrefcount(one)
        assert after2 - before2 == count * 3
        del arr
        gc.collect()
        assert sys.getrefcount(one) == before2
        assert sys.getrefcount(singleton) == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(
        ['shape', 'index', 'items_changed'],
        [((3,), ([0, 2],), 2),
         ((3, 2), ([0, 2], slice(None)), 4),
         ((3, 2), ([0, 2], [1]), 2),
         ((3,), ([True, False, True]), 2)])
    def test_structured_object_indexing(self, shape, index, items_changed,
                                        dt, pat, count, singleton):
        """Structured object reference counting for advanced indexing."""
        # Use two small negative values (should be singletons, but less likely
        # to run into race-conditions).  This failed in some threaded envs
        # When using 0 and 1.  If it fails again, should remove all explicit
        # checks, and rely on `pytest-leaks` reference count checker only.
        val0 = -4
        val1 = -5

        arr = np.full(shape, val0, dt)

        gc.collect()
        before_val0 = sys.getrefcount(val0)
        before_val1 = sys.getrefcount(val1)
        # Test item getting:
        part = arr[index]
        after_val0 = sys.getrefcount(val0)
        assert after_val0 - before_val0 == count * items_changed
        del part
        # Test item setting:
        arr[index] = val1
        gc.collect()
        after_val0 = sys.getrefcount(val0)
        after_val1 = sys.getrefcount(val1)
        assert before_val0 - after_val0 == count * items_changed
        assert after_val1 - before_val1 == count * items_changed

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_take_and_repeat(self, dt, pat, count, singleton):
        """Structured object reference counting for specialized functions.
        The older functions such as take and repeat use different code paths
        then item setting (when writing this).
        """
        indices = [0, 1]

        arr = np.array([pat] * 3, dt)
        gc.collect()
        before = sys.getrefcount(singleton)
        res = arr.take(indices)
        after = sys.getrefcount(singleton)
        assert after - before == count * 2
        new = res.repeat(10)
        gc.collect()
        after_repeat = sys.getrefcount(singleton)
        assert after_repeat - after == count * 2 * 10


class TestStructuredDtypeSparseFields:
    """Tests subarray fields which contain sparse dtypes so that
    not all memory is used by the dtype work. Such dtype's should
    leave the underlying memory unchanged.
    """
    dtype = np.dtype([('a', {'names':['aa', 'ab'], 'formats':['f', 'f'],
                             'offsets':[0, 4]}, (2, 3))])
    sparse_dtype = np.dtype([('a', {'names':['ab'], 'formats':['f'],
                                    'offsets':[4]}, (2, 3))])

    def test_sparse_field_assignment(self):
        arr = np.zeros(3, self.dtype)
        sparse_arr = arr.view(self.sparse_dtype)

        sparse_arr[...] = np.finfo(np.float32).max
        # dtype is reduced when accessing the field, so shape is (3, 2, 3):
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))

    def test_sparse_field_assignment_fancy(self):
        # Fancy assignment goes to the copyswap function for complex types:
        arr = np.zeros(3, self.dtype)
        sparse_arr = arr.view(self.sparse_dtype)

        sparse_arr[[0, 1, 2]] = np.finfo(np.float32).max
        # dtype is reduced when accessing the field, so shape is (3, 2, 3):
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))


class TestMonsterType:
    """Test deeply nested subtypes."""

    def test1(self):
        simple1 = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        a = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((int, (3, 2))))])
        b = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((int, (3, 2))))])
        assert_dtype_equal(a, b)

        c = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        d = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        assert_dtype_equal(c, d)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_list_recursion(self):
        l = list()
        l.append(('f', l))
        with pytest.raises(RecursionError):
            np.dtype(l)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_tuple_recursion(self):
        d = np.int32
        for i in range(100000):
            d = (d, (1,))
        with pytest.raises(RecursionError):
            np.dtype(d)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_dict_recursion(self):
        d = dict(names=['self'], formats=[None], offsets=[0])
        d['formats'][0] = d
        with pytest.raises(RecursionError):
            np.dtype(d)


class TestMetadata:
    def test_no_metadata(self):
        d = np.dtype(int)
        assert_(d.metadata is None)

    def test_metadata_takes_dict(self):
        d = np.dtype(int, metadata={'datum': 1})
        assert_(d.metadata == {'datum': 1})

    def test_metadata_rejects_nondict(self):
        assert_raises(TypeError, np.dtype, int, metadata='datum')
        assert_raises(TypeError, np.dtype, int, metadata=1)
        assert_raises(TypeError, np.dtype, int, metadata=None)

    def test_nested_metadata(self):
        d = np.dtype([('a', np.dtype(int, metadata={'datum': 1}))])
        assert_(d['a'].metadata == {'datum': 1})

    def test_base_metadata_copied(self):
        d = np.dtype((np.void, np.dtype('i4,i4', metadata={'datum': 1})))
        assert_(d.metadata == {'datum': 1})

class TestString:
    def test_complex_dtype_str(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(str(dt),
                     "[('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])]")

        # If the sticky aligned flag is set to True, it makes the
        # str() function use a dict representation with an 'aligned' flag
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))],
                                (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])],
                       align=True)
        assert_equal(str(dt),
                    "{'names': ['top', 'bottom'],"
                    " 'formats': [([('tiles', ('>f4', (64, 64)), (1,)), "
                                   "('rtile', '>f4', (64, 36))], (3,)), "
                                  "[('bleft', ('>f4', (8, 64)), (1,)), "
                                   "('bright', '>f4', (8, 36))]],"
                    " 'offsets': [0, 76800],"
                    " 'itemsize': 80000,"
                    " 'aligned': True}")
        with np.printoptions(legacy='1.21'):
            assert_equal(str(dt),
                        "{'names':['top','bottom'], "
                         "'formats':[([('tiles', ('>f4', (64, 64)), (1,)), "
                                      "('rtile', '>f4', (64, 36))], (3,)),"
                                     "[('bleft', ('>f4', (8, 64)), (1,)), "
                                      "('bright', '>f4', (8, 36))]], "
                         "'offsets':[0,76800], "
                         "'itemsize':80000, "
                         "'aligned':True}")
        assert_equal(np.dtype(eval(str(dt))), dt)

        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "[(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')]")

        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names': ['rgba', 'r', 'g', 'b'],"
                    " 'formats': ['<u4', 'u1', 'u1', 'u1'],"
                    " 'offsets': [0, 0, 1, 2],"
                    " 'titles': ['Color', 'Red pixel', "
                               "'Green pixel', 'Blue pixel'],"
                    " 'itemsize': 4}")

        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names': ['r', 'b'],"
                    " 'formats': ['u1', 'u1'],"
                    " 'offsets': [0, 2],"
                    " 'titles': ['Red pixel', 'Blue pixel'],"
                    " 'itemsize': 3}")

        dt = np.dtype([('a', '<m8[D]'), ('b', '<M8[us]')])
        assert_equal(str(dt),
                    "[('a', '<m8[D]'), ('b', '<M8[us]')]")

    def test_repr_structured(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(repr(dt),
                     "dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])])")

        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']},
                        align=True)
        assert_equal(repr(dt),
                    "dtype([(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')], align=True)")

    def test_repr_structured_not_packed(self):
        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']}, align=True)
        assert_equal(repr(dt),
                    "dtype({'names': ['rgba', 'r', 'g', 'b'],"
                    " 'formats': ['<u4', 'u1', 'u1', 'u1'],"
                    " 'offsets': [0, 0, 1, 2],"
                    " 'titles': ['Color', 'Red pixel', "
                                "'Green pixel', 'Blue pixel'],"
                    " 'itemsize': 4}, align=True)")

        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel'],
                        'itemsize': 4})
        assert_equal(repr(dt),
                    "dtype({'names': ['r', 'b'], "
                    "'formats': ['u1', 'u1'], "
                    "'offsets': [0, 2], "
                    "'titles': ['Red pixel', 'Blue pixel'], "
                    "'itemsize': 4})")

    def test_repr_structured_datetime(self):
        dt = np.dtype([('a', '<M8[D]'), ('b', '<m8[us]')])
        assert_equal(repr(dt),
                    "dtype([('a', '<M8[D]'), ('b', '<m8[us]')])")

    def test_repr_str_subarray(self):
        dt = np.dtype(('<i2', (1,)))
        assert_equal(repr(dt), "dtype(('<i2', (1,)))")
        assert_equal(str(dt), "('<i2', (1,))")

    def test_base_dtype_with_object_type(self):
        # Issue gh-2798, should not error.
        np.array(['a'], dtype="O").astype(("O", [("name", "O")]))

    def test_empty_string_to_object(self):
        # Pull request #4722
        np.array(["", ""]).astype(object)

    def test_void_subclass_unsized(self):
        dt = np.dtype(np.record)
        assert_equal(repr(dt), "dtype('V')")
        assert_equal(str(dt), '|V0')
        assert_equal(dt.name, 'record')

    def test_void_subclass_sized(self):
        dt = np.dtype((np.record, 2))
        assert_equal(repr(dt), "dtype('V2')")
        assert_equal(str(dt), '|V2')
        assert_equal(dt.name, 'record16')

    def test_void_subclass_fields(self):
        dt = np.dtype((np.record, [('a', '<u2')]))
        assert_equal(repr(dt), "dtype((numpy.record, [('a', '<u2')]))")
        assert_equal(str(dt), "(numpy.record, [('a', '<u2')])")
        assert_equal(dt.name, 'record16')


class TestDtypeAttributeDeletion:

    def test_dtype_non_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["subdtype", "descr", "str", "name", "base", "shape",
                "isbuiltin", "isnative", "isalignedstruct", "fields",
                "metadata", "hasobject"]

        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["names"]
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)


class TestDtypeAttributes:
    def test_descr_has_trailing_void(self):
        # see gh-6359
        dtype = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        new_dtype = np.dtype(dtype.descr)
        assert_equal(new_dtype.itemsize, 16)

    def test_name_dtype_subclass(self):
        # Ticket #4357
        class user_def_subcls(np.void):
            pass
        assert_equal(np.dtype(user_def_subcls).name, 'user_def_subcls')

    def test_zero_stride(self):
        arr = np.ones(1, dtype="i8")
        arr = np.broadcast_to(arr, 10)
        assert arr.strides == (0,)
        with pytest.raises(ValueError):
            arr.dtype = "i1"

class TestDTypeMakeCanonical:
    def check_canonical(self, dtype, canonical):
        """
        Check most properties relevant to "canonical" versions of a dtype,
        which is mainly native byte order for datatypes supporting this.

        The main work is checking structured dtypes with fields, where we
        reproduce most the actual logic used in the C-code.
        """
        assert type(dtype) is type(canonical)

        # a canonical DType should always have equivalent casting (both ways)
        assert np.can_cast(dtype, canonical, casting="equiv")
        assert np.can_cast(canonical, dtype, casting="equiv")
        # a canonical dtype (and its fields) is always native (checks fields):
        assert canonical.isnative

        # Check that canonical of canonical is the same (no casting):
        assert np.result_type(canonical) == canonical

        if not dtype.names:
            # The flags currently never change for unstructured dtypes
            assert dtype.flags == canonical.flags
            return

        # Must have all the needs API flag set:
        assert dtype.flags & 0b10000

        # Check that the fields are identical (including titles):
        assert dtype.fields.keys() == canonical.fields.keys()

        def aligned_offset(offset, alignment):
            # round up offset:
            return - (-offset // alignment) * alignment

        totalsize = 0
        max_alignment = 1
        for name in dtype.names:
            # each field is also canonical:
            new_field_descr = canonical.fields[name][0]
            self.check_canonical(dtype.fields[name][0], new_field_descr)

            # Must have the "inherited" object related flags:
            expected = 0b11011 & new_field_descr.flags
            assert (canonical.flags & expected) == expected

            if canonical.isalignedstruct:
                totalsize = aligned_offset(totalsize, new_field_descr.alignment)
                max_alignment = max(new_field_descr.alignment, max_alignment)

            assert canonical.fields[name][1] == totalsize
            # if a title exists, they must match (otherwise empty tuple):
            assert dtype.fields[name][2:] == canonical.fields[name][2:]

            totalsize += new_field_descr.itemsize

        if canonical.isalignedstruct:
            totalsize = aligned_offset(totalsize, max_alignment)
        assert canonical.itemsize == totalsize
        assert canonical.alignment == max_alignment

    def test_simple(self):
        dt = np.dtype(">i4")
        assert np.result_type(dt).isnative
        assert np.result_type(dt).num == dt.num

        # dtype with empty space:
        struct_dt = np.dtype(">i4,<i1,i8,V3")[["f0", "f2"]]
        canonical = np.result_type(struct_dt)
        assert canonical.itemsize == 4+8
        assert canonical.isnative

        # aligned struct dtype with empty space:
        struct_dt = np.dtype(">i1,<i4,i8,V3", align=True)[["f0", "f2"]]
        canonical = np.result_type(struct_dt)
        assert canonical.isalignedstruct
        assert canonical.itemsize == np.dtype("i8").alignment + 8
        assert canonical.isnative

    def test_object_flag_not_inherited(self):
        # The following dtype still indicates "object", because its included
        # in the unaccessible space (maybe this could change at some point):
        arr = np.ones(3, "i,O,i")[["f0", "f2"]]
        assert arr.dtype.hasobject
        canonical_dt = np.result_type(arr.dtype)
        assert not canonical_dt.hasobject

    @pytest.mark.slow
    @hypothesis.given(dtype=hynp.nested_dtypes())
    def test_make_canonical_hypothesis(self, dtype):
        canonical = np.result_type(dtype)
        self.check_canonical(dtype, canonical)
        # result_type with two arguments should always give identical results:
        two_arg_result = np.result_type(dtype, dtype)
        assert np.can_cast(two_arg_result, canonical, casting="no")

    @pytest.mark.slow
    @hypothesis.given(
            dtype=hypothesis.extra.numpy.array_dtypes(
                subtype_strategy=hypothesis.extra.numpy.array_dtypes(),
                min_size=5, max_size=10, allow_subarrays=True))
    def test_structured(self, dtype):
        # Pick 4 of the fields at random.  This will leave empty space in the
        # dtype (since we do not canonicalize it here).
        field_subset = random.sample(dtype.names, k=4)
        dtype_with_empty_space = dtype[field_subset]
        assert dtype_with_empty_space.itemsize == dtype.itemsize
        canonicalized = np.result_type(dtype_with_empty_space)
        self.check_canonical(dtype_with_empty_space, canonicalized)
        # promotion with two arguments should always give identical results:
        two_arg_result = np.promote_types(
                dtype_with_empty_space, dtype_with_empty_space)
        assert np.can_cast(two_arg_result, canonicalized, casting="no")

        # Ensure that we also check aligned struct (check the opposite, in
        # case hypothesis grows support for `align`.  Then repeat the test:
        dtype_aligned = np.dtype(dtype.descr, align=not dtype.isalignedstruct)
        dtype_with_empty_space = dtype_aligned[field_subset]
        assert dtype_with_empty_space.itemsize == dtype_aligned.itemsize
        canonicalized = np.result_type(dtype_with_empty_space)
        self.check_canonical(dtype_with_empty_space, canonicalized)
        # promotion with two arguments should always give identical results:
        two_arg_result = np.promote_types(
            dtype_with_empty_space, dtype_with_empty_space)
        assert np.can_cast(two_arg_result, canonicalized, casting="no")


class TestPickling:

    def check_pickling(self, dtype):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            buf = pickle.dumps(dtype, proto)
            # The dtype pickling itself pickles `np.dtype` if it is pickled
            # as a singleton `dtype` should be stored in the buffer:
            assert b"_DType_reconstruct" not in buf
            assert b"dtype" in buf
            pickled = pickle.loads(buf)
            assert_equal(pickled, dtype)
            assert_equal(pickled.descr, dtype.descr)
            if dtype.metadata is not None:
                assert_equal(pickled.metadata, dtype.metadata)
            # Check the reconstructed dtype is functional
            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object,
                                   np.compat.unicode, bool])
    def test_builtin(self, t):
        self.check_pickling(np.dtype(t))

    def test_structured(self):
        dt = np.dtype(([('a', '>f4', (2, 1)), ('b', '<f8', (1, 3))], (2, 2)))
        self.check_pickling(dt)

    def test_structured_aligned(self):
        dt = np.dtype('i4, i1', align=True)
        self.check_pickling(dt)

    def test_structured_unaligned(self):
        dt = np.dtype('i4, i1', align=False)
        self.check_pickling(dt)

    def test_structured_padded(self):
        dt = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        self.check_pickling(dt)

    def test_structured_titles(self):
        dt = np.dtype({'names': ['r', 'b'],
                       'formats': ['u1', 'u1'],
                       'titles': ['Red pixel', 'Blue pixel']})
        self.check_pickling(dt)

    @pytest.mark.parametrize('base', ['m8', 'M8'])
    @pytest.mark.parametrize('unit', ['', 'Y', 'M', 'W', 'D', 'h', 'm', 's',
                                      'ms', 'us', 'ns', 'ps', 'fs', 'as'])
    def test_datetime(self, base, unit):
        dt = np.dtype('%s[%s]' % (base, unit) if unit else base)
        self.check_pickling(dt)
        if unit:
            dt = np.dtype('%s[7%s]' % (base, unit))
            self.check_pickling(dt)

    def test_metadata(self):
        dt = np.dtype(int, metadata={'datum': 1})
        self.check_pickling(dt)

    @pytest.mark.parametrize("DType",
        [type(np.dtype(t)) for t in np.typecodes['All']] +
        [np.dtype(rational), np.dtype])
    def test_pickle_types(self, DType):
        # Check that DTypes (the classes/types) roundtrip when pickling
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            assert roundtrip_DType is DType


class TestPromotion:
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """
    @np._no_nep50_warning()
    @pytest.mark.parametrize(["other", "expected", "expected_weak"],
            [(2**16-1, np.complex64, None),
             (2**32-1, np.complex128, np.complex64),
             (np.float16(2), np.complex64, None),
             (np.float32(2), np.complex64, None),
             (np.longdouble(2), np.complex64, np.clongdouble),
             # Base of the double value to sidestep any rounding issues:
             (np.longdouble(np.nextafter(1.7e308, 0.)),
                  np.complex128, np.clongdouble),
             # Additionally use "nextafter" so the cast can't round down:
             (np.longdouble(np.nextafter(1.7e308, np.inf)),
                  np.clongdouble, None),
             # repeat for complex scalars:
             (np.complex64(2), np.complex64, None),
             (np.clongdouble(2), np.complex64, np.clongdouble),
             # Base of the double value to sidestep any rounding issues:
             (np.clongdouble(np.nextafter(1.7e308, 0.) * 1j),
                  np.complex128, np.clongdouble),
             # Additionally use "nextafter" so the cast can't round down:
             (np.clongdouble(np.nextafter(1.7e308, np.inf)),
                  np.clongdouble, None),
             ])
    def test_complex_other_value_based(self,
            weak_promotion, other, expected, expected_weak):
        if weak_promotion and expected_weak is not None:
            expected = expected_weak

        # This would change if we modify the value based promotion
        min_complex = np.dtype(np.complex64)

        res = np.result_type(other, min_complex)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    @pytest.mark.parametrize(["other", "expected"],
                 [(np.bool_, np.complex128),
                  (np.int64, np.complex128),
                  (np.float16, np.complex64),
                  (np.float32, np.complex64),
                  (np.float64, np.complex128),
                  (np.longdouble, np.clongdouble),
                  (np.complex64, np.complex64),
                  (np.complex128, np.complex128),
                  (np.clongdouble, np.clongdouble),
                  ])
    def test_complex_scalar_value_based(self, other, expected):
        # This would change if we modify the value based promotion
        complex_scalar = 1j

        res = np.result_type(other, complex_scalar)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected

    def test_complex_pyscalar_promote_rational(self):
        with pytest.raises(TypeError,
                match=r".* no common DType exists for the given inputs"):
            np.result_type(1j, rational)

        with pytest.raises(TypeError,
                match=r".* no common DType exists for the given inputs"):
            np.result_type(1j, rational(1, 2))

    @pytest.mark.parametrize("val", [2, 2**32, 2**63, 2**64, 2*100])
    def test_python_integer_promotion(self, val):
        # If we only path scalars (mainly python ones!), the result must take
        # into account that the integer may be considered int32, int64, uint64,
        # or object depending on the input value.  So test those paths!
        expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
        assert np.result_type(val, 0) == expected_dtype
        # For completeness sake, also check with a NumPy scalar as second arg:
        assert np.result_type(val, np.int8(0)) == expected_dtype

    @pytest.mark.parametrize(["other", "expected"],
            [(1, rational), (1., np.float64)])
    @np._no_nep50_warning()
    def test_float_int_pyscalar_promote_rational(
            self, weak_promotion, other, expected):
        # Note that rationals are a bit akward as they promote with float64
        # or default ints, but not float16 or uint8/int8 (which looks
        # inconsistent here).  The new promotion fixes this (partially?)
        if not weak_promotion and type(other) == float:
            # The float version, checks float16 in the legacy path, which fails
            # the integer version seems to check int8 (also), so it can
            # pass.
            with pytest.raises(TypeError,
                    match=r".* do not have a common DType"):
                np.result_type(other, rational)
        else:
            assert np.result_type(other, rational) == expected

        assert np.result_type(other, rational(1, 2)) == expected

    @pytest.mark.parametrize(["dtypes", "expected"], [
             # These promotions are not associative/commutative:
             ([np.uint16, np.int16, np.float16], np.float32),
             ([np.uint16, np.int8, np.float16], np.float32),
             ([np.uint8, np.int16, np.float16], np.float32),
             # The following promotions are not ambiguous, but cover code
             # paths of abstract promotion (no particular logic being tested)
             ([1, 1, np.float64], np.float64),
             ([1, 1., np.complex128], np.complex128),
             ([1, 1j, np.float64], np.complex128),
             ([1., 1., np.int64], np.float64),
             ([1., 1j, np.float64], np.complex128),
             ([1j, 1j, np.float64], np.complex128),
             ([1, True, np.bool_], np.int_),
            ])
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # Tests that most permutations do not influence the result.  In the
        # above some uint and int combintations promote to a larger integer
        # type, which would then promote to a larger than necessary float.
        for perm in permutations(dtypes):
            assert np.result_type(*perm) == expected


def test_rational_dtype():
    # test for bug gh-5719
    a = np.array([1111], dtype=rational).astype
    assert_raises(OverflowError, a, 'int8')

    # test that dtype detection finds user-defined types
    x = rational(1)
    assert_equal(np.array([x,x]).dtype, np.dtype(rational))


def test_dtypes_are_true():
    # test for gh-6294
    assert bool(np.dtype('f8'))
    assert bool(np.dtype('i8'))
    assert bool(np.dtype([('a', 'i8'), ('b', 'f4')]))


def test_invalid_dtype_string():
    # test for gh-10440
    assert_raises(TypeError, np.dtype, 'f8,i8,[f8,i8]')
    assert_raises(TypeError, np.dtype, 'Fl\xfcgel')


def test_keyword_argument():
    # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
    assert np.dtype(dtype=np.float64) == np.dtype(np.float64)


def test_ulong_dtype():
    # test for gh-21063
    assert np.dtype("ulong") == np.dtype(np.uint)


class TestFromDTypeAttribute:
    def test_simple(self):
        class dt:
            dtype = np.dtype("f8")

        assert np.dtype(dt) == np.float64
        assert np.dtype(dt()) == np.float64

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)

    def test_void_subtype(self):
        class dt(np.void):
            # This code path is fully untested before, so it is unclear
            # what this should be useful for. Note that if np.void is used
            # numpy will think we are deallocating a base type [1.17, 2019-02].
            dtype = np.dtype("f,f")

        np.dtype(dt)
        np.dtype(dt(1))

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_void_subtype_recursion(self):
        class vdt(np.void):
            pass

        vdt.dtype = vdt

        with pytest.raises(RecursionError):
            np.dtype(vdt)

        with pytest.raises(RecursionError):
            np.dtype(vdt(1))


class TestDTypeClasses:
    @pytest.mark.parametrize("dtype", list(np.typecodes['All']) + [rational])
    def test_basic_dtypes_subclass_properties(self, dtype):
        # Note: Except for the isinstance and type checks, these attributes
        #       are considered currently private and may change.
        dtype = np.dtype(dtype)
        assert isinstance(dtype, np.dtype)
        assert type(dtype) is not np.dtype
        assert type(dtype).__name__ == f"dtype[{dtype.type.__name__}]"
        assert type(dtype).__module__ == "numpy"
        assert not type(dtype)._abstract

        # the flexible dtypes and datetime/timedelta have additional parameters
        # which are more than just storage information, these would need to be
        # given when creating a dtype:
        parametric = (np.void, np.str_, np.bytes_, np.datetime64, np.timedelta64)
        if dtype.type not in parametric:
            assert not type(dtype)._parametric
            assert type(dtype)() is dtype
        else:
            assert type(dtype)._parametric
            with assert_raises(TypeError):
                type(dtype)()

    def test_dtype_superclass(self):
        assert type(np.dtype) is not type
        assert isinstance(np.dtype, type)

        assert type(np.dtype).__name__ == "_DTypeMeta"
        assert type(np.dtype).__module__ == "numpy"
        assert np.dtype._abstract


class TestFromCTypes:

    @staticmethod
    def check(ctype, dtype):
        dtype = np.dtype(dtype)
        assert_equal(np.dtype(ctype), dtype)
        assert_equal(np.dtype(ctype()), dtype)

    def test_array(self):
        c8 = ctypes.c_uint8
        self.check(     3 * c8,  (np.uint8, (3,)))
        self.check(     1 * c8,  (np.uint8, (1,)))
        self.check(     0 * c8,  (np.uint8, (0,)))
        self.check(1 * (3 * c8), ((np.uint8, (3,)), (1,)))
        self.check(3 * (1 * c8), ((np.uint8, (1,)), (3,)))

    def test_padded_structure(self):
        class PaddedStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_bit_fields(self):
        class BitfieldStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8, 7),
                ('b', ctypes.c_uint8, 1)
            ]
        assert_raises(TypeError, np.dtype, BitfieldStruct)
        assert_raises(TypeError, np.dtype, BitfieldStruct())

    def test_pointer(self):
        p_uint8 = ctypes.POINTER(ctypes.c_uint8)
        assert_raises(TypeError, np.dtype, p_uint8)

    def test_void_pointer(self):
        self.check(ctypes.c_void_p, np.uintp)

    def test_union(self):
        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
            ]
        expected = np.dtype(dict(
            names=['a', 'b'],
            formats=[np.uint8, np.uint16],
            offsets=[0, 0],
            itemsize=2
        ))
        self.check(Union, expected)

    def test_union_with_struct_packed(self):
        class Struct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]

        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        expected = np.dtype(dict(
            names=['a', 'b', 'c', 'd'],
            formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            offsets=[0, 0, 0, 0],
            itemsize=ctypes.sizeof(Union)
        ))
        self.check(Union, expected)

    def test_union_packed(self):
        class Struct(ctypes.Structure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        class Union(ctypes.Union):
            _pack_ = 1
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        expected = np.dtype(dict(
            names=['a', 'b', 'c', 'd'],
            formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            offsets=[0, 0, 0, 0],
            itemsize=ctypes.sizeof(Union)
        ))
        self.check(Union, expected)

    def test_packed_structure(self):
        class PackedStructure(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ])
        self.check(PackedStructure, expected)

    def test_large_packed_structure(self):
        class PackedStructure(ctypes.Structure):
            _pack_ = 2
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint8),
                ('d', ctypes.c_uint16),
                ('e', ctypes.c_uint32),
                ('f', ctypes.c_uint32),
                ('g', ctypes.c_uint8)
                ]
        expected = np.dtype(dict(
            formats=[np.uint8, np.uint16, np.uint8, np.uint16, np.uint32, np.uint32, np.uint8 ],
            offsets=[0, 2, 4, 6, 8, 12, 16],
            names=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            itemsize=18))
        self.check(PackedStructure, expected)

    def test_big_endian_structure_packed(self):
        class BigEndStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '>u4')])
        self.check(BigEndStruct, expected)

    def test_little_endian_structure_packed(self):
        class LittleEndStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '<u4')])
        self.check(LittleEndStruct, expected)

    def test_little_endian_structure(self):
        class PaddedStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', '<B'),
            ('b', '<H')
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_big_endian_structure(self):
        class PaddedStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', '>B'),
            ('b', '>H')
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_simple_endian_types(self):
        self.check(ctypes.c_uint16.__ctype_le__, np.dtype('<u2'))
        self.check(ctypes.c_uint16.__ctype_be__, np.dtype('>u2'))
        self.check(ctypes.c_uint8.__ctype_le__, np.dtype('u1'))
        self.check(ctypes.c_uint8.__ctype_be__, np.dtype('u1'))

    all_types = set(np.typecodes['All'])
    all_pairs = permutations(all_types, 2)

    @pytest.mark.parametrize("pair", all_pairs)
    def test_pairs(self, pair):
        """
        Check that np.dtype('x,y') matches [np.dtype('x'), np.dtype('y')]
        Example: np.dtype('d,I') -> dtype([('f0', '<f8'), ('f1', '<u4')])
        """
        # gh-5645: check that np.dtype('i,L') can be used
        pair_type = np.dtype('{},{}'.format(*pair))
        expected = np.dtype([('f0', pair[0]), ('f1', pair[1])])
        assert_equal(pair_type, expected)


class TestUserDType:
    @pytest.mark.leaks_references(reason="dynamically creates custom dtype.")
    def test_custom_structured_dtype(self):
        class mytype:
            pass

        blueprint = np.dtype([("field", object)])
        dt = create_custom_field_dtype(blueprint, mytype, 0)
        assert dt.type == mytype
        # We cannot (currently) *create* this dtype with `np.dtype` because
        # mytype does not inherit from `np.generic`.  This seems like an
        # unnecessary restriction, but one that has been around forever:
        assert np.dtype(mytype) == np.dtype("O")

    def test_custom_structured_dtype_errors(self):
        class mytype:
            pass

        blueprint = np.dtype([("field", object)])

        with pytest.raises(ValueError):
            # Tests what happens if fields are unset during creation
            # which is currently rejected due to the containing object
            # (see PyArray_RegisterDataType).
            create_custom_field_dtype(blueprint, mytype, 1)

        with pytest.raises(RuntimeError):
            # Tests that a dtype must have its type field set up to np.dtype
            # or in this case a builtin instance.
            create_custom_field_dtype(blueprint, mytype, 2)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python 3.9")
class TestClassGetItem:
    def test_dtype(self) -> None:
        alias = np.dtype[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.dtype

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_dtype_subclass(self, code: str) -> None:
        cls = type(np.dtype(code))
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.dtype[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.dtype[Any]


def test_result_type_integers_and_unitless_timedelta64():
    # Regression test for gh-20077.  The following call of `result_type`
    # would cause a seg. fault.
    td = np.timedelta64(4)
    result = np.result_type(0, td)
    assert_dtype_equal(result, td.dtype)


@pytest.mark.skipif(sys.version_info >= (3, 9), reason="Requires python 3.8")
def test_class_getitem_38() -> None:
    match = "Type subscription requires python >= 3.9"
    with pytest.raises(TypeError, match=match):
        np.dtype[Any]
