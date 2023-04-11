import pytest

import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
    drop_fields, rename_fields, get_fieldstructure, recursive_fill_fields,
    find_duplicates, merge_arrays, append_fields, stack_arrays, join_by,
    repack_fields, unstructured_to_structured, structured_to_unstructured,
    apply_along_fields, require_fields, assign_fields_by_name)
get_fieldspec = np.lib.recfunctions._get_fieldspec
get_names = np.lib.recfunctions.get_names
get_names_flat = np.lib.recfunctions.get_names_flat
zip_descr = np.lib.recfunctions._zip_descr
zip_dtype = np.lib.recfunctions._zip_dtype


class TestRecFunctions:
    # Misc tests

    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.), ('B', 2.)],
                     dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_zip_descr(self):
        # Test zip_descr
        (w, x, y, z) = self.data

        # Std array
        test = zip_descr((x, x), flatten=True)
        assert_equal(test,
                     np.dtype([('', int), ('', int)]))
        test = zip_descr((x, x), flatten=False)
        assert_equal(test,
                     np.dtype([('', int), ('', int)]))

        # Std & flexible-dtype
        test = zip_descr((x, z), flatten=True)
        assert_equal(test,
                     np.dtype([('', int), ('A', '|S3'), ('B', float)]))
        test = zip_descr((x, z), flatten=False)
        assert_equal(test,
                     np.dtype([('', int),
                               ('', [('A', '|S3'), ('B', float)])]))

        # Standard & nested dtype
        test = zip_descr((x, w), flatten=True)
        assert_equal(test,
                     np.dtype([('', int),
                               ('a', int),
                               ('ba', float), ('bb', int)]))
        test = zip_descr((x, w), flatten=False)
        assert_equal(test,
                     np.dtype([('', int),
                               ('', [('a', int),
                                     ('b', [('ba', float), ('bb', int)])])]))

    def test_drop_fields(self):
        # Test drop_fields
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])

        # A basic field
        test = drop_fields(a, 'a')
        control = np.array([((2, 3.0),), ((5, 6.0),)],
                           dtype=[('b', [('ba', float), ('bb', int)])])
        assert_equal(test, control)

        # Another basic field (but nesting two fields)
        test = drop_fields(a, 'b')
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)

        # A nested sub-field
        test = drop_fields(a, ['ba', ])
        control = np.array([(1, (3.0,)), (4, (6.0,))],
                           dtype=[('a', int), ('b', [('bb', int)])])
        assert_equal(test, control)

        # All the nested sub-field from a field: zap that field
        test = drop_fields(a, ['ba', 'bb'])
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)

        # dropping all fields results in an array with no fields
        test = drop_fields(a, ['a', 'b'])
        control = np.array([(), ()], dtype=[])
        assert_equal(test, control)

    def test_rename_fields(self):
        # Test rename fields
        a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
                     dtype=[('a', int),
                            ('b', [('ba', float), ('bb', (float, 2))])])
        test = rename_fields(a, {'a': 'A', 'bb': 'BB'})
        newdtype = [('A', int), ('b', [('ba', float), ('BB', (float, 2))])]
        control = a.view(newdtype)
        assert_equal(test.dtype, newdtype)
        assert_equal(test, control)

    def test_get_names(self):
        # Test get_names
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_names(ndtype)
        assert_equal(test, ('A', 'B'))

        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ('ba', 'bb'))))

        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ())))

        ndtype = np.dtype([])
        test = get_names(ndtype)
        assert_equal(test, ())

    def test_get_names_flat(self):
        # Test get_names_flat
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_names_flat(ndtype)
        assert_equal(test, ('A', 'B'))

        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b', 'ba', 'bb'))

        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b'))

        ndtype = np.dtype([])
        test = get_names_flat(ndtype)
        assert_equal(test, ())

    def test_get_fieldstructure(self):
        # Test get_fieldstructure

        # No nested fields
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {'A': [], 'B': []})

        # One 1-nested field
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {'A': [], 'B': [], 'BA': ['B', ], 'BB': ['B']})

        # One 2-nested fields
        ndtype = np.dtype([('A', int),
                           ('B', [('BA', int),
                                  ('BB', [('BBA', int), ('BBB', int)])])])
        test = get_fieldstructure(ndtype)
        control = {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'],
                   'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
        assert_equal(test, control)

        # 0 fields
        ndtype = np.dtype([])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {})

    def test_find_duplicates(self):
        # Test find_duplicates
        a = ma.array([(2, (2., 'B')), (1, (2., 'B')), (2, (2., 'B')),
                      (1, (1., 'B')), (2, (2., 'B')), (2, (2., 'C'))],
                     mask=[(0, (0, 0)), (0, (0, 0)), (0, (0, 0)),
                           (0, (0, 0)), (1, (0, 0)), (0, (1, 0))],
                     dtype=[('A', int), ('B', [('BA', float), ('BB', '|S1')])])

        test = find_duplicates(a, ignoremask=False, return_index=True)
        control = [0, 2]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='A', return_index=True)
        control = [0, 1, 2, 3, 5]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='B', return_index=True)
        control = [0, 1, 2, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='BA', return_index=True)
        control = [0, 1, 2, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, key='BB', return_index=True)
        control = [0, 1, 2, 3, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

    def test_find_duplicates_ignoremask(self):
        # Test the ignoremask option of find_duplicates
        ndtype = [('a', int)]
        a = ma.array([1, 1, 1, 2, 2, 3, 3],
                     mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
        test = find_duplicates(a, ignoremask=True, return_index=True)
        control = [0, 1, 3, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

        test = find_duplicates(a, ignoremask=False, return_index=True)
        control = [0, 1, 2, 3, 4, 6]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

    def test_repack_fields(self):
        dt = np.dtype('u1,f4,i8', align=True)
        a = np.zeros(2, dtype=dt)

        assert_equal(repack_fields(dt), np.dtype('u1,f4,i8'))
        assert_equal(repack_fields(a).itemsize, 13)
        assert_equal(repack_fields(repack_fields(dt), align=True), dt)

        # make sure type is preserved
        dt = np.dtype((np.record, dt))
        assert_(repack_fields(dt).type is np.record)

    def test_structured_to_unstructured(self):
        a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
        out = structured_to_unstructured(a)
        assert_equal(out, np.zeros((4,5), dtype='f8'))

        b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
                     dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
        out = np.mean(structured_to_unstructured(b[['x', 'z']]), axis=-1)
        assert_equal(out, np.array([ 3. ,  5.5,  9. , 11. ]))
        out = np.mean(structured_to_unstructured(b[['x']]), axis=-1)
        assert_equal(out, np.array([ 1. ,  4. ,  7. , 10. ]))

        c = np.arange(20).reshape((4,5))
        out = unstructured_to_structured(c, a.dtype)
        want = np.array([( 0, ( 1.,  2), [ 3.,  4.]),
                         ( 5, ( 6.,  7), [ 8.,  9.]),
                         (10, (11., 12), [13., 14.]),
                         (15, (16., 17), [18., 19.])],
                     dtype=[('a', 'i4'),
                            ('b', [('f0', 'f4'), ('f1', 'u2')]),
                            ('c', 'f4', (2,))])
        assert_equal(out, want)

        d = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
                     dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
        assert_equal(apply_along_fields(np.mean, d),
                     np.array([ 8.0/3,  16.0/3,  26.0/3, 11. ]))
        assert_equal(apply_along_fields(np.mean, d[['x', 'z']]),
                     np.array([ 3. ,  5.5,  9. , 11. ]))

        # check that for uniform field dtypes we get a view, not a copy:
        d = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
                     dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'i4')])
        dd = structured_to_unstructured(d)
        ddd = unstructured_to_structured(dd, d.dtype)
        assert_(dd.base is d)
        assert_(ddd.base is d)

        # including uniform fields with subarrays unpacked
        d = np.array([(1, [2,  3], [[ 4,  5], [ 6,  7]]),
                      (8, [9, 10], [[11, 12], [13, 14]])],
                     dtype=[('x0', 'i4'), ('x1', ('i4', 2)),
                            ('x2', ('i4', (2, 2)))])
        dd = structured_to_unstructured(d)
        ddd = unstructured_to_structured(dd, d.dtype)
        assert_(dd.base is d)
        assert_(ddd.base is d)

        # test that nested fields with identical names don't break anything
        point = np.dtype([('x', int), ('y', int)])
        triangle = np.dtype([('a', point), ('b', point), ('c', point)])
        arr = np.zeros(10, triangle)
        res = structured_to_unstructured(arr, dtype=int)
        assert_equal(res, np.zeros((10, 6), dtype=int))


        # test nested combinations of subarrays and structured arrays, gh-13333
        def subarray(dt, shape):
            return np.dtype((dt, shape))

        def structured(*dts):
            return np.dtype([('x{}'.format(i), dt) for i, dt in enumerate(dts)])

        def inspect(dt, dtype=None):
            arr = np.zeros((), dt)
            ret = structured_to_unstructured(arr, dtype=dtype)
            backarr = unstructured_to_structured(ret, dt)
            return ret.shape, ret.dtype, backarr.dtype

        dt = structured(subarray(structured(np.int32, np.int32), 3))
        assert_equal(inspect(dt), ((6,), np.int32, dt))

        dt = structured(subarray(subarray(np.int32, 2), 2))
        assert_equal(inspect(dt), ((4,), np.int32, dt))

        dt = structured(np.int32)
        assert_equal(inspect(dt), ((1,), np.int32, dt))

        dt = structured(np.int32, subarray(subarray(np.int32, 2), 2))
        assert_equal(inspect(dt), ((5,), np.int32, dt))

        dt = structured()
        assert_raises(ValueError, structured_to_unstructured, np.zeros(3, dt))

        # these currently don't work, but we may make it work in the future
        assert_raises(NotImplementedError, structured_to_unstructured,
                                           np.zeros(3, dt), dtype=np.int32)
        assert_raises(NotImplementedError, unstructured_to_structured,
                                           np.zeros((3,0), dtype=np.int32))

    def test_unstructured_to_structured(self):
        # test if dtype is the args of np.dtype
        a = np.zeros((20, 2))
        test_dtype_args = [('x', float), ('y', float)]
        test_dtype = np.dtype(test_dtype_args)
        field1 = unstructured_to_structured(a, dtype=test_dtype_args)  # now
        field2 = unstructured_to_structured(a, dtype=test_dtype)  # before
        assert_equal(field1, field2)

    def test_field_assignment_by_name(self):
        a = np.ones(2, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
        newdt = [('b', 'f4'), ('c', 'u1')]

        assert_equal(require_fields(a, newdt), np.ones(2, newdt))

        b = np.array([(1,2), (3,4)], dtype=newdt)
        assign_fields_by_name(a, b, zero_unassigned=False)
        assert_equal(a, np.array([(1,1,2),(1,3,4)], dtype=a.dtype))
        assign_fields_by_name(a, b)
        assert_equal(a, np.array([(0,1,2),(0,3,4)], dtype=a.dtype))

        # test nested fields
        a = np.ones(2, dtype=[('a', [('b', 'f8'), ('c', 'u1')])])
        newdt = [('a', [('c', 'u1')])]
        assert_equal(require_fields(a, newdt), np.ones(2, newdt))
        b = np.array([((2,),), ((3,),)], dtype=newdt)
        assign_fields_by_name(a, b, zero_unassigned=False)
        assert_equal(a, np.array([((1,2),), ((1,3),)], dtype=a.dtype))
        assign_fields_by_name(a, b)
        assert_equal(a, np.array([((0,2),), ((0,3),)], dtype=a.dtype))

        # test unstructured code path for 0d arrays
        a, b = np.array(3), np.array(0)
        assign_fields_by_name(b, a)
        assert_equal(b[()], 3)


class TestRecursiveFillFields:
    # Test recursive_fill_fields.
    def test_simple_flexible(self):
        # Test recursive_fill_fields on flexible-array
        a = np.array([(1, 10.), (2, 20.)], dtype=[('A', int), ('B', float)])
        b = np.zeros((3,), dtype=a.dtype)
        test = recursive_fill_fields(a, b)
        control = np.array([(1, 10.), (2, 20.), (0, 0.)],
                           dtype=[('A', int), ('B', float)])
        assert_equal(test, control)

    def test_masked_flexible(self):
        # Test recursive_fill_fields on masked flexible-array
        a = ma.array([(1, 10.), (2, 20.)], mask=[(0, 1), (1, 0)],
                     dtype=[('A', int), ('B', float)])
        b = ma.zeros((3,), dtype=a.dtype)
        test = recursive_fill_fields(a, b)
        control = ma.array([(1, 10.), (2, 20.), (0, 0.)],
                           mask=[(0, 1), (1, 0), (0, 0)],
                           dtype=[('A', int), ('B', float)])
        assert_equal(test, control)


class TestMergeArrays:
    # Test merge_arrays

    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array(
            [(1, (2, 3.0, ())), (4, (5, 6.0, ()))],
            dtype=[('a', int), ('b', [('ba', float), ('bb', int), ('bc', [])])])
        self.data = (w, x, y, z)

    def test_solo(self):
        # Test merge_arrays on a single array.
        (_, x, _, z) = self.data

        test = merge_arrays(x)
        control = np.array([(1,), (2,)], dtype=[('f0', int)])
        assert_equal(test, control)
        test = merge_arrays((x,))
        assert_equal(test, control)

        test = merge_arrays(z, flatten=False)
        assert_equal(test, z)
        test = merge_arrays(z, flatten=True)
        assert_equal(test, z)

    def test_solo_w_flatten(self):
        # Test merge_arrays on a single array w & w/o flattening
        w = self.data[0]
        test = merge_arrays(w, flatten=False)
        assert_equal(test, w)

        test = merge_arrays(w, flatten=True)
        control = np.array([(1, 2, 3.0), (4, 5, 6.0)],
                           dtype=[('a', int), ('ba', float), ('bb', int)])
        assert_equal(test, control)

    def test_standard(self):
        # Test standard & standard
        # Test merge arrays
        (_, x, y, _) = self.data
        test = merge_arrays((x, y), usemask=False)
        control = np.array([(1, 10), (2, 20), (-1, 30)],
                           dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)

        test = merge_arrays((x, y), usemask=True)
        control = ma.array([(1, 10), (2, 20), (-1, 30)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_flatten(self):
        # Test standard & flexible
        (_, x, _, z) = self.data
        test = merge_arrays((x, z), flatten=True)
        control = np.array([(1, 'A', 1.), (2, 'B', 2.)],
                           dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        assert_equal(test, control)

        test = merge_arrays((x, z), flatten=False)
        control = np.array([(1, ('A', 1.)), (2, ('B', 2.))],
                           dtype=[('f0', int),
                                  ('f1', [('A', '|S3'), ('B', float)])])
        assert_equal(test, control)

    def test_flatten_wflexible(self):
        # Test flatten standard & nested
        (w, x, _, _) = self.data
        test = merge_arrays((x, w), flatten=True)
        control = np.array([(1, 1, 2, 3.0), (2, 4, 5, 6.0)],
                           dtype=[('f0', int),
                                  ('a', int), ('ba', float), ('bb', int)])
        assert_equal(test, control)

        test = merge_arrays((x, w), flatten=False)
        controldtype = [('f0', int),
                                ('f1', [('a', int),
                                        ('b', [('ba', float), ('bb', int), ('bc', [])])])]
        control = np.array([(1., (1, (2, 3.0, ()))), (2, (4, (5, 6.0, ())))],
                           dtype=controldtype)
        assert_equal(test, control)

    def test_wmasked_arrays(self):
        # Test merge_arrays masked arrays
        (_, x, _, _) = self.data
        mx = ma.array([1, 2, 3], mask=[1, 0, 0])
        test = merge_arrays((x, mx), usemask=True)
        control = ma.array([(1, 1), (2, 2), (-1, 3)],
                           mask=[(0, 1), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        test = merge_arrays((x, mx), usemask=True, asrecarray=True)
        assert_equal(test, control)
        assert_(isinstance(test, MaskedRecords))

    def test_w_singlefield(self):
        # Test single field
        test = merge_arrays((np.array([1, 2]).view([('a', int)]),
                             np.array([10., 20., 30.])),)
        control = ma.array([(1, 10.), (2, 20.), (-1, 30.)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('a', int), ('f1', float)])
        assert_equal(test, control)

    def test_w_shorter_flex(self):
        # Test merge_arrays w/ a shorter flexndarray.
        z = self.data[-1]

        # Fixme, this test looks incomplete and broken
        #test = merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        #control = np.array([('A', 1., 10), ('B', 2., 20), ('-1', -1, 20)],
        #                   dtype=[('A', '|S3'), ('B', float), ('C', int)])
        #assert_equal(test, control)

        # Hack to avoid pyflakes warnings about unused variables
        merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        np.array([('A', 1., 10), ('B', 2., 20), ('-1', -1, 20)],
                 dtype=[('A', '|S3'), ('B', float), ('C', int)])

    def test_singlerecord(self):
        (_, x, y, z) = self.data
        test = merge_arrays((x[0], y[0], z[0]), usemask=False)
        control = np.array([(1, 10, ('A', 1))],
                           dtype=[('f0', int),
                                  ('f1', int),
                                  ('f2', [('A', '|S3'), ('B', float)])])
        assert_equal(test, control)


class TestAppendFields:
    # Test append_fields

    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_append_single(self):
        # Test simple case
        (_, x, _, _) = self.data
        test = append_fields(x, 'A', data=[10, 20, 30])
        control = ma.array([(1, 10), (2, 20), (-1, 30)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('A', int)],)
        assert_equal(test, control)

    def test_append_double(self):
        # Test simple case
        (_, x, _, _) = self.data
        test = append_fields(x, ('A', 'B'), data=[[10, 20, 30], [100, 200]])
        control = ma.array([(1, 10, 100), (2, 20, 200), (-1, 30, -1)],
                           mask=[(0, 0, 0), (0, 0, 0), (1, 0, 1)],
                           dtype=[('f0', int), ('A', int), ('B', int)],)
        assert_equal(test, control)

    def test_append_on_flex(self):
        # Test append_fields on flexible type arrays
        z = self.data[-1]
        test = append_fields(z, 'C', data=[10, 20, 30])
        control = ma.array([('A', 1., 10), ('B', 2., 20), (-1, -1., 30)],
                           mask=[(0, 0, 0), (0, 0, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('C', int)],)
        assert_equal(test, control)

    def test_append_on_nested(self):
        # Test append_fields on nested fields
        w = self.data[0]
        test = append_fields(w, 'C', data=[10, 20, 30])
        control = ma.array([(1, (2, 3.0), 10),
                            (4, (5, 6.0), 20),
                            (-1, (-1, -1.), 30)],
                           mask=[(
                               0, (0, 0), 0), (0, (0, 0), 0), (1, (1, 1), 0)],
                           dtype=[('a', int),
                                  ('b', [('ba', float), ('bb', int)]),
                                  ('C', int)],)
        assert_equal(test, control)


class TestStackArrays:
    # Test stack_arrays
    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_solo(self):
        # Test stack_arrays on single arrays
        (_, x, _, _) = self.data
        test = stack_arrays((x,))
        assert_equal(test, x)
        assert_(test is x)

        test = stack_arrays(x)
        assert_equal(test, x)
        assert_(test is x)

    def test_unnamed_fields(self):
        # Tests combinations of arrays w/o named fields
        (_, x, y, _) = self.data

        test = stack_arrays((x, x), usemask=False)
        control = np.array([1, 2, 1, 2])
        assert_equal(test, control)

        test = stack_arrays((x, y), usemask=False)
        control = np.array([1, 2, 10, 20, 30])
        assert_equal(test, control)

        test = stack_arrays((y, x), usemask=False)
        control = np.array([10, 20, 30, 1, 2])
        assert_equal(test, control)

    def test_unnamed_and_named_fields(self):
        # Test combination of arrays w/ & w/o named fields
        (_, x, _, z) = self.data

        test = stack_arrays((x, z))
        control = ma.array([(1, -1, -1), (2, -1, -1),
                            (-1, 'A', 1), (-1, 'B', 2)],
                           mask=[(0, 1, 1), (0, 1, 1),
                                 (1, 0, 0), (1, 0, 0)],
                           dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

        test = stack_arrays((z, x))
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            (-1, -1, 1), (-1, -1, 2), ],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (1, 1, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

        test = stack_arrays((z, z, x))
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            ('A', 1, -1), ('B', 2, -1),
                            (-1, -1, 1), (-1, -1, 2), ],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 1), (0, 0, 1),
                                 (1, 1, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        assert_equal(test, control)

    def test_matching_named_fields(self):
        # Test combination of arrays w/ matching field names
        (_, x, _, z) = self.data
        zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                      dtype=[('A', '|S3'), ('B', float), ('C', float)])
        test = stack_arrays((z, zz))
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            (
                                'a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                           dtype=[('A', '|S3'), ('B', float), ('C', float)],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

        test = stack_arrays((z, zz, x))
        ndtype = [('A', '|S3'), ('B', float), ('C', float), ('f3', int)]
        control = ma.array([('A', 1, -1, -1), ('B', 2, -1, -1),
                            ('a', 10., 100., -1), ('b', 20., 200., -1),
                            ('c', 30., 300., -1),
                            (-1, -1, -1, 1), (-1, -1, -1, 2)],
                           dtype=ndtype,
                           mask=[(0, 0, 1, 1), (0, 0, 1, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1),
                                 (1, 1, 1, 0), (1, 1, 1, 0)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_defaults(self):
        # Test defaults: no exception raised if keys of defaults are not fields.
        (_, _, _, z) = self.data
        zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                      dtype=[('A', '|S3'), ('B', float), ('C', float)])
        defaults = {'A': '???', 'B': -999., 'C': -9999., 'D': -99999.}
        test = stack_arrays((z, zz), defaults=defaults)
        control = ma.array([('A', 1, -9999.), ('B', 2, -9999.),
                            (
                                'a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                           dtype=[('A', '|S3'), ('B', float), ('C', float)],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_autoconversion(self):
        # Tests autoconversion
        adtype = [('A', int), ('B', bool), ('C', float)]
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        bdtype = [('A', int), ('B', float), ('C', float)]
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)],
                           dtype=bdtype)
        test = stack_arrays((a, b), autoconvert=True)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        with assert_raises(TypeError):
            stack_arrays((a, b), autoconvert=False)

    def test_checktitles(self):
        # Test using titles in the field names
        adtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        bdtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        test = stack_arrays((a, b))
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)],
                           dtype=bdtype)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_subdtype(self):
        z = np.array([
            ('A', 1), ('B', 2)
        ], dtype=[('A', '|S3'), ('B', float, (1,))])
        zz = np.array([
            ('a', [10.], 100.), ('b', [20.], 200.), ('c', [30.], 300.)
        ], dtype=[('A', '|S3'), ('B', float, (1,)), ('C', float)])

        res = stack_arrays((z, zz))
        expected = ma.array(
            data=[
                (b'A', [1.0], 0),
                (b'B', [2.0], 0),
                (b'a', [10.0], 100.0),
                (b'b', [20.0], 200.0),
                (b'c', [30.0], 300.0)],
            mask=[
                (False, [False],  True),
                (False, [False],  True),
                (False, [False], False),
                (False, [False], False),
                (False, [False], False)
            ],
            dtype=zz.dtype
        )
        assert_equal(res.dtype, expected.dtype)
        assert_equal(res, expected)
        assert_equal(res.mask, expected.mask)


class TestJoinBy:
    def setup_method(self):
        self.a = np.array(list(zip(np.arange(10), np.arange(50, 60),
                                   np.arange(100, 110))),
                          dtype=[('a', int), ('b', int), ('c', int)])
        self.b = np.array(list(zip(np.arange(5, 15), np.arange(65, 75),
                                   np.arange(100, 110))),
                          dtype=[('a', int), ('b', int), ('d', int)])

    def test_inner_join(self):
        # Basic test of join_by
        a, b = self.a, self.b

        test = join_by('a', a, b, jointype='inner')
        control = np.array([(5, 55, 65, 105, 100), (6, 56, 66, 106, 101),
                            (7, 57, 67, 107, 102), (8, 58, 68, 108, 103),
                            (9, 59, 69, 109, 104)],
                           dtype=[('a', int), ('b1', int), ('b2', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_join(self):
        a, b = self.a, self.b

        # Fixme, this test is broken
        #test = join_by(('a', 'b'), a, b)
        #control = np.array([(5, 55, 105, 100), (6, 56, 106, 101),
        #                    (7, 57, 107, 102), (8, 58, 108, 103),
        #                    (9, 59, 109, 104)],
        #                   dtype=[('a', int), ('b', int),
        #                          ('c', int), ('d', int)])
        #assert_equal(test, control)

        # Hack to avoid pyflakes unused variable warnings
        join_by(('a', 'b'), a, b)
        np.array([(5, 55, 105, 100), (6, 56, 106, 101),
                  (7, 57, 107, 102), (8, 58, 108, 103),
                  (9, 59, 109, 104)],
                  dtype=[('a', int), ('b', int),
                         ('c', int), ('d', int)])

    def test_join_subdtype(self):
        # tests the bug in https://stackoverflow.com/q/44769632/102441
        foo = np.array([(1,)],
                       dtype=[('key', int)])
        bar = np.array([(1, np.array([1,2,3]))],
                       dtype=[('key', int), ('value', 'uint16', 3)])
        res = join_by('key', foo, bar)
        assert_equal(res, bar.view(ma.MaskedArray))

    def test_outer_join(self):
        a, b = self.a, self.b

        test = join_by(('a', 'b'), a, b, 'outer')
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1),
                            (2, 52, 102, -1), (3, 53, 103, -1),
                            (4, 54, 104, -1), (5, 55, 105, -1),
                            (5, 65, -1, 100), (6, 56, 106, -1),
                            (6, 66, -1, 101), (7, 57, 107, -1),
                            (7, 67, -1, 102), (8, 58, 108, -1),
                            (8, 68, -1, 103), (9, 59, 109, -1),
                            (9, 69, -1, 104), (10, 70, -1, 105),
                            (11, 71, -1, 106), (12, 72, -1, 107),
                            (13, 73, -1, 108), (14, 74, -1, 109)],
                           mask=[(0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 1, 0),
                                 (0, 0, 1, 0), (0, 0, 1, 0),
                                 (0, 0, 1, 0), (0, 0, 1, 0)],
                           dtype=[('a', int), ('b', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_leftouter_join(self):
        a, b = self.a, self.b

        test = join_by(('a', 'b'), a, b, 'leftouter')
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1),
                            (2, 52, 102, -1), (3, 53, 103, -1),
                            (4, 54, 104, -1), (5, 55, 105, -1),
                            (6, 56, 106, -1), (7, 57, 107, -1),
                            (8, 58, 108, -1), (9, 59, 109, -1)],
                           mask=[(0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1)],
                           dtype=[('a', int), ('b', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_different_field_order(self):
        # gh-8940
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        # this should not give a FutureWarning:
        j = join_by(['c', 'b'], a, b, jointype='inner', usemask=False)
        assert_equal(j.dtype.names, ['b', 'c', 'a1', 'a2'])

    def test_duplicate_keys(self):
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        assert_raises(ValueError, join_by, ['a', 'b', 'b'], a, b)

    def test_same_name_different_dtypes_key(self):
        a_dtype = np.dtype([('key', 'S5'), ('value', '<f4')])
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        expected_dtype = np.dtype([
            ('key', 'S10'), ('value1', '<f4'), ('value2', '<f4')])

        a = np.array([('Sarah',  8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        res = join_by('key', a, b)

        assert_equal(res.dtype, expected_dtype)

    def test_same_name_different_dtypes(self):
        # gh-9338
        a_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f8')])
        expected_dtype = np.dtype([
            ('key', '|S10'), ('value1', '<f4'), ('value2', '<f8')])

        a = np.array([('Sarah',  8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        res = join_by('key', a, b)

        assert_equal(res.dtype, expected_dtype)

    def test_subarray_key(self):
        a_dtype = np.dtype([('pos', int, 3), ('f', '<f4')])
        a = np.array([([1, 1, 1], np.pi), ([1, 2, 3], 0.0)], dtype=a_dtype)

        b_dtype = np.dtype([('pos', int, 3), ('g', '<f4')])
        b = np.array([([1, 1, 1], 3), ([3, 2, 1], 0.0)], dtype=b_dtype)

        expected_dtype = np.dtype([('pos', int, 3), ('f', '<f4'), ('g', '<f4')])
        expected = np.array([([1, 1, 1], np.pi, 3)], dtype=expected_dtype)

        res = join_by('pos', a, b)
        assert_equal(res.dtype, expected_dtype)
        assert_equal(res, expected)

    def test_padded_dtype(self):
        dt = np.dtype('i1,f4', align=True)
        dt.names = ('k', 'v')
        assert_(len(dt.descr), 3)  # padding field is inserted

        a = np.array([(1, 3), (3, 2)], dt)
        b = np.array([(1, 1), (2, 2)], dt)
        res = join_by('k', a, b)

        # no padding fields remain
        expected_dtype = np.dtype([
            ('k', 'i1'), ('v1', 'f4'), ('v2', 'f4')
        ])

        assert_equal(res.dtype, expected_dtype)


class TestJoinBy2:
    @classmethod
    def setup_method(cls):
        cls.a = np.array(list(zip(np.arange(10), np.arange(50, 60),
                                  np.arange(100, 110))),
                         dtype=[('a', int), ('b', int), ('c', int)])
        cls.b = np.array(list(zip(np.arange(10), np.arange(65, 75),
                                  np.arange(100, 110))),
                         dtype=[('a', int), ('b', int), ('d', int)])

    def test_no_r1postfix(self):
        # Basic test of join_by no_r1postfix
        a, b = self.a, self.b

        test = join_by(
            'a', a, b, r1postfix='', r2postfix='2', jointype='inner')
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101),
                            (2, 52, 67, 102, 102), (3, 53, 68, 103, 103),
                            (4, 54, 69, 104, 104), (5, 55, 70, 105, 105),
                            (6, 56, 71, 106, 106), (7, 57, 72, 107, 107),
                            (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)],
                           dtype=[('a', int), ('b', int), ('b2', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_no_postfix(self):
        assert_raises(ValueError, join_by, 'a', self.a, self.b,
                      r1postfix='', r2postfix='')

    def test_no_r2postfix(self):
        # Basic test of join_by no_r2postfix
        a, b = self.a, self.b

        test = join_by(
            'a', a, b, r1postfix='1', r2postfix='', jointype='inner')
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101),
                            (2, 52, 67, 102, 102), (3, 53, 68, 103, 103),
                            (4, 54, 69, 104, 104), (5, 55, 70, 105, 105),
                            (6, 56, 71, 106, 106), (7, 57, 72, 107, 107),
                            (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)],
                           dtype=[('a', int), ('b1', int), ('b', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_two_keys_two_vars(self):
        a = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2),
                              np.arange(50, 60), np.arange(10, 20))),
                     dtype=[('k', int), ('a', int), ('b', int), ('c', int)])

        b = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2),
                              np.arange(65, 75), np.arange(0, 10))),
                     dtype=[('k', int), ('a', int), ('b', int), ('c', int)])

        control = np.array([(10, 0, 50, 65, 10, 0), (11, 0, 51, 66, 11, 1),
                            (10, 1, 52, 67, 12, 2), (11, 1, 53, 68, 13, 3),
                            (10, 2, 54, 69, 14, 4), (11, 2, 55, 70, 15, 5),
                            (10, 3, 56, 71, 16, 6), (11, 3, 57, 72, 17, 7),
                            (10, 4, 58, 73, 18, 8), (11, 4, 59, 74, 19, 9)],
                           dtype=[('k', int), ('a', int), ('b1', int),
                                  ('b2', int), ('c1', int), ('c2', int)])
        test = join_by(
            ['a', 'k'], a, b, r1postfix='1', r2postfix='2', jointype='inner')
        assert_equal(test.dtype, control.dtype)
        assert_equal(test, control)

class TestAppendFieldsObj:
    """
    Test append_fields with arrays containing objects
    """
    # https://github.com/numpy/numpy/issues/2346

    def setup_method(self):
        from datetime import date
        self.data = dict(obj=date(2000, 1, 1))

    def test_append_to_objects(self):
        "Test append_fields when the base array contains objects"
        obj = self.data['obj']
        x = np.array([(obj, 1.), (obj, 2.)],
                      dtype=[('A', object), ('B', float)])
        y = np.array([10, 20], dtype=int)
        test = append_fields(x, 'C', data=y, usemask=False)
        control = np.array([(obj, 1.0, 10), (obj, 2.0, 20)],
                           dtype=[('A', object), ('B', float), ('C', int)])
        assert_equal(test, control)
