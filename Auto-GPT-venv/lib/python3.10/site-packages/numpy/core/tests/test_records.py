import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_array_almost_equal,
    assert_raises, temppath,
    )
from numpy.compat import pickle


class TestFromrecords:
    def test_fromrecords(self):
        r = np.rec.fromrecords([[456, 'dbe', 1.2], [2, 'de', 1.3]],
                            names='col1,col2,col3')
        assert_equal(r[0].item(), (456, 'dbe', 1.2))
        assert_equal(r['col1'].dtype.kind, 'i')
        assert_equal(r['col2'].dtype.kind, 'U')
        assert_equal(r['col2'].dtype.itemsize, 12)
        assert_equal(r['col3'].dtype.kind, 'f')

    def test_fromrecords_0len(self):
        """ Verify fromrecords works with a 0-length input """
        dtype = [('a', float), ('b', float)]
        r = np.rec.fromrecords([], dtype=dtype)
        assert_equal(r.shape, (0,))

    def test_fromrecords_2d(self):
        data = [
            [(1, 2), (3, 4), (5, 6)],
            [(6, 5), (4, 3), (2, 1)]
        ]
        expected_a = [[1, 3, 5], [6, 4, 2]]
        expected_b = [[2, 4, 6], [5, 3, 1]]

        # try with dtype
        r1 = np.rec.fromrecords(data, dtype=[('a', int), ('b', int)])
        assert_equal(r1['a'], expected_a)
        assert_equal(r1['b'], expected_b)

        # try with names
        r2 = np.rec.fromrecords(data, names=['a', 'b'])
        assert_equal(r2['a'], expected_a)
        assert_equal(r2['b'], expected_b)

        assert_equal(r1, r2)

    def test_method_array(self):
        r = np.rec.array(b'abcdefg' * 100, formats='i2,a3,i4', shape=3, byteorder='big')
        assert_equal(r[1].item(), (25444, b'efg', 1633837924))

    def test_method_array2(self):
        r = np.rec.array([(1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'), (5, 55, 'ex'),
                     (6, 66, 'f'), (7, 77, 'g')], formats='u1,f4,a1')
        assert_equal(r[1].item(), (2, 22.0, b'b'))

    def test_recarray_slices(self):
        r = np.rec.array([(1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'), (5, 55, 'ex'),
                     (6, 66, 'f'), (7, 77, 'g')], formats='u1,f4,a1')
        assert_equal(r[1::2][1].item(), (4, 44.0, b'd'))

    def test_recarray_fromarrays(self):
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array(['a', 'dd', 'xyz', '12'])
        x3 = np.array([1.1, 2, 3, 4])
        r = np.rec.fromarrays([x1, x2, x3], names='a,b,c')
        assert_equal(r[1].item(), (2, 'dd', 2.0))
        x1[1] = 34
        assert_equal(r.a, np.array([1, 2, 3, 4]))

    def test_recarray_fromfile(self):
        data_dir = path.join(path.dirname(__file__), 'data')
        filename = path.join(data_dir, 'recarray_from_file.fits')
        fd = open(filename, 'rb')
        fd.seek(2880 * 2)
        r1 = np.rec.fromfile(fd, formats='f8,i4,a5', shape=3, byteorder='big')
        fd.seek(2880 * 2)
        r2 = np.rec.array(fd, formats='f8,i4,a5', shape=3, byteorder='big')
        fd.seek(2880 * 2)
        bytes_array = BytesIO()
        bytes_array.write(fd.read())
        bytes_array.seek(0)
        r3 = np.rec.fromfile(bytes_array, formats='f8,i4,a5', shape=3, byteorder='big')
        fd.close()
        assert_equal(r1, r2)
        assert_equal(r2, r3)

    def test_recarray_from_obj(self):
        count = 10
        a = np.zeros(count, dtype='O')
        b = np.zeros(count, dtype='f8')
        c = np.zeros(count, dtype='f8')
        for i in range(len(a)):
            a[i] = list(range(1, 10))

        mine = np.rec.fromarrays([a, b, c], names='date,data1,data2')
        for i in range(len(a)):
            assert_((mine.date[i] == list(range(1, 10))))
            assert_((mine.data1[i] == 0.0))
            assert_((mine.data2[i] == 0.0))

    def test_recarray_repr(self):
        a = np.array([(1, 0.1), (2, 0.2)],
                     dtype=[('foo', '<i4'), ('bar', '<f8')])
        a = np.rec.array(a)
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            rec.array([(1, 0.1), (2, 0.2)],
                      dtype=[('foo', '<i4'), ('bar', '<f8')])""")
        )

        # make sure non-structured dtypes also show up as rec.array
        a = np.array(np.ones(4, dtype='f8'))
        assert_(repr(np.rec.array(a)).startswith('rec.array'))

        # check that the 'np.record' part of the dtype isn't shown
        a = np.rec.array(np.ones(3, dtype='i4,i4'))
        assert_equal(repr(a).find('numpy.record'), -1)
        a = np.rec.array(np.ones(3, dtype='i4'))
        assert_(repr(a).find('dtype=int32') != -1)

    def test_0d_recarray_repr(self):
        arr_0d = np.rec.array((1, 2.0, '2003'), dtype='<i4,<f8,<M8[Y]')
        assert_equal(repr(arr_0d), textwrap.dedent("""\
            rec.array((1, 2., '2003'),
                      dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])"""))

        record = arr_0d[()]
        assert_equal(repr(record), "(1, 2., '2003')")
        # 1.13 converted to python scalars before the repr
        try:
            np.set_printoptions(legacy='1.13')
            assert_equal(repr(record), '(1, 2.0, datetime.date(2003, 1, 1))')
        finally:
            np.set_printoptions(legacy=False)

    def test_recarray_from_repr(self):
        a = np.array([(1,'ABC'), (2, "DEF")],
                     dtype=[('foo', int), ('bar', 'S4')])
        recordarr = np.rec.array(a)
        recarr = a.view(np.recarray)
        recordview = a.view(np.dtype((np.record, a.dtype)))

        recordarr_r = eval("numpy." + repr(recordarr), {'numpy': np})
        recarr_r = eval("numpy." + repr(recarr), {'numpy': np})
        recordview_r = eval("numpy." + repr(recordview), {'numpy': np})

        assert_equal(type(recordarr_r), np.recarray)
        assert_equal(recordarr_r.dtype.type, np.record)
        assert_equal(recordarr, recordarr_r)

        assert_equal(type(recarr_r), np.recarray)
        assert_equal(recarr_r.dtype.type, np.record)
        assert_equal(recarr, recarr_r)

        assert_equal(type(recordview_r), np.ndarray)
        assert_equal(recordview.dtype.type, np.record)
        assert_equal(recordview, recordview_r)

    def test_recarray_views(self):
        a = np.array([(1,'ABC'), (2, "DEF")],
                     dtype=[('foo', int), ('bar', 'S4')])
        b = np.array([1,2,3,4,5], dtype=np.int64)

        #check that np.rec.array gives right dtypes
        assert_equal(np.rec.array(a).dtype.type, np.record)
        assert_equal(type(np.rec.array(a)), np.recarray)
        assert_equal(np.rec.array(b).dtype.type, np.int64)
        assert_equal(type(np.rec.array(b)), np.recarray)

        #check that viewing as recarray does the same
        assert_equal(a.view(np.recarray).dtype.type, np.record)
        assert_equal(type(a.view(np.recarray)), np.recarray)
        assert_equal(b.view(np.recarray).dtype.type, np.int64)
        assert_equal(type(b.view(np.recarray)), np.recarray)

        #check that view to non-structured dtype preserves type=np.recarray
        r = np.rec.array(np.ones(4, dtype="f4,i4"))
        rv = r.view('f8').view('f4,i4')
        assert_equal(type(rv), np.recarray)
        assert_equal(rv.dtype.type, np.record)

        #check that getitem also preserves np.recarray and np.record
        r = np.rec.array(np.ones(4, dtype=[('a', 'i4'), ('b', 'i4'),
                                           ('c', 'i4,i4')]))
        assert_equal(r['c'].dtype.type, np.record)
        assert_equal(type(r['c']), np.recarray)

        #and that it preserves subclasses (gh-6949)
        class C(np.recarray):
            pass

        c = r.view(C)
        assert_equal(type(c['c']), C)

        # check that accessing nested structures keep record type, but
        # not for subarrays, non-void structures, non-structured voids
        test_dtype = [('a', 'f4,f4'), ('b', 'V8'), ('c', ('f4',2)),
                      ('d', ('i8', 'i4,i4'))]
        r = np.rec.array([((1,1), b'11111111', [1,1], 1),
                          ((1,1), b'11111111', [1,1], 1)], dtype=test_dtype)
        assert_equal(r.a.dtype.type, np.record)
        assert_equal(r.b.dtype.type, np.void)
        assert_equal(r.c.dtype.type, np.float32)
        assert_equal(r.d.dtype.type, np.int64)
        # check the same, but for views
        r = np.rec.array(np.ones(4, dtype='i4,i4'))
        assert_equal(r.view('f4,f4').dtype.type, np.record)
        assert_equal(r.view(('i4',2)).dtype.type, np.int32)
        assert_equal(r.view('V8').dtype.type, np.void)
        assert_equal(r.view(('i8', 'i4,i4')).dtype.type, np.int64)

        #check that we can undo the view
        arrs = [np.ones(4, dtype='f4,i4'), np.ones(4, dtype='f8')]
        for arr in arrs:
            rec = np.rec.array(arr)
            # recommended way to view as an ndarray:
            arr2 = rec.view(rec.dtype.fields or rec.dtype, np.ndarray)
            assert_equal(arr2.dtype.type, arr.dtype.type)
            assert_equal(type(arr2), type(arr))

    def test_recarray_from_names(self):
        ra = np.rec.array([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
                       names='c1, c2, c3, c4')
        pa = np.rec.fromrecords([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
                       names='c1, c2, c3, c4')
        assert_(ra.dtype == pa.dtype)
        assert_(ra.shape == pa.shape)
        for k in range(len(ra)):
            assert_(ra[k].item() == pa[k].item())

    def test_recarray_conflict_fields(self):
        ra = np.rec.array([(1, 'abc', 2.3), (2, 'xyz', 4.2),
                        (3, 'wrs', 1.3)],
                       names='field, shape, mean')
        ra.mean = [1.1, 2.2, 3.3]
        assert_array_almost_equal(ra['mean'], [1.1, 2.2, 3.3])
        assert_(type(ra.mean) is type(ra.var))
        ra.shape = (1, 3)
        assert_(ra.shape == (1, 3))
        ra.shape = ['A', 'B', 'C']
        assert_array_equal(ra['shape'], [['A', 'B', 'C']])
        ra.field = 5
        assert_array_equal(ra['field'], [[5, 5, 5]])
        assert_(isinstance(ra.field, collections.abc.Callable))

    def test_fromrecords_with_explicit_dtype(self):
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')],
                                dtype=[('a', int), ('b', object)])
        assert_equal(a.a, [1, 2])
        assert_equal(a[0].a, 1)
        assert_equal(a.b, ['a', 'bbb'])
        assert_equal(a[-1].b, 'bbb')
        #
        ndtype = np.dtype([('a', int), ('b', object)])
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')], dtype=ndtype)
        assert_equal(a.a, [1, 2])
        assert_equal(a[0].a, 1)
        assert_equal(a.b, ['a', 'bbb'])
        assert_equal(a[-1].b, 'bbb')

    def test_recarray_stringtypes(self):
        # Issue #3993
        a = np.array([('abc ', 1), ('abc', 2)],
                     dtype=[('foo', 'S4'), ('bar', int)])
        a = a.view(np.recarray)
        assert_equal(a.foo[0] == a.foo[1], False)

    def test_recarray_returntypes(self):
        qux_fields = {'C': (np.dtype('S5'), 0), 'D': (np.dtype('S5'), 6)}
        a = np.rec.array([('abc ', (1,1), 1, ('abcde', 'fgehi')),
                          ('abc', (2,3), 1, ('abcde', 'jklmn'))],
                         dtype=[('foo', 'S4'),
                                ('bar', [('A', int), ('B', int)]),
                                ('baz', int), ('qux', qux_fields)])
        assert_equal(type(a.foo), np.ndarray)
        assert_equal(type(a['foo']), np.ndarray)
        assert_equal(type(a.bar), np.recarray)
        assert_equal(type(a['bar']), np.recarray)
        assert_equal(a.bar.dtype.type, np.record)
        assert_equal(type(a['qux']), np.recarray)
        assert_equal(a.qux.dtype.type, np.record)
        assert_equal(dict(a.qux.dtype.fields), qux_fields)
        assert_equal(type(a.baz), np.ndarray)
        assert_equal(type(a['baz']), np.ndarray)
        assert_equal(type(a[0].bar), np.record)
        assert_equal(type(a[0]['bar']), np.record)
        assert_equal(a[0].bar.A, 1)
        assert_equal(a[0].bar['A'], 1)
        assert_equal(a[0]['bar'].A, 1)
        assert_equal(a[0]['bar']['A'], 1)
        assert_equal(a[0].qux.D, b'fgehi')
        assert_equal(a[0].qux['D'], b'fgehi')
        assert_equal(a[0]['qux'].D, b'fgehi')
        assert_equal(a[0]['qux']['D'], b'fgehi')

    def test_zero_width_strings(self):
        # Test for #6430, based on the test case from #1901

        cols = [['test'] * 3, [''] * 3]
        rec = np.rec.fromarrays(cols)
        assert_equal(rec['f0'], ['test', 'test', 'test'])
        assert_equal(rec['f1'], ['', '', ''])

        dt = np.dtype([('f0', '|S4'), ('f1', '|S')])
        rec = np.rec.fromarrays(cols, dtype=dt)
        assert_equal(rec.itemsize, 4)
        assert_equal(rec['f0'], [b'test', b'test', b'test'])
        assert_equal(rec['f1'], [b'', b'', b''])


class TestPathUsage:
    # Test that pathlib.Path can be used
    def test_tofile_fromfile(self):
        with temppath(suffix='.bin') as path:
            path = Path(path)
            np.random.seed(123)
            a = np.random.rand(10).astype('f8,i4,a5')
            a[5] = (0.5,10,'abcde')
            with path.open("wb") as fd:
                a.tofile(fd)
            x = np.core.records.fromfile(path,
                                         formats='f8,i4,a5',
                                         shape=10)
            assert_array_equal(x, a)


class TestRecord:
    def setup_method(self):
        self.data = np.rec.fromrecords([(1, 2, 3), (4, 5, 6)],
                            dtype=[("col1", "<i4"),
                                   ("col2", "<i4"),
                                   ("col3", "<i4")])

    def test_assignment1(self):
        a = self.data
        assert_equal(a.col1[0], 1)
        a[0].col1 = 0
        assert_equal(a.col1[0], 0)

    def test_assignment2(self):
        a = self.data
        assert_equal(a.col1[0], 1)
        a.col1[0] = 0
        assert_equal(a.col1[0], 0)

    def test_invalid_assignment(self):
        a = self.data

        def assign_invalid_column(x):
            x[0].col5 = 1

        assert_raises(AttributeError, assign_invalid_column, a)

    def test_nonwriteable_setfield(self):
        # gh-8171
        r = np.rec.array([(0,), (1,)], dtype=[('f', 'i4')])
        r.flags.writeable = False
        with assert_raises(ValueError):
            r.f = [2, 3]
        with assert_raises(ValueError):
            r.setfield([2,3], *r.dtype.fields['f'])

    def test_out_of_order_fields(self):
        # names in the same order, padding added to descr
        x = self.data[['col1', 'col2']]
        assert_equal(x.dtype.names, ('col1', 'col2'))
        assert_equal(x.dtype.descr,
                     [('col1', '<i4'), ('col2', '<i4'), ('', '|V4')])

        # names change order to match indexing, as of 1.14 - descr can't
        # represent that
        y = self.data[['col2', 'col1']]
        assert_equal(y.dtype.names, ('col2', 'col1'))
        assert_raises(ValueError, lambda: y.dtype.descr)

    def test_pickle_1(self):
        # Issue #1529
        a = np.array([(1, [])], dtype=[('a', np.int32), ('b', np.int32, 0)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
            assert_equal(a[0], pickle.loads(pickle.dumps(a[0],
                                                         protocol=proto)))

    def test_pickle_2(self):
        a = self.data
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
            assert_equal(a[0], pickle.loads(pickle.dumps(a[0],
                                                         protocol=proto)))

    def test_pickle_3(self):
        # Issue #7140
        a = self.data
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            pa = pickle.loads(pickle.dumps(a[0], protocol=proto))
            assert_(pa.flags.c_contiguous)
            assert_(pa.flags.f_contiguous)
            assert_(pa.flags.writeable)
            assert_(pa.flags.aligned)

    def test_pickle_void(self):
        # issue gh-13593
        dt = np.dtype([('obj', 'O'), ('int', 'i')])
        a = np.empty(1, dtype=dt)
        data = (bytearray(b'eman'),)
        a['obj'] = data
        a['int'] = 42
        ctor, args = a[0].__reduce__()
        # check the constructor is what we expect before interpreting the arguments
        assert ctor is np.core.multiarray.scalar
        dtype, obj = args
        # make sure we did not pickle the address
        assert not isinstance(obj, bytes)

        assert_raises(RuntimeError, ctor, dtype, 13)

        # Test roundtrip:
        dump = pickle.dumps(a[0])
        unpickled = pickle.loads(dump)
        assert a[0] == unpickled

        # Also check the similar (impossible) "object scalar" path:
        with pytest.warns(DeprecationWarning):
            assert ctor(np.dtype("O"), data) is data

    def test_objview_record(self):
        # https://github.com/numpy/numpy/issues/2599
        dt = np.dtype([('foo', 'i8'), ('bar', 'O')])
        r = np.zeros((1,3), dtype=dt).view(np.recarray)
        r.foo = np.array([1, 2, 3])  # TypeError?

        # https://github.com/numpy/numpy/issues/3256
        ra = np.recarray((2,), dtype=[('x', object), ('y', float), ('z', int)])
        ra[['x','y']]  # TypeError?

    def test_record_scalar_setitem(self):
        # https://github.com/numpy/numpy/issues/3561
        rec = np.recarray(1, dtype=[('x', float, 5)])
        rec[0].x = 1
        assert_equal(rec[0].x, np.ones(5))

    def test_missing_field(self):
        # https://github.com/numpy/numpy/issues/4806
        arr = np.zeros((3,), dtype=[('x', int), ('y', int)])
        assert_raises(KeyError, lambda: arr[['nofield']])

    def test_fromarrays_nested_structured_arrays(self):
        arrays = [
            np.arange(10),
            np.ones(10, dtype=[('a', '<u2'), ('b', '<f4')]),
        ]
        arr = np.rec.fromarrays(arrays)  # ValueError?

    @pytest.mark.parametrize('nfields', [0, 1, 2])
    def test_assign_dtype_attribute(self, nfields):
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
        data = np.zeros(3, dt).view(np.recarray)

        # the original and resulting dtypes differ on whether they are records
        assert data.dtype.type == np.record
        assert dt.type != np.record

        # ensure that the dtype remains a record even when assigned
        data.dtype = dt
        assert data.dtype.type == np.record

    @pytest.mark.parametrize('nfields', [0, 1, 2])
    def test_nested_fields_are_records(self, nfields):
        """ Test that nested structured types are treated as records too """
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
        dt_outer = np.dtype([('inner', dt)])

        data = np.zeros(3, dt_outer).view(np.recarray)
        assert isinstance(data, np.recarray)
        assert isinstance(data['inner'], np.recarray)

        data0 = data[0]
        assert isinstance(data0, np.record)
        assert isinstance(data0['inner'], np.record)

    def test_nested_dtype_padding(self):
        """ test that trailing padding is preserved """
        # construct a dtype with padding at the end
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)])
        dt_padded_end = dt[['a', 'b']]
        assert dt_padded_end.itemsize == dt.itemsize

        dt_outer = np.dtype([('inner', dt_padded_end)])

        data = np.zeros(3, dt_outer).view(np.recarray)
        assert_equal(data['inner'].dtype, dt_padded_end)

        data0 = data[0]
        assert_equal(data0['inner'].dtype, dt_padded_end)


def test_find_duplicate():
    l1 = [1, 2, 3, 4, 5, 6]
    assert_(np.rec.find_duplicate(l1) == [])

    l2 = [1, 2, 1, 4, 5, 6]
    assert_(np.rec.find_duplicate(l2) == [1])

    l3 = [1, 2, 1, 4, 1, 6, 2, 3]
    assert_(np.rec.find_duplicate(l3) == [1, 2])

    l3 = [2, 2, 1, 4, 1, 6, 2, 3]
    assert_(np.rec.find_duplicate(l3) == [2, 1])
