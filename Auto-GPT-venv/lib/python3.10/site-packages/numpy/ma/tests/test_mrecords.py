# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for mrecords.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
    fromrecords as recfromrecords, fromarrays as recfromarrays
    )
from numpy.ma.mrecords import (
    MaskedRecords, mrecarray, fromarrays, fromtextfile, fromrecords,
    addfield
    )
from numpy.ma.testutils import (
    assert_, assert_equal,
    assert_equal_records,
    )
from numpy.compat import pickle


class TestMRecords:

    ilist = [1, 2, 3, 4, 5]
    flist = [1.1, 2.2, 3.3, 4.4, 5.5]
    slist = [b'one', b'two', b'three', b'four', b'five']
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mask = [0, 1, 0, 0, 1]
    base = ma.array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)

    def test_byview(self):
        # Test creation by view
        base = self.base
        mbase = base.view(mrecarray)
        assert_equal(mbase.recordmask, base.recordmask)
        assert_equal_records(mbase._mask, base._mask)
        assert_(isinstance(mbase._data, recarray))
        assert_equal_records(mbase._data, base._data.view(recarray))
        for field in ('a', 'b', 'c'):
            assert_equal(base[field], mbase[field])
        assert_equal_records(mbase.view(mrecarray), mbase)

    def test_get(self):
        # Tests fields retrieval
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # As fields..........
        for field in ('a', 'b', 'c'):
            assert_equal(getattr(mbase, field), mbase[field])
            assert_equal(base[field], mbase[field])
        # as elements .......
        mbase_first = mbase[0]
        assert_(isinstance(mbase_first, mrecarray))
        assert_equal(mbase_first.dtype, mbase.dtype)
        assert_equal(mbase_first.tolist(), (1, 1.1, b'one'))
        # Used to be mask, now it's recordmask
        assert_equal(mbase_first.recordmask, nomask)
        assert_equal(mbase_first._mask.item(), (False, False, False))
        assert_equal(mbase_first['a'], mbase['a'][0])
        mbase_last = mbase[-1]
        assert_(isinstance(mbase_last, mrecarray))
        assert_equal(mbase_last.dtype, mbase.dtype)
        assert_equal(mbase_last.tolist(), (None, None, None))
        # Used to be mask, now it's recordmask
        assert_equal(mbase_last.recordmask, True)
        assert_equal(mbase_last._mask.item(), (True, True, True))
        assert_equal(mbase_last['a'], mbase['a'][-1])
        assert_((mbase_last['a'] is masked))
        # as slice ..........
        mbase_sl = mbase[:2]
        assert_(isinstance(mbase_sl, mrecarray))
        assert_equal(mbase_sl.dtype, mbase.dtype)
        # Used to be mask, now it's recordmask
        assert_equal(mbase_sl.recordmask, [0, 1])
        assert_equal_records(mbase_sl.mask,
                             np.array([(False, False, False),
                                       (True, True, True)],
                                      dtype=mbase._mask.dtype))
        assert_equal_records(mbase_sl, base[:2].view(mrecarray))
        for field in ('a', 'b', 'c'):
            assert_equal(getattr(mbase_sl, field), base[:2][field])

    def test_set_fields(self):
        # Tests setting fields.
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase = mbase.copy()
        mbase.fill_value = (999999, 1e20, 'N/A')
        # Change the data, the mask should be conserved
        mbase.a._data[:] = 5
        assert_equal(mbase['a']._data, [5, 5, 5, 5, 5])
        assert_equal(mbase['a']._mask, [0, 1, 0, 0, 1])
        # Change the elements, and the mask will follow
        mbase.a = 1
        assert_equal(mbase['a']._data, [1]*5)
        assert_equal(ma.getmaskarray(mbase['a']), [0]*5)
        # Use to be _mask, now it's recordmask
        assert_equal(mbase.recordmask, [False]*5)
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 0),
                               (0, 1, 1),
                               (0, 0, 0),
                               (0, 0, 0),
                               (0, 1, 1)],
                              dtype=bool))
        # Set a field to mask ........................
        mbase.c = masked
        # Use to be mask, and now it's still mask !
        assert_equal(mbase.c.mask, [1]*5)
        assert_equal(mbase.c.recordmask, [1]*5)
        assert_equal(ma.getmaskarray(mbase['c']), [1]*5)
        assert_equal(ma.getdata(mbase['c']), [b'N/A']*5)
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 1),
                               (0, 1, 1),
                               (0, 0, 1),
                               (0, 0, 1),
                               (0, 1, 1)],
                              dtype=bool))
        # Set fields by slices .......................
        mbase = base.view(mrecarray).copy()
        mbase.a[3:] = 5
        assert_equal(mbase.a, [1, 2, 3, 5, 5])
        assert_equal(mbase.a._mask, [0, 1, 0, 0, 0])
        mbase.b[3:] = masked
        assert_equal(mbase.b, base['b'])
        assert_equal(mbase.b._mask, [0, 1, 0, 1, 1])
        # Set fields globally..........................
        ndtype = [('alpha', '|S1'), ('num', int)]
        data = ma.array([('a', 1), ('b', 2), ('c', 3)], dtype=ndtype)
        rdata = data.view(MaskedRecords)
        val = ma.array([10, 20, 30], mask=[1, 0, 0])

        rdata['num'] = val
        assert_equal(rdata.num, val)
        assert_equal(rdata.num.mask, [1, 0, 0])

    def test_set_fields_mask(self):
        # Tests setting the mask of a field.
        base = self.base.copy()
        # This one has already a mask....
        mbase = base.view(mrecarray)
        mbase['a'][-2] = masked
        assert_equal(mbase.a, [1, 2, 3, 4, 5])
        assert_equal(mbase.a._mask, [0, 1, 0, 1, 1])
        # This one has not yet
        mbase = fromarrays([np.arange(5), np.random.rand(5)],
                           dtype=[('a', int), ('b', float)])
        mbase['a'][-2] = masked
        assert_equal(mbase.a, [0, 1, 2, 3, 4])
        assert_equal(mbase.a._mask, [0, 0, 0, 1, 0])

    def test_set_mask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Set the mask to True .......................
        mbase.mask = masked
        assert_equal(ma.getmaskarray(mbase['b']), [1]*5)
        assert_equal(mbase['a']._mask, mbase['b']._mask)
        assert_equal(mbase['a']._mask, mbase['c']._mask)
        assert_equal(mbase._mask.tolist(),
                     np.array([(1, 1, 1)]*5, dtype=bool))
        # Delete the mask ............................
        mbase.mask = nomask
        assert_equal(ma.getmaskarray(mbase['c']), [0]*5)
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 0)]*5, dtype=bool))

    def test_set_mask_fromarray(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Sets the mask w/ an array
        mbase.mask = [1, 0, 0, 0, 1]
        assert_equal(mbase.a.mask, [1, 0, 0, 0, 1])
        assert_equal(mbase.b.mask, [1, 0, 0, 0, 1])
        assert_equal(mbase.c.mask, [1, 0, 0, 0, 1])
        # Yay, once more !
        mbase.mask = [0, 0, 0, 0, 1]
        assert_equal(mbase.a.mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.b.mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.c.mask, [0, 0, 0, 0, 1])

    def test_set_mask_fromfields(self):
        mbase = self.base.copy().view(mrecarray)

        nmask = np.array(
            [(0, 1, 0), (0, 1, 0), (1, 0, 1), (1, 0, 1), (0, 0, 0)],
            dtype=[('a', bool), ('b', bool), ('c', bool)])
        mbase.mask = nmask
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])
        # Reinitialize and redo
        mbase.mask = False
        mbase.fieldmask = nmask
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])

    def test_set_elements(self):
        base = self.base.copy()
        # Set an element to mask .....................
        mbase = base.view(mrecarray).copy()
        mbase[-2] = masked
        assert_equal(
            mbase._mask.tolist(),
            np.array([(0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1), (1, 1, 1)],
                     dtype=bool))
        # Used to be mask, now it's recordmask!
        assert_equal(mbase.recordmask, [0, 1, 0, 1, 1])
        # Set slices .................................
        mbase = base.view(mrecarray).copy()
        mbase[:2] = (5, 5, 5)
        assert_equal(mbase.a._data, [5, 5, 3, 4, 5])
        assert_equal(mbase.a._mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.b._data, [5., 5., 3.3, 4.4, 5.5])
        assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.c._data,
                     [b'5', b'5', b'three', b'four', b'five'])
        assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])

        mbase = base.view(mrecarray).copy()
        mbase[:2] = masked
        assert_equal(mbase.a._data, [1, 2, 3, 4, 5])
        assert_equal(mbase.a._mask, [1, 1, 0, 0, 1])
        assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 4.4, 5.5])
        assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])
        assert_equal(mbase.c._data,
                     [b'one', b'two', b'three', b'four', b'five'])
        assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])

    def test_setslices_hardmask(self):
        # Tests setting slices w/ hardmask.
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        try:
            mbase[-2:] = (5, 5, 5)
            assert_equal(mbase.a._data, [1, 2, 3, 5, 5])
            assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 5, 5.5])
            assert_equal(mbase.c._data,
                         [b'one', b'two', b'three', b'5', b'five'])
            assert_equal(mbase.a._mask, [0, 1, 0, 0, 1])
            assert_equal(mbase.b._mask, mbase.a._mask)
            assert_equal(mbase.b._mask, mbase.c._mask)
        except NotImplementedError:
            # OK, not implemented yet...
            pass
        except AssertionError:
            raise
        else:
            raise Exception("Flexible hard masks should be supported !")
        # Not using a tuple should crash
        try:
            mbase[-2:] = 3
        except (NotImplementedError, TypeError):
            pass
        else:
            raise TypeError("Should have expected a readable buffer object!")

    def test_hardmask(self):
        # Test hardmask
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        assert_(mbase._hardmask)
        mbase.mask = nomask
        assert_equal_records(mbase._mask, base._mask)
        mbase.soften_mask()
        assert_(not mbase._hardmask)
        mbase.mask = nomask
        # So, the mask of a field is no longer set to nomask...
        assert_equal_records(mbase._mask,
                             ma.make_mask_none(base.shape, base.dtype))
        assert_(ma.make_mask(mbase['b']._mask) is nomask)
        assert_equal(mbase['a']._mask, mbase['b']._mask)

    def test_pickling(self):
        # Test pickling
        base = self.base.copy()
        mrec = base.view(mrecarray)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            _ = pickle.dumps(mrec, protocol=proto)
            mrec_ = pickle.loads(_)
            assert_equal(mrec_.dtype, mrec.dtype)
            assert_equal_records(mrec_._data, mrec._data)
            assert_equal(mrec_._mask, mrec._mask)
            assert_equal_records(mrec_._mask, mrec._mask)

    def test_filled(self):
        # Test filling the array
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                          fill_value=(99999, 99999., 'N/A'))
        mrecfilled = mrec.filled()
        assert_equal(mrecfilled['a'], np.array((1, 2, 99999), dtype=int))
        assert_equal(mrecfilled['b'], np.array((1.1, 2.2, 99999.),
                                               dtype=float))
        assert_equal(mrecfilled['c'], np.array(('one', 'two', 'N/A'),
                                               dtype='|S8'))

    def test_tolist(self):
        # Test tolist.
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[1, 0, 0], dtype='|S8')
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                          fill_value=(99999, 99999., 'N/A'))

        assert_equal(mrec.tolist(),
                     [(1, 1.1, None), (2, 2.2, b'two'),
                      (None, None, b'three')])

    def test_withnames(self):
        # Test the creation w/ format and names
        x = mrecarray(1, formats=float, names='base')
        x[0]['base'] = 10
        assert_equal(x['base'][0], 10)

    def test_exotic_formats(self):
        # Test that 'exotic' formats are processed properly
        easy = mrecarray(1, dtype=[('i', int), ('s', '|S8'), ('f', float)])
        easy[0] = masked
        assert_equal(easy.filled(1).item(), (1, b'1', 1.))

        solo = mrecarray(1, dtype=[('f0', '<f8', (2, 2))])
        solo[0] = masked
        assert_equal(solo.filled(1).item(),
                     np.array((1,), dtype=solo.dtype).item())

        mult = mrecarray(2, dtype="i4, (2,3)float, float")
        mult[0] = masked
        mult[1] = (1, 1, 1)
        mult.filled(0)
        assert_equal_records(mult.filled(0),
                             np.array([(0, 0, 0), (1, 1, 1)],
                                      dtype=mult.dtype))


class TestView:

    def setup_method(self):
        (a, b) = (np.arange(10), np.random.rand(10))
        ndtype = [('a', float), ('b', float)]
        arr = np.array(list(zip(a, b)), dtype=ndtype)

        mrec = fromarrays([a, b], dtype=ndtype, fill_value=(-9., -99.))
        mrec.mask[3] = (False, True)
        self.data = (mrec, a, b, arr)

    def test_view_by_itself(self):
        (mrec, a, b, arr) = self.data
        test = mrec.view()
        assert_(isinstance(test, MaskedRecords))
        assert_equal_records(test, mrec)
        assert_equal_records(test._mask, mrec._mask)

    def test_view_simple_dtype(self):
        (mrec, a, b, arr) = self.data
        ntype = (float, 2)
        test = mrec.view(ntype)
        assert_(isinstance(test, ma.MaskedArray))
        assert_equal(test, np.array(list(zip(a, b)), dtype=float))
        assert_(test[3, 1] is ma.masked)

    def test_view_flexible_type(self):
        (mrec, a, b, arr) = self.data
        alttype = [('A', float), ('B', float)]
        test = mrec.view(alttype)
        assert_(isinstance(test, MaskedRecords))
        assert_equal_records(test, arr.view(alttype))
        assert_(test['B'][3] is masked)
        assert_equal(test.dtype, np.dtype(alttype))
        assert_(test._fill_value is None)


##############################################################################
class TestMRecordsImport:

    _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
    _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
    _c = ma.array([b'one', b'two', b'three'],
                  mask=[0, 0, 1], dtype='|S8')
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                      fill_value=(b'99999', b'99999.',
                                  b'N/A'))
    nrec = recfromarrays((_a._data, _b._data, _c._data), dtype=ddtype)
    data = (mrec, nrec, ddtype)

    def test_fromarrays(self):
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        (mrec, nrec, _) = self.data
        for (f, l) in zip(('a', 'b', 'c'), (_a, _b, _c)):
            assert_equal(getattr(mrec, f)._mask, l._mask)
        # One record only
        _x = ma.array([1, 1.1, 'one'], mask=[1, 0, 0], dtype=object)
        assert_equal_records(fromarrays(_x, dtype=mrec.dtype), mrec[0])

    def test_fromrecords(self):
        # Test construction from records.
        (mrec, nrec, ddtype) = self.data
        #......
        palist = [(1, 'abc', 3.7000002861022949, 0),
                  (2, 'xy', 6.6999998092651367, 1),
                  (0, ' ', 0.40000000596046448, 0)]
        pa = recfromrecords(palist, names='c1, c2, c3, c4')
        mpa = fromrecords(palist, names='c1, c2, c3, c4')
        assert_equal_records(pa, mpa)
        #.....
        _mrec = fromrecords(nrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        for field in _mrec.dtype.names:
            assert_equal(getattr(_mrec, field), getattr(mrec._data, field))

        _mrec = fromrecords(nrec.tolist(), names='c1,c2,c3')
        assert_equal(_mrec.dtype, [('c1', int), ('c2', float), ('c3', '|S5')])
        for (f, n) in zip(('c1', 'c2', 'c3'), ('a', 'b', 'c')):
            assert_equal(getattr(_mrec, f), getattr(mrec._data, n))

        _mrec = fromrecords(mrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        assert_equal_records(_mrec._data, mrec.filled())
        assert_equal_records(_mrec._mask, mrec._mask)

    def test_fromrecords_wmask(self):
        # Tests construction from records w/ mask.
        (mrec, nrec, ddtype) = self.data

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=[0, 1, 0,])
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), [(0, 0, 0), (1, 1, 1), (0, 0, 0)])

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=True)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), [(1, 1, 1), (1, 1, 1), (1, 1, 1)])

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype,
                            mask=mrec._mask.tolist())
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())

    def test_fromtextfile(self):
        # Tests reading from a text file.
        fcontent = (
"""#
'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'strings',1,1.0,'mixed column',,1
'with embedded "double quotes"',2,2.0,1.0,,1
'strings',3,3.0E5,3,,1
'strings',4,-1e-10,,,1
""")
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(fcontent)
            mrectxt = fromtextfile(path, delimiter=',', varnames='ABCDEFG')
        assert_(isinstance(mrectxt, MaskedRecords))
        assert_equal(mrectxt.F, [1, 1, 1, 1])
        assert_equal(mrectxt.E._mask, [1, 1, 1, 1])
        assert_equal(mrectxt.C, [1, 2, 3.e+5, -1e-10])

    def test_addfield(self):
        # Tests addfield
        (mrec, nrec, ddtype) = self.data
        (d, m) = ([100, 200, 300], [1, 0, 0])
        mrec = addfield(mrec, ma.array(d, mask=m))
        assert_equal(mrec.f3, d)
        assert_equal(mrec.f3._mask, m)


def test_record_array_with_object_field():
    # Trac #1839
    y = ma.masked_array(
        [(1, '2'), (3, '4')],
        mask=[(0, 0), (0, 1)],
        dtype=[('a', int), ('b', object)])
    # getting an item used to fail
    y[1]
