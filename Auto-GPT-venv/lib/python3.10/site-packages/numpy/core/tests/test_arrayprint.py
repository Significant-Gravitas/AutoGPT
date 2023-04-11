import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_warns, HAS_REFCOUNT,
    assert_raises_regex,
    )
import textwrap

class TestArrayRepr:
    def test_nan_inf(self):
        x = np.array([np.nan, np.inf])
        assert_equal(repr(x), 'array([nan, inf])')

    def test_subclass(self):
        class sub(np.ndarray): pass

        # one dimensional
        x1d = np.array([1, 2]).view(sub)
        assert_equal(repr(x1d), 'sub([1, 2])')

        # two dimensional
        x2d = np.array([[1, 2], [3, 4]]).view(sub)
        assert_equal(repr(x2d),
            'sub([[1, 2],\n'
            '     [3, 4]])')

        # two dimensional with flexible dtype
        xstruct = np.ones((2,2), dtype=[('a', '<i4')]).view(sub)
        assert_equal(repr(xstruct),
            "sub([[(1,), (1,)],\n"
            "     [(1,), (1,)]], dtype=[('a', '<i4')])"
        )

    @pytest.mark.xfail(reason="See gh-10544")
    def test_object_subclass(self):
        class sub(np.ndarray):
            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)

        # test that object + subclass is OK:
        x = sub([None, None])
        assert_equal(repr(x), 'sub([None, None], dtype=object)')
        assert_equal(str(x), '[None None]')

        x = sub([None, sub([None, None])])
        assert_equal(repr(x),
            'sub([None, sub([None, None], dtype=object)], dtype=object)')
        assert_equal(str(x), '[None sub([None, None], dtype=object)]')

    def test_0d_object_subclass(self):
        # make sure that subclasses which return 0ds instead
        # of scalars don't cause infinite recursion in str
        class sub(np.ndarray):
            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)

        x = sub(1)
        assert_equal(repr(x), 'sub(1)')
        assert_equal(str(x), '1')

        x = sub([1, 1])
        assert_equal(repr(x), 'sub([1, 1])')
        assert_equal(str(x), '[1 1]')

        # check it works properly with object arrays too
        x = sub(None)
        assert_equal(repr(x), 'sub(None, dtype=object)')
        assert_equal(str(x), 'None')

        # plus recursive object arrays (even depth > 1)
        y = sub(None)
        x[()] = y
        y[()] = x
        assert_equal(repr(x),
            'sub(sub(sub(..., dtype=object), dtype=object), dtype=object)')
        assert_equal(str(x), '...')
        x[()] = 0  # resolve circular references for garbage collector

        # nested 0d-subclass-object
        x = sub(None)
        x[()] = sub(None)
        assert_equal(repr(x), 'sub(sub(None, dtype=object), dtype=object)')
        assert_equal(str(x), 'None')

        # gh-10663
        class DuckCounter(np.ndarray):
            def __getitem__(self, item):
                result = super().__getitem__(item)
                if not isinstance(result, DuckCounter):
                    result = result[...].view(DuckCounter)
                return result

            def to_string(self):
                return {0: 'zero', 1: 'one', 2: 'two'}.get(self.item(), 'many')

            def __str__(self):
                if self.shape == ():
                    return self.to_string()
                else:
                    fmt = {'all': lambda x: x.to_string()}
                    return np.array2string(self, formatter=fmt)

        dc = np.arange(5).view(DuckCounter)
        assert_equal(str(dc), "[zero one two many many]")
        assert_equal(str(dc[0]), "zero")

    def test_self_containing(self):
        arr0d = np.array(None)
        arr0d[()] = arr0d
        assert_equal(repr(arr0d),
            'array(array(..., dtype=object), dtype=object)')
        arr0d[()] = 0  # resolve recursion for garbage collector

        arr1d = np.array([None, None])
        arr1d[1] = arr1d
        assert_equal(repr(arr1d),
            'array([None, array(..., dtype=object)], dtype=object)')
        arr1d[1] = 0  # resolve recursion for garbage collector

        first = np.array(None)
        second = np.array(None)
        first[()] = second
        second[()] = first
        assert_equal(repr(first),
            'array(array(array(..., dtype=object), dtype=object), dtype=object)')
        first[()] = 0  # resolve circular references for garbage collector

    def test_containing_list(self):
        # printing square brackets directly would be ambiguuous
        arr1d = np.array([None, None])
        arr1d[0] = [1, 2]
        arr1d[1] = [3]
        assert_equal(repr(arr1d),
            'array([list([1, 2]), list([3])], dtype=object)')

    def test_void_scalar_recursion(self):
        # gh-9345
        repr(np.void(b'test'))  # RecursionError ?

    def test_fieldless_structured(self):
        # gh-10366
        no_fields = np.dtype([])
        arr_no_fields = np.empty(4, dtype=no_fields)
        assert_equal(repr(arr_no_fields), 'array([(), (), (), ()], dtype=[])')


class TestComplexArray:
    def test_str(self):
        rvals = [0, 1, -1, np.inf, -np.inf, np.nan]
        cvals = [complex(rp, ip) for rp in rvals for ip in rvals]
        dtypes = [np.complex64, np.cdouble, np.clongdouble]
        actual = [str(np.array([c], dt)) for c in cvals for dt in dtypes]
        wanted = [
            '[0.+0.j]',    '[0.+0.j]',    '[0.+0.j]',
            '[0.+1.j]',    '[0.+1.j]',    '[0.+1.j]',
            '[0.-1.j]',    '[0.-1.j]',    '[0.-1.j]',
            '[0.+infj]',   '[0.+infj]',   '[0.+infj]',
            '[0.-infj]',   '[0.-infj]',   '[0.-infj]',
            '[0.+nanj]',   '[0.+nanj]',   '[0.+nanj]',
            '[1.+0.j]',    '[1.+0.j]',    '[1.+0.j]',
            '[1.+1.j]',    '[1.+1.j]',    '[1.+1.j]',
            '[1.-1.j]',    '[1.-1.j]',    '[1.-1.j]',
            '[1.+infj]',   '[1.+infj]',   '[1.+infj]',
            '[1.-infj]',   '[1.-infj]',   '[1.-infj]',
            '[1.+nanj]',   '[1.+nanj]',   '[1.+nanj]',
            '[-1.+0.j]',   '[-1.+0.j]',   '[-1.+0.j]',
            '[-1.+1.j]',   '[-1.+1.j]',   '[-1.+1.j]',
            '[-1.-1.j]',   '[-1.-1.j]',   '[-1.-1.j]',
            '[-1.+infj]',  '[-1.+infj]',  '[-1.+infj]',
            '[-1.-infj]',  '[-1.-infj]',  '[-1.-infj]',
            '[-1.+nanj]',  '[-1.+nanj]',  '[-1.+nanj]',
            '[inf+0.j]',   '[inf+0.j]',   '[inf+0.j]',
            '[inf+1.j]',   '[inf+1.j]',   '[inf+1.j]',
            '[inf-1.j]',   '[inf-1.j]',   '[inf-1.j]',
            '[inf+infj]',  '[inf+infj]',  '[inf+infj]',
            '[inf-infj]',  '[inf-infj]',  '[inf-infj]',
            '[inf+nanj]',  '[inf+nanj]',  '[inf+nanj]',
            '[-inf+0.j]',  '[-inf+0.j]',  '[-inf+0.j]',
            '[-inf+1.j]',  '[-inf+1.j]',  '[-inf+1.j]',
            '[-inf-1.j]',  '[-inf-1.j]',  '[-inf-1.j]',
            '[-inf+infj]', '[-inf+infj]', '[-inf+infj]',
            '[-inf-infj]', '[-inf-infj]', '[-inf-infj]',
            '[-inf+nanj]', '[-inf+nanj]', '[-inf+nanj]',
            '[nan+0.j]',   '[nan+0.j]',   '[nan+0.j]',
            '[nan+1.j]',   '[nan+1.j]',   '[nan+1.j]',
            '[nan-1.j]',   '[nan-1.j]',   '[nan-1.j]',
            '[nan+infj]',  '[nan+infj]',  '[nan+infj]',
            '[nan-infj]',  '[nan-infj]',  '[nan-infj]',
            '[nan+nanj]',  '[nan+nanj]',  '[nan+nanj]']

        for res, val in zip(actual, wanted):
            assert_equal(res, val)

class TestArray2String:
    def test_basic(self):
        """Basic test of array2string."""
        a = np.arange(3)
        assert_(np.array2string(a) == '[0 1 2]')
        assert_(np.array2string(a, max_line_width=4, legacy='1.13') == '[0 1\n 2]')
        assert_(np.array2string(a, max_line_width=4) == '[0\n 1\n 2]')

    def test_unexpected_kwarg(self):
        # ensure than an appropriate TypeError
        # is raised when array2string receives
        # an unexpected kwarg

        with assert_raises_regex(TypeError, 'nonsense'):
            np.array2string(np.array([1, 2, 3]),
                            nonsense=None)

    def test_format_function(self):
        """Test custom format function for each element in array."""
        def _format_function(x):
            if np.abs(x) < 1:
                return '.'
            elif np.abs(x) < 2:
                return 'o'
            else:
                return 'O'

        x = np.arange(3)
        x_hex = "[0x0 0x1 0x2]"
        x_oct = "[0o0 0o1 0o2]"
        assert_(np.array2string(x, formatter={'all':_format_function}) ==
                "[. o O]")
        assert_(np.array2string(x, formatter={'int_kind':_format_function}) ==
                "[. o O]")
        assert_(np.array2string(x, formatter={'all':lambda x: "%.4f" % x}) ==
                "[0.0000 1.0000 2.0000]")
        assert_equal(np.array2string(x, formatter={'int':lambda x: hex(x)}),
                x_hex)
        assert_equal(np.array2string(x, formatter={'int':lambda x: oct(x)}),
                x_oct)

        x = np.arange(3.)
        assert_(np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x}) ==
                "[0.00 1.00 2.00]")
        assert_(np.array2string(x, formatter={'float':lambda x: "%.2f" % x}) ==
                "[0.00 1.00 2.00]")

        s = np.array(['abc', 'def'])
        assert_(np.array2string(s, formatter={'numpystr':lambda s: s*2}) ==
                '[abcabc defdef]')


    def test_structure_format(self):
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        assert_equal(np.array2string(x),
                "[('Sarah', [8., 7.]) ('John', [6., 7.])]")

        np.set_printoptions(legacy='1.13')
        try:
            # for issue #5692
            A = np.zeros(shape=10, dtype=[("A", "M8[s]")])
            A[5:].fill(np.datetime64('NaT'))
            assert_equal(
                np.array2string(A),
                textwrap.dedent("""\
                [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
                 ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('NaT',) ('NaT',)
                 ('NaT',) ('NaT',) ('NaT',)]""")
            )
        finally:
            np.set_printoptions(legacy=False)

        # same again, but with non-legacy behavior
        assert_equal(
            np.array2string(A),
            textwrap.dedent("""\
            [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
             ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
             ('1970-01-01T00:00:00',) (                'NaT',)
             (                'NaT',) (                'NaT',)
             (                'NaT',) (                'NaT',)]""")
        )

        # and again, with timedeltas
        A = np.full(10, 123456, dtype=[("A", "m8[s]")])
        A[5:].fill(np.datetime64('NaT'))
        assert_equal(
            np.array2string(A),
            textwrap.dedent("""\
            [(123456,) (123456,) (123456,) (123456,) (123456,) ( 'NaT',) ( 'NaT',)
             ( 'NaT',) ( 'NaT',) ( 'NaT',)]""")
        )

        # See #8160
        struct_int = np.array([([1, -1],), ([123, 1],)], dtype=[('B', 'i4', 2)])
        assert_equal(np.array2string(struct_int),
                "[([  1,  -1],) ([123,   1],)]")
        struct_2dint = np.array([([[0, 1], [2, 3]],), ([[12, 0], [0, 0]],)],
                dtype=[('B', 'i4', (2, 2))])
        assert_equal(np.array2string(struct_2dint),
                "[([[ 0,  1], [ 2,  3]],) ([[12,  0], [ 0,  0]],)]")

        # See #8172
        array_scalar = np.array(
                (1., 2.1234567890123456789, 3.), dtype=('f8,f8,f8'))
        assert_equal(np.array2string(array_scalar), "(1., 2.12345679, 3.)")

    def test_unstructured_void_repr(self):
        a = np.array([27, 91, 50, 75,  7, 65, 10,  8,
                      27, 91, 51, 49,109, 82,101,100], dtype='u1').view('V8')
        assert_equal(repr(a[0]), r"void(b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08')")
        assert_equal(str(a[0]), r"b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08'")
        assert_equal(repr(a),
            r"array([b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08'," "\n"
            r"       b'\x1B\x5B\x33\x31\x6D\x52\x65\x64'], dtype='|V8')")

        assert_equal(eval(repr(a), vars(np)), a)
        assert_equal(eval(repr(a[0]), vars(np)), a[0])

    def test_edgeitems_kwarg(self):
        # previously the global print options would be taken over the kwarg
        arr = np.zeros(3, int)
        assert_equal(
            np.array2string(arr, edgeitems=1, threshold=0),
            "[0 ... 0]"
        )

    def test_summarize_1d(self):
        A = np.arange(1001)
        strA = '[   0    1    2 ...  998  999 1000]'
        assert_equal(str(A), strA)

        reprA = 'array([   0,    1,    2, ...,  998,  999, 1000])'
        assert_equal(repr(A), reprA)

    def test_summarize_2d(self):
        A = np.arange(1002).reshape(2, 501)
        strA = '[[   0    1    2 ...  498  499  500]\n' \
               ' [ 501  502  503 ...  999 1000 1001]]'
        assert_equal(str(A), strA)

        reprA = 'array([[   0,    1,    2, ...,  498,  499,  500],\n' \
                '       [ 501,  502,  503, ...,  999, 1000, 1001]])'
        assert_equal(repr(A), reprA)

    def test_linewidth(self):
        a = np.full(6, 1)

        def make_str(a, width, **kw):
            return np.array2string(a, separator="", max_line_width=width, **kw)

        assert_equal(make_str(a, 8, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 7, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 5, legacy='1.13'), '[1111\n'
                                                    ' 11]')

        assert_equal(make_str(a, 8), '[111111]')
        assert_equal(make_str(a, 7), '[11111\n'
                                     ' 1]')
        assert_equal(make_str(a, 5), '[111\n'
                                     ' 111]')

        b = a[None,None,:]

        assert_equal(make_str(b, 12, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b,  9, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b,  8, legacy='1.13'), '[[[11111\n'
                                                     '   1]]]')

        assert_equal(make_str(b, 12), '[[[111111]]]')
        assert_equal(make_str(b,  9), '[[[111\n'
                                      '   111]]]')
        assert_equal(make_str(b,  8), '[[[11\n'
                                      '   11\n'
                                      '   11]]]')

    def test_wide_element(self):
        a = np.array(['xxxxx'])
        assert_equal(
            np.array2string(a, max_line_width=5),
            "['xxxxx']"
        )
        assert_equal(
            np.array2string(a, max_line_width=5, legacy='1.13'),
            "[ 'xxxxx']"
        )

    def test_multiline_repr(self):
        class MultiLine:
            def __repr__(self):
                return "Line 1\nLine 2"

        a = np.array([[None, MultiLine()], [MultiLine(), None]])

        assert_equal(
            np.array2string(a),
            '[[None Line 1\n'
            '       Line 2]\n'
            ' [Line 1\n'
            '  Line 2 None]]'
        )
        assert_equal(
            np.array2string(a, max_line_width=5),
            '[[None\n'
            '  Line 1\n'
            '  Line 2]\n'
            ' [Line 1\n'
            '  Line 2\n'
            '  None]]'
        )
        assert_equal(
            repr(a),
            'array([[None, Line 1\n'
            '              Line 2],\n'
            '       [Line 1\n'
            '        Line 2, None]], dtype=object)'
        )

        class MultiLineLong:
            def __repr__(self):
                return "Line 1\nLooooooooooongestLine2\nLongerLine 3"

        a = np.array([[None, MultiLineLong()], [MultiLineLong(), None]])
        assert_equal(
            repr(a),
            'array([[None, Line 1\n'
            '              LooooooooooongestLine2\n'
            '              LongerLine 3          ],\n'
            '       [Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          , None]], dtype=object)'
        )
        assert_equal(
            np.array_repr(a, 20),
            'array([[None,\n'
            '        Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          ],\n'
            '       [Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          ,\n'
            '        None]],\n'
            '      dtype=object)'
        )

    def test_nested_array_repr(self):
        a = np.empty((2, 2), dtype=object)
        a[0, 0] = np.eye(2)
        a[0, 1] = np.eye(3)
        a[1, 0] = None
        a[1, 1] = np.ones((3, 1))
        assert_equal(
            repr(a),
            'array([[array([[1., 0.],\n'
            '               [0., 1.]]), array([[1., 0., 0.],\n'
            '                                  [0., 1., 0.],\n'
            '                                  [0., 0., 1.]])],\n'
            '       [None, array([[1.],\n'
            '                     [1.],\n'
            '                     [1.]])]], dtype=object)'
        )

    @given(hynp.from_dtype(np.dtype("U")))
    def test_any_text(self, text):
        # This test checks that, given any value that can be represented in an
        # array of dtype("U") (i.e. unicode string), ...
        a = np.array([text, text, text])
        # casting a list of them to an array does not e.g. truncate the value
        assert_equal(a[0], text)
        # and that np.array2string puts a newline in the expected location
        expected_repr = "[{0!r} {0!r}\n {0!r}]".format(text)
        result = np.array2string(a, max_line_width=len(repr(text)) * 2 + 3)
        assert_equal(result, expected_repr)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount(self):
        # make sure we do not hold references to the array due to a recursive
        # closure (gh-10620)
        gc.disable()
        a = np.arange(2)
        r1 = sys.getrefcount(a)
        np.array2string(a)
        np.array2string(a)
        r2 = sys.getrefcount(a)
        gc.collect()
        gc.enable()
        assert_(r1 == r2)

class TestPrintOptions:
    """Test getting and setting global print options."""

    def setup_method(self):
        self.oldopts = np.get_printoptions()

    def teardown_method(self):
        np.set_printoptions(**self.oldopts)

    def test_basic(self):
        x = np.array([1.5, 0, 1.234567890])
        assert_equal(repr(x), "array([1.5       , 0.        , 1.23456789])")
        np.set_printoptions(precision=4)
        assert_equal(repr(x), "array([1.5   , 0.    , 1.2346])")

    def test_precision_zero(self):
        np.set_printoptions(precision=0)
        for values, string in (
                ([0.], "0."), ([.3], "0."), ([-.3], "-0."), ([.7], "1."),
                ([1.5], "2."), ([-1.5], "-2."), ([-15.34], "-15."),
                ([100.], "100."), ([.2, -1, 122.51], "  0.,  -1., 123."),
                ([0], "0"), ([-12], "-12"), ([complex(.3, -.7)], "0.-1.j")):
            x = np.array(values)
            assert_equal(repr(x), "array([%s])" % string)

    def test_formatter(self):
        x = np.arange(3)
        np.set_printoptions(formatter={'all':lambda x: str(x-1)})
        assert_equal(repr(x), "array([-1, 0, 1])")

    def test_formatter_reset(self):
        x = np.arange(3)
        np.set_printoptions(formatter={'all':lambda x: str(x-1)})
        assert_equal(repr(x), "array([-1, 0, 1])")
        np.set_printoptions(formatter={'int':None})
        assert_equal(repr(x), "array([0, 1, 2])")

        np.set_printoptions(formatter={'all':lambda x: str(x-1)})
        assert_equal(repr(x), "array([-1, 0, 1])")
        np.set_printoptions(formatter={'all':None})
        assert_equal(repr(x), "array([0, 1, 2])")

        np.set_printoptions(formatter={'int':lambda x: str(x-1)})
        assert_equal(repr(x), "array([-1, 0, 1])")
        np.set_printoptions(formatter={'int_kind':None})
        assert_equal(repr(x), "array([0, 1, 2])")

        x = np.arange(3.)
        np.set_printoptions(formatter={'float':lambda x: str(x-1)})
        assert_equal(repr(x), "array([-1.0, 0.0, 1.0])")
        np.set_printoptions(formatter={'float_kind':None})
        assert_equal(repr(x), "array([0., 1., 2.])")

    def test_0d_arrays(self):
        assert_equal(str(np.array('café', '<U4')), 'café')

        assert_equal(repr(np.array('café', '<U4')),
                     "array('café', dtype='<U4')")
        assert_equal(str(np.array('test', np.str_)), 'test')

        a = np.zeros(1, dtype=[('a', '<i4', (3,))])
        assert_equal(str(a[0]), '([0, 0, 0],)')

        assert_equal(repr(np.datetime64('2005-02-25')[...]),
                     "array('2005-02-25', dtype='datetime64[D]')")

        assert_equal(repr(np.timedelta64('10', 'Y')[...]),
                     "array(10, dtype='timedelta64[Y]')")

        # repr of 0d arrays is affected by printoptions
        x = np.array(1)
        np.set_printoptions(formatter={'all':lambda x: "test"})
        assert_equal(repr(x), "array(test)")
        # str is unaffected
        assert_equal(str(x), "1")

        # check `style` arg raises
        assert_warns(DeprecationWarning, np.array2string,
                                         np.array(1.), style=repr)
        # but not in legacy mode
        np.array2string(np.array(1.), style=repr, legacy='1.13')
        # gh-10934 style was broken in legacy mode, check it works
        np.array2string(np.array(1.), legacy='1.13')

    def test_float_spacing(self):
        x = np.array([1., 2., 3.])
        y = np.array([1., 2., -10.])
        z = np.array([100., 2., -1.])
        w = np.array([-100., 2., 1.])

        assert_equal(repr(x), 'array([1., 2., 3.])')
        assert_equal(repr(y), 'array([  1.,   2., -10.])')
        assert_equal(repr(np.array(y[0])), 'array(1.)')
        assert_equal(repr(np.array(y[-1])), 'array(-10.)')
        assert_equal(repr(z), 'array([100.,   2.,  -1.])')
        assert_equal(repr(w), 'array([-100.,    2.,    1.])')

        assert_equal(repr(np.array([np.nan, np.inf])), 'array([nan, inf])')
        assert_equal(repr(np.array([np.nan, -np.inf])), 'array([ nan, -inf])')

        x = np.array([np.inf, 100000, 1.1234])
        y = np.array([np.inf, 100000, -1.1234])
        z = np.array([np.inf, 1.1234, -1e120])
        np.set_printoptions(precision=2)
        assert_equal(repr(x), 'array([     inf, 1.00e+05, 1.12e+00])')
        assert_equal(repr(y), 'array([      inf,  1.00e+05, -1.12e+00])')
        assert_equal(repr(z), 'array([       inf,  1.12e+000, -1.00e+120])')

    def test_bool_spacing(self):
        assert_equal(repr(np.array([True,  True])),
                     'array([ True,  True])')
        assert_equal(repr(np.array([True, False])),
                     'array([ True, False])')
        assert_equal(repr(np.array([True])),
                     'array([ True])')
        assert_equal(repr(np.array(True)),
                     'array(True)')
        assert_equal(repr(np.array(False)),
                     'array(False)')

    def test_sign_spacing(self):
        a = np.arange(4.)
        b = np.array([1.234e9])
        c = np.array([1.0 + 1.0j, 1.123456789 + 1.123456789j], dtype='c16')

        assert_equal(repr(a), 'array([0., 1., 2., 3.])')
        assert_equal(repr(np.array(1.)), 'array(1.)')
        assert_equal(repr(b), 'array([1.234e+09])')
        assert_equal(repr(np.array([0.])), 'array([0.])')
        assert_equal(repr(c),
            "array([1.        +1.j        , 1.12345679+1.12345679j])")
        assert_equal(repr(np.array([0., -0.])), 'array([ 0., -0.])')

        np.set_printoptions(sign=' ')
        assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
        assert_equal(repr(np.array(1.)), 'array( 1.)')
        assert_equal(repr(b), 'array([ 1.234e+09])')
        assert_equal(repr(c),
            "array([ 1.        +1.j        ,  1.12345679+1.12345679j])")
        assert_equal(repr(np.array([0., -0.])), 'array([ 0., -0.])')

        np.set_printoptions(sign='+')
        assert_equal(repr(a), 'array([+0., +1., +2., +3.])')
        assert_equal(repr(np.array(1.)), 'array(+1.)')
        assert_equal(repr(b), 'array([+1.234e+09])')
        assert_equal(repr(c),
            "array([+1.        +1.j        , +1.12345679+1.12345679j])")

        np.set_printoptions(legacy='1.13')
        assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
        assert_equal(repr(b),  'array([  1.23400000e+09])')
        assert_equal(repr(-b), 'array([ -1.23400000e+09])')
        assert_equal(repr(np.array(1.)), 'array(1.0)')
        assert_equal(repr(np.array([0.])), 'array([ 0.])')
        assert_equal(repr(c),
            "array([ 1.00000000+1.j        ,  1.12345679+1.12345679j])")
        # gh-10383
        assert_equal(str(np.array([-1., 10])), "[ -1.  10.]")

        assert_raises(TypeError, np.set_printoptions, wrongarg=True)

    def test_float_overflow_nowarn(self):
        # make sure internal computations in FloatingFormat don't
        # warn about overflow
        repr(np.array([1e4, 0.1], dtype='f2'))

    def test_sign_spacing_structured(self):
        a = np.ones(2, dtype='<f,<f')
        assert_equal(repr(a),
            "array([(1., 1.), (1., 1.)], dtype=[('f0', '<f4'), ('f1', '<f4')])")
        assert_equal(repr(a[0]), "(1., 1.)")

    def test_floatmode(self):
        x = np.array([0.6104, 0.922, 0.457, 0.0906, 0.3733, 0.007244,
                      0.5933, 0.947, 0.2383, 0.4226], dtype=np.float16)
        y = np.array([0.2918820979355541, 0.5064172631089138,
                      0.2848750619642916, 0.4342965294660567,
                      0.7326538397312751, 0.3459503329096204,
                      0.0862072768214508, 0.39112753029631175],
                      dtype=np.float64)
        z = np.arange(6, dtype=np.float16)/10
        c = np.array([1.0 + 1.0j, 1.123456789 + 1.123456789j], dtype='c16')

        # also make sure 1e23 is right (is between two fp numbers)
        w = np.array(['1e{}'.format(i) for i in range(25)], dtype=np.float64)
        # note: we construct w from the strings `1eXX` instead of doing
        # `10.**arange(24)` because it turns out the two are not equivalent in
        # python. On some architectures `1e23 != 10.**23`.
        wp = np.array([1.234e1, 1e2, 1e123])

        # unique mode
        np.set_printoptions(floatmode='unique')
        assert_equal(repr(x),
            "array([0.6104  , 0.922   , 0.457   , 0.0906  , 0.3733  , 0.007244,\n"
            "       0.5933  , 0.947   , 0.2383  , 0.4226  ], dtype=float16)")
        assert_equal(repr(y),
            "array([0.2918820979355541 , 0.5064172631089138 , 0.2848750619642916 ,\n"
            "       0.4342965294660567 , 0.7326538397312751 , 0.3459503329096204 ,\n"
            "       0.0862072768214508 , 0.39112753029631175])")
        assert_equal(repr(z),
            "array([0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float16)")
        assert_equal(repr(w),
            "array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,\n"
            "       1.e+08, 1.e+09, 1.e+10, 1.e+11, 1.e+12, 1.e+13, 1.e+14, 1.e+15,\n"
            "       1.e+16, 1.e+17, 1.e+18, 1.e+19, 1.e+20, 1.e+21, 1.e+22, 1.e+23,\n"
            "       1.e+24])")
        assert_equal(repr(wp), "array([1.234e+001, 1.000e+002, 1.000e+123])")
        assert_equal(repr(c),
            "array([1.         +1.j         , 1.123456789+1.123456789j])")

        # maxprec mode, precision=8
        np.set_printoptions(floatmode='maxprec', precision=8)
        assert_equal(repr(x),
            "array([0.6104  , 0.922   , 0.457   , 0.0906  , 0.3733  , 0.007244,\n"
            "       0.5933  , 0.947   , 0.2383  , 0.4226  ], dtype=float16)")
        assert_equal(repr(y),
            "array([0.2918821 , 0.50641726, 0.28487506, 0.43429653, 0.73265384,\n"
            "       0.34595033, 0.08620728, 0.39112753])")
        assert_equal(repr(z),
            "array([0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float16)")
        assert_equal(repr(w[::5]),
            "array([1.e+00, 1.e+05, 1.e+10, 1.e+15, 1.e+20])")
        assert_equal(repr(wp), "array([1.234e+001, 1.000e+002, 1.000e+123])")
        assert_equal(repr(c),
            "array([1.        +1.j        , 1.12345679+1.12345679j])")

        # fixed mode, precision=4
        np.set_printoptions(floatmode='fixed', precision=4)
        assert_equal(repr(x),
            "array([0.6104, 0.9219, 0.4570, 0.0906, 0.3733, 0.0072, 0.5933, 0.9468,\n"
            "       0.2383, 0.4226], dtype=float16)")
        assert_equal(repr(y),
            "array([0.2919, 0.5064, 0.2849, 0.4343, 0.7327, 0.3460, 0.0862, 0.3911])")
        assert_equal(repr(z),
            "array([0.0000, 0.1000, 0.2000, 0.3000, 0.3999, 0.5000], dtype=float16)")
        assert_equal(repr(w[::5]),
            "array([1.0000e+00, 1.0000e+05, 1.0000e+10, 1.0000e+15, 1.0000e+20])")
        assert_equal(repr(wp), "array([1.2340e+001, 1.0000e+002, 1.0000e+123])")
        assert_equal(repr(np.zeros(3)), "array([0.0000, 0.0000, 0.0000])")
        assert_equal(repr(c),
            "array([1.0000+1.0000j, 1.1235+1.1235j])")
        # for larger precision, representation error becomes more apparent:
        np.set_printoptions(floatmode='fixed', precision=8)
        assert_equal(repr(z),
            "array([0.00000000, 0.09997559, 0.19995117, 0.30004883, 0.39990234,\n"
            "       0.50000000], dtype=float16)")

        # maxprec_equal  mode, precision=8
        np.set_printoptions(floatmode='maxprec_equal', precision=8)
        assert_equal(repr(x),
            "array([0.610352, 0.921875, 0.457031, 0.090576, 0.373291, 0.007244,\n"
            "       0.593262, 0.946777, 0.238281, 0.422607], dtype=float16)")
        assert_equal(repr(y),
            "array([0.29188210, 0.50641726, 0.28487506, 0.43429653, 0.73265384,\n"
            "       0.34595033, 0.08620728, 0.39112753])")
        assert_equal(repr(z),
            "array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float16)")
        assert_equal(repr(w[::5]),
            "array([1.e+00, 1.e+05, 1.e+10, 1.e+15, 1.e+20])")
        assert_equal(repr(wp), "array([1.234e+001, 1.000e+002, 1.000e+123])")
        assert_equal(repr(c),
            "array([1.00000000+1.00000000j, 1.12345679+1.12345679j])")

        # test unique special case (gh-18609)
        a = np.float64.fromhex('-1p-97')
        assert_equal(np.float64(np.array2string(a, floatmode='unique')), a)

    def test_legacy_mode_scalars(self):
        # in legacy mode, str of floats get truncated, and complex scalars
        # use * for non-finite imaginary part
        np.set_printoptions(legacy='1.13')
        assert_equal(str(np.float64(1.123456789123456789)), '1.12345678912')
        assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nan*j)')

        np.set_printoptions(legacy=False)
        assert_equal(str(np.float64(1.123456789123456789)),
                     '1.1234567891234568')
        assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nanj)')

    def test_legacy_stray_comma(self):
        np.set_printoptions(legacy='1.13')
        assert_equal(str(np.arange(10000)), '[   0    1    2 ..., 9997 9998 9999]')

        np.set_printoptions(legacy=False)
        assert_equal(str(np.arange(10000)), '[   0    1    2 ... 9997 9998 9999]')

    def test_dtype_linewidth_wrapping(self):
        np.set_printoptions(linewidth=75)
        assert_equal(repr(np.arange(10,20., dtype='f4')),
            "array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.], dtype=float32)")
        assert_equal(repr(np.arange(10,23., dtype='f4')), textwrap.dedent("""\
            array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.],
                  dtype=float32)"""))

        styp = '<U4'
        assert_equal(repr(np.ones(3, dtype=styp)),
            "array(['1', '1', '1'], dtype='{}')".format(styp))
        assert_equal(repr(np.ones(12, dtype=styp)), textwrap.dedent("""\
            array(['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                  dtype='{}')""".format(styp)))

    def test_linewidth_repr(self):
        a = np.full(7, fill_value=2)
        np.set_printoptions(linewidth=17)
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2,
                   2])""")
        )
        np.set_printoptions(linewidth=17, legacy='1.13')
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2, 2])""")
        )

        a = np.full(8, fill_value=2)

        np.set_printoptions(linewidth=18, legacy=False)
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2,
                   2, 2])""")
        )

        np.set_printoptions(linewidth=18, legacy='1.13')
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2, 2,
                   2, 2, 2, 2])""")
        )

    def test_linewidth_str(self):
        a = np.full(18, fill_value=2)
        np.set_printoptions(linewidth=18)
        assert_equal(
            str(a),
            textwrap.dedent("""\
            [2 2 2 2 2 2 2 2
             2 2 2 2 2 2 2 2
             2 2]""")
        )
        np.set_printoptions(linewidth=18, legacy='1.13')
        assert_equal(
            str(a),
            textwrap.dedent("""\
            [2 2 2 2 2 2 2 2 2
             2 2 2 2 2 2 2 2 2]""")
        )

    def test_edgeitems(self):
        np.set_printoptions(edgeitems=1, threshold=1)
        a = np.arange(27).reshape((3, 3, 3))
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([[[ 0, ...,  2],
                    ...,
                    [ 6, ...,  8]],

                   ...,

                   [[18, ..., 20],
                    ...,
                    [24, ..., 26]]])""")
        )

        b = np.zeros((3, 3, 1, 1))
        assert_equal(
            repr(b),
            textwrap.dedent("""\
            array([[[[0.]],

                    ...,

                    [[0.]]],


                   ...,


                   [[[0.]],

                    ...,

                    [[0.]]]])""")
        )

        # 1.13 had extra trailing spaces, and was missing newlines
        np.set_printoptions(legacy='1.13')

        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([[[ 0, ...,  2],
                    ..., 
                    [ 6, ...,  8]],

                   ..., 
                   [[18, ..., 20],
                    ..., 
                    [24, ..., 26]]])""")
        )

        assert_equal(
            repr(b),
            textwrap.dedent("""\
            array([[[[ 0.]],

                    ..., 
                    [[ 0.]]],


                   ..., 
                   [[[ 0.]],

                    ..., 
                    [[ 0.]]]])""")
        )

    def test_bad_args(self):
        assert_raises(ValueError, np.set_printoptions, threshold=float('nan'))
        assert_raises(TypeError, np.set_printoptions, threshold='1')
        assert_raises(TypeError, np.set_printoptions, threshold=b'1')

        assert_raises(TypeError, np.set_printoptions, precision='1')
        assert_raises(TypeError, np.set_printoptions, precision=1.5)

def test_unicode_object_array():
    expected = "array(['é'], dtype=object)"
    x = np.array(['\xe9'], dtype=object)
    assert_equal(repr(x), expected)


class TestContextManager:
    def test_ctx_mgr(self):
        # test that context manager actually works
        with np.printoptions(precision=2):
            s = str(np.array([2.0]) / 3)
        assert_equal(s, '[0.67]')

    def test_ctx_mgr_restores(self):
        # test that print options are actually restrored
        opts = np.get_printoptions()
        with np.printoptions(precision=opts['precision'] - 1,
                             linewidth=opts['linewidth'] - 4):
            pass
        assert_equal(np.get_printoptions(), opts)

    def test_ctx_mgr_exceptions(self):
        # test that print options are restored even if an exception is raised
        opts = np.get_printoptions()
        try:
            with np.printoptions(precision=2, linewidth=11):
                raise ValueError
        except ValueError:
            pass
        assert_equal(np.get_printoptions(), opts)

    def test_ctx_mgr_as_smth(self):
        opts = {"precision": 2}
        with np.printoptions(**opts) as ctx:
            saved_opts = ctx.copy()
        assert_equal({k: saved_opts[k] for k in opts}, opts)
