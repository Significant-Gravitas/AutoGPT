import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises,
    assert_raises_regex
    )

kw_unicode_true = {'unicode': True}  # make 2to3 work properly
kw_unicode_false = {'unicode': False}

class TestBasic:
    def test_from_object_array(self):
        A = np.array([['abc', 2],
                      ['long   ', '0123456789']], dtype='O')
        B = np.char.array(A)
        assert_equal(B.dtype.itemsize, 10)
        assert_array_equal(B, [[b'abc', b'2'],
                               [b'long', b'0123456789']])

    def test_from_object_array_unicode(self):
        A = np.array([['abc', 'Sigma \u03a3'],
                      ['long   ', '0123456789']], dtype='O')
        assert_raises(ValueError, np.char.array, (A,))
        B = np.char.array(A, **kw_unicode_true)
        assert_equal(B.dtype.itemsize, 10 * np.array('a', 'U').dtype.itemsize)
        assert_array_equal(B, [['abc', 'Sigma \u03a3'],
                               ['long', '0123456789']])

    def test_from_string_array(self):
        A = np.array([[b'abc', b'foo'],
                      [b'long   ', b'0123456789']])
        assert_equal(A.dtype.type, np.string_)
        B = np.char.array(A)
        assert_array_equal(B, A)
        assert_equal(B.dtype, A.dtype)
        assert_equal(B.shape, A.shape)
        B[0, 0] = 'changed'
        assert_(B[0, 0] != A[0, 0])
        C = np.char.asarray(A)
        assert_array_equal(C, A)
        assert_equal(C.dtype, A.dtype)
        C[0, 0] = 'changed again'
        assert_(C[0, 0] != B[0, 0])
        assert_(C[0, 0] == A[0, 0])

    def test_from_unicode_array(self):
        A = np.array([['abc', 'Sigma \u03a3'],
                      ['long   ', '0123456789']])
        assert_equal(A.dtype.type, np.unicode_)
        B = np.char.array(A)
        assert_array_equal(B, A)
        assert_equal(B.dtype, A.dtype)
        assert_equal(B.shape, A.shape)
        B = np.char.array(A, **kw_unicode_true)
        assert_array_equal(B, A)
        assert_equal(B.dtype, A.dtype)
        assert_equal(B.shape, A.shape)

        def fail():
            np.char.array(A, **kw_unicode_false)

        assert_raises(UnicodeEncodeError, fail)

    def test_unicode_upconvert(self):
        A = np.char.array(['abc'])
        B = np.char.array(['\u03a3'])
        assert_(issubclass((A + B).dtype.type, np.unicode_))

    def test_from_string(self):
        A = np.char.array(b'abc')
        assert_equal(len(A), 1)
        assert_equal(len(A[0]), 3)
        assert_(issubclass(A.dtype.type, np.string_))

    def test_from_unicode(self):
        A = np.char.array('\u03a3')
        assert_equal(len(A), 1)
        assert_equal(len(A[0]), 1)
        assert_equal(A.itemsize, 4)
        assert_(issubclass(A.dtype.type, np.unicode_))

class TestVecString:
    def test_non_existent_method(self):

        def fail():
            _vec_string('a', np.string_, 'bogus')

        assert_raises(AttributeError, fail)

    def test_non_string_array(self):

        def fail():
            _vec_string(1, np.string_, 'strip')

        assert_raises(TypeError, fail)

    def test_invalid_args_tuple(self):

        def fail():
            _vec_string(['a'], np.string_, 'strip', 1)

        assert_raises(TypeError, fail)

    def test_invalid_type_descr(self):

        def fail():
            _vec_string(['a'], 'BOGUS', 'strip')

        assert_raises(TypeError, fail)

    def test_invalid_function_args(self):

        def fail():
            _vec_string(['a'], np.string_, 'strip', (1,))

        assert_raises(TypeError, fail)

    def test_invalid_result_type(self):

        def fail():
            _vec_string(['a'], np.int_, 'strip')

        assert_raises(TypeError, fail)

    def test_broadcast_error(self):

        def fail():
            _vec_string([['abc', 'def']], np.int_, 'find', (['a', 'd', 'j'],))

        assert_raises(ValueError, fail)


class TestWhitespace:
    def setup_method(self):
        self.A = np.array([['abc ', '123  '],
                           ['789 ', 'xyz ']]).view(np.chararray)
        self.B = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.chararray)

    def test1(self):
        assert_(np.all(self.A == self.B))
        assert_(np.all(self.A >= self.B))
        assert_(np.all(self.A <= self.B))
        assert_(not np.any(self.A > self.B))
        assert_(not np.any(self.A < self.B))
        assert_(not np.any(self.A != self.B))

class TestChar:
    def setup_method(self):
        self.A = np.array('abc1', dtype='c').view(np.chararray)

    def test_it(self):
        assert_equal(self.A.shape, (4,))
        assert_equal(self.A.upper()[:2].tobytes(), b'AB')

class TestComparisons:
    def setup_method(self):
        self.A = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.chararray)
        self.B = np.array([['efg', '123  '],
                           ['051', 'tuv']]).view(np.chararray)

    def test_not_equal(self):
        assert_array_equal((self.A != self.B), [[True, False], [True, True]])

    def test_equal(self):
        assert_array_equal((self.A == self.B), [[False, True], [False, False]])

    def test_greater_equal(self):
        assert_array_equal((self.A >= self.B), [[False, True], [True, True]])

    def test_less_equal(self):
        assert_array_equal((self.A <= self.B), [[True, True], [False, False]])

    def test_greater(self):
        assert_array_equal((self.A > self.B), [[False, False], [True, True]])

    def test_less(self):
        assert_array_equal((self.A < self.B), [[True, False], [False, False]])

    def test_type(self):
        out1 = np.char.equal(self.A, self.B)
        out2 = np.char.equal('a', 'a')
        assert_(isinstance(out1, np.ndarray))
        assert_(isinstance(out2, np.ndarray))

class TestComparisonsMixed1(TestComparisons):
    """Ticket #1276"""

    def setup_method(self):
        TestComparisons.setup_method(self)
        self.B = np.array([['efg', '123  '],
                           ['051', 'tuv']], np.unicode_).view(np.chararray)

class TestComparisonsMixed2(TestComparisons):
    """Ticket #1276"""

    def setup_method(self):
        TestComparisons.setup_method(self)
        self.A = np.array([['abc', '123'],
                           ['789', 'xyz']], np.unicode_).view(np.chararray)

class TestInformation:
    def setup_method(self):
        self.A = np.array([[' abc ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]).view(np.chararray)
        self.B = np.array([[' \u03a3 ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]).view(np.chararray)

    def test_len(self):
        assert_(issubclass(np.char.str_len(self.A).dtype.type, np.integer))
        assert_array_equal(np.char.str_len(self.A), [[5, 0], [5, 9], [12, 5]])
        assert_array_equal(np.char.str_len(self.B), [[3, 0], [5, 9], [12, 5]])

    def test_count(self):
        assert_(issubclass(self.A.count('').dtype.type, np.integer))
        assert_array_equal(self.A.count('a'), [[1, 0], [0, 1], [0, 0]])
        assert_array_equal(self.A.count('123'), [[0, 0], [1, 0], [1, 0]])
        # Python doesn't seem to like counting NULL characters
        # assert_array_equal(self.A.count('\0'), [[0, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.count('a', 0, 2), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.B.count('a'), [[0, 0], [0, 1], [0, 0]])
        assert_array_equal(self.B.count('123'), [[0, 0], [1, 0], [1, 0]])
        # assert_array_equal(self.B.count('\0'), [[0, 0], [0, 0], [1, 0]])

    def test_endswith(self):
        assert_(issubclass(self.A.endswith('').dtype.type, np.bool_))
        assert_array_equal(self.A.endswith(' '), [[1, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.endswith('3', 0, 3), [[0, 0], [1, 0], [1, 0]])

        def fail():
            self.A.endswith('3', 'fdjk')

        assert_raises(TypeError, fail)

    def test_find(self):
        assert_(issubclass(self.A.find('a').dtype.type, np.integer))
        assert_array_equal(self.A.find('a'), [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(self.A.find('3'), [[-1, -1], [2, -1], [2, -1]])
        assert_array_equal(self.A.find('a', 0, 2), [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(self.A.find(['1', 'P']), [[-1, -1], [0, -1], [0, 1]])

    def test_index(self):

        def fail():
            self.A.index('a')

        assert_raises(ValueError, fail)
        assert_(np.char.index('abcba', 'b') == 1)
        assert_(issubclass(np.char.index('abcba', 'b').dtype.type, np.integer))

    def test_isalnum(self):
        assert_(issubclass(self.A.isalnum().dtype.type, np.bool_))
        assert_array_equal(self.A.isalnum(), [[False, False], [True, True], [False, True]])

    def test_isalpha(self):
        assert_(issubclass(self.A.isalpha().dtype.type, np.bool_))
        assert_array_equal(self.A.isalpha(), [[False, False], [False, True], [False, True]])

    def test_isdigit(self):
        assert_(issubclass(self.A.isdigit().dtype.type, np.bool_))
        assert_array_equal(self.A.isdigit(), [[False, False], [True, False], [False, False]])

    def test_islower(self):
        assert_(issubclass(self.A.islower().dtype.type, np.bool_))
        assert_array_equal(self.A.islower(), [[True, False], [False, False], [False, False]])

    def test_isspace(self):
        assert_(issubclass(self.A.isspace().dtype.type, np.bool_))
        assert_array_equal(self.A.isspace(), [[False, False], [False, False], [False, False]])

    def test_istitle(self):
        assert_(issubclass(self.A.istitle().dtype.type, np.bool_))
        assert_array_equal(self.A.istitle(), [[False, False], [False, False], [False, False]])

    def test_isupper(self):
        assert_(issubclass(self.A.isupper().dtype.type, np.bool_))
        assert_array_equal(self.A.isupper(), [[False, False], [False, False], [False, True]])

    def test_rfind(self):
        assert_(issubclass(self.A.rfind('a').dtype.type, np.integer))
        assert_array_equal(self.A.rfind('a'), [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(self.A.rfind('3'), [[-1, -1], [2, -1], [6, -1]])
        assert_array_equal(self.A.rfind('a', 0, 2), [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(self.A.rfind(['1', 'P']), [[-1, -1], [0, -1], [0, 2]])

    def test_rindex(self):

        def fail():
            self.A.rindex('a')

        assert_raises(ValueError, fail)
        assert_(np.char.rindex('abcba', 'b') == 3)
        assert_(issubclass(np.char.rindex('abcba', 'b').dtype.type, np.integer))

    def test_startswith(self):
        assert_(issubclass(self.A.startswith('').dtype.type, np.bool_))
        assert_array_equal(self.A.startswith(' '), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.A.startswith('1', 0, 3), [[0, 0], [1, 0], [1, 0]])

        def fail():
            self.A.startswith('3', 'fdjk')

        assert_raises(TypeError, fail)


class TestMethods:
    def setup_method(self):
        self.A = np.array([[' abc ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']],
                          dtype='S').view(np.chararray)
        self.B = np.array([[' \u03a3 ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]).view(np.chararray)

    def test_capitalize(self):
        tgt = [[b' abc ', b''],
               [b'12345', b'Mixedcase'],
               [b'123 \t 345 \0 ', b'Upper']]
        assert_(issubclass(self.A.capitalize().dtype.type, np.string_))
        assert_array_equal(self.A.capitalize(), tgt)

        tgt = [[' \u03c3 ', ''],
               ['12345', 'Mixedcase'],
               ['123 \t 345 \0 ', 'Upper']]
        assert_(issubclass(self.B.capitalize().dtype.type, np.unicode_))
        assert_array_equal(self.B.capitalize(), tgt)

    def test_center(self):
        assert_(issubclass(self.A.center(10).dtype.type, np.string_))
        C = self.A.center([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        C = self.A.center(20, b'#')
        assert_(np.all(C.startswith(b'#')))
        assert_(np.all(C.endswith(b'#')))

        C = np.char.center(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'   FOO    ', b'        FOO         '],
               [b'      FOO      ', b'  FOO   ']]
        assert_(issubclass(C.dtype.type, np.string_))
        assert_array_equal(C, tgt)

    def test_decode(self):
        A = np.char.array([b'\\u03a3'])
        assert_(A.decode('unicode-escape')[0] == '\u03a3')

    def test_encode(self):
        B = self.B.encode('unicode_escape')
        assert_(B[0][0] == str(' \\u03a3 ').encode('latin1'))

    def test_expandtabs(self):
        T = self.A.expandtabs()
        assert_(T[2, 0] == b'123      345 \0')

    def test_join(self):
        # NOTE: list(b'123') == [49, 50, 51]
        #       so that b','.join(b'123') results to an error on Py3
        A0 = self.A.decode('ascii')

        A = np.char.join([',', '#'], A0)
        assert_(issubclass(A.dtype.type, np.unicode_))
        tgt = np.array([[' ,a,b,c, ', ''],
                        ['1,2,3,4,5', 'M#i#x#e#d#C#a#s#e'],
                        ['1,2,3, ,\t, ,3,4,5, ,\x00, ', 'U#P#P#E#R']])
        assert_array_equal(np.char.join([',', '#'], A0), tgt)

    def test_ljust(self):
        assert_(issubclass(self.A.ljust(10).dtype.type, np.string_))

        C = self.A.ljust([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        C = self.A.ljust(20, b'#')
        assert_array_equal(C.startswith(b'#'), [
                [False, True], [False, False], [False, False]])
        assert_(np.all(C.endswith(b'#')))

        C = np.char.ljust(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'FOO       ', b'FOO                 '],
               [b'FOO            ', b'FOO     ']]
        assert_(issubclass(C.dtype.type, np.string_))
        assert_array_equal(C, tgt)

    def test_lower(self):
        tgt = [[b' abc ', b''],
               [b'12345', b'mixedcase'],
               [b'123 \t 345 \0 ', b'upper']]
        assert_(issubclass(self.A.lower().dtype.type, np.string_))
        assert_array_equal(self.A.lower(), tgt)

        tgt = [[' \u03c3 ', ''],
               ['12345', 'mixedcase'],
               ['123 \t 345 \0 ', 'upper']]
        assert_(issubclass(self.B.lower().dtype.type, np.unicode_))
        assert_array_equal(self.B.lower(), tgt)

    def test_lstrip(self):
        tgt = [[b'abc ', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345 \0 ', b'UPPER']]
        assert_(issubclass(self.A.lstrip().dtype.type, np.string_))
        assert_array_equal(self.A.lstrip(), tgt)

        tgt = [[b' abc', b''],
               [b'2345', b'ixedCase'],
               [b'23 \t 345 \x00', b'UPPER']]
        assert_array_equal(self.A.lstrip([b'1', b'M']), tgt)

        tgt = [['\u03a3 ', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345 \0 ', 'UPPER']]
        assert_(issubclass(self.B.lstrip().dtype.type, np.unicode_))
        assert_array_equal(self.B.lstrip(), tgt)

    def test_partition(self):
        P = self.A.partition([b'3', b'M'])
        tgt = [[(b' abc ', b'', b''), (b'', b'', b'')],
               [(b'12', b'3', b'45'), (b'', b'M', b'ixedCase')],
               [(b'12', b'3', b' \t 345 \0 '), (b'UPPER', b'', b'')]]
        assert_(issubclass(P.dtype.type, np.string_))
        assert_array_equal(P, tgt)

    def test_replace(self):
        R = self.A.replace([b'3', b'a'],
                           [b'##########', b'@'])
        tgt = [[b' abc ', b''],
               [b'12##########45', b'MixedC@se'],
               [b'12########## \t ##########45 \x00', b'UPPER']]
        assert_(issubclass(R.dtype.type, np.string_))
        assert_array_equal(R, tgt)

    def test_rjust(self):
        assert_(issubclass(self.A.rjust(10).dtype.type, np.string_))

        C = self.A.rjust([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        C = self.A.rjust(20, b'#')
        assert_(np.all(C.startswith(b'#')))
        assert_array_equal(C.endswith(b'#'),
                           [[False, True], [False, False], [False, False]])

        C = np.char.rjust(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'       FOO', b'                 FOO'],
               [b'            FOO', b'     FOO']]
        assert_(issubclass(C.dtype.type, np.string_))
        assert_array_equal(C, tgt)

    def test_rpartition(self):
        P = self.A.rpartition([b'3', b'M'])
        tgt = [[(b'', b'', b' abc '), (b'', b'', b'')],
               [(b'12', b'3', b'45'), (b'', b'M', b'ixedCase')],
               [(b'123 \t ', b'3', b'45 \0 '), (b'', b'', b'UPPER')]]
        assert_(issubclass(P.dtype.type, np.string_))
        assert_array_equal(P, tgt)

    def test_rsplit(self):
        A = self.A.rsplit(b'3')
        tgt = [[[b' abc '], [b'']],
               [[b'12', b'45'], [b'MixedCase']],
               [[b'12', b' \t ', b'45 \x00 '], [b'UPPER']]]
        assert_(issubclass(A.dtype.type, np.object_))
        assert_equal(A.tolist(), tgt)

    def test_rstrip(self):
        assert_(issubclass(self.A.rstrip().dtype.type, np.string_))

        tgt = [[b' abc', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345', b'UPPER']]
        assert_array_equal(self.A.rstrip(), tgt)

        tgt = [[b' abc ', b''],
               [b'1234', b'MixedCase'],
               [b'123 \t 345 \x00', b'UPP']
               ]
        assert_array_equal(self.A.rstrip([b'5', b'ER']), tgt)

        tgt = [[' \u03a3', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345', 'UPPER']]
        assert_(issubclass(self.B.rstrip().dtype.type, np.unicode_))
        assert_array_equal(self.B.rstrip(), tgt)

    def test_strip(self):
        tgt = [[b'abc', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345', b'UPPER']]
        assert_(issubclass(self.A.strip().dtype.type, np.string_))
        assert_array_equal(self.A.strip(), tgt)

        tgt = [[b' abc ', b''],
               [b'234', b'ixedCas'],
               [b'23 \t 345 \x00', b'UPP']]
        assert_array_equal(self.A.strip([b'15', b'EReM']), tgt)

        tgt = [['\u03a3', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345', 'UPPER']]
        assert_(issubclass(self.B.strip().dtype.type, np.unicode_))
        assert_array_equal(self.B.strip(), tgt)

    def test_split(self):
        A = self.A.split(b'3')
        tgt = [
               [[b' abc '], [b'']],
               [[b'12', b'45'], [b'MixedCase']],
               [[b'12', b' \t ', b'45 \x00 '], [b'UPPER']]]
        assert_(issubclass(A.dtype.type, np.object_))
        assert_equal(A.tolist(), tgt)

    def test_splitlines(self):
        A = np.char.array(['abc\nfds\nwer']).splitlines()
        assert_(issubclass(A.dtype.type, np.object_))
        assert_(A.shape == (1,))
        assert_(len(A[0]) == 3)

    def test_swapcase(self):
        tgt = [[b' ABC ', b''],
               [b'12345', b'mIXEDcASE'],
               [b'123 \t 345 \0 ', b'upper']]
        assert_(issubclass(self.A.swapcase().dtype.type, np.string_))
        assert_array_equal(self.A.swapcase(), tgt)

        tgt = [[' \u03c3 ', ''],
               ['12345', 'mIXEDcASE'],
               ['123 \t 345 \0 ', 'upper']]
        assert_(issubclass(self.B.swapcase().dtype.type, np.unicode_))
        assert_array_equal(self.B.swapcase(), tgt)

    def test_title(self):
        tgt = [[b' Abc ', b''],
               [b'12345', b'Mixedcase'],
               [b'123 \t 345 \0 ', b'Upper']]
        assert_(issubclass(self.A.title().dtype.type, np.string_))
        assert_array_equal(self.A.title(), tgt)

        tgt = [[' \u03a3 ', ''],
               ['12345', 'Mixedcase'],
               ['123 \t 345 \0 ', 'Upper']]
        assert_(issubclass(self.B.title().dtype.type, np.unicode_))
        assert_array_equal(self.B.title(), tgt)

    def test_upper(self):
        tgt = [[b' ABC ', b''],
               [b'12345', b'MIXEDCASE'],
               [b'123 \t 345 \0 ', b'UPPER']]
        assert_(issubclass(self.A.upper().dtype.type, np.string_))
        assert_array_equal(self.A.upper(), tgt)

        tgt = [[' \u03a3 ', ''],
               ['12345', 'MIXEDCASE'],
               ['123 \t 345 \0 ', 'UPPER']]
        assert_(issubclass(self.B.upper().dtype.type, np.unicode_))
        assert_array_equal(self.B.upper(), tgt)

    def test_isnumeric(self):

        def fail():
            self.A.isnumeric()

        assert_raises(TypeError, fail)
        assert_(issubclass(self.B.isnumeric().dtype.type, np.bool_))
        assert_array_equal(self.B.isnumeric(), [
                [False, False], [True, False], [False, False]])

    def test_isdecimal(self):

        def fail():
            self.A.isdecimal()

        assert_raises(TypeError, fail)
        assert_(issubclass(self.B.isdecimal().dtype.type, np.bool_))
        assert_array_equal(self.B.isdecimal(), [
                [False, False], [True, False], [False, False]])


class TestOperations:
    def setup_method(self):
        self.A = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.chararray)
        self.B = np.array([['efg', '456'],
                           ['051', 'tuv']]).view(np.chararray)

    def test_add(self):
        AB = np.array([['abcefg', '123456'],
                       ['789051', 'xyztuv']]).view(np.chararray)
        assert_array_equal(AB, (self.A + self.B))
        assert_(len((self.A + self.B)[0][0]) == 6)

    def test_radd(self):
        QA = np.array([['qabc', 'q123'],
                       ['q789', 'qxyz']]).view(np.chararray)
        assert_array_equal(QA, ('q' + self.A))

    def test_mul(self):
        A = self.A
        for r in (2, 3, 5, 7, 197):
            Ar = np.array([[A[0, 0]*r, A[0, 1]*r],
                           [A[1, 0]*r, A[1, 1]*r]]).view(np.chararray)

            assert_array_equal(Ar, (self.A * r))

        for ob in [object(), 'qrs']:
            with assert_raises_regex(ValueError,
                                     'Can only multiply by integers'):
                A*ob

    def test_rmul(self):
        A = self.A
        for r in (2, 3, 5, 7, 197):
            Ar = np.array([[A[0, 0]*r, A[0, 1]*r],
                           [A[1, 0]*r, A[1, 1]*r]]).view(np.chararray)
            assert_array_equal(Ar, (r * self.A))

        for ob in [object(), 'qrs']:
            with assert_raises_regex(ValueError,
                                     'Can only multiply by integers'):
                ob * A

    def test_mod(self):
        """Ticket #856"""
        F = np.array([['%d', '%f'], ['%s', '%r']]).view(np.chararray)
        C = np.array([[3, 7], [19, 1]])
        FC = np.array([['3', '7.000000'],
                       ['19', '1']]).view(np.chararray)
        assert_array_equal(FC, F % C)

        A = np.array([['%.3f', '%d'], ['%s', '%r']]).view(np.chararray)
        A1 = np.array([['1.000', '1'], ['1', '1']]).view(np.chararray)
        assert_array_equal(A1, (A % 1))

        A2 = np.array([['1.000', '2'], ['3', '4']]).view(np.chararray)
        assert_array_equal(A2, (A % [[1, 2], [3, 4]]))

    def test_rmod(self):
        assert_(("%s" % self.A) == str(self.A))
        assert_(("%r" % self.A) == repr(self.A))

        for ob in [42, object()]:
            with assert_raises_regex(
                    TypeError, "unsupported operand type.* and 'chararray'"):
                ob % self.A

    def test_slice(self):
        """Regression test for https://github.com/numpy/numpy/issues/5982"""

        arr = np.array([['abc ', 'def '], ['geh ', 'ijk ']],
                       dtype='S4').view(np.chararray)
        sl1 = arr[:]
        assert_array_equal(sl1, arr)
        assert_(sl1.base is arr)
        assert_(sl1.base.base is arr.base)

        sl2 = arr[:, :]
        assert_array_equal(sl2, arr)
        assert_(sl2.base is arr)
        assert_(sl2.base.base is arr.base)

        assert_(arr[0, 0] == b'abc')


def test_empty_indexing():
    """Regression test for ticket 1948."""
    # Check that indexing a chararray with an empty list/array returns an
    # empty chararray instead of a chararray with a single empty string in it.
    s = np.chararray((4,))
    assert_(s[[]].size == 0)
