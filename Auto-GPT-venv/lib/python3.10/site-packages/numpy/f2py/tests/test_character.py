import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util


class TestCharacterString(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    suffix = '.f90'
    fprefix = 'test_character_string'
    length_list = ['1', '3', 'star']

    code = ''
    for length in length_list:
        fsuffix = length
        clength = dict(star='(*)').get(length, length)

        code += textwrap.dedent(f"""

        subroutine {fprefix}_input_{fsuffix}(c, o, n)
          character*{clength}, intent(in) :: c
          integer n
          !f2py integer, depend(c), intent(hide) :: n = slen(c)
          integer*1, dimension(n) :: o
          !f2py intent(out) o
          o = transfer(c, o)
        end subroutine {fprefix}_input_{fsuffix}

        subroutine {fprefix}_output_{fsuffix}(c, o, n)
          character*{clength}, intent(out) :: c
          integer n
          integer*1, dimension(n), intent(in) :: o
          !f2py integer, depend(o), intent(hide) :: n = len(o)
          c = transfer(o, c)
        end subroutine {fprefix}_output_{fsuffix}

        subroutine {fprefix}_array_input_{fsuffix}(c, o, m, n)
          integer m, i, n
          character*{clength}, intent(in), dimension(m) :: c
          !f2py integer, depend(c), intent(hide) :: m = len(c)
          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)
          integer*1, dimension(m, n), intent(out) :: o
          do i=1,m
            o(i, :) = transfer(c(i), o(i, :))
          end do
        end subroutine {fprefix}_array_input_{fsuffix}

        subroutine {fprefix}_array_output_{fsuffix}(c, o, m, n)
          character*{clength}, intent(out), dimension(m) :: c
          integer n
          integer*1, dimension(m, n), intent(in) :: o
          !f2py character(f2py_len=n) :: c
          !f2py integer, depend(o), intent(hide) :: m = len(o)
          !f2py integer, depend(o), intent(hide) :: n = shape(o, 1)
          do i=1,m
            c(i) = transfer(o(i, :), c(i))
          end do
        end subroutine {fprefix}_array_output_{fsuffix}

        subroutine {fprefix}_2d_array_input_{fsuffix}(c, o, m1, m2, n)
          integer m1, m2, i, j, n
          character*{clength}, intent(in), dimension(m1, m2) :: c
          !f2py integer, depend(c), intent(hide) :: m1 = len(c)
          !f2py integer, depend(c), intent(hide) :: m2 = shape(c, 1)
          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)
          integer*1, dimension(m1, m2, n), intent(out) :: o
          do i=1,m1
            do j=1,m2
              o(i, j, :) = transfer(c(i, j), o(i, j, :))
            end do
          end do
        end subroutine {fprefix}_2d_array_input_{fsuffix}
        """)

    @pytest.mark.parametrize("length", length_list)
    def test_input(self, length):
        fsuffix = {'(*)': 'star'}.get(length, length)
        f = getattr(self.module, self.fprefix + '_input_' + fsuffix)

        a = {'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length]

        assert_array_equal(f(a), np.array(list(map(ord, a)), dtype='u1'))

    @pytest.mark.parametrize("length", length_list[:-1])
    def test_output(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_output_' + fsuffix)

        a = {'1': 'a', '3': 'abc'}[length]

        assert_array_equal(f(np.array(list(map(ord, a)), dtype='u1')),
                           a.encode())

    @pytest.mark.parametrize("length", length_list)
    def test_array_input(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_array_input_' + fsuffix)

        a = np.array([{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
                      {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length],
                      ], dtype='S')

        expected = np.array([[c for c in s] for s in a], dtype='u1')
        assert_array_equal(f(a), expected)

    @pytest.mark.parametrize("length", length_list)
    def test_array_output(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_array_output_' + fsuffix)

        expected = np.array(
            [{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
             {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]], dtype='S')

        a = np.array([[c for c in s] for s in expected], dtype='u1')
        assert_array_equal(f(a), expected)

    @pytest.mark.parametrize("length", length_list)
    def test_2d_array_input(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_2d_array_input_' + fsuffix)

        a = np.array([[{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
                       {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]],
                      [{'1': 'f', '3': 'fgh', 'star': 'fghij' * 3}[length],
                       {'1': 'F', '3': 'FGH', 'star': 'FGHIJ' * 3}[length]]],
                     dtype='S')
        expected = np.array([[[c for c in item] for item in row] for row in a],
                            dtype='u1', order='F')
        assert_array_equal(f(a), expected)


class TestCharacter(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    suffix = '.f90'
    fprefix = 'test_character'

    code = textwrap.dedent(f"""
       subroutine {fprefix}_input(c, o)
          character, intent(in) :: c
          integer*1 o
          !f2py intent(out) o
          o = transfer(c, o)
       end subroutine {fprefix}_input

       subroutine {fprefix}_output(c, o)
          character :: c
          integer*1, intent(in) :: o
          !f2py intent(out) c
          c = transfer(o, c)
       end subroutine {fprefix}_output

       subroutine {fprefix}_input_output(c, o)
          character, intent(in) :: c
          character o
          !f2py intent(out) o
          o = c
       end subroutine {fprefix}_input_output

       subroutine {fprefix}_inout(c, n)
          character :: c, n
          !f2py intent(in) n
          !f2py intent(inout) c
          c = n
       end subroutine {fprefix}_inout

       function {fprefix}_return(o) result (c)
          character :: c
          character, intent(in) :: o
          c = transfer(o, c)
       end function {fprefix}_return

       subroutine {fprefix}_array_input(c, o)
          character, intent(in) :: c(3)
          integer*1 o(3)
          !f2py intent(out) o
          integer i
          do i=1,3
            o(i) = transfer(c(i), o(i))
          end do
       end subroutine {fprefix}_array_input

       subroutine {fprefix}_2d_array_input(c, o)
          character, intent(in) :: c(2, 3)
          integer*1 o(2, 3)
          !f2py intent(out) o
          integer i, j
          do i=1,2
            do j=1,3
              o(i, j) = transfer(c(i, j), o(i, j))
            end do
          end do
       end subroutine {fprefix}_2d_array_input

       subroutine {fprefix}_array_output(c, o)
          character :: c(3)
          integer*1, intent(in) :: o(3)
          !f2py intent(out) c
          do i=1,3
            c(i) = transfer(o(i), c(i))
          end do
       end subroutine {fprefix}_array_output

       subroutine {fprefix}_array_inout(c, n)
          character :: c(3), n(3)
          !f2py intent(in) n(3)
          !f2py intent(inout) c(3)
          do i=1,3
            c(i) = n(i)
          end do
       end subroutine {fprefix}_array_inout

       subroutine {fprefix}_2d_array_inout(c, n)
          character :: c(2, 3), n(2, 3)
          !f2py intent(in) n(2, 3)
          !f2py intent(inout) c(2. 3)
          integer i, j
          do i=1,2
            do j=1,3
              c(i, j) = n(i, j)
            end do
          end do
       end subroutine {fprefix}_2d_array_inout

       function {fprefix}_array_return(o) result (c)
          character, dimension(3) :: c
          character, intent(in) :: o(3)
          do i=1,3
            c(i) = o(i)
          end do
       end function {fprefix}_array_return

       function {fprefix}_optional(o) result (c)
          character, intent(in) :: o
          !f2py character o = "a"
          character :: c
          c = o
       end function {fprefix}_optional
    """)

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_input')

        assert_equal(f(np.array('a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(b'a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(['a'], dtype=dtype)), ord('a'))
        assert_equal(f(np.array('abc', dtype=dtype)), ord('a'))
        assert_equal(f(np.array([['a']], dtype=dtype)), ord('a'))

    def test_input_varia(self):
        f = getattr(self.module, self.fprefix + '_input')

        assert_equal(f('a'), ord('a'))
        assert_equal(f(b'a'), ord(b'a'))
        assert_equal(f(''), 0)
        assert_equal(f(b''), 0)
        assert_equal(f(b'\0'), 0)
        assert_equal(f('ab'), ord('a'))
        assert_equal(f(b'ab'), ord('a'))
        assert_equal(f(['a']), ord('a'))

        assert_equal(f(np.array(b'a')), ord('a'))
        assert_equal(f(np.array([b'a'])), ord('a'))
        a = np.array('a')
        assert_equal(f(a), ord('a'))
        a = np.array(['a'])
        assert_equal(f(a), ord('a'))

        try:
            f([])
        except IndexError as msg:
            if not str(msg).endswith(' got 0-list'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on empty list')

        try:
            f(97)
        except TypeError as msg:
            if not str(msg).endswith(' got int instance'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on int value')

    @pytest.mark.parametrize("dtype", ['c', 'S1', 'U1'])
    def test_array_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_array_input')

        assert_array_equal(f(np.array(['a', 'b', 'c'], dtype=dtype)),
                           np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f(np.array([b'a', b'b', b'c'], dtype=dtype)),
                           np.array(list(map(ord, 'abc')), dtype='i1'))

    def test_array_input_varia(self):
        f = getattr(self.module, self.fprefix + '_array_input')
        assert_array_equal(f(['a', 'b', 'c']),
                           np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f([b'a', b'b', b'c']),
                           np.array(list(map(ord, 'abc')), dtype='i1'))

        try:
            f(['a', 'b', 'c', 'd'])
        except ValueError as msg:
            if not str(msg).endswith(
                    'th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(
                f'{f.__name__} should have failed on wrong input')

    @pytest.mark.parametrize("dtype", ['c', 'S1', 'U1'])
    def test_2d_array_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_2d_array_input')

        a = np.array([['a', 'b', 'c'],
                      ['d', 'e', 'f']], dtype=dtype, order='F')
        expected = a.view(np.uint32 if dtype == 'U1' else np.uint8)
        assert_array_equal(f(a), expected)

    def test_output(self):
        f = getattr(self.module, self.fprefix + '_output')

        assert_equal(f(ord(b'a')), b'a')
        assert_equal(f(0), b'\0')

    def test_array_output(self):
        f = getattr(self.module, self.fprefix + '_array_output')

        assert_array_equal(f(list(map(ord, 'abc'))),
                           np.array(list('abc'), dtype='S1'))

    def test_input_output(self):
        f = getattr(self.module, self.fprefix + '_input_output')

        assert_equal(f(b'a'), b'a')
        assert_equal(f('a'), b'a')
        assert_equal(f(''), b'\0')

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_inout')

        a = np.array(list('abc'), dtype=dtype)
        f(a, 'A')
        assert_array_equal(a, np.array(list('Abc'), dtype=a.dtype))
        f(a[1:], 'B')
        assert_array_equal(a, np.array(list('ABc'), dtype=a.dtype))

        a = np.array(['abc'], dtype=dtype)
        f(a, 'A')
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

    def test_inout_varia(self):
        f = getattr(self.module, self.fprefix + '_inout')
        a = np.array('abc', dtype='S3')
        f(a, 'A')
        assert_array_equal(a, np.array('Abc', dtype=a.dtype))

        a = np.array(['abc'], dtype='S3')
        f(a, 'A')
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

        try:
            f('abc', 'A')
        except ValueError as msg:
            if not str(msg).endswith(' got 3-str'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on str value')

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_array_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_array_inout')
        n = np.array(['A', 'B', 'C'], dtype=dtype, order='F')

        a = np.array(['a', 'b', 'c'], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, n)

        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype)
        f(a[1:], n)
        assert_array_equal(a, np.array(['a', 'A', 'B', 'C'], dtype=dtype))

        a = np.array([['a', 'b', 'c']], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, np.array([['A', 'B', 'C']], dtype=dtype))

        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype, order='F')
        try:
            f(a, n)
        except ValueError as msg:
            if not str(msg).endswith(
                    'th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(
                f'{f.__name__} should have failed on wrong input')

    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_2d_array_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_2d_array_inout')
        n = np.array([['A', 'B', 'C'],
                      ['D', 'E', 'F']],
                     dtype=dtype, order='F')
        a = np.array([['a', 'b', 'c'],
                      ['d', 'e', 'f']],
                     dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, n)

    def test_return(self):
        f = getattr(self.module, self.fprefix + '_return')

        assert_equal(f('a'), b'a')

    @pytest.mark.skip('fortran function returning array segfaults')
    def test_array_return(self):
        f = getattr(self.module, self.fprefix + '_array_return')

        a = np.array(list('abc'), dtype='S1')
        assert_array_equal(f(a), a)

    def test_optional(self):
        f = getattr(self.module, self.fprefix + '_optional')

        assert_equal(f(), b"a")
        assert_equal(f(b'B'), b"B")


class TestMiscCharacter(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    suffix = '.f90'
    fprefix = 'test_misc_character'

    code = textwrap.dedent(f"""
       subroutine {fprefix}_gh18684(x, y, m)
         character(len=5), dimension(m), intent(in) :: x
         character*5, dimension(m), intent(out) :: y
         integer i, m
         !f2py integer, intent(hide), depend(x) :: m = f2py_len(x)
         do i=1,m
           y(i) = x(i)
         end do
       end subroutine {fprefix}_gh18684

       subroutine {fprefix}_gh6308(x, i)
         integer i
         !f2py check(i>=0 && i<12) i
         character*5 name, x
         common name(12)
         name(i + 1) = x
       end subroutine {fprefix}_gh6308

       subroutine {fprefix}_gh4519(x)
         character(len=*), intent(in) :: x(:)
         !f2py intent(out) x
         integer :: i
         do i=1, size(x)
           print*, "x(",i,")=", x(i)
         end do
       end subroutine {fprefix}_gh4519

       pure function {fprefix}_gh3425(x) result (y)
         character(len=*), intent(in) :: x
         character(len=len(x)) :: y
         integer :: i
         do i = 1, len(x)
           j = iachar(x(i:i))
           if (j>=iachar("a") .and. j<=iachar("z") ) then
             y(i:i) = achar(j-32)
           else
             y(i:i) = x(i:i)
           endif
         end do
       end function {fprefix}_gh3425

       subroutine {fprefix}_character_bc_new(x, y, z)
         character, intent(in) :: x
         character, intent(out) :: y
         !f2py character, depend(x) :: y = x
         !f2py character, dimension((x=='a'?1:2)), depend(x), intent(out) :: z
         character, dimension(*) :: z
         !f2py character, optional, check(x == 'a' || x == 'b') :: x = 'a'
         !f2py callstatement (*f2py_func)(&x, &y, z)
         !f2py callprotoargument character*, character*, character*
         if (y.eq.x) then
           y = x
         else
           y = 'e'
         endif
         z(1) = 'c'
       end subroutine {fprefix}_character_bc_new

       subroutine {fprefix}_character_bc_old(x, y, z)
         character, intent(in) :: x
         character, intent(out) :: y
         !f2py character, depend(x) :: y = x[0]
         !f2py character, dimension((*x=='a'?1:2)), depend(x), intent(out) :: z
         character, dimension(*) :: z
         !f2py character, optional, check(*x == 'a' || x[0] == 'b') :: x = 'a'
         !f2py callstatement (*f2py_func)(x, y, z)
         !f2py callprotoargument char*, char*, char*
          if (y.eq.x) then
           y = x
         else
           y = 'e'
         endif
         z(1) = 'c'
       end subroutine {fprefix}_character_bc_old
    """)

    def test_gh18684(self):
        # Test character(len=5) and character*5 usages
        f = getattr(self.module, self.fprefix + '_gh18684')
        x = np.array(["abcde", "fghij"], dtype='S5')
        y = f(x)

        assert_array_equal(x, y)

    def test_gh6308(self):
        # Test character string array in a common block
        f = getattr(self.module, self.fprefix + '_gh6308')

        assert_equal(self.module._BLNK_.name.dtype, np.dtype('S5'))
        assert_equal(len(self.module._BLNK_.name), 12)
        f("abcde", 0)
        assert_equal(self.module._BLNK_.name[0], b"abcde")
        f("12345", 5)
        assert_equal(self.module._BLNK_.name[5], b"12345")

    def test_gh4519(self):
        # Test array of assumed length strings
        f = getattr(self.module, self.fprefix + '_gh4519')

        for x, expected in [
                ('a', dict(shape=(), dtype=np.dtype('S1'))),
                ('text', dict(shape=(), dtype=np.dtype('S4'))),
                (np.array(['1', '2', '3'], dtype='S1'),
                 dict(shape=(3,), dtype=np.dtype('S1'))),
                (['1', '2', '34'],
                 dict(shape=(3,), dtype=np.dtype('S2'))),
                (['', ''], dict(shape=(2,), dtype=np.dtype('S1')))]:
            r = f(x)
            for k, v in expected.items():
                assert_equal(getattr(r, k), v)

    def test_gh3425(self):
        # Test returning a copy of assumed length string
        f = getattr(self.module, self.fprefix + '_gh3425')
        # f is equivalent to bytes.upper

        assert_equal(f('abC'), b'ABC')
        assert_equal(f(''), b'')
        assert_equal(f('abC12d'), b'ABC12D')

    @pytest.mark.parametrize("state", ['new', 'old'])
    def test_character_bc(self, state):
        f = getattr(self.module, self.fprefix + '_character_bc_' + state)

        c, a = f()
        assert_equal(c, b'a')
        assert_equal(len(a), 1)

        c, a = f(b'b')
        assert_equal(c, b'b')
        assert_equal(len(a), 2)

        assert_raises(Exception, lambda: f(b'c'))
